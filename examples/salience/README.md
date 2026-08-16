# salience — agent memory as an incrementally maintained view

**Category: agent support.**

An agent's memory is not a vector index. In a working agent runtime, recall is
a weighted product over several signals at once:

```
salience = similarity × reinforcement × decay × keyword_boost × kind_weight
```

This project rebuilds that recall on `fold` in order to answer one question:

> **How much of an agent's recall function can a materialized view actually
> hold?**

The answer is four terms out of five, and the fifth is interesting.

| term | source | maintained by fold? |
|---|---|---|
| `similarity` | HNSW over ese embeddings (`terminal::search::Hnsw`) | yes |
| `keyword_boost` | BM25 (`terminal::search::Bm25`) | yes |
| `reinforcement` | per-memory counter (`Aggregate`) | yes |
| `kind_weight` | constant on the record (`terminal::Table`) | yes |
| `decay` | the wall clock | **no** |

## Run it

```bash
cargo run -p salience
```

A scripted demo, then an interactive prompt (`<query>` recalls, `add <text>`,
`use <id>` reinforces, `rm <id>` forgets).

```bash
cargo run --release -p salience -- bench   # the measurements below
cargo test -p salience                     # properties
cargo test -p salience -- --ignored        # the upstream defect, below
```

## The finding: decay is not materializable, and it is not for the obvious reason

The obvious argument is "decay depends on the clock, so it can't be folded."
That argument is wrong, and the demo shows why.

With a **single global half-life**, exponential decay is rank-*preserving*.
The ratio between two memories' decay is

```
exp(-(t-a)/T) / exp(-(t-b)/T)  =  exp((b-a)/T)
```

— a constant. The clock cancels. You could fold decay in once at write time
and the ordering would stay correct forever, so a materialized view *would*
suffice.

It stops being true the moment half-lives differ. Give a standing preference a
365-day half-life and an incidental episode a 21-day one, and the ratio becomes
`exp(-t(1/T₁ - 1/T₂))`, which moves with `t`. Memories now cross over. The
demo's third step issues the same query against the same database with **zero
writes**, only a clock 180 days later:

```
== recall: "why is the dashboard slow" ==
    total     mat decay    sim    kw reinf  kind   memory
    0.927   1.378  0.67   0.49  1.27  2.61  0.85   [3/epis] customer complained that the dashboard takes ten seconds to load
    0.402   0.469  0.86   0.37  1.28  1.00  1.00   [12/fact] the dashboard render blocks on a synchronous aggregate query
    0.310   0.492  0.63   0.34  1.25  1.00  1.15   [11/proc] to profile a slow page, start with the browser waterfall...

-- no writes at all; only the clock advances by 180 days --

    total     mat decay    sim    kw reinf  kind   memory
    0.155   0.492  0.31   0.34  1.25  1.00  1.15   [11/proc] to profile a slow page, start with the browser waterfall...
    0.101   0.469  0.21   0.37  1.28  1.00  1.00   [12/fact] the dashboard render blocks on a synchronous aggregate query
    0.066   0.372  0.18   0.30  1.26  1.00  1.00   [2/fact] the postgres database was slow because the orders table was missing an index
```

The `mat` column — everything fold holds — is **identical** in both. The heavily
reinforced episode that led by 2× has fallen out of the top three, and a
procedure has taken first place. Nothing was written.

So the shape an agent memory wants is not "materialize everything" and not
"compute at read time", but both: a cheap materialized **candidate** stage that
fold maintains, and a **rescore** on top that applies the clock. That is the
same two-stage shape production systems reach for with a vector prefilter and
an application-side ranker — this just says *why* it is forced.

## Design note: the stream carries an enum, not a record

The obvious modelling is `KeyedStream<u64, Memory>` with a reinforcement count
on the record. It is wrong. fold's unit of retraction is the whole record, so
bumping a counter retracts and reinserts the memory through *every* branch —
re-tokenizing it and **re-embedding it**. Reinforcement is the hottest field;
embedding is the most expensive branch.

So the stream carries an `Event` enum and each branch opens with a `FilterMap`.
`Reinforced` carries only an id, so it reaches the counter and nothing else.
Measured, that is **6× cheaper** per reinforcement (1.02 µs vs 6.15 µs).

The cost is that `forget` gives up `KeyedStream`'s retract-by-key and must push
its own compensating deltas — reading current state through `Tx::rtx` inside
the write transaction, so the read and the retraction cannot disagree.

## Measurements

Apple M4 Pro, `--release`, ese dim 512, median of 5.

**1. Splitting the event type pays.**

```
500 reinforcements over 500 memories
id-only event    509.96µs  (420.00µs..537.88µs)     1.02 us/op
record update      3.08ms  (2.87ms..3.15ms)         6.15 us/op   -> 6.0x
```

**2. Incremental maintenance costs the same per event as a rebuild.**

```
log of 1000 memories:  incremental 528.98 us/event   refold 713.99 us/event
log of 4000 memories:  incremental 520.44 us/event   refold 559.41 us/event
```

This is the honest result and it is not the flattering one. fold buys **no
per-event advantage** over refolding — roughly 520–560 µs either way at 4000 events, dominated by
embedding and HNSW insertion. What incrementality buys is that you only touch
the delta: the 215× saving at 4000 events is exactly 4000/20 events not
processed. The useful property is that maintenance is linear and predictable in
the number of events, so rebuild cost is a straightforward function of log
length.

**3. Retraction does not degrade recall.**

```
2000 live vectors, 6 rounds of 200 deletes + 200 inserts
round   live   recall@10
    0   2000      1.000
    ...
    6   2000      1.000
```

Recall@10 against brute-force cosine ground truth, flat across 1200 deletions
and 1200 insertions. fold's HNSW removes nodes from the graph rather than
tombstoning them, and this is what that buys. Caveat stated plainly: this
corpus is easy enough that recall starts at 1.000, so the result is "churn
causes no degradation", not "recall is high under difficult load".

## A defect found in `fold` 0.0.1

Replacing a record's value inside one transaction silently loses the update in
the vector index.

`Hnsw::push` buffers pending work in a map keyed only by `K`, overwriting the
value slot but *accumulating* the delta slot:

```rust
let e = self.pending.entry(kenc).or_insert_with(|| (key, data.val, 0));
e.1 = data.val;      // last writer wins
e.2 += delta as i64; // ...but the delta accumulates
```

`commit` then dispatches on the accumulated delta, with `0 => {}`. A retraction
paired with an insertion under the same key nets to zero, so the write is
skipped and the **old vector survives** in both the graph and the store.

`KeyedStream::upsert` emits exactly that `-1`/`+1` pair in one transaction, so
this is reachable through the primary API — and bogkit's own `search` example
hits it. After its "staging moved off the pi" update, querying `raspberry pi`
still returns document 6 at cosine distance 0.239, showing the *new* text: the
BM25 lane updated correctly (it accumulates per `(term, key)`, so old and new
terms never cancel) and the `Table` is last-writer-wins, so only the vector lane
is stale, and only a query for the *old* wording reveals it.

Reproduction, isolated to intra-transaction cancellation:

```
cargo test -p salience -- --ignored

stale vector retained: distance to the new text (1.0249) should be
smaller than to the replaced text (0.0000)
```

Distance `0.0000` to the replaced text is the old vector, still present and
exact. The same replacement split across two transactions passes
(`hnsw_replacement_across_two_transactions_updates_the_vector`).

`terminal::Table` handles this correctly by *replacing* its pending entry
rather than accumulating (`self.pending.insert(key, (val, delta))`), so the
minimal fix is probably to make `Hnsw` agree with it — though the fully correct
rule has to distinguish "net zero because nothing happened" from "net zero
because a value was replaced", which needs the prior store state. Left to the
maintainers rather than patched here, since it is a semantics call on their
crate. Happy to send a separate PR.

This crate works around it by amending across two transactions, which is
correct and costs the atomicity fold otherwise provides — a crash between them
leaves a memory retracted but not reinserted. A real trade, not a free
workaround.

## Layout

- `src/memory.rs` — `Kind`, `Memory`, `Event`, and the salience terms
- `src/main.rs` — the pipeline, the demo script, the REPL
- `src/bench.rs` — the three measurements
- `src/tests.rs` — properties, plus the defect reproduction

Helpers are macros rather than functions throughout. The pipeline type contains
closures and cannot be written down, so nothing that reads a stream can be an
ordinary function — a real ergonomic cost of fold's static composition, and the
main obstacle to driving fold from a host language across an FFI boundary.
