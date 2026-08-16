//! # salience — agent memory as an incrementally maintained view
//!
//! An agent's memory is not a vector index. Recall in a working agent runtime
//! is a weighted product over several signals at once:
//!
//! ```text
//! salience = similarity x reinforcement x decay x keyword_boost x kind_weight
//! ```
//!
//! This crate reproduces that recall over `fold`, and exists to answer one
//! question: **how much of that formula can a materialized view actually
//! hold?**
//!
//! The answer is four terms out of five.
//!
//! - `similarity` — HNSW over ese embeddings (`terminal::search::Hnsw`)
//! - `keyword_boost` — BM25 (`terminal::search::Bm25`)
//! - `reinforcement` — a per-memory counter (`Aggregate`)
//! - `kind_weight` — a constant carried on the record (`terminal::Table`)
//! - `decay` — **not materializable.** It is a function of the wall clock, so
//!   it changes while no delta arrives. It has to be applied at read time.
//!
//! That split is the finding, and the demo below shows it directly: at step 4
//! the ranking changes with *zero* writes to the database, purely because time
//! passed. No incremental view can produce that, which is why a recall path
//! wants a cheap materialized candidate stage and a rescore on top — not one
//! or the other.
//!
//! ## Why the stream carries an enum
//!
//! The obvious modelling is a `KeyedStream<u64, Memory>` where `Memory` holds
//! a reinforcement count. It is also wrong: fold's unit of retraction is the
//! whole record, so bumping a counter retracts and reinserts the record
//! through *every* branch — re-tokenizing it for BM25 and re-embedding it for
//! HNSW. Reinforcement is the hottest field and embedding is the most
//! expensive branch, and that modelling couples them.
//!
//! So the stream carries an [`Event`] enum instead, and each branch opens with
//! a `FilterMap` that selects the events it cares about. `Reinforced` carries
//! only an id, so it reaches the counter and nothing else. The cost is that
//! `forget` must push its own compensating deltas — see [`forget`] — because
//! we gave up `KeyedStream`'s retract-by-key. That trade is measured in
//! `bench.rs`.
//!
//! Run the scripted demo (then an interactive prompt on a terminal):
//!
//! ```console
//! $ cargo run -p salience
//! ```
//!
//! Run the measurements behind the claims above:
//!
//! ```console
//! $ cargo run -p salience -- bench
//! ```

mod bench;
mod memory;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::io::{BufRead, IsTerminal, Write};

use anny::metric::Cosine;
use fold::pipeline::{Aggregate, FilterMap, Keyed, terminal};
use fold::stream::Stream;

use memory::{Event, Kind, Memory, Salience};

/// ese's embedding width, fixed at compile time by the crate's features.
const DIM: usize = ese::DIMENSIONS;

/// How many candidates each index contributes before rescoring. The rescore
/// is what applies `decay`, so the pool has to be wider than the answer —
/// this is the same shape as a pgvector prefilter feeding an application-side
/// ranker.
const POOL: usize = 24;

/// A fixed "now" for the scripted demo, so runs are reproducible. Real
/// callers would pass `SystemTime::now()`.
const NOW: u64 = 1_760_000_000;

const DAY: u64 = 86_400;

// The pipeline's type contains closures and therefore cannot be written down.
// Anything that reads the stream has to be expanded where that type is known,
// so the helpers below are macros rather than functions. This is a real
// ergonomic cost of fold's static composition and worth seeing plainly.

/// Rank memories for `query` at time `now`.
///
/// Two stages, deliberately separated:
///
/// 1. **Candidates** — pulled straight from the materialized sinks. Cheap, and
///    maintained incrementally by fold.
/// 2. **Rescore** — applies `decay`, which no sink can hold.
macro_rules! recall {
    ($st:expr, $query:expr, $now:expr, $limit:expr) => {{
        let query: &str = $query;
        let now: u64 = $now;
        let limit: usize = $limit;
        let embedded = ese::encode_single(query);

        $st.rtx(|(bm25, vecs, memories, reinforcements)| {
            // --- stage 1: candidates, entirely from materialized state ---
            let mut boosts: HashMap<u64, f64> = HashMap::new();
            for hit in bm25.search(query, POOL) {
                boosts.insert(hit.val, memory::keyword_boost(hit.score));
            }
            let mut sims: HashMap<u64, f64> = HashMap::new();
            for hit in vecs.search(&embedded).into_iter().take(POOL) {
                sims.insert(hit.val, memory::similarity(hit.score));
            }

            // --- stage 2: rescore, which is where the clock enters ---
            let mut scored: Vec<(u64, Memory, Salience)> = boosts
                .keys()
                .chain(sims.keys())
                .copied()
                .collect::<std::collections::BTreeSet<u64>>()
                .into_iter()
                .filter_map(|id| {
                    let memory: Memory = memories.get(&id)?;
                    let salience = Salience {
                        similarity: sims.get(&id).copied().unwrap_or(0.0),
                        keyword_boost: boosts.get(&id).copied().unwrap_or(1.0),
                        reinforcement: memory::reinforcement(reinforcements.get(&id).unwrap_or(0)),
                        kind_weight: memory.kind.weight(),
                        decay: memory::decay(memory.created_at, now, memory.kind),
                    };
                    Some((id, memory, salience))
                })
                .collect();

            scored.sort_by(|a, b| b.2.total().total_cmp(&a.2.total()));
            scored.truncate(limit);
            scored
        })
    }};
}

/// Write a memory.
macro_rules! record {
    ($st:expr, $id:expr, $text:expr, $kind:expr, $created_at:expr) => {{
        let memory = Memory {
            text: $text.to_string(),
            kind: $kind,
            created_at: $created_at,
        };
        $st.wtx(|tx| tx.push(&Event::Recorded { id: $id, memory }, 1));
    }};
}

/// Mark a memory as used again.
///
/// This is the cheap path: the event carries only an id, so it reaches the
/// reinforcement counter and never touches BM25 or the embedding index.
macro_rules! reinforce {
    ($st:expr, $id:expr) => {{
        $st.wtx(|tx| tx.push(&Event::Reinforced { id: $id }, 1));
    }};
}

/// Correct a memory: retract the old content, insert the new.
///
/// Reinforcement history survives, because it lives in a different sink keyed
/// by the same id and this touches only the content branches.
///
/// # Why this is two transactions
///
/// The natural spelling is one transaction pushing `-1` for the old record and
/// `+1` for the new. Against fold 0.0.1 that **silently loses the update in the
/// vector index**: `Hnsw` buffers pending work in a map keyed only by `K`,
/// overwriting the value but *accumulating* the delta, so `-1` and `+1` under
/// one key net to zero and `commit`'s `0 => {}` arm skips the write. BM25 is
/// unaffected (it accumulates per `(term, key)`), and the record table is
/// last-writer-wins, so the memory's *text* updates while its *embedding* stays
/// stale — the failure is invisible unless you query for the old wording.
///
/// `tests::hnsw_upsert_within_one_transaction_keeps_the_stale_vector`
/// reproduces it; `KeyedStream::upsert` emits exactly that `-1`/`+1` pair, so
/// bogkit's own `search` example has the same stale vector.
///
/// Splitting the retraction and the insertion into separate transactions is
/// correct, and costs the atomicity fold otherwise gives us: a crash between
/// the two leaves the memory retracted but not reinserted. That is a real
/// trade, not a free workaround.
macro_rules! amend {
    ($st:expr, $id:expr, $text:expr) => {{
        let id: u64 = $id;
        let old: Option<Memory> = $st.rtx(|(_, _, memories, _)| memories.get(&id));
        match old {
            None => false,
            Some(old) => {
                let new = Memory {
                    text: $text.to_string(),
                    ..old.clone()
                };
                $st.wtx(|tx| tx.push(&Event::Recorded { id, memory: old }, -1));
                $st.wtx(|tx| tx.push(&Event::Recorded { id, memory: new }, 1));
                true
            }
        }
    }};
}

/// Forget a memory, everywhere.
///
/// Because this pipeline is fed by a raw `Stream` rather than a `KeyedStream`,
/// nothing retracts by key for us: we read the current state through the
/// transaction's own view and push the exact compensating deltas. Reading
/// inside `wtx` is what `Tx::rtx` is for, and it means the read and the
/// retraction cannot disagree.
///
/// The reinforcement counter is retracted by pushing the `Reinforced` event
/// with a multiplicity of `-count`, which drives the `Aggregate` to zero and
/// drops the key.
macro_rules! forget {
    ($st:expr, $id:expr) => {{
        let id: u64 = $id;
        $st.wtx(|tx| {
            let (memory, count): (Option<Memory>, i64) =
                tx.rtx(|(_, _, memories, reinf)| (memories.get(&id), reinf.get(&id).unwrap_or(0)));
            match memory {
                None => false,
                Some(memory) => {
                    tx.push(&Event::Recorded { id, memory }, -1);
                    if count > 0 {
                        tx.push(&Event::Reinforced { id }, -(count as isize));
                    }
                    true
                }
            }
        })
    }};
}

/// Print a ranked recall with the term breakdown, so the *why* is visible.
fn show(title: &str, hits: &[(u64, Memory, Salience)]) {
    println!("== {title} ==");
    if hits.is_empty() {
        println!("   (nothing recalled)\n");
        return;
    }
    // `mat` is the part fold holds; `decay` is applied at read time; `total`
    // is their product. Watching `mat` sit still while `total` reorders is the
    // whole argument for keeping a rescore stage.
    println!(
        "   {:>6}  {:>6} {:>5}  {:>5} {:>5} {:>5} {:>5}   memory",
        "total", "mat", "decay", "sim", "kw", "reinf", "kind"
    );
    for (id, memory, s) in hits {
        println!(
            "   {:>6.3}  {:>6.3} {:>5.2}  {:>5.2} {:>5.2} {:>5.2} {:>5.2}   [{id}/{}] {}",
            s.total(),
            s.materialized(),
            s.decay,
            s.similarity,
            s.keyword_boost,
            s.reinforcement,
            s.kind_weight,
            memory.kind.label(),
            memory.text
        );
    }
    println!();
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("bench") {
        bench::run();
        return;
    }

    let db_path = std::env::temp_dir().join("bogkit-salience.db");
    let _ = std::fs::remove_dir_all(&db_path);

    // Four branches off one stream. Every branch opens with a FilterMap that
    // selects the events it cares about, so `Reinforced` reaches only the
    // counter and never pays for tokenization or embedding.
    let mut st = Stream::new(
        &db_path,
        (
            // keyword relevance -> the `keyword_boost` term
            FilterMap::new(
                |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.text.clone())),
                terminal::search::Bm25::new("bm25"),
            ),
            // semantic similarity -> the `similarity` term.
            // ese runs inside the pipeline: it is a pure function of the text,
            // which is exactly what fold needs for a retraction to cancel.
            FilterMap::new(
                |e: &Event| {
                    e.recorded()
                        .map(|(id, m)| Keyed::new(id, ese::encode_single(&m.text)))
                },
                terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
            ),
            // the record itself -> `kind_weight`, `created_at`, and the text
            FilterMap::new(
                |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.clone())),
                terminal::Table::new("memories"),
            ),
            // usage counter -> the `reinforcement` term.
            // Sum is invertible under negative deltas, which is what
            // `Aggregate` requires; max or min would not be.
            FilterMap::new(
                |e: &Event| e.reinforced().map(|id| Keyed::new(id, 1i64)),
                Aggregate::new(
                    "reinforcement",
                    |acc: &mut i64, v: &i64, delta: isize| *acc += *v * delta as i64,
                    terminal::Table::new("reinforcements"),
                ),
            ),
        ),
    );

    for (id, text, kind, age_days) in seed() {
        record!(st, id, text, kind, NOW - age_days * DAY);
    }
    println!("indexed {} memories\n", seed().len());

    show(
        "recall: \"why is the dashboard slow\"",
        &recall!(st, "why is the dashboard slow", NOW, 5),
    );

    // 1. reinforcement: using a memory lifts it, without re-embedding it
    println!("-- reinforcing [3] four times (id only; no re-embed) --\n");
    for _ in 0..4 {
        reinforce!(st, 3);
    }
    show(
        "recall: same query, after reinforcement",
        &recall!(st, "why is the dashboard slow", NOW, 5),
    );

    // 2. THE POINT. Same query, same database, zero writes — only the clock
    //    moved. `mat` is identical to the run above, because nothing was
    //    written; `total` reorders anyway, because episodes fade faster than
    //    facts. No materialized view can produce that reordering, which is
    //    exactly why the rescore stage has to exist.
    println!("-- no writes at all; only the clock advances by 180 days --\n");
    show(
        "recall: same query, 180 days later (note `mat` is unchanged)",
        &recall!(st, "why is the dashboard slow", NOW + 180 * DAY, 5),
    );

    // 3. correcting a memory: retract old content, insert new, atomically.
    //    Shown before and after, so the retraction is visible rather than
    //    asserted.
    show(
        "recall: \"raspberry pi\" BEFORE amending [6]",
        &recall!(st, "raspberry pi under the desk", NOW, 3),
    );
    println!("-- amending [6]: the pi was replaced by a cloud vm --\n");
    amend!(
        st,
        6,
        "the staging environment moved to a cloud vm in april"
    );
    show(
        "recall: \"raspberry pi\" AFTER amending [6]",
        &recall!(st, "raspberry pi under the desk", NOW, 3),
    );

    // 4. forgetting: gone from every sink, including the HNSW graph itself
    println!("-- forgetting [3] --\n");
    forget!(st, 3);
    show(
        "recall: \"why is the dashboard slow\" after forgetting",
        &recall!(st, "why is the dashboard slow", NOW, 5),
    );

    println!("run `cargo run -p salience -- bench` for the measurements.\n");

    // A small REPL: the bones of an agent's memory tool. Inline rather than a
    // function, for the same reason the helpers are macros — the pipeline type
    // cannot be written down in a signature.
    if std::io::stdin().is_terminal() {
        println!(
            "interactive: <query> recalls | add <text> | use <id> reinforces | rm <id> forgets | ctrl-d quits"
        );
        let stdin = std::io::stdin();
        let mut next_id: u64 = seed().len() as u64;
        loop {
            print!("> ");
            std::io::stdout().flush().unwrap();
            let Some(Ok(line)) = stdin.lock().lines().next() else {
                return;
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(text) = line.strip_prefix("add ") {
                record!(st, next_id, text, Kind::Fact, NOW);
                println!("remembered as [{next_id}]");
                next_id += 1;
            } else if let Some(id) = line.strip_prefix("use ") {
                match id.trim().parse::<u64>() {
                    Ok(id) => {
                        reinforce!(st, id);
                        println!("reinforced [{id}]");
                    }
                    Err(_) => println!("usage: use <numeric id>"),
                }
            } else if let Some(id) = line.strip_prefix("rm ") {
                match id.trim().parse::<u64>() {
                    Ok(id) => {
                        if forget!(st, id) {
                            println!("forgot [{id}]");
                        } else {
                            println!("no memory with id {id}");
                        }
                    }
                    Err(_) => println!("usage: rm <numeric id>"),
                }
            } else {
                show("recall", &recall!(st, line, NOW, 5));
            }
        }
    }
}

/// Seed memories: agent-shaped, with kinds and ages, because the embedding
/// quality numbers are meaningless over lorem ipsum.
///
/// `(id, text, kind, age in days)`
fn seed() -> Vec<(u64, &'static str, Kind, u64)> {
    vec![
        (
            0,
            "the user prefers rust over python for backend services",
            Kind::Preference,
            200,
        ),
        (
            1,
            "deployed the api service to the kubernetes cluster on friday",
            Kind::Episode,
            3,
        ),
        (
            2,
            "the postgres database was slow because the orders table was missing an index",
            Kind::Fact,
            45,
        ),
        (
            3,
            "customer complained that the dashboard takes ten seconds to load",
            Kind::Episode,
            12,
        ),
        (
            4,
            "switched the cache from redis to an in-process lru and cut latency in half",
            Kind::Procedure,
            90,
        ),
        (
            5,
            "the user's name is sam and they are hosting a hackathon this weekend",
            Kind::Fact,
            1,
        ),
        (
            6,
            "the staging environment runs on a raspberry pi under the desk",
            Kind::Fact,
            150,
        ),
        (
            7,
            "always write commit messages in the imperative mood",
            Kind::Preference,
            300,
        ),
        (
            8,
            "the search feature should rank recent documents higher",
            Kind::Preference,
            60,
        ),
        (
            9,
            "billing bug: invoices double-counted when a plan changed mid-month",
            Kind::Episode,
            30,
        ),
        (
            10,
            "the team decided to adopt fold for all incremental state at the offsite",
            Kind::Fact,
            7,
        ),
        (
            11,
            "to profile a slow page, start with the browser waterfall then the server trace",
            Kind::Procedure,
            120,
        ),
        (
            12,
            "the dashboard render blocks on a synchronous aggregate query",
            Kind::Fact,
            20,
        ),
        (
            13,
            "meeting notes: ship the embedding search demo before the conference",
            Kind::Episode,
            5,
        ),
    ]
}
