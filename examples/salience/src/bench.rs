//! Measurements behind the claims in `main.rs`.
//!
//! Three questions, each answered with a number rather than an assertion:
//!
//! 1. **Does splitting the event type actually pay?** Reinforcing a memory as
//!    a record update (which re-embeds) versus as its own event (which does
//!    not). This justifies the [`Event`] enum in `memory.rs`.
//! 2. **Is incremental maintenance cheaper than refolding?** Applying one
//!    batch of deltas versus rebuilding every sink from the event log at the
//!    same frontier. This is the question a log-and-projection runtime cares
//!    about: rebuild is linear in the log, incremental should be linear in the
//!    delta.
//! 3. **Does recall survive churn?** fold's HNSW removes nodes from the graph
//!    rather than tombstoning them. Whether that actually preserves recall
//!    under sustained insert/delete traffic is measurable, so we measure it
//!    against brute-force cosine ground truth instead of assuming.
//!
//! Each benchmark builds its own pipeline. That is not laziness: the pipeline
//! type contains closures and cannot be named, so a fixture cannot be handed
//! to a helper function.

use std::time::{Duration, Instant};

use anny::metric::Cosine;
use fold::pipeline::{Aggregate, FilterMap, Keyed, terminal};
use fold::stream::Stream;

use crate::memory::{Event, Kind, Memory};

const DIM: usize = ese::DIMENSIONS;
const NOW: u64 = 1_760_000_000;

/// Repeats for the timing runs. Small enough to stay inside a hackathon, large
/// enough that the spread is meaningful.
const REPS: usize = 5;

pub fn run() {
    println!("salience benchmarks (ese dim {DIM})\n");
    reinforcement_cost();
    incremental_vs_refold();
    recall_under_churn();
}

/// Fresh temp path per fixture, so runs never inherit each other's state.
fn fresh(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("bogkit-salience-bench-{tag}.db"));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn corpus(n: usize) -> Vec<(u64, String)> {
    // Deterministic, agent-shaped-ish text. Varied enough that BM25 and the
    // embedder both have something to work with.
    let subjects = [
        "the dashboard",
        "the billing service",
        "the search index",
        "the api gateway",
        "the staging cluster",
        "the auth flow",
        "the export job",
        "the webhook consumer",
    ];
    let verbs = [
        "was slow because of",
        "failed after",
        "was rewritten to avoid",
        "started retrying on",
        "logged a spike in",
        "was rolled back due to",
        "now caches",
        "stopped emitting",
    ];
    let objects = [
        "a missing index",
        "a synchronous aggregate query",
        "connection pool exhaustion",
        "a stale embedding",
        "duplicate deliveries",
        "a clock skew bug",
        "cold start latency",
        "an unbounded fan-out",
    ];
    (0..n)
        .map(|i| {
            let t = format!(
                "{} {} {}",
                subjects[i % subjects.len()],
                verbs[(i / subjects.len()) % verbs.len()],
                objects[(i / (subjects.len() * verbs.len())) % objects.len()],
            );
            (i as u64, format!("{t} (case {i})"))
        })
        .collect()
}

fn event(id: u64, text: &str) -> Event {
    Event::Recorded {
        id,
        memory: Memory {
            text: text.to_string(),
            kind: Kind::Fact,
            created_at: NOW,
        },
    }
}

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn spread(xs: &[Duration]) -> (Duration, Duration) {
    (*xs.iter().min().unwrap(), *xs.iter().max().unwrap())
}

// ---------------------------------------------------------------------------
// 1. reinforcement: id-only event vs full record update
// ---------------------------------------------------------------------------

fn reinforcement_cost() {
    println!("1. reinforcement cost — id-only event vs record update");

    const N: usize = 500;
    const BUMPS: usize = 500;
    let docs = corpus(N);

    // The pipeline used by the real crate: reinforcement is its own event and
    // reaches only the counter.
    let mut split = Stream::new(
        fresh("reinf-split"),
        (
            FilterMap::new(
                |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.text.clone())),
                terminal::search::Bm25::new("bm25"),
            ),
            FilterMap::new(
                |e: &Event| {
                    e.recorded()
                        .map(|(id, m)| Keyed::new(id, ese::encode_single(&m.text)))
                },
                terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
            ),
            FilterMap::new(
                |e: &Event| e.reinforced().map(|id| Keyed::new(id, 1i64)),
                Aggregate::new(
                    "reinforcement",
                    |acc: &mut i64, v: &i64, d: isize| *acc += *v * d as i64,
                    terminal::Table::new("reinforcements"),
                ),
            ),
        ),
    );
    split.wtx(|tx| {
        for (id, text) in &docs {
            tx.push(&event(*id, text), 1);
        }
    });

    let mut split_times = Vec::new();
    for _ in 0..REPS {
        let t = Instant::now();
        split.wtx(|tx| {
            for i in 0..BUMPS {
                tx.push(&Event::Reinforced { id: (i % N) as u64 }, 1);
            }
        });
        split_times.push(t.elapsed());
    }

    // The counterfactual: reinforcement stored on the record, so a bump is a
    // retract-and-reinsert of the whole memory through every branch.
    let mut coupled = Stream::new(
        fresh("reinf-coupled"),
        (
            FilterMap::new(
                |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.text.clone())),
                terminal::search::Bm25::new("bm25"),
            ),
            FilterMap::new(
                |e: &Event| {
                    e.recorded()
                        .map(|(id, m)| Keyed::new(id, ese::encode_single(&m.text)))
                },
                terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
            ),
        ),
    );
    coupled.wtx(|tx| {
        for (id, text) in &docs {
            tx.push(&event(*id, text), 1);
        }
    });

    let mut coupled_times = Vec::new();
    for _ in 0..REPS {
        let t = Instant::now();
        coupled.wtx(|tx| {
            for i in 0..BUMPS {
                let (id, text) = &docs[i % N];
                // a counter bump modelled as a record change: retract, reinsert
                tx.push(&event(*id, text), -1);
                tx.push(&event(*id, text), 1);
            }
        });
        coupled_times.push(t.elapsed());
    }

    let (s, c) = (median(split_times.clone()), median(coupled_times.clone()));
    let (smin, smax) = spread(&split_times);
    let (cmin, cmax) = spread(&coupled_times);
    println!("   {BUMPS} reinforcements over {N} memories, median of {REPS}");
    println!(
        "   id-only event   {:>9.2?}  ({:.2?}..{:.2?})  {:>7.2} us/op",
        s,
        smin,
        smax,
        s.as_secs_f64() * 1e6 / BUMPS as f64
    );
    println!(
        "   record update   {:>9.2?}  ({:.2?}..{:.2?})  {:>7.2} us/op",
        c,
        cmin,
        cmax,
        c.as_secs_f64() * 1e6 / BUMPS as f64
    );
    println!(
        "   -> splitting the event type: {:.1}x cheaper\n",
        c.as_secs_f64() / s.as_secs_f64().max(f64::MIN_POSITIVE)
    );
}

// ---------------------------------------------------------------------------
// 2. incremental delta application vs full refold
// ---------------------------------------------------------------------------

fn incremental_vs_refold() {
    println!("2. incremental application vs full refold");

    for &n in &[1_000usize, 4_000] {
        let docs = corpus(n);
        let log: Vec<Event> = docs.iter().map(|(id, t)| event(*id, t)).collect();

        // Build once, then time a small batch of new deltas on top.
        let mut st = Stream::new(
            fresh(&format!("inc-{n}")),
            (
                FilterMap::new(
                    |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.text.clone())),
                    terminal::search::Bm25::new("bm25"),
                ),
                FilterMap::new(
                    |e: &Event| {
                        e.recorded()
                            .map(|(id, m)| Keyed::new(id, ese::encode_single(&m.text)))
                    },
                    terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
                ),
                FilterMap::new(
                    |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.clone())),
                    terminal::Table::new("memories"),
                ),
            ),
        );
        st.wtx(|tx| {
            for e in &log {
                tx.push(e, 1);
            }
        });

        const BATCH: usize = 20;
        let mut inc = Vec::new();
        for r in 0..REPS {
            let extra: Vec<Event> = (0..BATCH)
                .map(|i| {
                    let id = (n + r * BATCH + i) as u64;
                    event(
                        id,
                        &format!("late arriving note number {id} about cold starts"),
                    )
                })
                .collect();
            let t = Instant::now();
            st.wtx(|tx| {
                for e in &extra {
                    tx.push(e, 1);
                }
            });
            inc.push(t.elapsed());
        }

        // The refold: throw everything away and rebuild from the log.
        let mut refold = Vec::new();
        for r in 0..REPS {
            let path = fresh(&format!("refold-{n}-{r}"));
            let t = Instant::now();
            let mut fresh_st = Stream::new(
                &path,
                (
                    FilterMap::new(
                        |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.text.clone())),
                        terminal::search::Bm25::new("bm25"),
                    ),
                    FilterMap::new(
                        |e: &Event| {
                            e.recorded()
                                .map(|(id, m)| Keyed::new(id, ese::encode_single(&m.text)))
                        },
                        terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
                    ),
                    FilterMap::new(
                        |e: &Event| e.recorded().map(|(id, m)| Keyed::new(id, m.clone())),
                        terminal::Table::new("memories"),
                    ),
                ),
            );
            fresh_st.wtx(|tx| {
                for e in &log {
                    tx.push(e, 1);
                }
            });
            refold.push(t.elapsed());
            let _ = std::fs::remove_dir_all(&path);
        }

        let i = median(inc.clone());
        let f = median(refold.clone());
        let per_delta = i.as_secs_f64() * 1e6 / BATCH as f64;
        let per_log = f.as_secs_f64() * 1e6 / n as f64;
        println!("   log of {n} memories, median of {REPS}");
        println!(
            "   incremental  {BATCH:>4} deltas  {:>9.2?}   {:>7.2} us/event",
            i, per_delta
        );
        println!(
            "   refold       {n:>4} events  {:>9.2?}   {:>7.2} us/event",
            f, per_log
        );
        // The honest reading: the per-event costs are the same, so fold buys no
        // per-event advantage. What it buys is that you only touch the delta.
        // The speedup is therefore exactly the work ratio, and the useful fact
        // is that maintenance is *linear and predictable* either way.
        println!(
            "   -> same cost per event ({:.0}x apart); the {:.0}x saving is just \
             {n}/{BATCH} events not touched",
            (per_delta / per_log).max(per_log / per_delta),
            f.as_secs_f64() / i.as_secs_f64().max(f64::MIN_POSITIVE),
        );
        println!();
    }
}

// ---------------------------------------------------------------------------
// 3. recall under churn
// ---------------------------------------------------------------------------

fn cosine(a: &[f32; DIM], b: &[f32; DIM]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..DIM {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

fn recall_under_churn() {
    println!("3. recall@10 under sustained insert/delete churn");

    const N: usize = 2_000;
    const ROUNDS: usize = 6;
    const CHURN: usize = 200;
    const K: usize = 10;

    let docs = corpus(N + ROUNDS * CHURN);
    let mut st = Stream::new(
        fresh("churn"),
        FilterMap::new(
            |e: &Event| {
                e.recorded()
                    .map(|(id, m)| Keyed::new(id, ese::encode_single(&m.text)))
            },
            terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
        ),
    );

    // Ground truth: every live vector, brute-forced.
    let mut live: Vec<(u64, [f32; DIM])> = Vec::new();

    st.wtx(|tx| {
        for (id, text) in docs.iter().take(N) {
            tx.push(&event(*id, text), 1);
        }
    });
    for (id, text) in docs.iter().take(N) {
        live.push((*id, ese::encode_single(text)));
    }

    let queries = [
        "the dashboard was slow because of a missing index",
        "connection pool exhaustion in the api gateway",
        "duplicate deliveries from the webhook consumer",
        "cold start latency on the export job",
    ];

    // A macro, not a closure: `st.rtx` resolves its reader type from the
    // concrete pipeline, which a `&Stream<_, _>` parameter erases.
    macro_rules! measure {
        ($st:expr, $live:expr) => {{
            let mut total = 0.0;
            for q in &queries {
                let qv = ese::encode_single(q);
                let mut truth: Vec<(u64, f64)> =
                    $live.iter().map(|(id, v)| (*id, cosine(&qv, v))).collect();
                truth.sort_by(|a, b| b.1.total_cmp(&a.1));
                let truth: std::collections::BTreeSet<u64> =
                    truth.into_iter().take(K).map(|(id, _)| id).collect();

                let got: std::collections::BTreeSet<u64> = $st
                    .rtx(|vecs| vecs.search(&qv))
                    .into_iter()
                    .take(K)
                    .map(|h| h.val)
                    .collect();

                total += truth.intersection(&got).count() as f64 / K as f64;
            }
            total / queries.len() as f64
        }};
    }

    println!("   {N} live vectors, {ROUNDS} rounds of {CHURN} deletes + {CHURN} inserts");
    println!("   round   live   recall@{K}");
    println!(
        "   {:>5}  {:>5}   {:>8.3}",
        0,
        live.len(),
        measure!(st, live)
    );

    let mut next = N;
    for round in 1..=ROUNDS {
        // delete the oldest CHURN, insert CHURN new ones
        let doomed: Vec<(u64, [f32; DIM])> = live.drain(..CHURN).collect();
        let added: Vec<(u64, String)> = docs[next..next + CHURN].to_vec();
        next += CHURN;

        st.wtx(|tx| {
            for (id, _) in &doomed {
                let (_, text) = &docs[*id as usize];
                tx.push(&event(*id, text), -1);
            }
            for (id, text) in &added {
                tx.push(&event(*id, text), 1);
            }
        });
        for (id, text) in &added {
            live.push((*id, ese::encode_single(text)));
        }

        println!(
            "   {:>5}  {:>5}   {:>8.3}",
            round,
            live.len(),
            measure!(st, live)
        );
    }
    println!(
        "   -> a flat line means retraction genuinely removes nodes from the graph.\n   \
         note this corpus is easy enough that recall starts at 1.000, so the result is\n   \
         \"churn causes no degradation\", not \"recall is high under load\"."
    );
    println!();
}
