//! Properties this crate relies on, plus one upstream defect it found.

use anny::metric::Cosine;
use fold::pipeline::{Aggregate, FilterMap, Keyed, Map, terminal};
use fold::stream::{KeyedStream, Stream};

use crate::memory::{Event, Kind, Memory};

const DIM: usize = ese::DIMENSIONS;
const NOW: u64 = 1_760_000_000;

fn tmp(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("bogkit-salience-test-{tag}.db"));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn mem(text: &str) -> Memory {
    Memory {
        text: text.to_string(),
        kind: Kind::Fact,
        created_at: NOW,
    }
}

const PI: &str = "the staging environment runs on a raspberry pi under the desk";
const QCD: &str = "quantum chromodynamics describes the lattice gauge interaction";

/// Cosine distance from `query` to the single vector stored under key 1.
macro_rules! dist_to {
    ($st:expr, $query:expr) => {{
        let q = ese::encode_single($query);
        $st.rtx(|vecs| {
            vecs.search(&q)
                .into_iter()
                .find(|h| h.val == 1u64)
                .map(|h| h.score)
        })
    }};
}

/// **Upstream defect.** `KeyedStream::upsert` replaces a record by pushing the
/// old value at delta `-1` and the new value at delta `+1` *inside one
/// transaction*. `Hnsw` buffers pending work in a map keyed only by `K`, where
/// the value slot is overwritten (`e.1 = data.val`) while the delta slot
/// accumulates (`e.2 += delta`). The pair therefore nets to zero, and
/// `commit`'s `0 => {}` arm drops the update on the floor: the stale vector
/// survives in both the graph and the store.
///
/// `Bm25` is immune because it accumulates per `(term, key)` posting rather
/// than per key, so old and new terms never cancel — which is why bogkit's own
/// `search` example *looks* correct: its BM25 lane updates, its text comes from
/// a last-writer-wins `Table`, and only the vector lane is silently stale.
///
/// Marked `#[ignore]` so the suite stays green against upstream; run it with
/// `cargo test -p salience -- --ignored` to see the defect.
#[test]
#[ignore = "reproduces an upstream fold defect; unignore once Hnsw keys pending state by value"]
fn hnsw_upsert_within_one_transaction_keeps_the_stale_vector() {
    let mut st = KeyedStream::new(
        tmp("hnsw-upsert"),
        Map::new(
            |d: &Keyed<u64, String>| Keyed::new(d.key, ese::encode_single(&d.val)),
            terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
        ),
    );

    st.wtx(|tx| {
        tx.upsert(&1u64, &PI.to_string());
    });
    st.wtx(|tx| {
        tx.upsert(&1u64, &QCD.to_string());
    });

    let to_pi = dist_to!(st, PI).expect("key 1 present");
    let to_qcd = dist_to!(st, QCD).expect("key 1 present");

    // After replacing the text, the stored vector should be the new one: close
    // to QCD, far from PI.
    assert!(
        to_qcd < to_pi,
        "stale vector retained: distance to the new text ({to_qcd:.4}) should be \
         smaller than to the replaced text ({to_pi:.4})"
    );
}

/// The same replacement, split across two transactions, works — which
/// isolates the cause to intra-transaction delta cancellation rather than
/// anything about the vectors themselves.
#[test]
fn hnsw_replacement_across_two_transactions_updates_the_vector() {
    let mut st = KeyedStream::new(
        tmp("hnsw-two-tx"),
        Map::new(
            |d: &Keyed<u64, String>| Keyed::new(d.key, ese::encode_single(&d.val)),
            terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
        ),
    );

    st.wtx(|tx| {
        tx.upsert(&1u64, &PI.to_string());
    });
    // retract in its own transaction...
    st.wtx(|tx| {
        tx.remove(&1u64);
    });
    // ...then insert in another
    st.wtx(|tx| {
        tx.upsert(&1u64, &QCD.to_string());
    });

    let to_pi = dist_to!(st, PI).expect("key 1 present");
    let to_qcd = dist_to!(st, QCD).expect("key 1 present");
    assert!(
        to_qcd < to_pi,
        "distance to new text {to_qcd:.4} should beat distance to old text {to_pi:.4}"
    );
}

/// Building the salience pipeline and driving it end to end: a memory is
/// recalled by both indexes, then forgetting it removes it from *every* sink —
/// the vector index included, not tombstoned.
#[test]
fn forgetting_retracts_from_every_sink() {
    let mut st = Stream::new(
        tmp("forget"),
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

    st.wtx(|tx| {
        tx.push(
            &Event::Recorded {
                id: 1,
                memory: mem(PI),
            },
            1,
        );
        tx.push(
            &Event::Recorded {
                id: 2,
                memory: mem(QCD),
            },
            1,
        );
        // reinforce 1 three times
        for _ in 0..3 {
            tx.push(&Event::Reinforced { id: 1 }, 1);
        }
    });

    // present in all four sinks
    st.rtx(|(bm25, vecs, memories, reinf)| {
        assert!(bm25.search("raspberry pi", 5).iter().any(|h| h.val == 1));
        assert!(
            vecs.search(&ese::encode_single(PI))
                .iter()
                .any(|h| h.val == 1)
        );
        assert!(memories.get(&1).is_some());
        assert_eq!(reinf.get(&1), Some(3));
    });

    // forget it: read current state inside the transaction, push the exact
    // compensating deltas
    st.wtx(|tx| {
        let (memory, count): (Option<Memory>, i64) =
            tx.rtx(|(_, _, memories, reinf)| (memories.get(&1), reinf.get(&1).unwrap_or(0)));
        let memory = memory.expect("memory 1 present");
        tx.push(&Event::Recorded { id: 1, memory }, -1);
        tx.push(&Event::Reinforced { id: 1 }, -(count as isize));
    });

    st.rtx(|(bm25, vecs, memories, reinf)| {
        assert!(
            !bm25.search("raspberry pi", 5).iter().any(|h| h.val == 1),
            "bm25 still returns the forgotten memory"
        );
        assert!(
            !vecs
                .search(&ese::encode_single(PI))
                .iter()
                .any(|h| h.val == 1),
            "hnsw still returns the forgotten memory — tombstoned rather than removed"
        );
        assert!(memories.get(&1).is_none(), "record table still holds it");
        assert_eq!(reinf.get(&1), None, "reinforcement counter survived");
        // the untouched neighbour is unaffected
        assert!(memories.get(&2).is_some());
    });
}

/// Reinforcement is invertible, which is what `Aggregate` requires: pushing
/// `+n` then `-n` must leave no trace of the key at all.
#[test]
fn reinforcement_aggregate_is_invertible() {
    let mut st = Stream::new(
        tmp("reinf"),
        FilterMap::new(
            |e: &Event| e.reinforced().map(|id| Keyed::new(id, 1i64)),
            Aggregate::new(
                "reinforcement",
                |acc: &mut i64, v: &i64, d: isize| *acc += *v * d as i64,
                terminal::Table::new("reinforcements"),
            ),
        ),
    );

    st.wtx(|tx| {
        for _ in 0..7 {
            tx.push(&Event::Reinforced { id: 1 }, 1);
        }
    });
    st.rtx(|reinf| assert_eq!(reinf.get(&1), Some(7)));

    st.wtx(|tx| tx.push(&Event::Reinforced { id: 1 }, -7));
    st.rtx(|reinf| assert_eq!(reinf.get(&1), None, "aggregate did not drop the key"));
}

/// Sinks resume from committed state when the store is reopened. Worth
/// asserting rather than assuming, because the HNSW graph itself is in-memory
/// and is rebuilt from the persisted vectors at `init`.
#[test]
fn sinks_resume_after_reopen() {
    let path = tmp("reopen");

    {
        let mut st = KeyedStream::new(
            &path,
            Map::new(
                |d: &Keyed<u64, String>| Keyed::new(d.key, ese::encode_single(&d.val)),
                terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
            ),
        );
        st.wtx(|tx| {
            tx.upsert(&1u64, &PI.to_string());
            tx.upsert(&2u64, &QCD.to_string());
        });
        st.checkpoint();
    }

    let st = KeyedStream::new(
        &path,
        Map::new(
            |d: &Keyed<u64, String>| Keyed::new(d.key, ese::encode_single(&d.val)),
            terminal::search::Hnsw::<u64, f32, Cosine, DIM>::new("vecs", Cosine, 42),
        ),
    );
    st.rtx(|vecs| {
        let hits = vecs.search(&ese::encode_single(PI));
        assert_eq!(
            hits.first().map(|h| h.val),
            Some(1),
            "reopened index lost or reordered its vectors"
        );
    });
}

/// `decay` is the term that cannot be materialized, and the reason is that
/// half-lives differ by kind. Two memories of the same age but different kinds
/// must cross over as time passes; if they never crossed, decay could be
/// folded in once at write time and this whole design would be unnecessary.
#[test]
fn per_kind_decay_reorders_over_time_and_a_global_rate_would_not() {
    let born = NOW;
    let (episode, preference) = (Kind::Episode, Kind::Preference);

    // Give the episode a head start so it leads initially.
    let ep_base = 1.0;
    let pref_base = 0.55;

    let at = |t: u64| {
        (
            ep_base * crate::memory::decay(born, t, episode),
            pref_base * crate::memory::decay(born, t, preference),
        )
    };

    let (e0, p0) = at(NOW);
    assert!(e0 > p0, "episode should lead at t=0");

    let (e1, p1) = at(NOW + 120 * 86_400);
    assert!(
        p1 > e1,
        "preference should overtake the episode after 120 days ({p1:.4} vs {e1:.4})"
    );

    // The control: with one shared half-life the ratio is constant, so no
    // crossover is possible and a materialized score would stay correct.
    let same = |t: u64| {
        let d = crate::memory::decay(born, t, Kind::Fact);
        (ep_base * d, pref_base * d)
    };
    let (a0, b0) = same(NOW);
    let (a1, b1) = same(NOW + 120 * 86_400);
    assert!(
        ((a0 / b0) - (a1 / b1)).abs() < 1e-9,
        "a single global half-life must preserve rank order"
    );
}
