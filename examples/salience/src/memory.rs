//! Memory records, the events that produce them, and the salience formula.
//!
//! The vocabulary here is borrowed from a working agent runtime (Herald),
//! where "recall" is not nearest-neighbour search but a weighted product:
//!
//! ```text
//! salience = similarity x reinforcement x decay x keyword_boost x kind_weight
//! ```
//!
//! The point of this crate is that those five terms do *not* all belong in
//! the same place. Four of them are functions of the data and can be
//! materialized incrementally by fold. One of them — `decay` — is a function
//! of the clock, and changes with no delta arriving. See [`Salience`].

use serde::{Deserialize, Serialize};

/// Seconds in a day, for the decay half-life.
const DAY: u64 = 86_400;

/// What sort of thing a memory is. Different kinds are worth different
/// amounts at recall time regardless of how well they match the query — a
/// standing preference outranks an incidental episode.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Kind {
    /// A durable statement about the world.
    Fact,
    /// A standing instruction from the user.
    Preference,
    /// Something that happened once.
    Episode,
    /// A learned way of doing something.
    Procedure,
}

impl Kind {
    /// The `kind_weight` term. A constant per kind, so it costs nothing to
    /// apply at read time and nothing to maintain.
    pub fn weight(self) -> f64 {
        match self {
            Kind::Preference => 1.30,
            Kind::Procedure => 1.15,
            Kind::Fact => 1.00,
            Kind::Episode => 0.85,
        }
    }

    /// How long this kind of memory takes to lose half its salience.
    ///
    /// **These differing by kind is what makes `decay` unmaterializable.** With
    /// a single global half-life, exponential decay is rank-*preserving*: the
    /// ratio `exp(-(t-a)/T) / exp(-(t-b)/T)` is `exp((b-a)/T)`, a constant, so
    /// the clock could be folded in once at write time and the ordering would
    /// stay correct forever. Give two memories different half-lives and the
    /// ratio becomes `exp(-t(1/T1 - 1/T2))`, which moves with `t`. A standing
    /// preference overtakes a stale incident with nobody writing anything.
    pub fn half_life_days(self) -> f64 {
        match self {
            Kind::Preference => 365.0,
            Kind::Procedure => 180.0,
            Kind::Fact => 90.0,
            Kind::Episode => 21.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Kind::Fact => "fact",
            Kind::Preference => "pref",
            Kind::Episode => "epis",
            Kind::Procedure => "proc",
        }
    }
}

/// A stored memory. This is the *content* of a memory and nothing else:
/// reinforcement counts deliberately live in a separate sink, because
/// changing a record is what forces fold to re-run its whole branch —
/// including the embedding. See the module docs on `main`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub text: String,
    pub kind: Kind,
    /// Unix seconds. Stored, not derived, so replay is deterministic.
    pub created_at: u64,
}

/// What arrives on the stream.
///
/// One enum rather than two streams, because fold gives atomicity *within* a
/// `Stream` and has no way to join across two. Keeping recording and
/// reinforcement on one stream is what lets a batch of both commit together.
#[derive(Clone, Serialize, Deserialize)]
pub enum Event {
    /// A memory was written. Carries the whole record: this is the delta that
    /// the content sinks (BM25, HNSW, the memory table) consume, so retracting
    /// it must reproduce it exactly.
    Recorded { id: u64, memory: Memory },
    /// A memory was used again. Carries only the id — deliberately, so that
    /// reinforcing a hot memory does not re-embed its text.
    Reinforced { id: u64 },
}

impl Event {
    /// The content projection: `Some` only for `Recorded`.
    ///
    /// Used by the BM25, HNSW and table branches. `Reinforced` returns `None`
    /// and so never touches them — that is the whole reason the event type is
    /// an enum rather than a record.
    pub fn recorded(&self) -> Option<(u64, &Memory)> {
        match self {
            Event::Recorded { id, memory } => Some((*id, memory)),
            Event::Reinforced { .. } => None,
        }
    }

    /// The reinforcement projection: `Some` only for `Reinforced`.
    pub fn reinforced(&self) -> Option<u64> {
        match self {
            Event::Reinforced { id } => Some(*id),
            Event::Recorded { .. } => None,
        }
    }
}

/// A recall score, kept broken into its terms so the demo can show *why*
/// something ranked where it did — and so the split between the materialized
/// terms and the read-time term stays visible in the type.
#[derive(Clone, Copy)]
pub struct Salience {
    // --- maintained by fold, read straight out of a sink ---
    /// From the HNSW sink: `1 - cosine_distance`.
    pub similarity: f64,
    /// From the BM25 sink, normalized to a multiplier >= 1.
    pub keyword_boost: f64,
    /// From the reinforcement `Aggregate`.
    pub reinforcement: f64,
    /// From the memory table (a constant per kind).
    pub kind_weight: f64,

    // --- computed at read time, from the clock ---
    /// Exponential decay in the memory's age. Not materializable: it changes
    /// continuously with wall-clock time while no delta arrives.
    pub decay: f64,
}

impl Salience {
    pub fn total(&self) -> f64 {
        self.similarity * self.keyword_boost * self.reinforcement * self.kind_weight * self.decay
    }

    /// The part fold can hold. Everything except `decay`.
    ///
    /// This is the number a materialized view could store; `total()` is what a
    /// caller actually wants, and the gap between them is exactly the argument
    /// for keeping a rescore stage.
    pub fn materialized(&self) -> f64 {
        self.similarity * self.keyword_boost * self.reinforcement * self.kind_weight
    }
}

/// `reinforcement`: a saturating boost, so a memory recalled a thousand times
/// does not drown out everything else.
pub fn reinforcement(count: i64) -> f64 {
    1.0 + (1.0 + count.max(0) as f64).ln()
}

/// `decay`: exponential in age, halving every [`Kind::half_life_days`].
///
/// This is the term that cannot be folded — see [`Kind::half_life_days`] for
/// exactly why the per-kind rate is what breaks materializability. It takes
/// `now` as an argument rather than reading the clock itself, so tests and
/// benchmarks stay deterministic.
pub fn decay(created_at: u64, now: u64, kind: Kind) -> f64 {
    let age_days = now.saturating_sub(created_at) as f64 / DAY as f64;
    0.5_f64.powf(age_days / kind.half_life_days())
}

/// `keyword_boost`: turn a raw BM25 relevance score into a multiplier >= 1.
///
/// BM25 is unbounded above and incomparable to a cosine similarity, so it is
/// damped rather than used directly.
pub fn keyword_boost(bm25: f64) -> f64 {
    1.0 + (1.0 + bm25.max(0.0)).ln() / 4.0
}

/// `similarity`: anny reports cosine *distance* (smaller is closer); recall
/// wants a similarity (larger is better).
pub fn similarity(cosine_distance: f32) -> f64 {
    (1.0 - cosine_distance as f64).clamp(0.0, 1.0)
}
