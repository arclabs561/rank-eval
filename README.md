# rank-eval

[![CI](https://github.com/arclabs561/rank-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-eval/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-eval.svg)](https://crates.io/crates/rank-eval)
[![Docs](https://docs.rs/rank-eval/badge.svg)](https://docs.rs/rank-eval)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Ranking evaluation metrics: NDCG, MAP, MRR, precision, recall. TREC format support.

rank-eval provides standardized implementations of ranking evaluation metrics and utilities for working with TREC-formatted datasets. Different projects implement IR metrics differently, leading to inconsistent results. This crate is the single source of truth for evaluation across ranking projects (`rank-fusion`, `rank-refine`, `rank-relax`).

## Installation

```bash
cargo add rank-eval
```

## Usage

### Binary Relevance

```rust
use std::collections::HashSet;
use rank_eval::binary::ndcg_at_k;

let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

let ndcg = ndcg_at_k(&ranked, &relevant, 10);
assert!(ndcg >= 0.0 && ndcg <= 1.0);
```

Available binary metrics:
- `precision_at_k()` - Precision at rank k
- `recall_at_k()` - Recall at rank k
- `mrr()` - Mean Reciprocal Rank
- `dcg_at_k()` - Discounted Cumulative Gain
- `idcg_at_k()` - Ideal DCG
- `ndcg_at_k()` - Normalized DCG
- `average_precision()` - Average Precision (MAP for single query)
- `err_at_k()` - Expected Reciprocal Rank
- `rbp_at_k()` - Rank-Biased Precision
- `f_measure_at_k()` - F-measure (F1, F2, etc.)
- `success_at_k()` - Success at k (binary)
- `r_precision()` - R-Precision

### Graded Relevance

```rust
use std::collections::HashMap;
use rank_eval::graded::{compute_ndcg, compute_map};

let ranked = vec![
    ("doc1".to_string(), 0.9),
    ("doc2".to_string(), 0.8),
    ("doc3".to_string(), 0.7),
];
let mut qrels = HashMap::new();
qrels.insert("doc1".to_string(), 2); // Highly relevant
qrels.insert("doc2".to_string(), 1); // Relevant
qrels.insert("doc3".to_string(), 0); // Not relevant

let ndcg = compute_ndcg(&ranked, &qrels, 10);
let map = compute_map(&ranked, &qrels);
```

Available graded metrics:
- `compute_ndcg()` - nDCG@k for graded relevance
- `compute_map()` - Mean Average Precision for graded relevance

### TREC Format Parsing

```rust
use rank_eval::trec::{load_trec_runs, load_qrels, group_runs_by_query, group_qrels_by_query};

// Load TREC run file
let runs = load_trec_runs("runs.txt")?;

// Load TREC qrels file
let qrels = load_qrels("qrels.txt")?;

// Group by query for evaluation
let runs_by_query = group_runs_by_query(&runs);
let qrels_by_query = group_qrels_by_query(&qrels);
```

### Metrics Struct

With the `serde` feature:

```rust
use rank_eval::binary::Metrics;
use std::collections::HashSet;

let ranked = vec!["doc1", "doc2", "doc3"];
let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

let metrics = Metrics::compute(&ranked, &relevant);
println!("nDCG@10: {}", metrics.ndcg_at_10);
println!("MAP: {}", metrics.average_precision);
```

## Modules

- `binary`: Binary relevance metrics
- `graded`: Graded relevance metrics (relevance scores 0, 1, 2, 3...)
- `trec`: TREC format parsing (`TrecRun`, `Qrel`, loading functions, grouping utilities)
- `dataset`: Dataset loaders, validators, statistics (requires `serde` feature)

## Features

- `serde` (default): Serialization support for the `Metrics` struct

## TREC Format

### Run Files

Format: `query_id Q0 doc_id rank score run_tag`

Example:
```
1 Q0 doc1 1 0.95 bm25
1 Q0 doc2 2 0.90 bm25
2 Q0 doc3 1 0.88 dense
```

### Qrels Files

Format: `query_id 0 doc_id relevance`

Example:
```
1 0 doc1 2
1 0 doc2 1
2 0 doc3 2
```

## Comparison

- `trec_eval`: Rust-native implementations with clean API. `trec_eval` is a wrapper around C trec_eval.
- `ir-measures`: Similar goals, but this crate is designed for the ranking workspace and integrates with TREC format parsing.

## Related

- [rank-fusion](https://crates.io/crates/rank-fusion): Combine ranked lists from multiple retrievers
- [rank-refine](https://crates.io/crates/rank-refine): Score embeddings with MaxSim (ColBERT)
- [rank-relax](https://crates.io/crates/rank-relax): Differentiable ranking operations for ML training

## License

MIT OR Apache-2.0

