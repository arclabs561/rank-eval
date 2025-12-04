# rank-eval

IR evaluation metrics and TREC format parsing for Rust.

This crate provides standardized implementations of information retrieval evaluation metrics and utilities for working with TREC-formatted datasets. It's designed to be shared across multiple ranking projects (`rank-fusion`, `rank-refine`, `rank-relax`) to ensure consistent evaluation.

## Features

- **TREC Format Parsing**: Load and parse TREC run files and qrels
- **Binary Relevance Metrics**: NDCG, MAP, MRR, Precision@K, Recall@K for binary relevance judgments
- **Graded Relevance Metrics**: NDCG and MAP for graded relevance judgments (0, 1, 2, 3...)
- **Lightweight**: Minimal dependencies, fast compilation
- **Well-Tested**: Comprehensive test coverage

## Quick Start

### Rust

```bash
cargo add rank-eval
```

```rust
use rank_eval::binary::ndcg_at_k;
use std::collections::HashSet;

let ranked = vec!["doc1", "doc2", "doc3"];
let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
let ndcg = ndcg_at_k(&ranked, &relevant, 10);
```

### Python

**Install from PyPI:**

```bash
pip install rank-eval
```

```python
import rank_eval

ranked = ["doc1", "doc2", "doc3"]
relevant = {"doc1", "doc3"}
ndcg = rank_eval.ndcg_at_k(ranked, relevant, k=10)
```

**For development/contributing:**

```bash
cd rank-eval-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

### Using as Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
rank-eval = { path = "../../rank-eval" }
# Or when published:
# rank-eval = "0.1"
```

## Usage

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

### Binary Relevance Metrics

For scenarios where documents are either relevant or not relevant:

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

### Graded Relevance Metrics

For scenarios with graded relevance judgments (0 = not relevant, 1+ = relevant, higher = more relevant):

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

### Convenience Struct

With the `serde` feature enabled, you can use the `Metrics` struct:

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

- **`trec`**: TREC format parsing (`TrecRun`, `Qrel`, loading functions, grouping utilities)
- **`binary`**: Binary relevance metrics (all documents are either relevant or not)
- **`graded`**: Graded relevance metrics (documents have relevance scores 0, 1, 2, 3...)
- **`trec`**: TREC format parsing (run files and qrels)
- **`dataset`**: Dataset loaders, validators, and statistics (requires `serde` feature)

## Features

- **`serde`** (default): Enables serialization support for the `Metrics` struct

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

## Design Philosophy

This crate is designed to be:

1. **Standardized**: Single source of truth for IR metrics across all ranking projects
2. **Lightweight**: Minimal dependencies, fast compilation
3. **Well-Tested**: Comprehensive test coverage for correctness
4. **Documented**: Clear examples and API documentation
5. **Extensible**: Easy to add new metrics or formats

## Comparison with Other Crates

- **`trec_eval`**: This crate focuses on Rust-native implementations with a clean API, while `trec_eval` is a wrapper around the C trec_eval tool.
- **`ir-measures`**: Similar goals, but this crate is designed specifically for the ranking workspace and integrates with TREC format parsing.

## Contributing

When adding new metrics:

1. Add to the appropriate module (`binary.rs` or `graded.rs`)
2. Include comprehensive tests
3. Add documentation with examples
4. Update this README

## See Also

- **[rank-fusion](https://crates.io/crates/rank-fusion)**: Combine ranked lists from multiple retrievers
- **[rank-refine](https://crates.io/crates/rank-refine)**: Score embeddings with MaxSim (ColBERT)
- **[rank-relax](https://crates.io/crates/rank-relax)**: Differentiable ranking operations for ML training

## License

MIT OR Apache-2.0

