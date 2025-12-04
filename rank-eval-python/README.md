# rank-eval-python

Python bindings for [`rank-eval`](../README.md) — IR evaluation metrics and TREC format parsing.

[![PyPI](https://img.shields.io/pypi/v/rank-eval.svg)](https://pypi.org/project/rank-eval/)

## Installation

**Install from PyPI:**

```bash
pip install rank-eval
```

**For development/contributing:**

```bash
cd rank-eval-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

## Usage

### Binary Relevance Metrics

```python
import rank_eval

ranked = ["doc1", "doc2", "doc3", "doc4"]
relevant = {"doc1", "doc3"}

# Precision at k
precision = rank_eval.precision_at_k(ranked, relevant, k=10)

# Recall at k
recall = rank_eval.recall_at_k(ranked, relevant, k=10)

# Mean Reciprocal Rank
mrr = rank_eval.mrr(ranked, relevant)

# nDCG@k
ndcg = rank_eval.ndcg_at_k(ranked, relevant, k=10)

# Average Precision (MAP for single query)
ap = rank_eval.average_precision(ranked, relevant)
```

### Graded Relevance Metrics

```python
import rank_eval

ranked = [
    ("doc1", 0.9),
    ("doc2", 0.8),
    ("doc3", 0.7),
]
qrels = {
    "doc1": 2,  # Highly relevant
    "doc2": 1,  # Relevant
    "doc3": 0,  # Not relevant
}

# nDCG@k for graded relevance
ndcg = rank_eval.compute_ndcg(ranked, qrels, k=10)

# MAP for graded relevance
map_score = rank_eval.compute_map(ranked, qrels)
```

## API

### Binary Relevance Metrics

| Function | Description | Returns |
|----------|-------------|---------|
| `precision_at_k(ranked, relevant, k)` | Precision at rank k | `float` |
| `recall_at_k(ranked, relevant, k)` | Recall at rank k | `float` |
| `mrr(ranked, relevant)` | Mean Reciprocal Rank | `float` |
| `dcg_at_k(ranked, relevant, k)` | Discounted Cumulative Gain | `float` |
| `idcg_at_k(ranked, relevant, k)` | Ideal DCG | `float` |
| `ndcg_at_k(ranked, relevant, k)` | Normalized DCG | `float` |
| `average_precision(ranked, relevant)` | Average Precision | `float` |
| `err_at_k(ranked, relevant, k)` | Expected Reciprocal Rank | `float` |
| `rbp_at_k(ranked, relevant, k, persistence)` | Rank-Biased Precision | `float` |
| `f_measure_at_k(ranked, relevant, k, beta)` | F-measure (F1, F2, etc.) | `float` |
| `success_at_k(ranked, relevant, k)` | Success at k (binary) | `float` |
| `r_precision(ranked, relevant)` | R-Precision | `float` |

### Graded Relevance Metrics

| Function | Description | Returns |
|----------|-------------|---------|
| `compute_ndcg(ranked, qrels, k)` | nDCG@k for graded relevance | `float` |
| `compute_map(ranked, qrels)` | MAP for graded relevance | `float` |

## See Also

- **[rank-eval Rust crate](../README.md)**: Core library documentation
- **[rank-fusion](https://crates.io/crates/rank-fusion)**: Combine ranked lists from multiple retrievers
- **[rank-refine](https://crates.io/crates/rank-refine)**: Score embeddings with MaxSim (ColBERT)
- **[rank-relax](https://crates.io/crates/rank-relax)**: Differentiable ranking operations for ML training

## License

MIT OR Apache-2.0

