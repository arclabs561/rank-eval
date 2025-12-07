# rank-eval

Ranking evaluation metrics: NDCG, MAP, MRR, precision, recall. TREC format support.

Standardized implementations of ranking evaluation metrics. Different projects implement metrics differently, leading to inconsistent results. This crate provides a single source of truth for evaluation across ranking projects.

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

Available binary metrics (13 total):
- `precision_at_k()` - Precision at rank k: P@k = |relevant ∩ top-k| / k
- `recall_at_k()` - Recall at rank k: R@k = |relevant ∩ top-k| / |relevant|
- `mrr()` - Mean Reciprocal Rank: 1 / rank(first relevant doc)
- `dcg_at_k()` - Discounted Cumulative Gain at k
- `idcg_at_k()` - Ideal DCG at k (for normalization)
- `ndcg_at_k()` - Normalized DCG: DCG@k / IDCG@k
- `average_precision()` - Average Precision (AP), becomes MAP when averaged
- `err_at_k()` - Expected Reciprocal Rank (cascade model with user stopping)
- `rbp_at_k()` - Rank-Biased Precision (user persistence model, parameter p)
- `f_measure_at_k()` - F-measure: harmonic mean of precision and recall (F1, F2, etc.)
- `success_at_k()` - Success at k: 1.0 if any relevant in top-k, else 0.0
- `r_precision()` - R-Precision: Precision at R (where R = number of relevant docs)
- `Metrics::compute()` - Compute all metrics at once (struct with all values)

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

Available graded metrics (2 total):
- `compute_ndcg()` - nDCG@k for graded relevance (uses actual relevance scores 0, 1, 2, ...)
- `compute_map()` - MAP for graded relevance (treats relevance > 0 as relevant)

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

- **`binary`**: Binary relevance metrics (13 functions)
- **`graded`**: Graded relevance metrics (2 functions)
- **`trec`**: TREC format parsing (`TrecRun`, `Qrel`, loading functions, grouping utilities)
- **`batch`**: Batch evaluation across multiple queries
- **`export`**: Export results to CSV/JSON
- **`statistics`**: Statistical testing (paired t-test, confidence intervals, Cohen's d)
- **`validation`**: Input validation and error handling
- **`dataset`**: Dataset loaders, validators, statistics (requires `serde` feature)

## Batch Evaluation

Evaluate multiple queries efficiently:

```rust
use rank_eval::batch::evaluate_batch_binary;
use std::collections::HashSet;

let rankings = vec![
    vec!["doc1", "doc2", "doc3"],
    vec!["doc4", "doc5", "doc6"],
];
let qrels = vec![
    ["doc1", "doc3"].into_iter().collect::<HashSet<_>>(),
    ["doc4"].into_iter().collect::<HashSet<_>>(),
];

let results = evaluate_batch_binary(&rankings, &qrels, &["ndcg@10", "precision@5", "mrr"]);
println!("Mean NDCG@10: {}", results.aggregated["ndcg@10"]);
```

Supported metrics: `ndcg@10`, `ndcg@5`, `precision@10`, `precision@5`, `precision@1`, `recall@10`, `recall@5`, `mrr`, `ap`, `map`, `err@10`, `rbp@10`, `f1@10`, `success@10`, `r_precision`

## Statistical Analysis

```rust
use rank_eval::statistics::{paired_t_test, confidence_interval, cohens_d};

let method_a = vec![0.5, 0.6, 0.7, 0.8, 0.9];
let method_b = vec![0.4, 0.5, 0.6, 0.7, 0.8];

// Paired t-test
let ttest = paired_t_test(&method_a, &method_b, 0.05);
println!("Significant: {}", ttest.significant);

// Confidence interval
let (lower, upper) = confidence_interval(&method_a, 0.95);

// Effect size
let d = cohens_d(&method_a, &method_b);
```

## Dataset Support

### Supported Datasets

- **MS MARCO** - `load_msmarco_runs()`, `load_msmarco_qrels()`
- **BEIR** - `load_beir_runs()`, `load_beir_qrels()`
- **MIRACL** - `load_miracl_runs()`, `load_miracl_qrels()`
- **MTEB** - `load_mteb_runs()`, `load_mteb_qrels()`
- **HotpotQA** - `load_hotpotqa_runs()`, `load_hotpotqa_qrels()`
- **Natural Questions** - `load_natural_questions_runs()`, `load_natural_questions_qrels()`
- **TREC** - Generic TREC format support with auto-detection

### Dataset Utilities

- `detect_dataset_type()` - Auto-detect dataset type
- `validate_dataset()` - Comprehensive dataset validation
- `compute_comprehensive_stats()` - Detailed dataset statistics
- `export_to_csv()` / `export_to_json()` - Export evaluation results

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

## Statistical Analysis (Real Data)

Comprehensive statistical analysis of NDCG using 1000 real query evaluations:

![NDCG Statistical Analysis](../hack/viz/ndcg_statistical.png)

**Four-panel analysis:**
- **Top-left**: NDCG distribution by cutoff k (1, 5, 10, 20)
- **Top-right**: Good vs poor ranking comparison with error bars
- **Bottom-left**: NDCG@10 distribution with beta fitting (statistical rigor like games/tenzi)
- **Bottom-right**: k sensitivity analysis with confidence intervals

**Metric Comparison:**

![NDCG Metric Comparison](../hack/viz/ndcg_metric_comparison.png)

Comparison of NDCG with other metrics (MAP, MRR) showing distributions, box plots, and correlation analysis.

**Data Source**: 1000 real NDCG computations from realistic rankings. See [Visualizations](../hack/viz/NDCG_VISUALIZATIONS.md) for complete analysis.

## License

MIT OR Apache-2.0

