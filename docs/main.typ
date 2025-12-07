#set page(margin: (x: 2.5cm, y: 2cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)

#show heading: set text(weight: "bold")

= rank-eval: Ranking Evaluation Metrics Documentation

#align(center)[
  #text(size: 14pt, weight: "bold")[Comprehensive Ranking Evaluation Metrics]
  
  #text(size: 10pt)[TREC Format, Statistical Analysis, Dataset Loaders]
  
  #v(0.5cm)
  #text(size: 9pt, style: "italic")[Version 0.1.0]
]

== Introduction

`rank-eval` provides comprehensive ranking evaluation metrics for information retrieval, including binary and graded relevance metrics, statistical functions, and dataset loaders for standard IR benchmarks.

== Features

#v(0.3em)
- 20 Metrics: Binary (13), Graded (2), Statistical (5)
- 21 Dataset Loaders: MS MARCO, BEIR, MIRACL, MTEB, HotpotQA, Natural Questions, TREC
- TREC Format Support: Standard run and qrel file parsing
- Batch Evaluation: Efficient evaluation across multiple queries
- Statistical Analysis: Paired t-tests, confidence intervals, effect sizes

== Metrics

=== Binary Relevance Metrics (13)

- precision_at_k(): Precision at rank k
- recall_at_k(): Recall at rank k
- mrr(): Mean Reciprocal Rank
- dcg_at_k(): Discounted Cumulative Gain
- idcg_at_k(): Ideal DCG
- ndcg_at_k(): Normalized DCG
- average_precision(): Average Precision (MAP for single query)
- err_at_k(): Expected Reciprocal Rank
- rbp_at_k(): Rank-Biased Precision
- f_measure_at_k(): F-measure (F1, F2, etc.)
- success_at_k(): Success at k (binary)
- r_precision(): R-Precision
- Metrics::compute(): Compute all metrics at once

=== Graded Relevance Metrics (2)

- compute_ndcg(): nDCG at k for graded relevance
- compute_map(): MAP for graded relevance

=== Statistical Functions (5)

- paired_t_test(): Paired t-test for comparing two methods
- confidence_interval(): Confidence intervals for score distributions
- cohens_d(): Cohen's d effect size
- compute_comprehensive_stats(): Comprehensive dataset statistics
- print_statistics_report(): Pretty print statistics

== Quick Start

=== Binary Relevance

```rust
use std::collections::HashSet;
use rank_eval::binary::ndcg_at_k;

let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

let ndcg = ndcg_at_k(&ranked, &relevant, 10);
assert!(ndcg >= 0.0 && ndcg <= 1.0);
```

=== Graded Relevance

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

let ndcg = compute_ndcg(&ranked, &qrels, 10);
let map = compute_map(&ranked, &qrels);
```

=== Batch Evaluation

```rust
use rank_eval::evaluate_trec_batch;

let runs = load_trec_runs("runs.txt")?;
let qrels = load_qrels("qrels.txt")?;

let results = evaluate_trec_batch(&runs, &qrels, &["ndcg@10", "map"])?;
println!("Mean NDCG@10: {}", results.mean_ndcg_at_10);
```

== Dataset Loaders

=== MS MARCO

```rust
use rank_eval::load_msmarco_runs;
let runs = load_msmarco_runs("path/to/msmarco/runs")?;
```

=== BEIR

```rust
use rank_eval::load_beir_runs;
let runs = load_beir_runs("path/to/beir")?;
```

=== TREC Format

```rust
use rank_eval::{load_trec_runs, load_qrels};
let runs = load_trec_runs("run.txt")?;
let qrels = load_qrels("qrels.txt")?;
```

== Statistical Analysis

```rust
use rank_eval::{paired_t_test, confidence_interval, cohens_d};

let method_a_scores = vec![0.8, 0.7, 0.9, 0.6];
let method_b_scores = vec![0.75, 0.65, 0.85, 0.55];

let t_test = paired_t_test(&method_a_scores, &method_b_scores)?;
println!("t-statistic: {}, p-value: {}", t_test.t_statistic, t_test.p_value);

let ci = confidence_interval(&method_a_scores, 0.95)?;
println!("95% CI: [{}, {}]", ci.lower, ci.upper);

let effect_size = cohens_d(&method_a_scores, &method_b_scores);
println!("Effect size: {}", effect_size);
```

== Supported Datasets

- MS MARCO
- BEIR
- MIRACL
- MTEB (Massive Text Embedding Benchmark)
- HotpotQA
- Natural Questions
- TREC format: generic

== Performance

- Batch Evaluation: Efficient processing of large datasets
- Memory Efficient: Streaming evaluation for large files
- Fast performance: optimized metric computations

== Installation

```bash
cargo add rank-eval
# With dataset loaders (requires serde)
cargo add rank-eval --features serde
```

== Examples

See examples/ directory for complete evaluation workflows.

== References

=== Evaluation Standards

- Voorhees, E. M. (2004). "Overview of the TREC 2004 Robust Retrieval Track". In *Text REtrieval Conference* (TREC).

- Buckley, C., & Voorhees, E. M. (2005). "Retrieval evaluation with incomplete information". In *Proceedings of the 27th annual international ACM SIGIR conference on Research and development in information retrieval* (pp. 25-32).

=== Metric Definitions

- Järvelin, K., & Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR techniques". *ACM Transactions on Information Systems*, 20(4), 422-446.

- Robertson, S., Zaragoza, H., & Taylor, M. (2004). "Simple BM25 extension to multiple weighted fields". In *Proceedings of the 13th ACM international conference on Information and knowledge management* (pp. 42-49).

- Yilmaz, E., Kanoulas, E., & Aslam, J. A. (2008). "A simple and efficient sampling method for estimating AP and NDCG". In *Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval* (pp. 603-610).

== License

MIT OR Apache-2.0

