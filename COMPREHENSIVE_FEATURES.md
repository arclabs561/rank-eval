# rank-eval: Comprehensive Feature Documentation

**Generated from actual codebase analysis** - This documents what's actually implemented, not just what's in the README.

## Metrics (20 total)

### Binary Relevance Metrics (13)

1. **`precision_at_k()`** - Precision at rank k: P@k = |relevant ∩ top-k| / k
2. **`recall_at_k()`** - Recall at rank k: R@k = |relevant ∩ top-k| / |relevant|
3. **`mrr()`** - Mean Reciprocal Rank: 1 / rank(first relevant doc)
4. **`dcg_at_k()`** - Discounted Cumulative Gain at k
5. **`idcg_at_k()`** - Ideal DCG at k (for normalization)
6. **`ndcg_at_k()`** - Normalized DCG at k: DCG@k / IDCG@k
7. **`average_precision()`** - Average Precision (AP), becomes MAP when averaged
8. **`err_at_k()`** - Expected Reciprocal Rank (cascade model)
9. **`rbp_at_k()`** - Rank-Biased Precision (user persistence model)
10. **`f_measure_at_k()`** - F-measure (F1, F2, etc.) - harmonic mean of precision/recall
11. **`success_at_k()`** - Success at k (binary: 1.0 if any relevant in top-k, else 0.0)
12. **`r_precision()`** - R-Precision: Precision at R (where R = number of relevant docs)
13. **`Metrics::compute()`** - Compute all metrics at once (struct with all values)

### Graded Relevance Metrics (2)

1. **`compute_ndcg()`** - nDCG@k for graded relevance (uses actual relevance scores 0, 1, 2, ...)
2. **`compute_map()`** - MAP for graded relevance (treats relevance > 0 as relevant)

### Statistical Functions (5)

1. **`paired_t_test()`** - Paired t-test for comparing two methods
2. **`confidence_interval()`** - Confidence intervals for score distributions
3. **`cohens_d()`** - Cohen's d effect size (standardized difference between means)
4. **`compute_comprehensive_stats()`** - Comprehensive dataset statistics
5. **`print_statistics_report()`** - Pretty-print statistics report

## Dataset Loaders (21 functions)

### Dataset-Specific Loaders

1. **MS MARCO**
   - `load_msmarco_runs()` - Load MS MARCO run files
   - `load_msmarco_qrels()` - Load MS MARCO qrels

2. **BEIR**
   - `load_beir_runs()` - Load BEIR runs
   - `load_beir_qrels()` - Load BEIR qrels

3. **MIRACL**
   - `load_miracl_runs()` - Load MIRACL runs
   - `load_miracl_qrels()` - Load MIRACL qrels

4. **MTEB (Massive Text Embedding Benchmark)**
   - `load_mteb_runs()` - Load MTEB runs
   - `load_mteb_qrels()` - Load MTEB qrels

5. **HotpotQA**
   - `load_hotpotqa_runs()` - Load HotpotQA runs
   - `load_hotpotqa_qrels()` - Load HotpotQA qrels

6. **Natural Questions**
   - `load_natural_questions_runs()` - Load Natural Questions runs
   - `load_natural_questions_qrels()` - Load Natural Questions qrels

### Generic TREC Loaders

7. **TREC Format**
   - `load_trec_runs_from_dir()` - Load multiple TREC run files from directory
   - `load_trec_qrels_from_dir()` - Load TREC qrels from directory (auto-detects filename)

### Dataset Utilities

8. **Detection & Configuration**
   - `detect_dataset_type()` - Auto-detect dataset type from directory structure
   - `DatasetType::detect()` - Enum method for detection
   - `DatasetType::name()` - Get human-readable dataset name
   - `create_dataset_config()` - Create dataset configuration
   - `list_datasets()` - List available datasets in directory
   - `validate_dataset_dir()` - Validate dataset directory structure
   - `get_dataset_stats()` - Get basic statistics for dataset

### Dataset Types (Enum)

- `DatasetType::MsMarco`
- `DatasetType::Beir`
- `DatasetType::Trec`
- `DatasetType::Miracl`
- `DatasetType::Mteb`
- `DatasetType::HotpotQA`
- `DatasetType::NaturalQuestions`
- `DatasetType::Squad`
- `DatasetType::Custom`

## Modules (8 total)

1. **`binary`** - Binary relevance metrics (13 functions)
2. **`graded`** - Graded relevance metrics (2 functions)
3. **`trec`** - TREC format parsing and utilities
4. **`batch`** - Batch evaluation across multiple queries
5. **`export`** - Export results to CSV/JSON
6. **`statistics`** - Statistical testing and analysis
7. **`validation`** - Input validation and error handling
8. **`dataset`** - Dataset loaders, statistics, validation (requires `serde` feature)

## Batch Evaluation

### Functions

- **`evaluate_batch_binary()`** - Evaluate multiple queries with binary metrics
- **`evaluate_trec_batch()`** - Evaluate TREC runs in batch

### Supported Metrics in Batch

All metrics supported via string names:
- `"ndcg@10"`, `"ndcg@5"`
- `"precision@10"`, `"precision@5"`, `"precision@1"`
- `"recall@10"`, `"recall@5"`
- `"mrr"`
- `"ap"` or `"map"`
- `"err@10"`
- `"rbp@10"`
- `"f1@10"`
- `"success@10"`
- `"r_precision"`

### Output

- **`BatchResults`** - Contains per-query results and aggregated means
- **`QueryResults`** - Individual query metrics

## Export Formats

1. **CSV Export** - `export_to_csv()` - Export batch results to CSV
2. **JSON Export** - `export_to_json()` - Export batch results to JSON (requires `serde` feature)

## Dataset Statistics

### Comprehensive Statistics

- **`ComprehensiveStats`** - Complete dataset analysis including:
  - Run file statistics (entries, queries, documents, score distributions)
  - Qrel statistics (relevance distributions, queries with relevant docs)
  - Overlap statistics (queries/documents in both runs and qrels)
  - Quality metrics (fusion readiness, runs per query)

### Score Distribution

- Min, max, mean, median, std dev
- Percentiles: P25, P50, P75, P90, P95, P99

## Validation

### Input Validation

- **`validate_metric_inputs()`** - Validate k, ranked list, relevant set
- **`validate_persistence()`** - Validate RBP persistence parameter (0 < p < 1)
- **`validate_beta()`** - Validate F-measure beta parameter (beta > 0)

### Dataset Validation

- **`validate_dataset()`** - Comprehensive dataset validation
- **`DatasetValidationResult`** - Detailed validation report with errors/warnings
- **`print_validation_report()`** - Pretty-print validation results

## TREC Format Support

### Data Structures

- **`TrecRun`** - Run file entry (query_id, doc_id, rank, score, run_tag)
- **`Qrel`** - Qrel entry (query_id, doc_id, relevance)

### Loading Functions

- **`load_trec_runs()`** - Load TREC run file
- **`load_qrels()`** - Load TREC qrels file

### Grouping Utilities

- **`group_runs_by_query()`** - Group runs by query_id
- **`group_qrels_by_query()`** - Group qrels by query_id (returns HashMap<query_id, HashMap<doc_id, relevance>>)

## Features

- **`serde`** (default) - Serialization support for Metrics struct and dataset types
- **`serde_json`** (default) - JSON export functionality

## Python Bindings

Available via `rank-eval-python` package:
- All binary metrics
- All graded metrics
- TREC format loading
- Batch evaluation
- Statistical functions

## Summary

**Total Public API Surface:**
- **20 metric functions** (13 binary + 2 graded + 5 statistical)
- **21 dataset loader functions**
- **8 modules**
- **17 public structs**
- **2 public enums**
- **7 re-exports**

**Documentation Coverage:** Needs improvement - many features not in README.

