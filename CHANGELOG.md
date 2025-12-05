# Changelog

All notable changes to `rank-eval` will be documented in this file.

## [0.2.0] - 2025-01-XX

### Added

#### New Metrics
- **ERR (Expected Reciprocal Rank)**: `err_at_k()` - Cascade model for user behavior
- **RBP (Rank-Biased Precision)**: `rbp_at_k()` - User persistence model
- **F-measure@K**: `f_measure_at_k()` - Harmonic mean of precision and recall
- **Success@K**: `success_at_k()` - Binary success metric
- **R-Precision**: `r_precision()` - Precision at R (number of relevant docs)

#### New Modules
- **`validation`**: Input validation utilities (`validate_metric_inputs`, `validate_persistence`, `validate_beta`)
- **`batch`**: Batch evaluation utilities (`evaluate_batch_binary`, `evaluate_trec_batch`)
- **`statistics`**: Statistical testing (`paired_t_test`, `confidence_interval`, `cohens_d`)
- **`export`**: Export utilities (`export_to_csv`, `export_to_json`)

#### Enhanced Dataset Support
- **MTEB loader**: `load_mteb_runs()`, `load_mteb_qrels()`
- **HotpotQA loader**: `load_hotpotqa_runs()`, `load_hotpotqa_qrels()`
- **Natural Questions loader**: `load_natural_questions_runs()`, `load_natural_questions_qrels()`
- **Auto-detection**: `detect_dataset_type()` function

#### Integration
- **rank-relax**: Added integration tests validating differentiable ranking quality
- **Extended Metrics struct**: Added `err_at_10`, `rbp_at_10`, `f1_at_10`, `success_at_10`, `r_precision` fields

### Changed
- **Error messages**: Enhanced with line numbers, context, and suggestions
- **Documentation**: Updated README with all new metrics and modules

### Testing
- Added 20+ new tests for all new functionality
- Total test count: 80+ tests
- All tests passing across workspace

## [0.1.0] - Initial Release

### Added
- TREC format parsing (`TrecRun`, `Qrel`, loading functions)
- Binary relevance metrics (NDCG, MAP, MRR, Precision, Recall, AP)
- Graded relevance metrics (nDCG, MAP for graded judgments)
- Dataset loaders (MS MARCO, BEIR, MIRACL, TREC)
- Dataset validation and statistics
- Integration with rank-fusion/evals and rank-refine

