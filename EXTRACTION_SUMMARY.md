# rank-eval Extraction Summary

This document summarizes the extraction of evaluation infrastructure from `rank-fusion/evals` into the shared `rank-eval` crate.

## What Was Extracted

### 1. TREC Format Parsing (`trec.rs`)

**From:** `rank-fusion/evals/src/real_world.rs`

**Extracted:**
- `TrecRun` struct - TREC run file entry
- `Qrel` struct - Ground truth relevance judgments
- `load_trec_runs()` - Load TREC run files
- `load_qrels()` - Load TREC qrels files
- `group_runs_by_query()` - Group runs by query and run tag
- `group_qrels_by_query()` - Group qrels by query

**Dependencies:** `anyhow` (error handling)

### 2. Binary Relevance Metrics (`binary.rs`)

**From:** `rank-fusion/evals/src/metrics.rs`

**Extracted:**
- `precision_at_k()` - Precision at rank k
- `recall_at_k()` - Recall at rank k
- `mrr()` - Mean Reciprocal Rank
- `dcg_at_k()` - Discounted Cumulative Gain
- `idcg_at_k()` - Ideal DCG
- `ndcg_at_k()` - Normalized DCG
- `average_precision()` - Average Precision
- `Metrics` struct (with `serde` feature) - Convenience struct for all metrics

**Dependencies:** `serde` (optional, for `Metrics` struct)

### 3. Graded Relevance Metrics (`graded.rs`)

**From:** `rank-fusion/evals/src/real_world.rs`

**Extracted:**
- `compute_ndcg()` - nDCG@k for graded relevance
- `compute_map()` - Mean Average Precision for graded relevance

**Dependencies:** None (uses `std::collections::HashMap`)

## Integration in rank-fusion/evals

### Changes Made

1. **`Cargo.toml`**: Added `rank-eval = { path = "../../rank-eval" }` dependency

2. **`src/real_world.rs`**:
   - Removed `TrecRun` and `Qrel` struct definitions
   - Removed `load_trec_runs()`, `load_qrels()`, `group_runs_by_query()`, `group_qrels_by_query()` functions
   - Removed `compute_ndcg()` and `compute_map()` functions
   - Added imports from `rank-eval::trec` and `rank-eval::graded`
   - Re-exported types/functions for backward compatibility

3. **`src/metrics.rs`**:
   - Removed all metric function implementations
   - Re-exported functions from `rank-eval::binary`
   - Kept `Metrics` struct wrapper for convenience

### Backward Compatibility

All existing code in `rank-fusion/evals` continues to work without changes because:
- Types are re-exported: `pub use rank_eval::trec::{TrecRun, Qrel, ...}`
- Functions are re-exported: `pub use rank_eval::trec::{load_trec_runs, load_qrels, ...}`
- Module structure remains the same

## Test Results

- ✅ All 31 tests in `rank-fusion/evals` pass
- ✅ All 14 tests in `rank-eval` pass
- ✅ All doctests pass

## Benefits

1. **Code Reuse**: TREC parsing and metrics can now be shared across `rank-fusion`, `rank-refine`, and `rank-relax`
2. **Standardization**: Single source of truth for IR metrics
3. **Maintainability**: Easier to maintain and extend metrics in one place
4. **Testing**: Centralized test coverage for evaluation infrastructure

## Future Use Cases

### rank-refine

The `rank-refine` crate mentions nDCG improvements in its implementation plans. It could use `rank-eval` to:
- Evaluate reranking performance on TREC datasets
- Compare different reranking methods using standardized metrics
- Validate improvements mentioned in research papers

### rank-relax

The `rank-relax` crate focuses on differentiable ranking. It could use `rank-eval` to:
- Evaluate differentiable ranking methods
- Compare with traditional ranking approaches
- Validate that differentiable methods maintain quality

## Next Steps

1. ✅ Create `rank-eval` crate
2. ✅ Extract TREC parsing
3. ✅ Extract binary metrics
4. ✅ Extract graded metrics
5. ✅ Integrate into `rank-fusion/evals`
6. ✅ Test everything works
7. ⚠️ Consider adding `rank-eval` to `rank-refine` if evaluation is needed
8. ⚠️ Consider publishing to crates.io if useful to broader community

## Files Created

- `/Users/arc/Documents/dev/rank-eval/Cargo.toml`
- `/Users/arc/Documents/dev/rank-eval/src/lib.rs`
- `/Users/arc/Documents/dev/rank-eval/src/trec.rs`
- `/Users/arc/Documents/dev/rank-eval/src/binary.rs`
- `/Users/arc/Documents/dev/rank-eval/src/graded.rs`
- `/Users/arc/Documents/dev/rank-eval/README.md`
- `/Users/arc/Documents/dev/rank-eval/EXTRACTION_SUMMARY.md` (this file)

## Files Modified

- `/Users/arc/Documents/dev/rank-fusion/evals/Cargo.toml` - Added dependency
- `/Users/arc/Documents/dev/rank-fusion/evals/src/real_world.rs` - Removed code, added imports
- `/Users/arc/Documents/dev/rank-fusion/evals/src/metrics.rs` - Removed code, added re-exports

