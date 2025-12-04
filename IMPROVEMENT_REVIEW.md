# rank-eval Improvement and Usage Review

## Executive Summary

This document reviews how `rank-eval` can be improved and used more effectively across your dev repositories. Based on analysis of the codebase and related projects, here are key opportunities.

## Current Integration Status

### ✅ Fully Integrated
- **rank-fusion/evals**: Complete integration, all dataset functionality moved here
- **rank-refine**: Dev dependency for evaluation tests (14 tests)

### ⚠️ Partially Integrated
- **rank-relax**: Not integrated, but has potential use cases

### 🔍 Potential New Integrations
- **allRank**: Learning-to-rank framework (Python/Rust?)
- **pytorch-listnet**, **pytorch-ranknet**: Learning-to-rank models
- **ann-benchmarks**: Could use evaluation metrics
- **matryoshka-box**: Vector search evaluation

## Improvement Opportunities

### 1. Enhanced Metrics (High Priority)

#### Missing Metrics
- **ERR (Expected Reciprocal Rank)**: Important for web search evaluation
- **RBP (Rank-Biased Precision)**: User model-based metric
- **nERR (Normalized ERR)**: Normalized version
- **F-measure@K**: Combined precision/recall
- **Success@K**: Binary success metric
- **R-Precision**: Precision at R (where R = number of relevant docs)

#### Implementation Priority
```rust
// High priority (commonly used)
pub fn err_at_k(ranked: &[I], relevant: &HashSet<I>, persistence: f64) -> f64
pub fn rbp_at_k(ranked: &[I], relevant: &HashSet<I>, persistence: f64) -> f64
pub fn f_measure_at_k(ranked: &[I], relevant: &HashSet<I>, k: usize, beta: f64) -> f64

// Medium priority (specialized use cases)
pub fn success_at_k(ranked: &[I], relevant: &HashSet<I>, k: usize) -> bool
pub fn r_precision(ranked: &[I], relevant: &HashSet<I>) -> f64
```

### 2. Performance Optimizations

#### Current Bottlenecks
- Repeated HashSet lookups in metric calculations
- No SIMD optimizations for large-scale evaluation
- No parallel evaluation across queries

#### Optimization Opportunities
```rust
// Batch evaluation
pub fn evaluate_batch(
    rankings: &[Vec<I>],
    qrels: &[HashSet<I>],
    metrics: &[Metric],
) -> Vec<MetricResults>

// SIMD-accelerated DCG calculation (for large k)
pub fn dcg_at_k_simd(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64

// Parallel query evaluation
pub fn evaluate_parallel(
    runs_by_query: &HashMap<String, Vec<TrecRun>>,
    qrels_by_query: &HashMap<String, HashMap<String, u32>>,
) -> HashMap<String, Metrics>
```

### 3. Enhanced Dataset Support

#### Additional Dataset Loaders
- **MTEB (Massive Text Embedding Benchmark)**: Standard benchmark
- **HotpotQA**: Multi-hop QA evaluation
- **Natural Questions**: Open-domain QA
- **SQuAD**: Reading comprehension
- **FIRE**: Multilingual IR
- **CLEF**: Cross-language evaluation

#### Dataset Utilities
```rust
// Auto-detect dataset format
pub fn detect_dataset_format(path: &Path) -> Result<DatasetType>

// Download datasets automatically
pub fn download_dataset(name: &str, output_dir: &Path) -> Result<()>

// Dataset registry with metadata
pub fn list_available_datasets() -> Vec<DatasetInfo>
pub fn get_dataset_info(name: &str) -> Result<DatasetInfo>
```

### 4. Better Error Messages and Validation

#### Current Issues
- Generic error messages when TREC parsing fails
- No validation of metric inputs (e.g., k > ranked length)
- Limited debugging information

#### Improvements
```rust
// Detailed error context
pub struct ParseError {
    pub line_number: usize,
    pub line_content: String,
    pub expected_format: String,
    pub suggestion: String,
}

// Input validation
pub fn validate_metric_inputs(
    ranked: &[I],
    relevant: &HashSet<I>,
    k: usize,
) -> Result<(), ValidationError>

// Debug mode with detailed logging
#[cfg(feature = "debug")]
pub fn ndcg_at_k_debug(...) -> (f64, DebugInfo)
```

### 5. Integration with rank-relax

#### Use Cases
- **Validation**: Compare differentiable ranking quality vs. discrete
- **Training Monitoring**: Track ranking metrics during training
- **Ablation Studies**: Evaluate impact of different regularization strengths

#### Implementation
```rust
// In rank-relax
use rank_eval::binary::ndcg_at_k;

// After soft ranking, convert to discrete and evaluate
let soft_ranks = soft_rank(&values, regularization);
let discrete_ranking = convert_to_discrete(&soft_ranks);
let ndcg = ndcg_at_k(&discrete_ranking, &relevant, 10);
```

### 6. Benchmarking and Profiling Utilities

#### Performance Benchmarking
```rust
// Benchmark metric computation
pub fn benchmark_metrics(
    rankings: &[Vec<I>],
    qrels: &[HashSet<I>],
    iterations: usize,
) -> BenchmarkResults

// Profile slow queries
pub fn profile_evaluation(
    runs: &[TrecRun],
    qrels: &[Qrel],
) -> QueryProfile
```

### 7. Statistical Analysis

#### Statistical Testing
```rust
// Significance testing
pub fn paired_t_test(
    method_a: &[f64],
    method_b: &[f64],
) -> Result<TTestResult>

// Confidence intervals
pub fn confidence_interval(
    scores: &[f64],
    confidence: f64,
) -> (f64, f64)

// Effect size
pub fn cohens_d(method_a: &[f64], method_b: &[f64]) -> f64
```

### 8. Export and Reporting

#### Additional Formats
- **CSV export**: For spreadsheet analysis
- **JSON export**: For programmatic analysis
- **LaTeX tables**: For papers
- **Interactive HTML**: Enhanced visualization

#### Enhanced Reporting
```rust
// Comprehensive report generation
pub fn generate_report(
    results: &EvaluationResults,
    format: ReportFormat,
) -> Result<String>

// Comparison reports
pub fn compare_methods(
    methods: &[(&str, &EvaluationResults)],
) -> ComparisonReport
```

## Usage Opportunities in Other Repos

### 1. allRank (Learning-to-Rank)

**Current State**: Unknown (need to investigate)
**Opportunity**: Use `rank-eval` for:
- Evaluating learned ranking models
- Comparing different LTR algorithms
- Standardized evaluation across experiments

**Integration**:
```rust
// In allRank evaluation
use rank_eval::binary::{ndcg_at_k, Metrics};
use rank_eval::trec::{load_qrels, group_qrels_by_query};

// Evaluate learned model
let predictions = model.predict(queries, documents);
let metrics = Metrics::compute(&predictions, &qrels);
```

### 2. pytorch-listnet / pytorch-ranknet

**Current State**: Python-based learning-to-rank
**Opportunity**: 
- Create Rust bindings that use `rank-eval`
- Faster evaluation for large-scale experiments
- Consistent metrics across Python/Rust codebases

**Integration**:
```python
# Python wrapper using rank-eval
import rank_eval_rs

predictions = model.predict(queries, documents)
ndcg = rank_eval_rs.ndcg_at_k(predictions, qrels, k=10)
```

### 3. ann-benchmarks

**Current State**: Approximate nearest neighbor benchmarking
**Opportunity**: Add ranking evaluation to benchmarks
- Evaluate retrieval quality, not just speed
- Compare ANN methods on ranking metrics
- Standardized evaluation across ANN libraries

**Integration**:
```rust
// In ann-benchmarks
use rank_eval::binary::ndcg_at_k;

// After ANN search, evaluate ranking quality
let results = ann_index.search(query, k);
let ndcg = ndcg_at_k(&results, &ground_truth, k);
```

### 4. matryoshka-box

**Current State**: Matryoshka representation learning
**Opportunity**: Evaluate ranking quality at different dimensions
- Track nDCG@10 as dimension changes
- Compare full vs. compressed representations
- Validate quality preservation

**Integration**:
```rust
// In matryoshka-box evaluation
use rank_eval::binary::ndcg_at_k;

for dim in [64, 128, 256, 512] {
    let embeddings = project_to_dim(embeddings, dim);
    let ranking = search(embeddings, query);
    let ndcg = ndcg_at_k(&ranking, &qrels, 10);
    println!("nDCG@10 at dim {}: {}", dim, ndcg);
}
```

### 5. rank-relax (Differentiable Ranking)

**Current State**: Not integrated
**Opportunity**: Validate differentiable ranking quality

**Integration Plan**:
1. Add `rank-eval` as dev dependency
2. Create validation tests comparing soft vs. discrete ranking
3. Add training monitoring with ranking metrics
4. Evaluate convergence of differentiable methods

**Example**:
```rust
// In rank-relax tests
use rank_eval::binary::ndcg_at_k;

#[test]
fn test_soft_rank_quality() {
    let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    let soft_ranks = soft_rank(&values, 10.0); // High regularization
    
    // Convert to discrete ranking
    let mut indexed: Vec<_> = soft_ranks.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let ranking: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
    
    // Evaluate (assuming ground truth)
    let relevant: HashSet<_> = [0, 2, 4].into_iter().collect();
    let ndcg = ndcg_at_k(&ranking, &relevant, 10);
    
    // Soft ranking should approach discrete quality
    assert!(ndcg > 0.8, "Soft ranking should maintain quality");
}
```

## Code Quality Improvements

### 1. Documentation

#### Current State
- Good README
- Module-level docs
- Some examples

#### Improvements Needed
- More usage examples in docs
- Tutorial-style guides
- API reference with more detail
- Performance characteristics documented

### 2. Testing

#### Current State
- 58 tests total
- Property-based tests
- Integration tests

#### Improvements Needed
- More edge case tests
- Fuzzing for TREC parsing
- Performance regression tests
- Cross-platform testing

### 3. Error Handling

#### Current State
- Uses `anyhow::Result`
- Basic error messages

#### Improvements Needed
- More specific error types
- Better error context
- Recovery suggestions
- Error codes for programmatic handling

## Implementation Priority

### Phase 1: High-Value, Low-Effort (Do First)
1. ✅ Add ERR and RBP metrics (commonly requested)
2. ✅ Better error messages with line numbers
3. ✅ Input validation for metrics
4. ✅ More dataset loaders (MTEB, HotpotQA)

### Phase 2: High-Value, Medium-Effort (Do Next)
1. Batch evaluation utilities
2. Statistical testing functions
3. Enhanced reporting (CSV, JSON)
4. rank-relax integration

### Phase 3: Medium-Value, High-Effort (Consider)
1. SIMD optimizations
2. Parallel evaluation
3. Auto-download datasets
4. Python bindings

### Phase 4: Nice-to-Have (Future)
1. Interactive HTML reports
2. LaTeX table generation
3. Advanced statistical analysis
4. Integration with more repos

## Recommendations

### Immediate Actions (This Week)
1. **Add ERR and RBP metrics** - High demand, relatively easy
2. **Improve error messages** - Better developer experience
3. **Add input validation** - Prevent subtle bugs
4. **Document integration with rank-relax** - Clear path forward

### Short-Term (This Month)
1. **Integrate with rank-relax** - Add validation tests
2. **Add more dataset loaders** - MTEB, HotpotQA
3. **Batch evaluation utilities** - Performance improvement
4. **Enhanced reporting** - CSV/JSON export

### Long-Term (Next Quarter)
1. **Performance optimizations** - SIMD, parallelization
2. **Statistical testing** - Significance testing, confidence intervals
3. **Python bindings** - Broader adoption
4. **Integration with other repos** - allRank, ann-benchmarks

## Success Metrics

### Adoption Metrics
- Number of repos using `rank-eval`
- Number of downloads (when published)
- GitHub stars/forks

### Quality Metrics
- Test coverage (aim for >90%)
- Documentation coverage
- Performance benchmarks

### Usage Metrics
- Number of metrics computed
- Number of datasets evaluated
- Number of queries processed

## Conclusion

`rank-eval` is well-positioned to become the standard IR evaluation library for Rust. Key opportunities:

1. **Expand metrics**: Add ERR, RBP, and other commonly used metrics
2. **Improve DX**: Better errors, validation, documentation
3. **Performance**: Batch evaluation, parallelization, SIMD
4. **Integration**: More repos, especially rank-relax
5. **Features**: Statistical testing, enhanced reporting

The foundation is solid. Focus on high-value, low-effort improvements first, then tackle more ambitious features as adoption grows.

