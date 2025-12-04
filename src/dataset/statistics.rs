//! Dataset statistics and analysis utilities.
//!
//! Provides detailed statistics about datasets, run files, and qrels.

use crate::trec::{Qrel, TrecRun};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Comprehensive dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveStats {
    pub runs: RunStatistics,
    pub qrels: QrelStatistics,
    pub overlap: OverlapStatistics,
    pub quality: QualityMetrics,
}

/// Run file statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStatistics {
    pub total_entries: usize,
    pub unique_queries: usize,
    pub unique_documents: usize,
    pub unique_run_tags: usize,
    pub run_tags: Vec<String>,
    pub queries_per_run: HashMap<String, usize>,
    pub documents_per_run: HashMap<String, usize>,
    pub avg_docs_per_query: f64,
    pub max_docs_per_query: usize,
    pub min_docs_per_query: usize,
    pub score_distribution: ScoreDistribution,
}

/// Qrel statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QrelStatistics {
    pub total_entries: usize,
    pub unique_queries: usize,
    pub unique_documents: usize,
    pub queries_with_relevant: usize,
    pub total_relevant: usize,
    pub avg_relevance_per_query: f64,
    pub relevance_distribution: HashMap<u32, usize>,
}

/// Overlap statistics between runs and qrels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapStatistics {
    pub queries_in_both: usize,
    pub queries_only_in_runs: usize,
    pub queries_only_in_qrels: usize,
    pub documents_in_both: usize,
    pub documents_only_in_runs: usize,
    pub documents_only_in_qrels: usize,
    pub query_overlap_ratio: f64,
    pub document_overlap_ratio: f64,
}

/// Quality metrics for the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub queries_with_multiple_runs: usize,
    pub queries_with_single_run: usize,
    pub queries_ready_for_fusion: usize,
    pub fusion_readiness_ratio: f64,
    pub avg_runs_per_query: f64,
}

/// Score distribution statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDistribution {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub percentiles: Percentiles,
}

/// Percentile values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p90: f32,
    pub p95: f32,
    pub p99: f32,
}

/// Compute comprehensive statistics for a dataset.
pub fn compute_comprehensive_stats(
    runs: &[TrecRun],
    qrels: &[Qrel],
) -> ComprehensiveStats {
    let runs_stats = compute_run_statistics(runs);
    let qrels_stats = compute_qrel_statistics(qrels);
    let overlap = compute_overlap_statistics(runs, qrels);
    let quality = compute_quality_metrics(runs);

    ComprehensiveStats {
        runs: runs_stats,
        qrels: qrels_stats,
        overlap,
        quality,
    }
}

/// Compute statistics for run files.
fn compute_run_statistics(runs: &[TrecRun]) -> RunStatistics {
    if runs.is_empty() {
        return RunStatistics {
            total_entries: 0,
            unique_queries: 0,
            unique_documents: 0,
            unique_run_tags: 0,
            run_tags: Vec::new(),
            queries_per_run: HashMap::new(),
            documents_per_run: HashMap::new(),
            avg_docs_per_query: 0.0,
            max_docs_per_query: 0,
            min_docs_per_query: 0,
            score_distribution: ScoreDistribution {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                percentiles: Percentiles {
                    p25: 0.0,
                    p50: 0.0,
                    p75: 0.0,
                    p90: 0.0,
                    p95: 0.0,
                    p99: 0.0,
                },
            },
        };
    }

    let unique_queries: HashSet<String> = runs.iter().map(|r| r.query_id.clone()).collect();
    let unique_documents: HashSet<String> = runs.iter().map(|r| r.doc_id.clone()).collect();
    let unique_run_tags: HashSet<String> = runs.iter().map(|r| r.run_tag.clone()).collect();
    
    let mut queries_per_run: HashMap<String, usize> = HashMap::new();
    let mut documents_per_run: HashMap<String, usize> = HashMap::new();
    let mut docs_per_query: HashMap<String, usize> = HashMap::new();
    let mut scores: Vec<f32> = runs.iter().map(|r| r.score).collect();

    for run in runs {
        *queries_per_run.entry(run.run_tag.clone()).or_insert(0) += 1;
        *documents_per_run.entry(run.run_tag.clone()).or_insert(0) += 1;
        *docs_per_query.entry(run.query_id.clone()).or_insert(0) += 1;
    }

    // Remove duplicates for document counting per run
    let mut unique_docs_per_run: HashMap<String, HashSet<String>> = HashMap::new();
    for run in runs {
        unique_docs_per_run
            .entry(run.run_tag.clone())
            .or_default()
            .insert(run.doc_id.clone());
    }
    for (tag, docs) in &unique_docs_per_run {
        documents_per_run.insert(tag.clone(), docs.len());
    }

    let docs_per_query_values: Vec<usize> = docs_per_query.values().copied().collect();
    let avg_docs_per_query = if !docs_per_query_values.is_empty() {
        docs_per_query_values.iter().sum::<usize>() as f64 / docs_per_query_values.len() as f64
    } else {
        0.0
    };

    let max_docs_per_query = docs_per_query_values.iter().max().copied().unwrap_or(0);
    let min_docs_per_query = docs_per_query_values.iter().min().copied().unwrap_or(0);

    // Compute score distribution
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let score_dist = compute_score_distribution(&scores);

    RunStatistics {
        total_entries: runs.len(),
        unique_queries: unique_queries.len(),
        unique_documents: unique_documents.len(),
        unique_run_tags: unique_run_tags.len(),
        run_tags: unique_run_tags.iter().cloned().collect::<Vec<_>>(),
        queries_per_run,
        documents_per_run,
        avg_docs_per_query,
        max_docs_per_query,
        min_docs_per_query,
        score_distribution: score_dist,
    }
}

/// Compute statistics for qrels.
fn compute_qrel_statistics(qrels: &[Qrel]) -> QrelStatistics {
    if qrels.is_empty() {
        return QrelStatistics {
            total_entries: 0,
            unique_queries: 0,
            unique_documents: 0,
            queries_with_relevant: 0,
            total_relevant: 0,
            avg_relevance_per_query: 0.0,
            relevance_distribution: HashMap::new(),
        };
    }

    let unique_queries: HashSet<String> = qrels.iter().map(|q| q.query_id.clone()).collect();
    let unique_documents: HashSet<String> = qrels.iter().map(|q| q.doc_id.clone()).collect();
    
    let mut relevance_dist: HashMap<u32, usize> = HashMap::new();
    let mut queries_with_relevant: HashSet<String> = HashSet::new();
    let mut total_relevant = 0;

    for qrel in qrels {
        *relevance_dist.entry(qrel.relevance).or_insert(0) += 1;
        if qrel.relevance > 0 {
            queries_with_relevant.insert(qrel.query_id.clone());
            total_relevant += 1;
        }
    }

    let avg_relevance_per_query = if !unique_queries.is_empty() {
        total_relevant as f64 / unique_queries.len() as f64
    } else {
        0.0
    };

    QrelStatistics {
        total_entries: qrels.len(),
        unique_queries: unique_queries.len(),
        unique_documents: unique_documents.len(),
        queries_with_relevant: queries_with_relevant.len(),
        total_relevant,
        avg_relevance_per_query,
        relevance_distribution: relevance_dist,
    }
}

/// Compute overlap statistics.
fn compute_overlap_statistics(runs: &[TrecRun], qrels: &[Qrel]) -> OverlapStatistics {
    let runs_queries: HashSet<String> = runs.iter().map(|r| r.query_id.clone()).collect();
    let qrels_queries: HashSet<String> = qrels.iter().map(|q| q.query_id.clone()).collect();
    
    let queries_in_both: HashSet<_> = runs_queries.intersection(&qrels_queries).cloned().collect();
    let queries_only_in_runs: HashSet<_> = runs_queries.difference(&qrels_queries).cloned().collect();
    let queries_only_in_qrels: HashSet<_> = qrels_queries.difference(&runs_queries).cloned().collect();

    let runs_docs: HashSet<String> = runs.iter().map(|r| r.doc_id.clone()).collect();
    let qrels_docs: HashSet<String> = qrels.iter().map(|q| q.doc_id.clone()).collect();
    
    let documents_in_both: HashSet<_> = runs_docs.intersection(&qrels_docs).cloned().collect();
    let documents_only_in_runs: HashSet<_> = runs_docs.difference(&qrels_docs).cloned().collect();
    let documents_only_in_qrels: HashSet<_> = qrels_docs.difference(&runs_docs).cloned().collect();

    let query_overlap_ratio = if !runs_queries.is_empty() {
        queries_in_both.len() as f64 / runs_queries.len() as f64
    } else {
        0.0
    };

    let document_overlap_ratio = if !runs_docs.is_empty() {
        documents_in_both.len() as f64 / runs_docs.len() as f64
    } else {
        0.0
    };

    OverlapStatistics {
        queries_in_both: queries_in_both.len(),
        queries_only_in_runs: queries_only_in_runs.len(),
        queries_only_in_qrels: queries_only_in_qrels.len(),
        documents_in_both: documents_in_both.len(),
        documents_only_in_runs: documents_only_in_runs.len(),
        documents_only_in_qrels: documents_only_in_qrels.len(),
        query_overlap_ratio,
        document_overlap_ratio,
    }
}

/// Compute quality metrics.
fn compute_quality_metrics(runs: &[TrecRun]) -> QualityMetrics {
    let mut queries_with_runs: HashMap<String, HashSet<String>> = HashMap::new();
    
    for run in runs {
        queries_with_runs
            .entry(run.query_id.clone())
            .or_default()
            .insert(run.run_tag.clone());
    }

    let mut queries_with_multiple_runs = 0;
    let mut queries_with_single_run = 0;
    let mut total_runs = 0;

    for run_tags in queries_with_runs.values() {
        let count = run_tags.len();
        total_runs += count;
        if count >= 2 {
            queries_with_multiple_runs += 1;
        } else {
            queries_with_single_run += 1;
        }
    }

    let total_queries = queries_with_runs.len();
    let avg_runs_per_query = if total_queries > 0 {
        total_runs as f64 / total_queries as f64
    } else {
        0.0
    };

    let fusion_readiness_ratio = if total_queries > 0 {
        queries_with_multiple_runs as f64 / total_queries as f64
    } else {
        0.0
    };

    QualityMetrics {
        queries_with_multiple_runs,
        queries_with_single_run,
        queries_ready_for_fusion: queries_with_multiple_runs,
        fusion_readiness_ratio,
        avg_runs_per_query,
    }
}

/// Compute score distribution.
fn compute_score_distribution(scores: &[f32]) -> ScoreDistribution {
    if scores.is_empty() {
        return ScoreDistribution {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            percentiles: Percentiles {
                p25: 0.0,
                p50: 0.0,
                p75: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
            },
        };
    }

    let min = scores[0];
    let max = scores[scores.len() - 1];
    let mean = scores.iter().sum::<f32>() as f64 / scores.len() as f64;
    
    let variance = scores.iter()
        .map(|&s| {
            let diff = s as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / scores.len() as f64;
    let std_dev = variance.sqrt();

    let median = if scores.len() % 2 == 0 {
        (scores[scores.len() / 2 - 1] as f64 + scores[scores.len() / 2] as f64) / 2.0
    } else {
        scores[scores.len() / 2] as f64
    };

    let p25_idx = (scores.len() as f64 * 0.25) as usize;
    let p50_idx = (scores.len() as f64 * 0.50) as usize;
    let p75_idx = (scores.len() as f64 * 0.75) as usize;
    let p90_idx = (scores.len() as f64 * 0.90) as usize;
    let p95_idx = (scores.len() as f64 * 0.95) as usize;
    let p99_idx = ((scores.len() as f64 * 0.99) as usize).min(scores.len() - 1);

    ScoreDistribution {
        min,
        max,
        mean,
        median,
        std_dev,
        percentiles: Percentiles {
            p25: scores[p25_idx],
            p50: scores[p50_idx],
            p75: scores[p75_idx],
            p90: scores[p90_idx],
            p95: scores[p95_idx],
            p99: scores[p99_idx],
        },
    }
}

/// Print comprehensive statistics report.
pub fn print_statistics_report(stats: &ComprehensiveStats) {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              Dataset Statistics Report                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("┌─ Run File Statistics ──────────────────────────────────────────┐");
    println!("│ Total entries:        {:>10}                                │", stats.runs.total_entries);
    println!("│ Unique queries:       {:>10}                                │", stats.runs.unique_queries);
    println!("│ Unique documents:     {:>10}                                │", stats.runs.unique_documents);
    println!("│ Unique run tags:      {:>10}                                │", stats.runs.unique_run_tags);
    println!("│ Run tags:             {:>10}                                │", stats.runs.run_tags.join(", "));
    println!("│ Avg docs per query:   {:>10.2}                                │", stats.runs.avg_docs_per_query);
    println!("│ Min docs per query:   {:>10}                                │", stats.runs.min_docs_per_query);
    println!("│ Max docs per query:   {:>10}                                │", stats.runs.max_docs_per_query);
    println!("└────────────────────────────────────────────────────────────────┘\n");

    println!("┌─ Score Distribution ───────────────────────────────────────────┐");
    println!("│ Min:        {:>10.6}  │  Max:        {:>10.6}              │", 
        stats.runs.score_distribution.min, stats.runs.score_distribution.max);
    println!("│ Mean:       {:>10.6}  │  Median:     {:>10.6}              │", 
        stats.runs.score_distribution.mean, stats.runs.score_distribution.median);
    println!("│ Std Dev:    {:>10.6}  │  P25:        {:>10.6}              │", 
        stats.runs.score_distribution.std_dev, stats.runs.score_distribution.percentiles.p25);
    println!("│ P50:        {:>10.6}  │  P75:        {:>10.6}              │", 
        stats.runs.score_distribution.percentiles.p50, stats.runs.score_distribution.percentiles.p75);
    println!("│ P90:        {:>10.6}  │  P95:        {:>10.6}              │", 
        stats.runs.score_distribution.percentiles.p90, stats.runs.score_distribution.percentiles.p95);
    println!("└────────────────────────────────────────────────────────────────┘\n");

    println!("┌─ Qrel Statistics ─────────────────────────────────────────────┐");
    println!("│ Total entries:        {:>10}                                │", stats.qrels.total_entries);
    println!("│ Unique queries:       {:>10}                                │", stats.qrels.unique_queries);
    println!("│ Unique documents:      {:>10}                                │", stats.qrels.unique_documents);
    println!("│ Queries with relevant: {:>10}                                │", stats.qrels.queries_with_relevant);
    println!("│ Total relevant docs:   {:>10}                                │", stats.qrels.total_relevant);
    println!("│ Avg relevance/query:   {:>10.2}                                │", stats.qrels.avg_relevance_per_query);
    println!("└────────────────────────────────────────────────────────────────┘\n");

    println!("┌─ Overlap Statistics ───────────────────────────────────────────┐");
    println!("│ Queries in both:      {:>10}  ({:.1}% overlap)            │", 
        stats.overlap.queries_in_both, stats.overlap.query_overlap_ratio * 100.0);
    println!("│ Queries only in runs: {:>10}                                │", stats.overlap.queries_only_in_runs);
    println!("│ Queries only in qrels: {:>10}                                │", stats.overlap.queries_only_in_qrels);
    println!("│ Documents in both:    {:>10}  ({:.1}% overlap)            │", 
        stats.overlap.documents_in_both, stats.overlap.document_overlap_ratio * 100.0);
    println!("└────────────────────────────────────────────────────────────────┘\n");

    println!("┌─ Quality Metrics ─────────────────────────────────────────────┐");
    println!("│ Queries with 2+ runs: {:>10}  ({:.1}% ready)              │", 
        stats.quality.queries_with_multiple_runs, stats.quality.fusion_readiness_ratio * 100.0);
    println!("│ Queries with 1 run:   {:>10}                                │", stats.quality.queries_with_single_run);
    println!("│ Avg runs per query:   {:>10.2}                                │", stats.quality.avg_runs_per_query);
    println!("│ Fusion readiness:     {:>10.1}%                                │", stats.quality.fusion_readiness_ratio * 100.0);
    println!("└────────────────────────────────────────────────────────────────┘\n");
}


