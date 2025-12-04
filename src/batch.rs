//! Batch evaluation utilities for processing multiple queries efficiently.

use crate::binary::*;
use crate::trec::{Qrel, TrecRun};
use std::collections::{HashMap, HashSet};

/// Results for a single query evaluation.
#[derive(Debug, Clone)]
pub struct QueryResults {
    pub query_id: String,
    pub metrics: HashMap<String, f64>,
}

/// Batch evaluation results across multiple queries.
#[derive(Debug, Clone)]
pub struct BatchResults {
    pub query_results: Vec<QueryResults>,
    pub aggregated: HashMap<String, f64>, // Mean across queries
}

/// Evaluate a batch of rankings using binary relevance metrics.
///
/// # Arguments
///
/// * `rankings` - Vector of ranked document lists, one per query
/// * `qrels` - Vector of relevant document sets, one per query
/// * `metrics` - List of metric names to compute (e.g., ["ndcg@10", "precision@5"])
///
/// # Returns
///
/// `BatchResults` with per-query results and aggregated means.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::batch::evaluate_batch_binary;
///
/// let rankings = vec![
///     vec!["doc1", "doc2", "doc3"],
///     vec!["doc4", "doc5", "doc6"],
/// ];
/// let qrels = vec![
///     ["doc1", "doc3"].into_iter().collect::<HashSet<_>>(),
///     ["doc4"].into_iter().collect::<HashSet<_>>(),
/// ];
///
/// let results = evaluate_batch_binary(&rankings, &qrels, &["ndcg@10", "precision@5"]);
/// assert_eq!(results.query_results.len(), 2);
/// ```
pub fn evaluate_batch_binary<I: Eq + std::hash::Hash + Clone>(
    rankings: &[Vec<I>],
    qrels: &[HashSet<I>],
    metrics: &[&str],
) -> BatchResults {
    assert_eq!(
        rankings.len(),
        qrels.len(),
        "rankings and qrels must have same length"
    );

    let mut query_results = Vec::new();
    let mut metric_sums: HashMap<String, f64> = HashMap::new();
    let mut metric_counts: HashMap<String, usize> = HashMap::new();

    for (_i, (ranked, relevant)) in rankings.iter().zip(qrels.iter()).enumerate() {
        let mut query_metrics = HashMap::new();

        for metric_name in metrics {
            let value = match *metric_name {
                "ndcg@10" => ndcg_at_k(ranked, relevant, 10),
                "ndcg@5" => ndcg_at_k(ranked, relevant, 5),
                "precision@10" => precision_at_k(ranked, relevant, 10),
                "precision@5" => precision_at_k(ranked, relevant, 5),
                "precision@1" => precision_at_k(ranked, relevant, 1),
                "recall@10" => recall_at_k(ranked, relevant, 10),
                "recall@5" => recall_at_k(ranked, relevant, 5),
                "mrr" => mrr(ranked, relevant),
                "ap" | "map" => average_precision(ranked, relevant),
                "err@10" => err_at_k(ranked, relevant, 10),
                "rbp@10" => rbp_at_k(ranked, relevant, 10, 0.95),
                "f1@10" => f_measure_at_k(ranked, relevant, 10, 1.0),
                "success@10" => success_at_k(ranked, relevant, 10),
                "r_precision" => r_precision(ranked, relevant),
                _ => {
                    eprintln!("Unknown metric: {}", metric_name);
                    continue;
                }
            };

            query_metrics.insert(metric_name.to_string(), value);
            *metric_sums.entry(metric_name.to_string()).or_insert(0.0) += value;
            *metric_counts.entry(metric_name.to_string()).or_insert(0) += 1;
        }

        query_results.push(QueryResults {
            query_id: format!("query_{}", _i),
            metrics: query_metrics,
        });
    }

    // Compute aggregated means
    let aggregated: HashMap<String, f64> = metric_sums
        .into_iter()
        .map(|(name, sum)| {
            let count = metric_counts.get(&name).copied().unwrap_or(1);
            (name, sum / count as f64)
        })
        .collect();

    BatchResults {
        query_results,
        aggregated,
    }
}

/// Evaluate TREC runs and qrels in batch.
///
/// Groups runs and qrels by query, then evaluates each query.
///
/// # Arguments
///
/// * `runs` - TREC run entries
/// * `qrels` - TREC qrel entries
/// * `metrics` - List of metric names to compute
///
/// # Returns
///
/// `BatchResults` with per-query results and aggregated means.
pub fn evaluate_trec_batch(
    runs: &[TrecRun],
    qrels: &[Qrel],
    metrics: &[&str],
) -> BatchResults {
    use crate::trec::{group_qrels_by_query, group_runs_by_query};

    let runs_by_query = group_runs_by_query(runs);
    let qrels_by_query = group_qrels_by_query(qrels);

    let mut query_results = Vec::new();
    let mut metric_sums: HashMap<String, f64> = HashMap::new();
    let mut metric_counts: HashMap<String, usize> = HashMap::new();

    for (query_id, query_qrels) in &qrels_by_query {
        // Get first run for this query (or skip if no runs)
        let query_runs = match runs_by_query.get(query_id) {
            Some(runs) => runs,
            None => continue,
        };

        // Use first run tag (or combine all runs)
        let first_run_tag = query_runs.keys().next();
        if first_run_tag.is_none() {
            continue;
        }

        let run_tag = first_run_tag.unwrap();
        let ranked_run = &query_runs[run_tag];

        // Convert to ranked list
        let mut ranked: Vec<(&String, f32)> = ranked_run.iter().map(|(id, score)| (id, *score)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ranked_ids: Vec<&String> = ranked.iter().map(|(id, _)| *id).collect();

        // Convert qrels to HashSet
        let relevant: HashSet<_> = query_qrels
            .iter()
            .filter(|(_, &rel)| rel > 0)
            .map(|(id, _)| id)
            .collect();

        let mut query_metrics = HashMap::new();

        for metric_name in metrics {
            let value = match *metric_name {
                "ndcg@10" => ndcg_at_k(&ranked_ids, &relevant, 10),
                "ndcg@5" => ndcg_at_k(&ranked_ids, &relevant, 5),
                "precision@10" => precision_at_k(&ranked_ids, &relevant, 10),
                "precision@5" => precision_at_k(&ranked_ids, &relevant, 5),
                "precision@1" => precision_at_k(&ranked_ids, &relevant, 1),
                "recall@10" => recall_at_k(&ranked_ids, &relevant, 10),
                "recall@5" => recall_at_k(&ranked_ids, &relevant, 5),
                "mrr" => mrr(&ranked_ids, &relevant),
                "ap" | "map" => average_precision(&ranked_ids, &relevant),
                "err@10" => err_at_k(&ranked_ids, &relevant, 10),
                "rbp@10" => rbp_at_k(&ranked_ids, &relevant, 10, 0.95),
                "f1@10" => f_measure_at_k(&ranked_ids, &relevant, 10, 1.0),
                "success@10" => success_at_k(&ranked_ids, &relevant, 10),
                "r_precision" => r_precision(&ranked_ids, &relevant),
                _ => {
                    eprintln!("Unknown metric: {}", metric_name);
                    continue;
                }
            };

            query_metrics.insert(metric_name.to_string(), value);
            *metric_sums.entry(metric_name.to_string()).or_insert(0.0) += value;
            *metric_counts.entry(metric_name.to_string()).or_insert(0) += 1;
        }

        query_results.push(QueryResults {
            query_id: query_id.clone(),
            metrics: query_metrics,
        });
    }

    // Compute aggregated means
    let aggregated: HashMap<String, f64> = metric_sums
        .into_iter()
        .map(|(name, sum)| {
            let count = metric_counts.get(&name).copied().unwrap_or(1);
            (name, sum / count as f64)
        })
        .collect();

    BatchResults {
        query_results,
        aggregated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_evaluate_batch_binary() {
        let rankings = vec![
            vec!["doc1", "doc2", "doc3"],
            vec!["doc4", "doc5", "doc6"],
        ];
        let qrels = vec![
            ["doc1", "doc3"].into_iter().collect::<HashSet<_>>(),
            ["doc4"].into_iter().collect::<HashSet<_>>(),
        ];

        let results = evaluate_batch_binary(&rankings, &qrels, &["ndcg@10", "precision@5"]);

        assert_eq!(results.query_results.len(), 2);
        assert!(results.aggregated.contains_key("ndcg@10"));
        assert!(results.aggregated.contains_key("precision@5"));
    }
}

