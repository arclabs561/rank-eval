//! Graded relevance IR evaluation metrics.
//!
//! These metrics support graded relevance judgments where documents can have
//! different relevance levels (0 = not relevant, 1 = relevant, 2 = highly relevant, etc.).
//!
//! Unlike binary metrics, these use the actual relevance scores in calculations,
//! making them more suitable for real-world datasets with graded judgments.

use std::collections::HashMap;

/// Compute nDCG@k for graded relevance.
///
/// Uses actual relevance scores (u32) in the DCG calculation, not just binary relevance.
///
/// # Arguments
///
/// * `ranked` - List of (document_id, score) tuples in ranked order
/// * `qrels` - Map from document_id to relevance score (0 = not relevant, 1+ = relevant)
/// * `k` - Cutoff rank for nDCG calculation
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use rank_eval::graded::compute_ndcg;
///
/// let ranked = vec![
///     ("doc1".to_string(), 0.9),
///     ("doc2".to_string(), 0.8),
///     ("doc3".to_string(), 0.7),
/// ];
/// let mut qrels = HashMap::new();
/// qrels.insert("doc1".to_string(), 2); // Highly relevant
/// qrels.insert("doc2".to_string(), 1); // Relevant
/// qrels.insert("doc3".to_string(), 0); // Not relevant
///
/// let ndcg = compute_ndcg(&ranked, &qrels, 3);
/// assert!(ndcg >= 0.0 && ndcg <= 1.0);
/// ```
pub fn compute_ndcg(
    ranked: &[(String, f32)],
    qrels: &HashMap<String, u32>,
    k: usize,
) -> f64 {
    let mut dcg = 0.0;
    let mut ideal_gains: Vec<u32> = qrels.values().copied().filter(|&r| r > 0).collect();
    ideal_gains.sort_by(|a, b| b.cmp(a));

    for (rank, (doc_id, _)) in ranked.iter().take(k).enumerate() {
        if let Some(&relevance) = qrels.get(doc_id.as_str()) {
            if relevance > 0 {
                // Use log2(rank + 2) for DCG calculation
                dcg += (relevance as f64) / ((rank + 2) as f64).log2();
            }
        }
    }

    let mut idcg = 0.0;
    for (rank, &gain) in ideal_gains.iter().take(k).enumerate() {
        idcg += (gain as f64) / ((rank + 2) as f64).log2();
    }

    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

/// Compute Mean Average Precision (MAP) for graded relevance.
///
/// Uses binary relevance (relevance > 0) for MAP calculation, as MAP is
/// traditionally defined for binary relevance.
///
/// # Arguments
///
/// * `ranked` - List of (document_id, score) tuples in ranked order
/// * `qrels` - Map from document_id to relevance score
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use rank_eval::graded::compute_map;
///
/// let ranked = vec![
///     ("doc1".to_string(), 0.9),
///     ("doc2".to_string(), 0.8),
///     ("doc3".to_string(), 0.7),
/// ];
/// let mut qrels = HashMap::new();
/// qrels.insert("doc1".to_string(), 2);
/// qrels.insert("doc2".to_string(), 1);
/// qrels.insert("doc3".to_string(), 0);
///
/// let map = compute_map(&ranked, &qrels);
/// assert!(map >= 0.0 && map <= 1.0);
/// ```
pub fn compute_map(ranked: &[(String, f32)], qrels: &HashMap<String, u32>) -> f64 {
    let relevant_docs: Vec<&String> = qrels
        .iter()
        .filter(|(_, &rel)| rel > 0)
        .map(|(doc_id, _)| doc_id)
        .collect();

    if relevant_docs.is_empty() {
        return 0.0;
    }

    let mut sum_precision = 0.0;
    let mut relevant_found = 0;

    for (rank, (doc_id, _)) in ranked.iter().enumerate() {
        if qrels.get(doc_id.as_str()).unwrap_or(&0) > &0 {
            relevant_found += 1;
            sum_precision += relevant_found as f64 / (rank + 1) as f64;
        }
    }

    if relevant_found > 0 {
        sum_precision / relevant_docs.len() as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_ndcg_graded() {
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 2); // Highly relevant
        qrels.insert("doc2".to_string(), 1); // Relevant
        qrels.insert("doc3".to_string(), 0); // Not relevant

        let ndcg = compute_ndcg(&ranked, &qrels, 3);
        assert!(ndcg > 0.0 && ndcg <= 1.0);
        
        // doc1 (relevance 2) at rank 0 should contribute more than doc2 (relevance 1) at rank 1
        // So nDCG should be > 0
        assert!(ndcg > 0.5); // Should be reasonably high since highly relevant doc is first
    }

    #[test]
    fn test_compute_map_graded() {
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 2);
        qrels.insert("doc2".to_string(), 1);
        qrels.insert("doc3".to_string(), 0);

        let map = compute_map(&ranked, &qrels);
        assert!(map >= 0.0 && map <= 1.0);
        
        // Both doc1 and doc2 are relevant (relevance > 0)
        // doc1 at rank 0: precision = 1/1 = 1.0
        // doc2 at rank 1: precision = 2/2 = 1.0
        // MAP = (1.0 + 1.0) / 2 = 1.0
        assert!((map - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_ndcg_no_relevant() {
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
        ];
        let qrels = HashMap::new(); // No relevant documents

        let ndcg = compute_ndcg(&ranked, &qrels, 2);
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_compute_map_no_relevant() {
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
        ];
        let qrels = HashMap::new(); // No relevant documents

        let map = compute_map(&ranked, &qrels);
        assert_eq!(map, 0.0);
    }
}

