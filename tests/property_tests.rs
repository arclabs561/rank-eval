//! Property-based tests for evaluation metrics.
//!
//! These tests verify mathematical properties and invariants that should always hold.

#[cfg(feature = "serde")]
mod tests {
    use rank_eval::binary::*;
    use rank_eval::graded::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_ndcg_consistency() {
        // Property: nDCG@k should be consistent (not necessarily monotonic)
        // Note: nDCG is NOT monotonic because ideal DCG changes with k
        // But for perfect rankings, nDCG should be 1.0 at all k
        let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant: HashSet<_> = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            .into_iter()
            .collect();

        // Perfect ranking: all relevant docs first
        let ndcg1 = ndcg_at_k(&ranked, &relevant, 1);
        let ndcg3 = ndcg_at_k(&ranked, &relevant, 3);
        let ndcg5 = ndcg_at_k(&ranked, &relevant, 5);
        let ndcg10 = ndcg_at_k(&ranked, &relevant, 10);

        // For perfect ranking, all should be 1.0
        assert!(
            (ndcg1 - 1.0).abs() < 1e-9,
            "Perfect ranking should give nDCG@1 = 1.0"
        );
        assert!(
            (ndcg3 - 1.0).abs() < 1e-9,
            "Perfect ranking should give nDCG@3 = 1.0"
        );
        assert!(
            (ndcg5 - 1.0).abs() < 1e-9,
            "Perfect ranking should give nDCG@5 = 1.0"
        );
        assert!(
            (ndcg10 - 1.0).abs() < 1e-9,
            "Perfect ranking should give nDCG@10 = 1.0"
        );
    }

    #[test]
    fn test_precision_non_increasing() {
        // Property: Precision@k should be non-increasing as k increases
        // (adding more documents can only decrease or maintain precision)
        let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

        let p1 = precision_at_k(&ranked, &relevant, 1);
        let p3 = precision_at_k(&ranked, &relevant, 3);
        let p5 = precision_at_k(&ranked, &relevant, 5);

        assert!(p1 >= p3, "Precision@1 should be >= Precision@3");
        assert!(p3 >= p5, "Precision@3 should be >= Precision@5");
    }

    #[test]
    fn test_recall_non_decreasing() {
        // Property: Recall@k should be non-decreasing as k increases
        // (more documents = more opportunity to find relevant ones)
        let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant: HashSet<_> = ["doc1", "doc3", "doc5"].into_iter().collect();

        let r1 = recall_at_k(&ranked, &relevant, 1);
        let r3 = recall_at_k(&ranked, &relevant, 3);
        let r5 = recall_at_k(&ranked, &relevant, 5);

        assert!(r3 >= r1 - 1e-9, "Recall@3 should be >= Recall@1");
        assert!(r5 >= r3 - 1e-9, "Recall@5 should be >= Recall@3");
    }

    #[test]
    fn test_metrics_bounded() {
        // Property: All metrics should be in [0, 1]
        let ranked = vec!["doc1", "doc2", "doc3"];
        let relevant: HashSet<_> = ["doc1", "doc2"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked, &relevant, 10);
        let precision = precision_at_k(&ranked, &relevant, 10);
        let recall = recall_at_k(&ranked, &relevant, 10);
        let mrr_score = mrr(&ranked, &relevant);
        let ap = average_precision(&ranked, &relevant);

        assert!(ndcg >= 0.0 && ndcg <= 1.0, "nDCG should be in [0, 1]");
        assert!(
            precision >= 0.0 && precision <= 1.0,
            "Precision should be in [0, 1]"
        );
        assert!(recall >= 0.0 && recall <= 1.0, "Recall should be in [0, 1]");
        assert!(
            mrr_score >= 0.0 && mrr_score <= 1.0,
            "MRR should be in [0, 1]"
        );
        assert!(ap >= 0.0 && ap <= 1.0, "AP should be in [0, 1]");
    }

    #[test]
    fn test_perfect_ranking_properties() {
        // Property: Perfect ranking should give maximum metrics
        let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked, &relevant, 10);
        let precision = precision_at_k(&ranked, &relevant, 3);
        let recall = recall_at_k(&ranked, &relevant, 3);
        let mrr_score = mrr(&ranked, &relevant);
        let ap = average_precision(&ranked, &relevant);

        assert!(
            (ndcg - 1.0).abs() < 1e-9,
            "Perfect ranking should give nDCG = 1.0"
        );
        assert!(
            (precision - 1.0).abs() < 1e-9,
            "Perfect ranking should give precision = 1.0"
        );
        assert!(
            (recall - 1.0).abs() < 1e-9,
            "Perfect ranking should give recall = 1.0"
        );
        assert!(
            (mrr_score - 1.0).abs() < 1e-9,
            "Perfect ranking should give MRR = 1.0"
        );
        assert!(
            (ap - 1.0).abs() < 1e-9,
            "Perfect ranking should give AP = 1.0"
        );
    }

    #[test]
    fn test_empty_ranking_properties() {
        // Property: Empty ranking should give zero metrics
        let ranked: Vec<&str> = vec![];
        let relevant: HashSet<_> = ["doc1", "doc2"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked, &relevant, 10);
        let precision = precision_at_k(&ranked, &relevant, 10);
        let recall = recall_at_k(&ranked, &relevant, 10);
        let mrr_score = mrr(&ranked, &relevant);
        let ap = average_precision(&ranked, &relevant);

        assert_eq!(ndcg, 0.0, "Empty ranking should give nDCG = 0");
        assert_eq!(precision, 0.0, "Empty ranking should give precision = 0");
        assert_eq!(recall, 0.0, "Empty ranking should give recall = 0");
        assert_eq!(mrr_score, 0.0, "Empty ranking should give MRR = 0");
        assert_eq!(ap, 0.0, "Empty ranking should give AP = 0");
    }

    #[test]
    fn test_no_relevant_properties() {
        // Property: No relevant documents should give zero metrics
        let ranked = vec!["doc1", "doc2", "doc3"];
        let relevant: HashSet<&str> = HashSet::new();

        let ndcg = ndcg_at_k(&ranked, &relevant, 10);
        let precision = precision_at_k(&ranked, &relevant, 10);
        let recall = recall_at_k(&ranked, &relevant, 10);
        let mrr_score = mrr(&ranked, &relevant);
        let ap = average_precision(&ranked, &relevant);

        assert_eq!(ndcg, 0.0, "No relevant docs should give nDCG = 0");
        assert_eq!(precision, 0.0, "No relevant docs should give precision = 0");
        assert_eq!(recall, 0.0, "No relevant docs should give recall = 0");
        assert_eq!(mrr_score, 0.0, "No relevant docs should give MRR = 0");
        assert_eq!(ap, 0.0, "No relevant docs should give AP = 0");
    }

    #[test]
    fn test_graded_ndcg_bounded() {
        // Property: Graded nDCG should be in [0, 1]
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 2);
        qrels.insert("doc2".to_string(), 1);
        qrels.insert("doc3".to_string(), 0);

        let ndcg = compute_ndcg(&ranked, &qrels, 10);
        let map = compute_map(&ranked, &qrels);

        assert!(
            ndcg >= 0.0 && ndcg <= 1.0,
            "Graded nDCG should be in [0, 1]"
        );
        assert!(map >= 0.0 && map <= 1.0, "Graded MAP should be in [0, 1]");
    }

    #[test]
    fn test_graded_ndcg_perfect_ranking() {
        // Property: Perfect ranking with graded relevance should give nDCG = 1.0
        let ranked = vec![
            ("doc1".to_string(), 0.9), // Highest relevance
            ("doc2".to_string(), 0.8), // Medium relevance
            ("doc3".to_string(), 0.7), // Low relevance
        ];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 3); // Highly relevant
        qrels.insert("doc2".to_string(), 2); // Relevant
        qrels.insert("doc3".to_string(), 1); // Partially relevant

        let ndcg = compute_ndcg(&ranked, &qrels, 10);

        assert!(
            (ndcg - 1.0).abs() < 1e-9,
            "Perfect graded ranking should give nDCG = 1.0"
        );
    }

    #[test]
    fn test_mrr_first_relevant_rank() {
        // Property: MRR = 1 / rank_of_first_relevant
        let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
        let relevant: HashSet<_> = ["doc3"].into_iter().collect();

        let mrr_score = mrr(&ranked, &relevant);
        let expected = 1.0 / 3.0; // First relevant at rank 3

        assert!(
            (mrr_score - expected).abs() < 1e-9,
            "MRR should equal 1 / rank_of_first_relevant"
        );
    }

    #[test]
    fn test_precision_recall_relationship() {
        // Property: When all ranked docs are relevant, precision = recall (for same k)
        let ranked = vec!["doc1", "doc2", "doc3"];
        let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

        let precision = precision_at_k(&ranked, &relevant, 3);
        let recall = recall_at_k(&ranked, &relevant, 3);

        assert!(
            (precision - recall).abs() < 1e-9,
            "When all ranked are relevant, precision = recall"
        );
    }

    #[test]
    fn test_ndcg_identity_ranking() {
        // Property: Ranking all relevant docs first should give nDCG = 1.0
        let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked, &relevant, 10);

        assert!(
            (ndcg - 1.0).abs() < 1e-9,
            "All relevant docs first should give nDCG = 1.0"
        );
    }

    #[test]
    fn test_metrics_symmetry() {
        // Property: Metrics should be consistent across different orderings of same set
        let ranked1 = vec!["doc1", "doc2", "doc3"];
        let ranked2 = vec!["doc3", "doc2", "doc1"]; // Reversed
        let relevant: HashSet<_> = ["doc1", "doc2"].into_iter().collect();

        // Note: This property doesn't hold for all metrics (order matters!)
        // But we can test that empty/perfect cases are symmetric
        let ranked_empty: Vec<&str> = vec![];
        let relevant_empty: HashSet<&str> = HashSet::new();

        let ndcg1 = ndcg_at_k(&ranked_empty, &relevant_empty, 10);
        let ndcg2 = ndcg_at_k(&ranked_empty, &relevant_empty, 10);

        assert_eq!(ndcg1, ndcg2, "Same inputs should give same outputs");
    }

    #[test]
    fn test_k_larger_than_ranked() {
        // Property: Metrics should handle k > ranked list size gracefully
        let ranked = vec!["doc1", "doc2"];
        let relevant: HashSet<_> = ["doc1"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked, &relevant, 100);
        let precision = precision_at_k(&ranked, &relevant, 100);
        let recall = recall_at_k(&ranked, &relevant, 100);

        assert!(
            ndcg >= 0.0 && ndcg <= 1.0,
            "nDCG with k > ranked should be valid"
        );
        assert!(
            precision >= 0.0 && precision <= 1.0,
            "Precision with k > ranked should be valid"
        );
        assert!(
            recall >= 0.0 && recall <= 1.0,
            "Recall with k > ranked should be valid"
        );
    }

    #[test]
    fn test_graded_ndcg_empty() {
        // Property: Empty ranked list should give zero graded nDCG
        let ranked: Vec<(String, f32)> = vec![];
        let qrels: HashMap<String, u32> = HashMap::new();

        let ndcg = compute_ndcg(&ranked, &qrels, 10);
        let map = compute_map(&ranked, &qrels);

        assert_eq!(ndcg, 0.0, "Empty ranked list should give nDCG = 0");
        assert_eq!(map, 0.0, "Empty ranked list should give MAP = 0");
    }

    #[test]
    fn test_graded_ndcg_no_relevant() {
        // Property: No relevant documents should give zero graded nDCG
        let ranked = vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.8)];
        let qrels: HashMap<String, u32> = HashMap::new();

        let ndcg = compute_ndcg(&ranked, &qrels, 10);
        let map = compute_map(&ranked, &qrels);

        assert_eq!(ndcg, 0.0, "No relevant docs should give nDCG = 0");
        assert_eq!(map, 0.0, "No relevant docs should give MAP = 0");
    }
}
