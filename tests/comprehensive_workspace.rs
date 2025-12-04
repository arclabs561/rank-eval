//! Comprehensive tests that verify rank-eval works correctly across different usage patterns.
//!
//! These tests simulate real-world usage scenarios from different parts of the workspace.

#[cfg(feature = "serde")]
mod tests {
    use rank_eval::trec::{load_trec_runs, load_qrels, group_runs_by_query, group_qrels_by_query, TrecRun, Qrel};
    use rank_eval::binary::*;
    use rank_eval::graded::*;
    use rank_eval::dataset::*;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_rank_fusion_usage_pattern() {
        // Simulate how rank-fusion/evals uses rank-eval
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("runs.txt");
        let qrels_path = dir.path().join("qrels.txt");

        let mut runs_file = fs::File::create(&runs_path).unwrap();
        writeln!(runs_file, "1 Q0 doc1 1 0.95 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 0.87 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 1 0.93 dense").unwrap();
        writeln!(runs_file, "1 Q0 doc1 2 0.88 dense").unwrap();

        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();
        writeln!(qrels_file, "1 0 doc2 1").unwrap();

        // Load and group (rank-fusion pattern)
        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();
        let runs_by_query = group_runs_by_query(&runs);
        let qrels_by_query = group_qrels_by_query(&qrels);

        // Evaluate fusion result (simulated)
        let query1_runs = &runs_by_query["1"];
        let query1_qrels = &qrels_by_query["1"];

        // Simulate RRF fusion
        let bm25 = &query1_runs["bm25"];
        let dense = &query1_runs["dense"];
        
        // Simple fusion: average scores
        let mut fused: HashMap<String, f32> = HashMap::new();
        for (id, score) in bm25 {
            *fused.entry(id.clone()).or_insert(0.0) += score;
        }
        for (id, score) in dense {
            *fused.entry(id.clone()).or_insert(0.0) += score;
        }
        let mut fused_vec: Vec<(String, f32)> = fused.into_iter().collect();
        fused_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Evaluate fused result
        let ranked_ids: Vec<&str> = fused_vec.iter().map(|(id, _)| id.as_str()).collect();
        let relevant: HashSet<_> = query1_qrels
            .iter()
            .filter(|(_, &rel)| rel > 0)
            .map(|(id, _)| id.as_str())
            .collect();

        let ndcg = ndcg_at_k(&ranked_ids, &relevant, 10);
        assert!(ndcg > 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn test_rank_refine_usage_pattern() {
        // Simulate how rank-refine uses rank-eval for reranking evaluation
        let _query = "Rust memory safety";
        let candidates = vec![
            ("doc1", 0.9), // Highly relevant
            ("doc2", 0.7), // Partially relevant
            ("doc3", 0.5), // Less relevant
        ];

        // Simulate reranking (e.g., MaxSim scoring)
        let mut reranked: Vec<(&str, f32)> = candidates
            .iter()
            .map(|(id, score)| (*id, *score * 1.2)) // Boost scores
            .collect();
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Evaluate reranking improvement
        let ranked_ids: Vec<&str> = reranked.iter().map(|(id, _)| *id).collect();
        let relevant: HashSet<_> = ["doc1", "doc2"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked_ids, &relevant, 10);
        let precision = precision_at_k(&ranked_ids, &relevant, 3);

        assert!(ndcg > 0.0 && ndcg <= 1.0);
        assert!(precision > 0.0 && precision <= 1.0);

        // Verify reranking improved order
        assert_eq!(ranked_ids[0], "doc1", "Most relevant should be first");
    }

    #[test]
    fn test_graded_relevance_evaluation() {
        // Test graded relevance metrics (used in real-world evaluation)
        let ranked = vec![
            ("doc1".to_string(), 0.95),
            ("doc2".to_string(), 0.87),
            ("doc3".to_string(), 0.75),
        ];

        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 3); // Highly relevant
        qrels.insert("doc2".to_string(), 2); // Relevant
        qrels.insert("doc3".to_string(), 1); // Partially relevant

        let ndcg = compute_ndcg(&ranked, &qrels, 10);
        let map = compute_map(&ranked, &qrels);

        assert!(ndcg > 0.0 && ndcg <= 1.0);
        assert!(map > 0.0 && map <= 1.0);
    }

    #[test]
    fn test_dataset_statistics_usage() {
        // Test comprehensive statistics (used in dataset analysis)
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("runs.txt");
        let qrels_path = dir.path().join("qrels.txt");

        let mut runs_file = fs::File::create(&runs_path).unwrap();
        writeln!(runs_file, "1 Q0 doc1 1 0.95 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 0.87 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 1 0.93 dense").unwrap();

        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();
        writeln!(qrels_file, "1 0 doc2 1").unwrap();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        let stats = compute_comprehensive_stats(&runs, &qrels);

        assert_eq!(stats.runs.total_entries, 3);
        assert_eq!(stats.runs.unique_queries, 1);
        assert_eq!(stats.runs.unique_run_tags, 2);
        assert!(stats.quality.queries_with_multiple_runs > 0);
    }

    #[test]
    fn test_validation_workflow() {
        // Test validation workflow (used in dataset validation)
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("runs.txt");
        let qrels_path = dir.path().join("qrels.txt");

        let mut runs_file = fs::File::create(&runs_path).unwrap();
        writeln!(runs_file, "1 Q0 doc1 1 0.95 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 0.87 bm25").unwrap();

        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();
        writeln!(qrels_file, "1 0 doc2 1").unwrap();

        let validation = validate_dataset(&runs_path, &qrels_path).unwrap();

        assert!(validation.is_valid);
        assert!(validation.runs_valid);
        assert!(validation.qrels_valid);
        assert!(validation.consistency_valid);
        assert_eq!(validation.statistics.queries_in_both, 1);
    }

    #[test]
    fn test_metrics_consistency_across_queries() {
        // Test that metrics are computed consistently across multiple queries
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("runs.txt");
        let qrels_path = dir.path().join("qrels.txt");

        let mut runs_file = fs::File::create(&runs_path).unwrap();
        writeln!(runs_file, "1 Q0 doc1 1 0.95 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 0.87 bm25").unwrap();
        writeln!(runs_file, "2 Q0 doc3 1 0.92 bm25").unwrap();
        writeln!(runs_file, "2 Q0 doc4 2 0.85 bm25").unwrap();

        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();
        writeln!(qrels_file, "1 0 doc2 1").unwrap();
        writeln!(qrels_file, "2 0 doc3 2").unwrap();
        writeln!(qrels_file, "2 0 doc4 1").unwrap();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();
        let runs_by_query = group_runs_by_query(&runs);
        let qrels_by_query = group_qrels_by_query(&qrels);

        // Evaluate both queries
        let mut all_ndcgs = Vec::new();
        for query_id in ["1", "2"] {
            let query_runs = &runs_by_query[query_id];
            let query_qrels = &qrels_by_query[query_id];
            let bm25 = &query_runs["bm25"];
            let ranked_ids: Vec<&str> = bm25.iter().map(|(id, _)| id.as_str()).collect();
            let relevant: HashSet<_> = query_qrels
                .iter()
                .filter(|(_, &rel)| rel > 0)
                .map(|(id, _)| id.as_str())
                .collect();
            let ndcg = ndcg_at_k(&ranked_ids, &relevant, 10);
            all_ndcgs.push(ndcg);
        }

        assert_eq!(all_ndcgs.len(), 2);
        assert!(all_ndcgs.iter().all(|&n| n > 0.0 && n <= 1.0));
    }
}

