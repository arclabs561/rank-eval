//! Comprehensive tests for dataset loading, validation, and statistics.

#[cfg(feature = "serde")]
mod tests {
    use rank_eval::dataset::*;
    use rank_eval::trec::{load_trec_runs, load_qrels};
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_temp_trec_runs() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("runs.txt");
        let mut file = fs::File::create(&file_path).unwrap();
        
        writeln!(file, "1 Q0 doc1 1 0.95 bm25").unwrap();
        writeln!(file, "1 Q0 doc2 2 0.87 bm25").unwrap();
        writeln!(file, "1 Q0 doc3 3 0.75 bm25").unwrap();
        writeln!(file, "2 Q0 doc4 1 0.92 bm25").unwrap();
        writeln!(file, "2 Q0 doc5 2 0.85 bm25").unwrap();
        writeln!(file, "1 Q0 doc2 1 0.93 dense").unwrap();
        writeln!(file, "1 Q0 doc1 2 0.88 dense").unwrap();
        writeln!(file, "1 Q0 doc3 3 0.82 dense").unwrap();
        
        (dir, file_path)
    }

    fn create_temp_trec_qrels() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("qrels.txt");
        let mut file = fs::File::create(&file_path).unwrap();
        
        writeln!(file, "1 0 doc1 2").unwrap();
        writeln!(file, "1 0 doc2 1").unwrap();
        writeln!(file, "1 0 doc3 0").unwrap();
        writeln!(file, "2 0 doc4 2").unwrap();
        writeln!(file, "2 0 doc5 1").unwrap();
        
        (dir, file_path)
    }

    #[test]
    fn test_dataset_loading() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        assert_eq!(runs.len(), 8);
        assert_eq!(qrels.len(), 5);
    }

    #[test]
    fn test_dataset_validation() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let result = validate_dataset(&runs_path, &qrels_path).unwrap();

        assert!(result.is_valid);
        assert!(result.runs_valid);
        assert!(result.qrels_valid);
        assert!(result.consistency_valid);
        assert_eq!(result.statistics.queries_in_both, 2);
    }

    #[test]
    fn test_dataset_statistics() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        let stats = compute_comprehensive_stats(&runs, &qrels);

        assert_eq!(stats.runs.total_entries, 8);
        assert_eq!(stats.runs.unique_queries, 2);
        assert_eq!(stats.runs.unique_run_tags, 2);
        assert_eq!(stats.qrels.total_entries, 5);
        assert_eq!(stats.qrels.unique_queries, 2);
        assert_eq!(stats.overlap.queries_in_both, 2);
        assert!(stats.quality.queries_with_multiple_runs > 0);
    }

    #[test]
    fn test_dataset_loaders() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        // Test MS MARCO loader (just wraps TREC loader)
        let runs = load_msmarco_runs(runs_path.parent().unwrap(), &["runs.txt"]).unwrap();
        assert_eq!(runs.len(), 8);

        let qrels = load_msmarco_qrels(&qrels_path).unwrap();
        assert_eq!(qrels.len(), 5);
    }

    #[test]
    fn test_dataset_stats_helper() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        let stats = get_dataset_stats(&runs, &qrels);

        assert_eq!(stats.total_runs, 8);
        assert_eq!(stats.unique_queries, 2);
        assert_eq!(stats.unique_documents, 5);
        assert_eq!(stats.unique_run_tags, 2);
        assert_eq!(stats.total_qrels, 5);
        assert_eq!(stats.queries_with_qrels, 2);
        assert_eq!(stats.relevant_documents, 4);
    }

    #[test]
    fn test_validate_dataset_dir() {
        let dir = TempDir::new().unwrap();
        
        // Create a run file
        let runs_file = dir.path().join("runs.txt");
        let mut file = fs::File::create(&runs_file).unwrap();
        writeln!(file, "1 Q0 doc1 1 0.9 bm25").unwrap();
        
        // Create a qrels file
        let qrels_file = dir.path().join("qrels.txt");
        let mut file = fs::File::create(&qrels_file).unwrap();
        writeln!(file, "1 0 doc1 2").unwrap();
        
        let is_valid = validate_dataset_dir(dir.path()).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_list_datasets() {
        let dir = TempDir::new().unwrap();
        let dataset1 = dir.path().join("dataset1");
        let dataset2 = dir.path().join("dataset2");
        
        fs::create_dir(&dataset1).unwrap();
        fs::create_dir(&dataset2).unwrap();

        let datasets = list_datasets(dir.path()).unwrap();
        assert_eq!(datasets.len(), 2);
        assert!(datasets.contains(&"dataset1".to_string()));
        assert!(datasets.contains(&"dataset2".to_string()));
    }

    #[test]
    fn test_dataset_type_detection() {
        let dir = TempDir::new().unwrap();
        
        // Test default (TREC)
        let dataset_type = DatasetType::detect(dir.path()).unwrap();
        assert_eq!(dataset_type, DatasetType::Trec);
        
        // Test MS MARCO
        let msmarco_dir = dir.path().join("msmarco");
        fs::create_dir(&msmarco_dir).unwrap();
        let dataset_type = DatasetType::detect(dir.path()).unwrap();
        assert_eq!(dataset_type, DatasetType::MsMarco);
    }

    #[test]
    fn test_validation_with_errors() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("empty_runs.txt");
        let qrels_path = dir.path().join("empty_qrels.txt");

        fs::File::create(&runs_path).unwrap();
        fs::File::create(&qrels_path).unwrap();

        let result = validate_dataset(&runs_path, &qrels_path).unwrap();
        assert!(!result.is_valid);
        assert!(!result.runs_valid);
        assert!(!result.qrels_valid);
    }

    #[test]
    fn test_validation_with_warnings() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("runs.txt");
        let qrels_path = dir.path().join("qrels.txt");

        let mut runs_file = fs::File::create(&runs_path).unwrap();
        writeln!(runs_file, "1 Q0 doc1 1 0.9 bm25").unwrap();
        writeln!(runs_file, "2 Q0 doc2 1 0.8 bm25").unwrap();

        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();
        // Query 2 missing from qrels

        let result = validate_dataset(&runs_path, &qrels_path).unwrap();
        assert!(result.is_valid); // Still valid, just warnings
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_statistics_empty_dataset() {
        let runs = vec![];
        let qrels = vec![];

        let stats = compute_comprehensive_stats(&runs, &qrels);

        assert_eq!(stats.runs.total_entries, 0);
        assert_eq!(stats.qrels.total_entries, 0);
        assert_eq!(stats.overlap.queries_in_both, 0);
        assert_eq!(stats.quality.queries_with_multiple_runs, 0);
    }

    #[test]
    fn test_score_distribution() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        let stats = compute_comprehensive_stats(&runs, &qrels);

        assert!(stats.runs.score_distribution.min >= 0.0);
        assert!(stats.runs.score_distribution.max <= 1.0);
        assert!(stats.runs.score_distribution.mean > 0.0);
        assert!(stats.runs.score_distribution.median > 0.0);
        assert!(stats.runs.score_distribution.std_dev >= 0.0);
    }

    #[test]
    fn test_relevance_distribution() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        let stats = compute_comprehensive_stats(&runs, &qrels);

        // Should have relevance levels 0, 1, 2
        assert!(stats.qrels.relevance_distribution.contains_key(&0));
        assert!(stats.qrels.relevance_distribution.contains_key(&1));
        assert!(stats.qrels.relevance_distribution.contains_key(&2));
    }

    #[test]
    fn test_fusion_readiness() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        let stats = compute_comprehensive_stats(&runs, &qrels);

        // Query 1 has 2 runs (bm25 and dense), so should be ready for fusion
        assert!(stats.quality.queries_with_multiple_runs > 0);
        assert!(stats.quality.fusion_readiness_ratio > 0.0);
        assert!(stats.quality.avg_runs_per_query > 1.0);
    }
}

