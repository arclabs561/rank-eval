//! E2E test: rank-eval integration with batch operations.

//! E2E test: rank-eval integration with batch operations.

use rank_eval::batch::evaluate_batch_binary;
use rank_eval::binary::ndcg_at_k;
use rank_eval::graded::{compute_map, compute_ndcg};
use rank_eval::trec::{group_qrels_by_query, group_runs_by_query, load_qrels, load_trec_runs};
use std::collections::HashSet;
use std::io::Write;
use tempfile::TempDir;

fn main() {
    println!("Testing rank-eval integration features...");

    // Create test TREC files
    let temp_dir = TempDir::new().unwrap();
    let runs_file = temp_dir.path().join("runs.txt");
    let qrels_file = temp_dir.path().join("qrels.txt");

    // Write test data
    let mut runs_writer = std::fs::File::create(&runs_file).unwrap();
    writeln!(runs_writer, "1 Q0 doc1 1 0.95 run1").unwrap();
    writeln!(runs_writer, "1 Q0 doc2 2 0.87 run1").unwrap();
    writeln!(runs_writer, "1 Q0 doc3 3 0.75 run1").unwrap();
    writeln!(runs_writer, "2 Q0 doc4 1 0.92 run1").unwrap();
    writeln!(runs_writer, "2 Q0 doc5 2 0.85 run1").unwrap();

    let mut qrels_writer = std::fs::File::create(&qrels_file).unwrap();
    writeln!(qrels_writer, "1 0 doc1 2").unwrap();
    writeln!(qrels_writer, "1 0 doc2 1").unwrap();
    writeln!(qrels_writer, "1 0 doc3 0").unwrap();
    writeln!(qrels_writer, "2 0 doc4 2").unwrap();
    writeln!(qrels_writer, "2 0 doc5 1").unwrap();

    // Load and group
    let runs = load_trec_runs(&runs_file).unwrap();
    let qrels = load_qrels(&qrels_file).unwrap();

    let runs_by_query = group_runs_by_query(&runs);
    let qrels_by_query = group_qrels_by_query(&qrels);

    assert_eq!(runs_by_query.len(), 2);
    assert_eq!(qrels_by_query.len(), 2);
    println!("✅ Loaded and grouped {} queries", runs_by_query.len());

    // Test batch evaluation
    let mut rankings = Vec::new();
    let mut relevants = Vec::new();

    for (query_id, query_runs_by_tag) in &runs_by_query {
        // Get first run tag's results
        if let Some((_, run_results)) = query_runs_by_tag.iter().next() {
            let ranked: Vec<String> = run_results
                .iter()
                .map(|(doc_id, _)| doc_id.clone())
                .collect();
            rankings.push(ranked);
        }

        if let Some(query_qrels) = qrels_by_query.get(query_id) {
            let relevant: HashSet<String> = query_qrels.keys().cloned().collect();
            relevants.push(relevant);
        }
    }

    // Evaluate batch with metrics list
    let metrics = ["ndcg@10", "precision@10", "recall@10"];
    let batch_results = evaluate_batch_binary(&rankings, &relevants, &metrics);
    assert!(!batch_results.query_results.is_empty());
    assert!(batch_results.aggregated.contains_key("ndcg@10"));
    println!(
        "✅ Batch evaluation: {} queries",
        batch_results.query_results.len()
    );
    println!(
        "   Average nDCG@10: {:.4}",
        batch_results.aggregated.get("ndcg@10").unwrap_or(&0.0)
    );

    // Test individual query evaluation
    let query1_runs = &runs_by_query["1"]["run1"];
    let query1_ranked: Vec<String> = query1_runs
        .iter()
        .map(|(doc_id, _)| doc_id.clone())
        .collect();
    let query1_relevant: HashSet<String> = qrels_by_query["1"].keys().cloned().collect();

    let ndcg = ndcg_at_k(&query1_ranked, &query1_relevant, 10);
    assert!(ndcg >= 0.0 && ndcg <= 1.0);
    println!("✅ Query 1 nDCG@10: {:.4}", ndcg);

    // Test graded evaluation
    let query1_graded: Vec<(String, f32)> = query1_runs.clone();

    let graded_ndcg = compute_ndcg(&query1_graded, &qrels_by_query["1"], 10);
    let map_score = compute_map(&query1_graded, &qrels_by_query["1"]);

    assert!(graded_ndcg >= 0.0 && graded_ndcg <= 1.0);
    assert!(map_score >= 0.0 && map_score <= 1.0);
    println!("✅ Query 1 graded metrics:");
    println!("   nDCG@10: {:.4}", graded_ndcg);
    println!("   MAP: {:.4}", map_score);

    println!("\n✅ All rank-eval integration tests passed!");
}
