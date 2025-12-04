//! Dataset validation utilities.
//!
//! Validates TREC format files, checks consistency between runs and qrels,
//! and provides detailed validation reports.

use crate::trec::{load_qrels, load_trec_runs, TrecRun};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Comprehensive validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetValidationResult {
    pub is_valid: bool,
    pub runs_valid: bool,
    pub qrels_valid: bool,
    pub consistency_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub statistics: ValidationStatistics,
}

/// Validation statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStatistics {
    pub runs_count: usize,
    pub qrels_count: usize,
    pub unique_queries_in_runs: usize,
    pub unique_queries_in_qrels: usize,
    pub queries_in_both: usize,
    pub queries_only_in_runs: usize,
    pub queries_only_in_qrels: usize,
    pub unique_documents_in_runs: usize,
    pub unique_documents_in_qrels: usize,
    pub documents_in_both: usize,
}

/// Validate a complete dataset (runs + qrels).
pub fn validate_dataset(
    runs_path: impl AsRef<Path>,
    qrels_path: impl AsRef<Path>,
) -> Result<DatasetValidationResult> {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Load and validate runs
    let runs = match load_trec_runs(runs_path.as_ref()) {
        Ok(r) => {
            if r.is_empty() {
                errors.push("Runs file is empty".to_string());
            }
            r
        }
        Err(e) => {
            errors.push(format!("Failed to load runs: {}", e));
            return Ok(DatasetValidationResult {
                is_valid: false,
                runs_valid: false,
                qrels_valid: false,
                consistency_valid: false,
                errors,
                warnings,
                statistics: ValidationStatistics::default(),
            });
        }
    };

    // Load and validate qrels
    let qrels = match load_qrels(qrels_path.as_ref()) {
        Ok(q) => {
            if q.is_empty() {
                errors.push("Qrels file is empty".to_string());
            }
            q
        }
        Err(e) => {
            errors.push(format!("Failed to load qrels: {}", e));
            return Ok(DatasetValidationResult {
                is_valid: false,
                runs_valid: runs.is_empty(),
                qrels_valid: false,
                consistency_valid: false,
                errors,
                warnings,
                statistics: ValidationStatistics::default(),
            });
        }
    };

    // Compute statistics
    let runs_queries: HashSet<String> = runs.iter().map(|r| r.query_id.clone()).collect();
    let qrels_queries: HashSet<String> = qrels.iter().map(|q| q.query_id.clone()).collect();
    let queries_in_both: HashSet<_> = runs_queries.intersection(&qrels_queries).cloned().collect();
    let queries_only_in_runs: HashSet<_> = runs_queries.difference(&qrels_queries).cloned().collect();
    let queries_only_in_qrels: HashSet<_> = qrels_queries.difference(&runs_queries).cloned().collect();

    let runs_docs: HashSet<String> = runs.iter().map(|r| r.doc_id.clone()).collect();
    let qrels_docs: HashSet<String> = qrels.iter().map(|q| q.doc_id.clone()).collect();
    let docs_in_both: HashSet<_> = runs_docs.intersection(&qrels_docs).cloned().collect();

    // Check consistency
    if !queries_only_in_runs.is_empty() {
        warnings.push(format!(
            "{} queries in runs but not in qrels (will be skipped in evaluation)",
            queries_only_in_runs.len()
        ));
    }

    if !queries_only_in_qrels.is_empty() {
        warnings.push(format!(
            "{} queries in qrels but not in runs (no evaluation possible)",
            queries_only_in_qrels.len()
        ));
    }

    if queries_in_both.is_empty() {
        errors.push("No queries in common between runs and qrels".to_string());
    }

    // Check for duplicate query-doc pairs in runs
    let mut seen_runs: HashSet<(String, String, String)> = HashSet::new();
    for run in &runs {
        let key = (run.query_id.clone(), run.doc_id.clone(), run.run_tag.clone());
        if !seen_runs.insert(key.clone()) {
            warnings.push(format!(
                "Duplicate run entry: query={}, doc={}, tag={}",
                run.query_id, run.doc_id, run.run_tag
            ));
        }
    }

    // Check for duplicate query-doc pairs in qrels
    let mut seen_qrels: HashSet<(String, String)> = HashSet::new();
    for qrel in &qrels {
        let key = (qrel.query_id.clone(), qrel.doc_id.clone());
        if !seen_qrels.insert(key.clone()) {
            warnings.push(format!(
                "Duplicate qrel entry: query={}, doc={}",
                qrel.query_id, qrel.doc_id
            ));
        }
    }

    // Check rank ordering within queries
    let mut by_query_run: HashMap<String, Vec<&TrecRun>> = HashMap::new();
    for run in &runs {
        by_query_run
            .entry(run.query_id.clone())
            .or_default()
            .push(run);
    }

    for (query_id, query_runs) in &by_query_run {
        // Group by run_tag
        let mut by_tag: HashMap<String, Vec<&TrecRun>> = HashMap::new();
        for run in query_runs {
            by_tag
                .entry(run.run_tag.clone())
                .or_default()
                .push(run);
        }

        for (tag, tag_runs) in &by_tag {
            let mut sorted = tag_runs.clone();
            sorted.sort_by_key(|r| r.rank);

            for (expected_rank, run) in sorted.iter().enumerate() {
                if run.rank != expected_rank + 1 {
                    warnings.push(format!(
                        "Query {} (tag {}): rank {} not sequential (expected {})",
                        query_id, tag, run.rank, expected_rank + 1
                    ));
                }
            }
        }
    }

    let statistics = ValidationStatistics {
        runs_count: runs.len(),
        qrels_count: qrels.len(),
        unique_queries_in_runs: runs_queries.len(),
        unique_queries_in_qrels: qrels_queries.len(),
        queries_in_both: queries_in_both.len(),
        queries_only_in_runs: queries_only_in_runs.len(),
        queries_only_in_qrels: queries_only_in_qrels.len(),
        unique_documents_in_runs: runs_docs.len(),
        unique_documents_in_qrels: qrels_docs.len(),
        documents_in_both: docs_in_both.len(),
    };

    let runs_valid = !runs.is_empty() && errors.iter().all(|e| !e.contains("runs"));
    let qrels_valid = !qrels.is_empty() && errors.iter().all(|e| !e.contains("qrels"));
    let consistency_valid = !queries_in_both.is_empty() && errors.is_empty();

    Ok(DatasetValidationResult {
        is_valid: runs_valid && qrels_valid && consistency_valid && errors.is_empty(),
        runs_valid,
        qrels_valid,
        consistency_valid,
        errors,
        warnings,
        statistics,
    })
}

/// Print validation report to stdout.
pub fn print_validation_report(result: &DatasetValidationResult) {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              Dataset Validation Report                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let status = if result.is_valid { "✓ VALID" } else { "✗ INVALID" };
    println!("Status: {}\n", status);

    println!("Validation Checks:");
    println!("  Runs:        {}", if result.runs_valid { "✓" } else { "✗" });
    println!("  Qrels:       {}", if result.qrels_valid { "✓" } else { "✗" });
    println!("  Consistency: {}", if result.consistency_valid { "✓" } else { "✗" });

    if !result.errors.is_empty() {
        println!("\nErrors:");
        for error in &result.errors {
            println!("  ✗ {}", error);
        }
    }

    if !result.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &result.warnings {
            println!("  ⚠ {}", warning);
        }
    }

    println!("\nStatistics:");
    println!("  Runs:        {} entries", result.statistics.runs_count);
    println!("  Qrels:       {} entries", result.statistics.qrels_count);
    println!("  Queries:     {} in runs, {} in qrels, {} in both",
        result.statistics.unique_queries_in_runs,
        result.statistics.unique_queries_in_qrels,
        result.statistics.queries_in_both
    );
    println!("  Documents:   {} in runs, {} in qrels, {} in both",
        result.statistics.unique_documents_in_runs,
        result.statistics.unique_documents_in_qrels,
        result.statistics.documents_in_both
    );

    if result.statistics.queries_only_in_runs > 0 {
        println!("\n  Note: {} queries in runs but not in qrels (will be skipped)",
            result.statistics.queries_only_in_runs);
    }

    if result.statistics.queries_only_in_qrels > 0 {
        println!("\n  Note: {} queries in qrels but not in runs (cannot evaluate)",
            result.statistics.queries_only_in_qrels);
    }
}


