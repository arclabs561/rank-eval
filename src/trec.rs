//! TREC format parsing utilities.
//!
//! Provides functions to load and parse TREC run files and qrels files.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A TREC run file entry.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecRun {
    pub query_id: String,
    pub doc_id: String,
    pub rank: usize,
    pub score: f32,
    pub run_tag: String,
}

/// Ground truth relevance judgments (qrels).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qrel {
    pub query_id: String,
    pub doc_id: String,
    pub relevance: u32, // 0 = not relevant, 1+ = relevant (higher = more relevant)
}

/// Load TREC run file.
///
/// Format: query_id Q0 doc_id rank score run_tag
///
/// # Example
///
/// ```rust,no_run
/// use rank_eval::trec::load_trec_runs;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runs = load_trec_runs("runs.txt")?;
/// # Ok(())
/// # }
/// ```
pub fn load_trec_runs(path: impl AsRef<Path>) -> Result<Vec<TrecRun>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open TREC runs file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);
    let mut runs = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context("Failed to read line")?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            // Try to provide helpful error for common issues
            if parts.len() == 5 && parts[1] != "Q0" {
                return Err(anyhow::anyhow!(
                    "Line {}: Expected 'Q0' as second field, found '{}'. Format: query_id Q0 doc_id rank score run_tag",
                    line_num + 1, parts.get(1).unwrap_or(&"<missing>")
                ));
            }
            return Err(anyhow::anyhow!(
                "Line {}: Invalid TREC run format. Expected 6 fields, found {}. Format: query_id Q0 doc_id rank score run_tag\nLine: {}",
                line_num + 1, parts.len(), line
            ));
        }

        // Validate Q0 field (TREC format requirement)
        if parts[1] != "Q0" {
            return Err(anyhow::anyhow!(
                "Line {}: Expected 'Q0' as second field, found '{}'. Format: query_id Q0 doc_id rank score run_tag",
                line_num + 1, parts[1]
            ));
        }

        let query_id = parts[0].to_string();
        let doc_id = parts[2].to_string();
        let rank: usize = parts[3]
            .parse()
            .with_context(|| format!("Invalid rank on line {}: {}", line_num + 1, parts[3]))?;
        let score: f32 = parts[4]
            .parse()
            .with_context(|| format!("Invalid score on line {}: {}", line_num + 1, parts[4]))?;

        // Validate score is finite
        if !score.is_finite() {
            return Err(anyhow::anyhow!(
                "Line {}: Invalid score (NaN or Infinity): {}",
                line_num + 1,
                score
            ));
        }

        // Handle run_tag that might contain spaces (join remaining parts)
        let run_tag = if parts.len() > 6 {
            parts[5..].join(" ")
        } else {
            parts[5].to_string()
        };

        runs.push(TrecRun {
            query_id,
            doc_id,
            rank,
            score,
            run_tag,
        });
    }

    Ok(runs)
}

/// Load TREC qrels file.
///
/// Format: query_id 0 doc_id relevance
///
/// # Example
///
/// ```rust,no_run
/// use rank_eval::trec::load_qrels;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let qrels = load_qrels("qrels.txt")?;
/// # Ok(())
/// # }
/// ```
pub fn load_qrels(path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open qrels file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);
    let mut qrels = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context("Failed to read line")?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(anyhow::anyhow!(
                "Line {}: Invalid TREC qrels format. Expected 4 fields, found {}. Format: query_id 0 doc_id relevance\nLine: {}",
                line_num + 1, parts.len(), line
            ));
        }

        // Validate "0" field (TREC format requirement for qrels)
        if parts[1] != "0" {
            return Err(anyhow::anyhow!(
                "Line {}: Expected '0' as second field in qrels, found '{}'. Format: query_id 0 doc_id relevance",
                line_num + 1, parts[1]
            ));
        }

        let query_id = parts[0].to_string();
        let doc_id = parts[2].to_string();
        let relevance: u32 = parts[3]
            .parse()
            .with_context(|| format!("Invalid relevance on line {}: {}", line_num + 1, parts[3]))?;

        qrels.push(Qrel {
            query_id,
            doc_id,
            relevance,
        });
    }

    Ok(qrels)
}

/// Group runs by query and run tag.
///
/// Returns a nested HashMap: query_id -> run_tag -> Vec<(doc_id, score)>
/// Each run is sorted by score (descending).
pub fn group_runs_by_query(
    runs: &[TrecRun],
) -> HashMap<String, HashMap<String, Vec<(String, f32)>>> {
    let mut grouped: HashMap<String, HashMap<String, Vec<(String, f32)>>> = HashMap::new();

    for run in runs {
        grouped
            .entry(run.query_id.clone())
            .or_default()
            .entry(run.run_tag.clone())
            .or_default()
            .push((run.doc_id.clone(), run.score));
    }

    // Sort each run by score descending
    for query_runs in grouped.values_mut() {
        for run_results in query_runs.values_mut() {
            run_results.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
    }

    grouped
}

/// Group qrels by query.
///
/// Returns a HashMap: query_id -> doc_id -> relevance
pub fn group_qrels_by_query(qrels: &[Qrel]) -> HashMap<String, HashMap<String, u32>> {
    let mut grouped: HashMap<String, HashMap<String, u32>> = HashMap::new();

    for qrel in qrels {
        grouped
            .entry(qrel.query_id.clone())
            .or_default()
            .insert(qrel.doc_id.clone(), qrel.relevance);
    }

    grouped
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_load_trec_runs() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("runs.txt");
        let mut file = fs::File::create(&file_path).unwrap();

        writeln!(file, "1 Q0 doc1 1 0.9 run1").unwrap();
        writeln!(file, "1 Q0 doc2 2 0.8 run1").unwrap();
        writeln!(file, "2 Q0 doc3 1 0.95 run1").unwrap();

        let runs = load_trec_runs(&file_path).unwrap();
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].query_id, "1");
        assert_eq!(runs[0].doc_id, "doc1");
        assert_eq!(runs[0].rank, 1);
        assert_eq!(runs[0].score, 0.9);
        assert_eq!(runs[0].run_tag, "run1");
    }

    #[test]
    fn test_load_qrels() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("qrels.txt");
        let mut file = fs::File::create(&file_path).unwrap();

        writeln!(file, "1 0 doc1 2").unwrap();
        writeln!(file, "1 0 doc2 1").unwrap();
        writeln!(file, "2 0 doc3 2").unwrap();

        let qrels = load_qrels(&file_path).unwrap();
        assert_eq!(qrels.len(), 3);
        assert_eq!(qrels[0].query_id, "1");
        assert_eq!(qrels[0].doc_id, "doc1");
        assert_eq!(qrels[0].relevance, 2);
    }

    #[test]
    fn test_group_runs_by_query() {
        let runs = vec![
            TrecRun {
                query_id: "1".to_string(),
                doc_id: "doc1".to_string(),
                rank: 1,
                score: 0.9,
                run_tag: "run1".to_string(),
            },
            TrecRun {
                query_id: "1".to_string(),
                doc_id: "doc2".to_string(),
                rank: 2,
                score: 0.8,
                run_tag: "run1".to_string(),
            },
            TrecRun {
                query_id: "2".to_string(),
                doc_id: "doc3".to_string(),
                rank: 1,
                score: 0.95,
                run_tag: "run1".to_string(),
            },
        ];

        let grouped = group_runs_by_query(&runs);
        assert_eq!(grouped.len(), 2);
        assert!(grouped.contains_key("1"));
        assert!(grouped.contains_key("2"));
        assert_eq!(grouped["1"]["run1"].len(), 2);
    }

    #[test]
    fn test_group_qrels_by_query() {
        let qrels = vec![
            Qrel {
                query_id: "1".to_string(),
                doc_id: "doc1".to_string(),
                relevance: 2,
            },
            Qrel {
                query_id: "1".to_string(),
                doc_id: "doc2".to_string(),
                relevance: 1,
            },
            Qrel {
                query_id: "2".to_string(),
                doc_id: "doc3".to_string(),
                relevance: 2,
            },
        ];

        let grouped = group_qrels_by_query(&qrels);
        assert_eq!(grouped.len(), 2);
        assert!(grouped.contains_key("1"));
        assert!(grouped.contains_key("2"));
        assert_eq!(grouped["1"]["doc1"], 2);
    }

    #[test]
    fn test_run_tag_with_spaces() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("runs.txt");
        let mut file = fs::File::create(&file_path).unwrap();

        writeln!(file, "1 Q0 doc1 1 0.9 my run tag").unwrap();

        let runs = load_trec_runs(&file_path).unwrap();
        assert_eq!(runs[0].run_tag, "my run tag");
    }

    #[test]
    fn test_error_invalid_format() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("bad_runs.txt");
        let mut file = fs::File::create(&file_path).unwrap();

        writeln!(file, "1 doc1 1 0.9").unwrap(); // Missing Q0

        let result = load_trec_runs(&file_path);
        assert!(result.is_err());
    }
}
