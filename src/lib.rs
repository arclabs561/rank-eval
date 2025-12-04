//! IR evaluation metrics and TREC format parsing for Rust.
//!
//! This crate provides:
//! - **TREC format parsing**: Load and parse TREC run files and qrels
//! - **Binary relevance metrics**: NDCG, MAP, MRR, Precision@K, Recall@K for binary relevance
//! - **Graded relevance metrics**: NDCG and MAP for graded relevance judgments
//!
//! # Quick Start
//!
//! ## TREC Format Parsing
//!
//! ```rust,no_run
//! use rank_eval::trec::{load_trec_runs, load_qrels, TrecRun, Qrel};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load TREC run file
//! let runs = load_trec_runs("runs.txt")?;
//!
//! // Load TREC qrels file
//! let qrels = load_qrels("qrels.txt")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Binary Relevance Metrics
//!
//! ```rust
//! use std::collections::HashSet;
//! use rank_eval::binary::ndcg_at_k;
//!
//! let ranked = vec!["doc1", "doc2", "doc3"];
//! let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
//!
//! let ndcg = ndcg_at_k(&ranked, &relevant, 10);
//! ```
//!
//! ## Graded Relevance Metrics
//!
//! ```rust
//! use std::collections::HashMap;
//! use rank_eval::graded::compute_ndcg;
//!
//! let ranked = vec![
//!     ("doc1".to_string(), 0.9),
//!     ("doc2".to_string(), 0.8),
//! ];
//! let mut qrels = HashMap::new();
//! qrels.insert("doc1".to_string(), 2); // Highly relevant
//! qrels.insert("doc2".to_string(), 1); // Relevant
//!
//! let ndcg = compute_ndcg(&ranked, &qrels, 10);
//! ```

pub mod trec;
pub mod binary;
pub mod graded;
pub mod validation;
pub mod batch;
pub mod statistics;
pub mod export;

#[cfg(feature = "serde")]
pub mod dataset;

// Re-export commonly used items
pub use trec::{TrecRun, Qrel, load_trec_runs, load_qrels, group_runs_by_query, group_qrels_by_query};
pub use validation::{ValidationError, validate_metric_inputs, validate_persistence, validate_beta};
pub use batch::{BatchResults, QueryResults, evaluate_batch_binary, evaluate_trec_batch};
pub use statistics::{TTestResult, paired_t_test, confidence_interval, cohens_d};
pub use export::{export_to_csv};

#[cfg(feature = "serde")]
pub use binary::Metrics;
#[cfg(feature = "serde")]
pub use export::export_to_json;
