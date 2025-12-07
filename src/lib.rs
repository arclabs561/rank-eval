//! Ranking evaluation metrics: NDCG, MAP, MRR, precision, recall. TREC format support.
//!
//! This crate provides:
//! - **Ranking evaluation metrics**: NDCG, MAP, MRR, Precision@K, Recall@K for binary and graded relevance
//! - **TREC format support**: Load and parse TREC run files and qrels
//!
//! # Quick Start
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

pub mod batch;
pub mod binary;
pub mod export;
pub mod graded;
pub mod statistics;
pub mod trec;
pub mod validation;

#[cfg(feature = "serde")]
pub mod dataset;

// Re-export commonly used items
pub use batch::{evaluate_batch_binary, evaluate_trec_batch, BatchResults, QueryResults};
pub use export::export_to_csv;
pub use statistics::{cohens_d, confidence_interval, paired_t_test, TTestResult};
pub use trec::{
    group_qrels_by_query, group_runs_by_query, load_qrels, load_trec_runs, Qrel, TrecRun,
};
pub use validation::{
    validate_beta, validate_metric_inputs, validate_persistence, ValidationError,
};

#[cfg(feature = "serde")]
pub use binary::Metrics;
#[cfg(feature = "serde")]
pub use export::export_to_json;
