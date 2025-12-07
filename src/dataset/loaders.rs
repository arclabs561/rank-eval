//! Dataset loaders for MS MARCO, BEIR, TREC, MIRACL, MTEB, and other IR datasets.
//!
//! Provides utilities to download and load evaluation datasets.

use crate::trec::{load_qrels, load_trec_runs, Qrel, TrecRun};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Dataset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub description: String,
    pub url: Option<String>,
    pub queries: usize,
    pub documents: usize,
    pub format: String,         // "trec", "msmarco", "beir", "miracl", "mteb"
    pub languages: Vec<String>, // For multilingual datasets
}

/// Load MS MARCO passage ranking runs from a directory.
///
/// MS MARCO runs are typically in TREC format.
pub fn load_msmarco_runs(runs_dir: impl AsRef<Path>, run_files: &[&str]) -> Result<Vec<TrecRun>> {
    let mut all_runs = Vec::new();

    for run_file in run_files {
        let run_path = runs_dir.as_ref().join(run_file);
        let runs = load_trec_runs(&run_path)
            .with_context(|| format!("Failed to load run file: {:?}", run_path))?;
        all_runs.extend(runs);
    }

    Ok(all_runs)
}

/// Load MS MARCO qrels.
pub fn load_msmarco_qrels(qrels_path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    load_qrels(qrels_path)
}

/// Load BEIR dataset runs.
///
/// BEIR datasets typically come with pre-computed runs or require generating them.
/// This function loads runs that are already in TREC format.
pub fn load_beir_runs(runs_path: impl AsRef<Path>) -> Result<Vec<TrecRun>> {
    load_trec_runs(runs_path)
}

/// Load BEIR qrels.
pub fn load_beir_qrels(qrels_path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    load_qrels(qrels_path)
}

/// Load MIRACL dataset runs.
///
/// MIRACL is available via HuggingFace and may need conversion to TREC format.
/// This function loads runs that are already in TREC format.
pub fn load_miracl_runs(runs_path: impl AsRef<Path>) -> Result<Vec<TrecRun>> {
    load_trec_runs(runs_path)
}

/// Load MIRACL qrels.
pub fn load_miracl_qrels(qrels_path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    load_qrels(qrels_path)
}

/// Load TREC run files from a directory.
pub fn load_trec_runs_from_dir(
    runs_dir: impl AsRef<Path>,
    run_files: &[&str],
) -> Result<Vec<TrecRun>> {
    let mut all_runs = Vec::new();

    for run_file in run_files {
        let run_path = runs_dir.as_ref().join(run_file);
        let runs = load_trec_runs(&run_path)
            .with_context(|| format!("Failed to load TREC run file: {:?}", run_path))?;
        all_runs.extend(runs);
    }

    Ok(all_runs)
}

/// Load TREC qrels from a directory.
pub fn load_trec_qrels_from_dir(qrels_dir: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    // Try multiple possible qrels file names
    let possible_names = ["qrels.txt", "qrels", "qrels.dev.txt", "qrels.test.txt"];

    for name in &possible_names {
        let qrels_path = qrels_dir.as_ref().join(name);
        if qrels_path.exists() {
            return load_qrels(&qrels_path);
        }
    }

    anyhow::bail!(
        "No qrels file found in {:?}. Tried: {:?}",
        qrels_dir.as_ref(),
        possible_names
    );
}

/// Create a dataset configuration file.
pub fn create_dataset_config(
    name: &str,
    description: &str,
    _runs_path: Option<&Path>,
    _qrels_path: Option<&Path>,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let config = DatasetMetadata {
        name: name.to_string(),
        description: description.to_string(),
        url: None,
        queries: 0,   // Will be filled in during evaluation
        documents: 0, // Will be filled in during evaluation
        format: "trec".to_string(),
        languages: vec!["en".to_string()], // Default to English
    };

    let json = serde_json::to_string_pretty(&config)?;
    std::fs::write(output_path.as_ref(), json.as_bytes())
        .with_context(|| format!("Failed to create config file: {:?}", output_path.as_ref()))?;

    Ok(())
}

/// List available datasets in a directory.
pub fn list_datasets(datasets_dir: impl AsRef<Path>) -> Result<Vec<String>> {
    let dir = std::fs::read_dir(datasets_dir.as_ref()).with_context(|| {
        format!(
            "Failed to read datasets directory: {:?}",
            datasets_dir.as_ref()
        )
    })?;

    let mut datasets = Vec::new();
    for entry in dir {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                datasets.push(name.to_string());
            }
        }
    }

    Ok(datasets)
}

/// Check if a dataset directory has the required files.
pub fn validate_dataset_dir(dataset_dir: impl AsRef<Path>) -> Result<bool> {
    let dir = dataset_dir.as_ref();

    // Check for at least one run file
    let has_runs = if let Ok(entries) = std::fs::read_dir(dir) {
        entries.filter_map(|e| e.ok()).any(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".run") || n.ends_with(".txt"))
                .unwrap_or(false)
        })
    } else {
        false
    };

    // Check for qrels file (try multiple names)
    let possible_qrels = ["qrels.txt", "qrels", "qrels.dev.txt", "qrels.test.txt"];
    let has_qrels = possible_qrels.iter().any(|name| dir.join(name).exists());

    Ok(has_runs && has_qrels)
}

/// Get dataset statistics.
pub fn get_dataset_stats(runs: &[TrecRun], qrels: &[Qrel]) -> DatasetStats {
    let unique_queries: std::collections::HashSet<_> = runs.iter().map(|r| &r.query_id).collect();
    let unique_docs: std::collections::HashSet<_> = runs.iter().map(|r| &r.doc_id).collect();
    let unique_run_tags: std::collections::HashSet<_> = runs.iter().map(|r| &r.run_tag).collect();

    let queries_with_qrels: std::collections::HashSet<_> =
        qrels.iter().map(|q| &q.query_id).collect();
    let relevant_docs: usize = qrels.iter().filter(|q| q.relevance > 0).count();

    DatasetStats {
        total_runs: runs.len(),
        unique_queries: unique_queries.len(),
        unique_documents: unique_docs.len(),
        unique_run_tags: unique_run_tags.len(),
        total_qrels: qrels.len(),
        queries_with_qrels: queries_with_qrels.len(),
        relevant_documents: relevant_docs,
    }
}

/// Dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_runs: usize,
    pub unique_queries: usize,
    pub unique_documents: usize,
    pub unique_run_tags: usize,
    pub total_qrels: usize,
    pub queries_with_qrels: usize,
    pub relevant_documents: usize,
}

/// Load MTEB (Massive Text Embedding Benchmark) dataset runs.
///
/// MTEB datasets are typically in TREC format or can be converted to TREC format.
/// This function loads runs that are already in TREC format.
pub fn load_mteb_runs(runs_path: impl AsRef<Path>) -> Result<Vec<TrecRun>> {
    load_trec_runs(runs_path)
}

/// Load MTEB qrels.
pub fn load_mteb_qrels(qrels_path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    load_qrels(qrels_path)
}

/// Load HotpotQA dataset runs.
///
/// HotpotQA is a multi-hop question answering dataset.
/// This function loads runs that are already in TREC format.
pub fn load_hotpotqa_runs(runs_path: impl AsRef<Path>) -> Result<Vec<TrecRun>> {
    load_trec_runs(runs_path)
}

/// Load HotpotQA qrels.
pub fn load_hotpotqa_qrels(qrels_path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    load_qrels(qrels_path)
}

/// Load Natural Questions dataset runs.
///
/// Natural Questions is an open-domain question answering dataset.
/// This function loads runs that are already in TREC format.
pub fn load_natural_questions_runs(runs_path: impl AsRef<Path>) -> Result<Vec<TrecRun>> {
    load_trec_runs(runs_path)
}

/// Load Natural Questions qrels.
pub fn load_natural_questions_qrels(qrels_path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    load_qrels(qrels_path)
}

/// Supported dataset types for automatic loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetType {
    MsMarco,
    Beir,
    Trec,
    Miracl,
    Mteb,
    HotpotQA,
    NaturalQuestions,
    Squad,
    Custom, // Generic TREC format
}

impl DatasetType {
    /// Detect dataset type from directory structure or metadata.
    pub fn detect(dataset_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dataset_dir.as_ref();

        // Check for dataset-specific markers
        if dir.join("msmarco").exists() || dir.join("MSMARCO").exists() {
            return Ok(Self::MsMarco);
        }

        if dir.join("beir").exists() || dir.join("BEIR").exists() {
            return Ok(Self::Beir);
        }

        if dir.join("miracl").exists() || dir.join("MIRACL").exists() {
            return Ok(Self::Miracl);
        }

        // Default to TREC format (most common)
        Ok(Self::Trec)
    }

    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::MsMarco => "MS MARCO",
            Self::Beir => "BEIR",
            Self::Trec => "TREC",
            Self::Miracl => "MIRACL",
            Self::Mteb => "MTEB",
            Self::HotpotQA => "HotpotQA",
            Self::NaturalQuestions => "Natural Questions",
            Self::Squad => "SQuAD",
            Self::Custom => "Custom",
        }
    }
}

/// Detect dataset type from directory structure or metadata.
///
/// Attempts to automatically detect the dataset type by checking for
/// dataset-specific markers in the directory.
pub fn detect_dataset_type(dataset_dir: impl AsRef<Path>) -> Result<DatasetType> {
    let dir = dataset_dir.as_ref();

    // Check for dataset-specific markers
    if dir.join("msmarco").exists() || dir.join("MSMARCO").exists() {
        return Ok(DatasetType::MsMarco);
    }

    if dir.join("beir").exists() || dir.join("BEIR").exists() {
        return Ok(DatasetType::Beir);
    }

    if dir.join("miracl").exists() || dir.join("MIRACL").exists() {
        return Ok(DatasetType::Miracl);
    }

    if dir.join("mteb").exists() || dir.join("MTEB").exists() {
        return Ok(DatasetType::Mteb);
    }

    if dir.join("hotpotqa").exists() || dir.join("HotpotQA").exists() {
        return Ok(DatasetType::HotpotQA);
    }

    if dir.join("natural_questions").exists() || dir.join("NaturalQuestions").exists() {
        return Ok(DatasetType::NaturalQuestions);
    }

    // Default to TREC format (most common)
    Ok(DatasetType::Trec)
}
