//! Python bindings for rank-eval using PyO3.
//!
//! Provides a Python API that mirrors the Rust API, enabling seamless
//! integration with Python evaluation scripts.
//!
//! # Usage
//!
//! ```python
//! import rank_eval
//!
//! # Binary relevance metrics
//! ranked = ["doc1", "doc2", "doc3"]
//! relevant = {"doc1", "doc3"}
//! ndcg = rank_eval.ndcg_at_k(ranked, relevant, k=10)
//!
//! # Graded relevance metrics
//! ranked = [("doc1", 0.9), ("doc2", 0.8)]
//! qrels = {"doc1": 2, "doc2": 1}  # Highly relevant, relevant
//! ndcg = rank_eval.compute_ndcg(ranked, qrels, k=10)
//! ```

// TODO: Remove allow(deprecated) when upgrading to pyo3 0.25+ which uses IntoPyObject
#![allow(deprecated)]

use ::rank_eval::binary;
use ::rank_eval::graded;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

/// Python module for rank-eval.
#[pymodule]
#[pyo3(name = "rank_eval")]
fn rank_eval_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Binary relevance metrics
    m.add_function(wrap_pyfunction!(precision_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(recall_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(mrr_py, m)?)?;
    m.add_function(wrap_pyfunction!(dcg_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(idcg_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(ndcg_at_k_py, m)?)?;
    m.add_function(wrap_pyfunction!(average_precision_py, m)?)?;

    // Graded relevance metrics
    m.add_function(wrap_pyfunction!(compute_ndcg_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_map_py, m)?)?;

    Ok(())
}

/// Precision at rank k.
#[pyfunction]
fn precision_at_k_py(
    ranked: &Bound<'_, PyList>,
    relevant: &Bound<'_, PySet>,
    k: usize,
) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::precision_at_k(&ranked_vec, &relevant_set, k))
}

/// Recall at rank k.
#[pyfunction]
fn recall_at_k_py(
    ranked: &Bound<'_, PyList>,
    relevant: &Bound<'_, PySet>,
    k: usize,
) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::recall_at_k(&ranked_vec, &relevant_set, k))
}

/// Mean Reciprocal Rank.
#[pyfunction]
fn mrr_py(ranked: &Bound<'_, PyList>, relevant: &Bound<'_, PySet>) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::mrr(&ranked_vec, &relevant_set))
}

/// Discounted Cumulative Gain at rank k.
#[pyfunction]
fn dcg_at_k_py(ranked: &Bound<'_, PyList>, relevant: &Bound<'_, PySet>, k: usize) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::dcg_at_k(&ranked_vec, &relevant_set, k))
}

/// Ideal DCG at rank k.
#[pyfunction]
fn idcg_at_k_py(
    ranked: &Bound<'_, PyList>,
    relevant: &Bound<'_, PySet>,
    k: usize,
) -> PyResult<f64> {
    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    let n_relevant = relevant_set.len();
    Ok(binary::idcg_at_k(n_relevant, k))
}

/// Normalized DCG at rank k.
#[pyfunction]
fn ndcg_at_k_py(
    ranked: &Bound<'_, PyList>,
    relevant: &Bound<'_, PySet>,
    k: usize,
) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::ndcg_at_k(&ranked_vec, &relevant_set, k))
}

/// Average Precision (MAP for single query).
#[pyfunction]
fn average_precision_py(ranked: &Bound<'_, PyList>, relevant: &Bound<'_, PySet>) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::average_precision(&ranked_vec, &relevant_set))
}

/// Expected Reciprocal Rank (ERR) at k.
#[pyfunction]
fn err_at_k_py(ranked: &Bound<'_, PyList>, relevant: &Bound<'_, PySet>, k: usize) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::err_at_k(&ranked_vec, &relevant_set, k))
}

/// Rank-Biased Precision (RBP) at k.
#[pyfunction]
fn rbp_at_k_py(
    ranked: &Bound<'_, PyList>,
    relevant: &Bound<'_, PySet>,
    k: usize,
    persistence: f64,
) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::rbp_at_k(&ranked_vec, &relevant_set, k, persistence))
}

/// F-measure at k (F1, F2, etc.).
#[pyfunction]
fn f_measure_at_k_py(
    ranked: &Bound<'_, PyList>,
    relevant: &Bound<'_, PySet>,
    k: usize,
    beta: f64,
) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::f_measure_at_k(&ranked_vec, &relevant_set, k, beta))
}

/// Success at k: whether at least one relevant document is in top-k.
#[pyfunction]
fn success_at_k_py(
    ranked: &Bound<'_, PyList>,
    relevant: &Bound<'_, PySet>,
    k: usize,
) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::success_at_k(&ranked_vec, &relevant_set, k))
}

/// R-Precision: Precision at R (where R is the number of relevant documents).
#[pyfunction]
fn r_precision_py(ranked: &Bound<'_, PyList>, relevant: &Bound<'_, PySet>) -> PyResult<f64> {
    let ranked_vec: Vec<String> = ranked
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<Vec<_>, _>>()?;

    let relevant_set: std::collections::HashSet<String> = relevant
        .iter()
        .map(|v| v.extract::<String>())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;

    Ok(binary::r_precision(&ranked_vec, &relevant_set))
}

/// Compute nDCG@k for graded relevance.
#[pyfunction]
fn compute_ndcg_py(
    ranked: &Bound<'_, PyList>,
    qrels: &Bound<'_, PyDict>,
    k: usize,
) -> PyResult<f64> {
    let ranked_vec: Vec<(String, f32)> = ranked
        .iter()
        .map(|v| {
            let tuple = v.downcast::<PyTuple>()?;
            let id: String = tuple.get_item(0)?.extract()?;
            let score: f64 = tuple.get_item(1)?.extract()?;
            Ok((id, score as f32))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut qrels_map = std::collections::HashMap::new();
    for (key, value) in qrels.iter() {
        let id: String = key.extract()?;
        let relevance: u32 = value.extract()?;
        qrels_map.insert(id, relevance);
    }

    Ok(graded::compute_ndcg(&ranked_vec, &qrels_map, k) as f64)
}

/// Compute MAP for graded relevance.
#[pyfunction]
fn compute_map_py(ranked: &Bound<'_, PyList>, qrels: &Bound<'_, PyDict>) -> PyResult<f64> {
    let ranked_vec: Vec<(String, f32)> = ranked
        .iter()
        .map(|v| {
            let tuple = v.downcast::<PyTuple>()?;
            let id: String = tuple.get_item(0)?.extract()?;
            let score: f64 = tuple.get_item(1)?.extract()?;
            Ok((id, score as f32))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut qrels_map = std::collections::HashMap::new();
    for (key, value) in qrels.iter() {
        let id: String = key.extract()?;
        let relevance: u32 = value.extract()?;
        qrels_map.insert(id, relevance);
    }

    Ok(graded::compute_map(&ranked_vec, &qrels_map) as f64)
}
