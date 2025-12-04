"""Tests for rank-eval Python bindings."""

import pytest
import rank_eval


def test_precision_at_k():
    """Test precision at k."""
    ranked = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3"}
    
    precision = rank_eval.precision_at_k(ranked, relevant, k=2)
    assert 0.0 <= precision <= 1.0
    assert precision == 0.5  # 1 relevant out of 2


def test_recall_at_k():
    """Test recall at k."""
    ranked = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3"}
    
    recall = rank_eval.recall_at_k(ranked, relevant, k=2)
    assert 0.0 <= recall <= 1.0
    assert recall == 0.5  # 1 relevant out of 2 total


def test_ndcg_at_k():
    """Test nDCG@k."""
    ranked = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc3"}
    
    ndcg = rank_eval.ndcg_at_k(ranked, relevant, k=10)
    assert 0.0 <= ndcg <= 1.0


def test_mrr():
    """Test Mean Reciprocal Rank."""
    ranked = ["doc1", "doc2", "doc3"]
    relevant = {"doc2"}
    
    mrr = rank_eval.mrr(ranked, relevant)
    assert 0.0 <= mrr <= 1.0
    assert mrr == 0.5  # First relevant at position 2


def test_compute_ndcg_graded():
    """Test nDCG for graded relevance."""
    ranked = [
        ("doc1", 0.9),
        ("doc2", 0.8),
        ("doc3", 0.7),
    ]
    qrels = {
        "doc1": 2,  # Highly relevant
        "doc2": 1,  # Relevant
        "doc3": 0,  # Not relevant
    }
    
    ndcg = rank_eval.compute_ndcg(ranked, qrels, k=10)
    assert 0.0 <= ndcg <= 1.0


def test_compute_map_graded():
    """Test MAP for graded relevance."""
    ranked = [
        ("doc1", 0.9),
        ("doc2", 0.8),
    ]
    qrels = {
        "doc1": 2,
        "doc2": 1,
    }
    
    map_score = rank_eval.compute_map(ranked, qrels)
    assert 0.0 <= map_score <= 1.0


def test_err_at_k():
    """Test Expected Reciprocal Rank."""
    ranked = ["doc1", "doc2", "doc3"]
    relevant = {"doc2"}
    
    err = rank_eval.err_at_k(ranked, relevant, k=10)
    assert 0.0 <= err <= 1.0
    # First relevant at position 2, so ERR = 1/2 = 0.5 (same as MRR for binary)
    assert abs(err - 0.5) < 0.1


def test_rbp_at_k():
    """Test Rank-Biased Precision."""
    ranked = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3"}
    
    rbp = rank_eval.rbp_at_k(ranked, relevant, k=10, persistence=0.95)
    assert 0.0 <= rbp <= 1.0


def test_f_measure_at_k():
    """Test F-measure."""
    ranked = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc3"}
    
    # F1 score (beta=1.0)
    f1 = rank_eval.f_measure_at_k(ranked, relevant, k=3, beta=1.0)
    assert 0.0 <= f1 <= 1.0
    
    # F2 score (beta=2.0, emphasizes recall)
    f2 = rank_eval.f_measure_at_k(ranked, relevant, k=3, beta=2.0)
    assert 0.0 <= f2 <= 1.0


def test_success_at_k():
    """Test Success at k."""
    ranked = ["doc1", "doc2", "doc3"]
    relevant = {"doc2"}
    
    assert rank_eval.success_at_k(ranked, relevant, k=1) == 0.0
    assert rank_eval.success_at_k(ranked, relevant, k=2) == 1.0
    assert rank_eval.success_at_k(ranked, relevant, k=3) == 1.0


def test_r_precision():
    """Test R-Precision."""
    ranked = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3"}
    
    # R = 2, so we check precision at top-2
    r_prec = rank_eval.r_precision(ranked, relevant)
    assert 0.0 <= r_prec <= 1.0
    # Top-2: doc1 (relevant), doc2 (not relevant) -> precision = 1/2 = 0.5
    assert abs(r_prec - 0.5) < 0.1

