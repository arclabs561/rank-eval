"""Type stubs for rank-eval Python bindings."""

def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Precision at rank k."""
    ...

def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Recall at rank k."""
    ...

def mrr(ranked: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank."""
    ...

def dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    ...

def idcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Ideal DCG at rank k."""
    ...

def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG at rank k."""
    ...

def average_precision(ranked: list[str], relevant: set[str]) -> float:
    """Average Precision (MAP for single query)."""
    ...

def err_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Expected Reciprocal Rank (ERR) at k."""
    ...

def rbp_at_k(ranked: list[str], relevant: set[str], k: int, persistence: float) -> float:
    """Rank-Biased Precision (RBP) at k."""
    ...

def f_measure_at_k(ranked: list[str], relevant: set[str], k: int, beta: float) -> float:
    """F-measure at k (F1, F2, etc.)."""
    ...

def success_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Success at k: whether at least one relevant document is in top-k."""
    ...

def r_precision(ranked: list[str], relevant: set[str]) -> float:
    """R-Precision: Precision at R (where R is the number of relevant documents)."""
    ...

def compute_ndcg(ranked: list[tuple[str, float]], qrels: dict[str, int], k: int) -> float:
    """Compute nDCG@k for graded relevance."""
    ...

def compute_map(ranked: list[tuple[str, float]], qrels: dict[str, int]) -> float:
    """Compute MAP for graded relevance."""
    ...

