//! Binary relevance IR evaluation metrics.
//!
//! All metrics assume:
//! - `ranked`: List of document IDs in ranked order (best first)
//! - `relevant`: Set of document IDs that are relevant (ground truth)
//!
//! These metrics use binary relevance: a document is either relevant (in the set) or not.

use std::collections::HashSet;

/// Precision at k: fraction of top-k that are relevant.
///
/// P@k = |relevant ∩ top-k| / k
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::precision_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
/// let relevant: HashSet<_> = ["doc1", "doc3", "doc5"].into_iter().collect();
///
/// let p_at_5 = precision_at_k(&ranked, &relevant, 5);
/// assert!((p_at_5 - 0.6).abs() < 1e-9); // 3 relevant out of 5
/// ```
pub fn precision_at_k<I: Eq + std::hash::Hash>(
    ranked: &[I],
    relevant: &HashSet<I>,
    k: usize,
) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / k as f64
}

/// Recall at k: fraction of relevant docs in top-k.
///
/// R@k = |relevant ∩ top-k| / |relevant|
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::recall_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3"];
/// let relevant: HashSet<_> = ["doc1", "doc3", "doc5"].into_iter().collect();
///
/// let r_at_3 = recall_at_k(&ranked, &relevant, 3);
/// assert!((r_at_3 - 2.0/3.0).abs() < 1e-9); // 2 out of 3 relevant found
/// ```
pub fn recall_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / relevant.len() as f64
}

/// Mean Reciprocal Rank: 1 / rank of first relevant document.
///
/// Formula: `MRR = 1 / rank(first relevant doc)`
///
/// For a single query, this is Reciprocal Rank (RR). When averaged across queries,
/// it becomes Mean Reciprocal Rank (MRR).
///
/// Returns 0.0 if no relevant docs found.
///
/// Reference: Manning et al. (2008) [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::mrr;
///
/// let ranked = vec!["doc1", "doc2", "doc3"];
/// let relevant: HashSet<_> = ["doc2"].into_iter().collect();
///
/// let mrr_score = mrr(&ranked, &relevant);
/// assert!((mrr_score - 0.5).abs() < 1e-9); // First relevant at rank 2, so 1/2 = 0.5
/// ```
pub fn mrr<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    for (i, id) in ranked.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

/// Discounted Cumulative Gain at k.
///
/// Formula: `DCG@k = Σᵢ (rel(i) / log₂(i + 2))`
///
/// Where:
/// - `rel(i)` = 1 if document at position i is relevant, 0 otherwise (binary relevance)
/// - `i` is 0-indexed position (so log₂(i + 2) discounts position 0 by log₂(2) = 1.0)
///
/// DCG accumulates relevance scores with logarithmic discounting, giving more weight
/// to relevant documents appearing earlier in the ranking.
///
/// Reference: Järvelin & Kekäläinen (2002) "Cumulated gain-based evaluation of IR techniques"
/// Manning et al. (2008) [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
///
/// Uses binary relevance: rel(i) = 1 if relevant, 0 otherwise.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::dcg_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3"];
/// let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
///
/// let dcg = dcg_at_k(&ranked, &relevant, 3);
/// // doc1 at rank 0: 1.0 / log2(2) = 1.0
/// // doc3 at rank 2: 1.0 / log2(4) = 0.5
/// // Total: 1.5
/// ```
pub fn dcg_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, id)| relevant.contains(id))
        .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2())
        .sum()
}

/// Ideal DCG at k (all relevant docs at top).
///
/// IDCG@k is the maximum possible DCG@k when all relevant documents
/// are ranked at the top positions.
///
/// # Example
///
/// ```
/// use rank_eval::binary::idcg_at_k;
///
/// // If we have 3 relevant documents and compute IDCG@5
/// let idcg = idcg_at_k(3, 5);
/// // This gives the DCG if all 3 relevant docs were at positions 0, 1, 2
/// ```
pub fn idcg_at_k(n_relevant: usize, k: usize) -> f64 {
    (0..k.min(n_relevant))
        .map(|i| 1.0 / (i as f64 + 2.0).log2())
        .sum()
}

/// Normalized DCG at k.
///
/// Formula: `nDCG@k = DCG@k / IDCG@k`
///
/// Where `IDCG@k` is the ideal DCG (all relevant documents ranked at the top).
/// Normalization ensures the metric is bounded [0.0, 1.0] and comparable across queries.
///
/// Returns a value between 0.0 and 1.0, where 1.0 indicates perfect ranking.
///
/// Reference: Järvelin & Kekäläinen (2002) "Cumulated gain-based evaluation of IR techniques"
/// Manning et al. (2008) [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::ndcg_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
/// let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
///
/// let ndcg = ndcg_at_k(&ranked, &relevant, 4);
/// assert!(ndcg >= 0.0 && ndcg <= 1.0);
/// ```
pub fn ndcg_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    let ideal = idcg_at_k(relevant.len(), k);
    if ideal == 0.0 {
        return 0.0;
    }
    dcg_at_k(ranked, relevant, k) / ideal
}

/// Average Precision: average of precision at each relevant doc.
///
/// Formula: `AP = (1/|R|) × Σᵢ (P@i × rel(i))`
///
/// Where:
/// - `R` is the set of relevant documents
/// - `P@i` is precision at position i (number of relevant docs up to position i / i)
/// - `rel(i)` = 1 if document at position i is relevant, 0 otherwise
///
/// AP rewards systems that rank relevant documents higher. For a single query,
/// this is Average Precision; averaged across queries, it becomes Mean Average Precision (MAP).
///
/// Reference: Manning et al. (2008) [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::average_precision;
///
/// let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
/// let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
///
/// let ap = average_precision(&ranked, &relevant);
/// assert!(ap >= 0.0 && ap <= 1.0);
/// ```
pub fn average_precision<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut hits = 0;

    for (i, id) in ranked.iter().enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum += hits as f64 / (i + 1) as f64;
        }
    }

    sum / relevant.len() as f64
}

/// Expected Reciprocal Rank (ERR).
///
/// ERR models user behavior using a cascade model where users scan results
/// sequentially and stop once they find a satisfactory document.
///
/// For binary relevance, ERR reduces to Reciprocal Rank (RR).
///
/// Formula: ERR = Σᵢ (1/i) × P(user stops at position i)
/// where P(stop at i) = R(i) × Πⱼ<ᵢ (1 - R(j))
///
/// For binary relevance: R(i) = 1 if relevant, 0 otherwise.
///
/// # Arguments
///
/// * `ranked` - List of document IDs in ranked order
/// * `relevant` - Set of relevant document IDs
/// * `max_grade` - Maximum relevance grade (for graded relevance, use graded::err_at_k)
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::err_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3"];
/// let relevant: HashSet<_> = ["doc2"].into_iter().collect();
///
/// let err = err_at_k(&ranked, &relevant, 10);
/// // First relevant at rank 2, so ERR = 1/2 = 0.5 (same as MRR for binary)
/// ```
pub fn err_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let mut p_stop = 1.0; // Probability of continuing to this position
    let mut err = 0.0;

    for (i, id) in ranked.iter().take(k).enumerate() {
        let rank = i + 1;
        if relevant.contains(id) {
            // User finds relevant doc at this position
            // R(i) = 1 for binary relevance
            let r = 1.0;
            // Probability of stopping here = p_stop * r
            err += p_stop * r / rank as f64;
            // Update probability of continuing
            p_stop *= 1.0 - r;
        }
        // If not relevant, p_stop remains the same (user continues)
    }

    err
}

/// Rank-Biased Precision (RBP).
///
/// RBP models user behavior where a user examining a document at rank r
/// continues to the next document with probability p (persistence parameter)
/// or stops with probability (1-p).
///
/// Formula: RBP = (1-p) × Σᵢ p^(i-1) × rel(i)
///
/// where rel(i) = 1 if relevant, 0 otherwise (for binary relevance).
///
/// # Arguments
///
/// * `ranked` - List of document IDs in ranked order
/// * `relevant` - Set of relevant document IDs
/// * `persistence` - Persistence parameter p (typically 0.8 or 0.95)
///   - Higher p = users examine more deeply
///   - Lower p = users stop earlier
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::rbp_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
/// let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
///
/// let rbp = rbp_at_k(&ranked, &relevant, 10, 0.95);
/// assert!(rbp >= 0.0 && rbp <= 1.0);
/// ```
pub fn rbp_at_k<I: Eq + std::hash::Hash>(
    ranked: &[I],
    relevant: &HashSet<I>,
    k: usize,
    persistence: f64,
) -> f64 {
    if persistence <= 0.0 || persistence >= 1.0 {
        return 0.0;
    }

    let mut rbp = 0.0;
    let mut p_power = 1.0; // p^0 = 1

    for id in ranked.iter().take(k) {
        if relevant.contains(id) {
            rbp += p_power;
        }
        // Update p_power for next rank: p^(i) = p^(i-1) * p
        p_power *= persistence;
    }

    (1.0 - persistence) * rbp
}

/// F-measure at k: harmonic mean of precision and recall.
///
/// F@k = (1 + β²) × (P@k × R@k) / (β² × P@k + R@k)
///
/// When β=1, this is the standard F1 score.
///
/// # Arguments
///
/// * `ranked` - List of document IDs in ranked order
/// * `relevant` - Set of relevant document IDs
/// * `k` - Cutoff rank
/// * `beta` - Weight parameter (β=1 gives F1, β>1 emphasizes recall, β<1 emphasizes precision)
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::f_measure_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3"];
/// let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
///
/// let f1 = f_measure_at_k(&ranked, &relevant, 3, 1.0);
/// assert!(f1 >= 0.0 && f1 <= 1.0);
/// ```
pub fn f_measure_at_k<I: Eq + std::hash::Hash>(
    ranked: &[I],
    relevant: &HashSet<I>,
    k: usize,
    beta: f64,
) -> f64 {
    let precision = precision_at_k(ranked, relevant, k);
    let recall = recall_at_k(ranked, relevant, k);

    if precision == 0.0 && recall == 0.0 {
        return 0.0;
    }

    let beta_sq = beta * beta;
    (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
}

/// Success at k: whether at least one relevant document is in top-k.
///
/// Returns 1.0 if at least one relevant doc is in top-k, 0.0 otherwise.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::success_at_k;
///
/// let ranked = vec!["doc1", "doc2", "doc3"];
/// let relevant: HashSet<_> = ["doc2"].into_iter().collect();
///
/// assert_eq!(success_at_k(&ranked, &relevant, 3), 1.0);
/// assert_eq!(success_at_k(&ranked, &relevant, 1), 0.0);
/// ```
pub fn success_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    if ranked.iter().take(k).any(|id| relevant.contains(id)) {
        1.0
    } else {
        0.0
    }
}

/// R-Precision: Precision at R, where R is the number of relevant documents.
///
/// R-Precision = |relevant ∩ top-R| / R
///
/// This metric is useful because it evaluates precision at a cutoff that
/// depends on the number of relevant documents for each query.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::binary::r_precision;
///
/// let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
/// let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();
///
/// // R = 2, so we check precision at top-2
/// let r_prec = r_precision(&ranked, &relevant);
/// assert!(r_prec >= 0.0 && r_prec <= 1.0);
/// ```
pub fn r_precision<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let r = relevant.len();
    precision_at_k(ranked, relevant, r)
}

/// All metrics for a single ranking (binary relevance).
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    pub precision_at_1: f64,
    pub precision_at_5: f64,
    pub precision_at_10: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub average_precision: f64,
    pub err_at_10: f64,
    pub rbp_at_10: f64,
    pub f1_at_10: f64,
    pub success_at_10: f64,
    pub r_precision: f64,
}

#[cfg(feature = "serde")]
impl Metrics {
    /// Compute all metrics for a ranking.
    pub fn compute<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> Self {
        Self {
            precision_at_1: precision_at_k(ranked, relevant, 1),
            precision_at_5: precision_at_k(ranked, relevant, 5),
            precision_at_10: precision_at_k(ranked, relevant, 10),
            recall_at_5: recall_at_k(ranked, relevant, 5),
            recall_at_10: recall_at_k(ranked, relevant, 10),
            mrr: mrr(ranked, relevant),
            ndcg_at_5: ndcg_at_k(ranked, relevant, 5),
            ndcg_at_10: ndcg_at_k(ranked, relevant, 10),
            average_precision: average_precision(ranked, relevant),
            err_at_10: err_at_k(ranked, relevant, 10),
            rbp_at_10: rbp_at_k(ranked, relevant, 10, 0.95),
            f1_at_10: f_measure_at_k(ranked, relevant, 10, 1.0),
            success_at_10: success_at_k(ranked, relevant, 10),
            r_precision: r_precision(ranked, relevant),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_at_k() {
        let ranked = vec!["a", "b", "c", "d", "e"];
        let relevant: HashSet<_> = ["a", "c", "e"].into_iter().collect();

        assert!((precision_at_k(&ranked, &relevant, 1) - 1.0).abs() < 1e-9);
        assert!((precision_at_k(&ranked, &relevant, 2) - 0.5).abs() < 1e-9);
        assert!((precision_at_k(&ranked, &relevant, 5) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_recall_at_k() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<_> = ["a", "c", "e"].into_iter().collect();

        assert!((recall_at_k(&ranked, &relevant, 3) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_mrr() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<_> = ["b"].into_iter().collect();

        assert!((mrr(&ranked, &relevant) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg() {
        let ranked = vec!["a", "b", "c", "d"];
        let relevant: HashSet<_> = ["a", "c"].into_iter().collect();

        // a at pos 0, c at pos 2
        let dcg = 1.0 / 2.0_f64.log2() + 1.0 / 4.0_f64.log2();
        // ideal: both at top
        let idcg = 1.0 / 2.0_f64.log2() + 1.0 / 3.0_f64.log2();

        assert!((ndcg_at_k(&ranked, &relevant, 4) - dcg / idcg).abs() < 1e-9);
    }

    #[test]
    fn test_average_precision() {
        let ranked = vec!["a", "b", "c", "d"];
        let relevant: HashSet<_> = ["a", "c"].into_iter().collect();

        // a at pos 0: precision = 1/1 = 1.0
        // c at pos 2: precision = 2/3 ≈ 0.667
        // AP = (1.0 + 0.667) / 2 = 0.8335
        let ap = average_precision(&ranked, &relevant);
        assert!(ap > 0.8 && ap < 0.85);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_metrics_struct() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<_> = ["a", "c"].into_iter().collect();

        let metrics = Metrics::compute(&ranked, &relevant);
        assert!(metrics.precision_at_1 > 0.0);
        assert!(metrics.ndcg_at_10 > 0.0);
        assert!(metrics.err_at_10 >= 0.0 && metrics.err_at_10 <= 1.0);
        assert!(metrics.rbp_at_10 >= 0.0 && metrics.rbp_at_10 <= 1.0);
        assert!(metrics.f1_at_10 >= 0.0 && metrics.f1_at_10 <= 1.0);
        assert!(metrics.success_at_10 >= 0.0 && metrics.success_at_10 <= 1.0);
        assert!(metrics.r_precision >= 0.0 && metrics.r_precision <= 1.0);
    }

    #[test]
    fn test_err_at_k() {
        let ranked = vec!["doc1", "doc2", "doc3"];
        let relevant: HashSet<_> = ["doc2"].into_iter().collect();

        // First relevant at rank 2, so ERR = 1/2 = 0.5 (same as MRR for binary)
        let err = err_at_k(&ranked, &relevant, 10);
        assert!((err - 0.5).abs() < 1e-9);

        // Perfect ranking: relevant at rank 1
        let ranked2 = vec!["doc1", "doc2", "doc3"];
        let relevant2: HashSet<_> = ["doc1"].into_iter().collect();
        let err2 = err_at_k(&ranked2, &relevant2, 10);
        assert!((err2 - 1.0).abs() < 1e-9);

        // No relevant docs
        let relevant3: HashSet<&str> = HashSet::new();
        let err3 = err_at_k(&ranked, &relevant3, 10);
        assert_eq!(err3, 0.0);
    }

    #[test]
    fn test_rbp_at_k() {
        let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
        let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

        // With p=0.95
        let rbp = rbp_at_k(&ranked, &relevant, 10, 0.95);
        // RBP = (1-0.95) * (1.0 + 0.95^2) = 0.05 * (1.0 + 0.9025) = 0.05 * 1.9025 = 0.095125
        assert!(rbp > 0.0 && rbp <= 1.0);

        // With p=0.8
        let rbp2 = rbp_at_k(&ranked, &relevant, 10, 0.8);
        assert!(rbp2 > 0.0 && rbp2 <= 1.0);
        // Lower persistence should give different (typically lower) RBP
        assert!(rbp2 != rbp);

        // Invalid persistence
        assert_eq!(rbp_at_k(&ranked, &relevant, 10, 0.0), 0.0);
        assert_eq!(rbp_at_k(&ranked, &relevant, 10, 1.0), 0.0);
    }

    #[test]
    fn test_f_measure_at_k() {
        let ranked = vec!["doc1", "doc2", "doc3"];
        let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

        // F1 score (beta=1.0)
        let f1 = f_measure_at_k(&ranked, &relevant, 3, 1.0);
        let precision = precision_at_k(&ranked, &relevant, 3);
        let recall = recall_at_k(&ranked, &relevant, 3);
        let expected_f1 = 2.0 * (precision * recall) / (precision + recall);
        assert!((f1 - expected_f1).abs() < 1e-9);

        // F2 score (beta=2.0, emphasizes recall)
        let f2 = f_measure_at_k(&ranked, &relevant, 3, 2.0);
        assert!(f2 >= 0.0 && f2 <= 1.0);
        // F2 should be >= F1 when recall > precision
        if recall > precision {
            assert!(f2 >= f1);
        }
    }

    #[test]
    fn test_success_at_k() {
        let ranked = vec!["doc1", "doc2", "doc3"];
        let relevant: HashSet<_> = ["doc2"].into_iter().collect();

        assert_eq!(success_at_k(&ranked, &relevant, 1), 0.0);
        assert_eq!(success_at_k(&ranked, &relevant, 2), 1.0);
        assert_eq!(success_at_k(&ranked, &relevant, 3), 1.0);

        // No relevant docs
        let relevant2: HashSet<&str> = HashSet::new();
        assert_eq!(success_at_k(&ranked, &relevant2, 10), 0.0);
    }

    #[test]
    fn test_r_precision() {
        let ranked = vec!["doc1", "doc2", "doc3", "doc4"];
        let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

        // R = 2, so we check precision at top-2
        let r_prec = r_precision(&ranked, &relevant);
        // Top-2: doc1 (relevant), doc2 (not relevant) -> precision = 1/2 = 0.5
        assert!((r_prec - 0.5).abs() < 1e-9);

        // Perfect case: all relevant docs in top-R
        let ranked2 = vec!["doc1", "doc3", "doc2", "doc4"];
        let r_prec2 = r_precision(&ranked2, &relevant);
        assert!((r_prec2 - 1.0).abs() < 1e-9);
    }
}
