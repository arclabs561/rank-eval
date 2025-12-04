//! Input validation utilities for metrics and evaluation.

use anyhow::Result;

/// Validation error for metric inputs.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// k is larger than ranked list size
    KTooLarge { k: usize, ranked_len: usize },
    /// k is zero
    KZero,
    /// Ranked list is empty
    EmptyRanked,
    /// No relevant documents
    NoRelevant,
    /// Invalid persistence parameter (must be in (0, 1))
    InvalidPersistence { persistence: f64 },
    /// Invalid beta parameter (must be positive)
    InvalidBeta { beta: f64 },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::KTooLarge { k, ranked_len } => {
                write!(
                    f,
                    "k ({}) is larger than ranked list size ({})",
                    k, ranked_len
                )
            }
            ValidationError::KZero => write!(f, "k must be greater than 0"),
            ValidationError::EmptyRanked => write!(f, "ranked list cannot be empty"),
            ValidationError::NoRelevant => write!(f, "no relevant documents provided"),
            ValidationError::InvalidPersistence { persistence } => {
                write!(
                    f,
                    "persistence parameter ({}) must be in (0, 1)",
                    persistence
                )
            }
            ValidationError::InvalidBeta { beta } => {
                write!(f, "beta parameter ({}) must be positive", beta)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validate metric inputs.
///
/// Checks that:
/// - k > 0
/// - k <= ranked.len() (or warns if k > ranked.len())
/// - ranked is not empty
/// - relevant is not empty (optional, some metrics handle this)
///
/// # Arguments
///
/// * `ranked` - Ranked list of document IDs
/// * `relevant` - Set of relevant document IDs
/// * `k` - Cutoff rank
/// * `require_relevant` - Whether to require at least one relevant document
///
/// # Returns
///
/// `Ok(())` if valid, `Err(ValidationError)` otherwise.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use rank_eval::validation::validate_metric_inputs;
///
/// let ranked = vec!["doc1", "doc2", "doc3"];
/// let relevant: HashSet<_> = ["doc1"].into_iter().collect();
///
/// // Valid inputs
/// assert!(validate_metric_inputs(&ranked, &relevant, 3, false).is_ok());
///
/// // Invalid: k = 0
/// assert!(validate_metric_inputs(&ranked, &relevant, 0, false).is_err());
/// ```
pub fn validate_metric_inputs<I>(
    ranked: &[I],
    relevant: &std::collections::HashSet<I>,
    k: usize,
    require_relevant: bool,
) -> Result<(), ValidationError> {
    if k == 0 {
        return Err(ValidationError::KZero);
    }

    if ranked.is_empty() {
        return Err(ValidationError::EmptyRanked);
    }

    if require_relevant && relevant.is_empty() {
        return Err(ValidationError::NoRelevant);
    }

    // Note: k > ranked.len() is allowed (metrics handle this gracefully)
    // but we could warn if needed

    Ok(())
}

/// Validate RBP persistence parameter.
///
/// Persistence must be in (0, 1).
pub fn validate_persistence(persistence: f64) -> Result<(), ValidationError> {
    if persistence <= 0.0 || persistence >= 1.0 {
        return Err(ValidationError::InvalidPersistence { persistence });
    }
    Ok(())
}

/// Validate F-measure beta parameter.
///
/// Beta must be positive.
pub fn validate_beta(beta: f64) -> Result<(), ValidationError> {
    if beta <= 0.0 {
        return Err(ValidationError::InvalidBeta { beta });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_validate_metric_inputs() {
        let ranked = vec!["doc1", "doc2", "doc3"];
        let relevant: HashSet<_> = ["doc1"].into_iter().collect();

        // Valid
        assert!(validate_metric_inputs(&ranked, &relevant, 3, false).is_ok());
        assert!(validate_metric_inputs(&ranked, &relevant, 3, true).is_ok());

        // Invalid: k = 0
        assert!(matches!(
            validate_metric_inputs(&ranked, &relevant, 0, false),
            Err(ValidationError::KZero)
        ));

        // Invalid: empty ranked
        let empty: Vec<&str> = vec![];
        assert!(matches!(
            validate_metric_inputs(&empty, &relevant, 3, false),
            Err(ValidationError::EmptyRanked)
        ));

        // Invalid: no relevant (when required)
        let empty_relevant: HashSet<&str> = HashSet::new();
        assert!(matches!(
            validate_metric_inputs(&ranked, &empty_relevant, 3, true),
            Err(ValidationError::NoRelevant)
        ));

        // Valid: no relevant (when not required)
        assert!(validate_metric_inputs(&ranked, &empty_relevant, 3, false).is_ok());
    }

    #[test]
    fn test_validate_persistence() {
        assert!(validate_persistence(0.5).is_ok());
        assert!(validate_persistence(0.95).is_ok());
        assert!(validate_persistence(0.01).is_ok());
        assert!(validate_persistence(0.99).is_ok());

        assert!(matches!(
            validate_persistence(0.0),
            Err(ValidationError::InvalidPersistence { .. })
        ));
        assert!(matches!(
            validate_persistence(1.0),
            Err(ValidationError::InvalidPersistence { .. })
        ));
        assert!(matches!(
            validate_persistence(-1.0),
            Err(ValidationError::InvalidPersistence { .. })
        ));
    }

    #[test]
    fn test_validate_beta() {
        assert!(validate_beta(1.0).is_ok());
        assert!(validate_beta(2.0).is_ok());
        assert!(validate_beta(0.5).is_ok());

        assert!(matches!(
            validate_beta(0.0),
            Err(ValidationError::InvalidBeta { .. })
        ));
        assert!(matches!(
            validate_beta(-1.0),
            Err(ValidationError::InvalidBeta { .. })
        ));
    }
}

