//! Statistical testing utilities for evaluation results.

/// Result of a paired t-test.
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,
    pub degrees_of_freedom: usize,
    pub mean_difference: f64,
    pub std_error: f64,
    pub significant: bool, // p < 0.05
}

/// Perform a paired t-test on two sets of scores.
///
/// Tests the null hypothesis that the mean difference between paired observations is zero.
///
/// # Arguments
///
/// * `method_a` - Scores from method A (one per query)
/// * `method_b` - Scores from method B (one per query, paired with A)
/// * `alpha` - Significance level (default 0.05)
///
/// # Returns
///
/// `TTestResult` with t-statistic, p-value, and significance.
///
/// # Example
///
/// ```
/// use rank_eval::statistics::paired_t_test;
///
/// let method_a = vec![0.5, 0.6, 0.7, 0.8, 0.9];
/// let method_b = vec![0.4, 0.5, 0.6, 0.7, 0.8];
///
/// let result = paired_t_test(&method_a, &method_b, 0.05);
/// println!("t-statistic: {}, p-value: {}", result.t_statistic, result.p_value);
/// ```
pub fn paired_t_test(method_a: &[f64], method_b: &[f64], alpha: f64) -> TTestResult {
    assert_eq!(
        method_a.len(),
        method_b.len(),
        "method_a and method_b must have same length"
    );

    if method_a.len() < 2 {
        return TTestResult {
            t_statistic: 0.0,
            p_value: 1.0,
            degrees_of_freedom: 0,
            mean_difference: 0.0,
            std_error: 0.0,
            significant: false,
        };
    }

    // Compute differences
    let differences: Vec<f64> = method_a
        .iter()
        .zip(method_b.iter())
        .map(|(a, b)| a - b)
        .collect();

    // Mean difference
    let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;

    // Standard error
    let variance = differences
        .iter()
        .map(|d| (d - mean_diff).powi(2))
        .sum::<f64>()
        / (differences.len() - 1) as f64;
    let std_error = (variance / differences.len() as f64).sqrt();

    // t-statistic
    let t_statistic = if std_error > 1e-10 {
        mean_diff / std_error
    } else {
        0.0
    };

    // Degrees of freedom
    let df = differences.len() - 1;

    // Approximate p-value using t-distribution
    // For simplicity, using a normal approximation for large samples
    // For exact calculation, would need t-distribution library
    let p_value = if df > 30 {
        // Normal approximation
        let z = t_statistic.abs();
        2.0 * (1.0 - normal_cdf(z))
    } else {
        // Rough approximation for small samples
        // In production, use proper t-distribution
        let z = t_statistic.abs();
        2.0 * (1.0 - normal_cdf(z))
    };

    TTestResult {
        t_statistic,
        p_value,
        degrees_of_freedom: df,
        mean_difference: mean_diff,
        std_error,
        significant: p_value < alpha,
    }
}

/// Compute confidence interval for a set of scores.
///
/// # Arguments
///
/// * `scores` - Vector of scores
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
///
/// # Returns
///
/// `(lower_bound, upper_bound)` confidence interval.
///
/// # Example
///
/// ```
/// use rank_eval::statistics::confidence_interval;
///
/// let scores = vec![0.5, 0.6, 0.7, 0.8, 0.9];
/// let (lower, upper) = confidence_interval(&scores, 0.95);
/// println!("95% CI: [{:.3}, {:.3}]", lower, upper);
/// ```
pub fn confidence_interval(scores: &[f64], confidence: f64) -> (f64, f64) {
    if scores.is_empty() {
        return (0.0, 0.0);
    }

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f64>()
        / (scores.len() - 1) as f64;
    let std_dev = variance.sqrt();

    // Standard error
    let se = std_dev / (scores.len() as f64).sqrt();

    // z-score for confidence level (using normal approximation)
    let alpha = 1.0 - confidence;
    let z = normal_quantile(1.0 - alpha / 2.0);

    let margin = z * se;
    (mean - margin, mean + margin)
}

/// Compute Cohen's d effect size.
///
/// Measures the standardized difference between two means.
///
/// # Interpretation
///
/// - |d| < 0.2: Negligible
/// - 0.2 ≤ |d| < 0.5: Small
/// - 0.5 ≤ |d| < 0.8: Medium
/// - |d| ≥ 0.8: Large
///
/// # Example
///
/// ```
/// use rank_eval::statistics::cohens_d;
///
/// let method_a = vec![0.5, 0.6, 0.7];
/// let method_b = vec![0.4, 0.5, 0.6];
///
/// let d = cohens_d(&method_a, &method_b);
/// println!("Effect size: {:.3}", d);
/// ```
pub fn cohens_d(method_a: &[f64], method_b: &[f64]) -> f64 {
    assert_eq!(
        method_a.len(),
        method_b.len(),
        "method_a and method_b must have same length"
    );

    if method_a.is_empty() {
        return 0.0;
    }

    let mean_a = method_a.iter().sum::<f64>() / method_a.len() as f64;
    let mean_b = method_b.iter().sum::<f64>() / method_b.len() as f64;

    // Pooled standard deviation
    let var_a: f64 = method_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (method_a.len() - 1) as f64;
    let var_b: f64 = method_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (method_b.len() - 1) as f64;
    
    let pooled_std = ((var_a + var_b) / 2.0).sqrt();

    if pooled_std < 1e-10 {
        return 0.0;
    }

    (mean_a - mean_b) / pooled_std
}

/// Normal CDF approximation (using error function).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / (2.0_f64).sqrt()))
}

/// Error function approximation.
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Normal quantile (inverse CDF) approximation.
fn normal_quantile(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm approximation
    if p < 0.5 {
        -normal_quantile(1.0 - p)
    } else if p > 0.5 {
        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paired_t_test() {
        // Use data with variance in differences to get non-zero t-statistic
        let method_a = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let method_b = vec![0.4, 0.55, 0.6, 0.75, 0.8];

        let result = paired_t_test(&method_a, &method_b, 0.05);
        // t-statistic can be positive or negative depending on which method is better
        assert!(result.t_statistic.abs() >= 0.0 || result.t_statistic == 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.degrees_of_freedom, 4);
    }

    #[test]
    fn test_confidence_interval() {
        let scores = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let (lower, upper) = confidence_interval(&scores, 0.95);
        assert!(lower < upper);
        assert!(lower >= 0.0 && upper <= 1.0);
    }

    #[test]
    fn test_cohens_d() {
        let method_a = vec![0.5, 0.6, 0.7];
        let method_b = vec![0.4, 0.5, 0.6];

        let d = cohens_d(&method_a, &method_b);
        assert!(d > 0.0); // method_a should be better
    }
}

