//! MCMC diagnostics and convergence assessment.
//!
//! This module summarizes Markov chain Monte Carlo output with common
//! user-facing diagnostics:
//!
//! - **Effective sample size (ESS)** estimates how many independent draws would
//!   carry the same information as an autocorrelated chain. For a single chain
//!   with `n` draws this implementation computes autocorrelations `rho_t`, an
//!   integrated autocorrelation time `tau = sum_t rho_t`, and returns
//!   `n / (1 + 2 * tau)` when `tau > 0`; otherwise it falls back to `n`.
//! - **R-hat** is the Gelman-Rubin potential scale reduction factor. For `m`
//!   equal-length chains of length `n`, it computes within-chain variance `W`,
//!   between-chain variance `B`, `var+ = ((n - 1) / n) * W + B / n`, and
//!   `sqrt(var+ / W)`.
//! - **Monte Carlo standard error (MCSE)** is `sqrt(variance / ESS)` for a
//!   single chain. For multiple chains, MCSE uses the pooled variance divided by
//!   the sum of per-chain ESS estimates so unrelated chain boundaries are not
//!   treated as adjacent Markov states.
//! - **Autocorrelation** at lag `t` is calculated as the lagged covariance sum
//!   `sum_i ((x_i - mean) * (x_{i+t} - mean)) / ((n - t) * sample_variance)`.
//!   Lags are evaluated in the half-open range `0..min(n / 4, 100)`. The
//!   integrated autocorrelation sum truncates when correlation drops below
//!   `0.01` or when the lag exceeds `6 * tau`.
//!
//! Edge cases are handled explicitly. Empty sample sets, zero-dimensional
//! samples, non-finite values, inconsistent dimensions, unequal chain lengths,
//! and fewer than two samples per chain for R-hat return errors. Constant chains
//! have zero variance, zero MCSE, and ESS equal to the chain length. Identical
//! constant chains have R-hat `1.0`; constant chains with different means make
//! R-hat undefined and return an error.
//!
//! Convergence helpers use the public [`R_HAT_CONVERGENCE_THRESHOLD`] and
//! [`LOW_ESS_THRESHOLD`] constants so applications and documentation can share
//! the same thresholds.

use crate::error::{BayesError, Result};
use nalgebra::DVector;

/// Default R-hat convergence threshold used by [`McmcDiagnostics::has_converged`].
///
/// A parameter is treated as converged only when its R-hat is strictly less than
/// this value. A value exactly equal to the threshold does not pass.
pub const R_HAT_CONVERGENCE_THRESHOLD: f64 = 1.1;

/// Default low effective sample size threshold used by [`McmcDiagnostics::low_ess_params`].
///
/// A parameter is reported as low-ESS only when its ESS is strictly less than
/// this value. A value exactly equal to the threshold is not reported.
pub const LOW_ESS_THRESHOLD: f64 = 400.0;

/// MCMC diagnostic statistics for each model parameter.
///
/// Diagnostics can be created from one chain with [`Self::from_single_chain`] or
/// from multiple chains with [`Self::from_multiple_chains`]. Vectors are ordered
/// by parameter index.
#[derive(Debug, Clone)]
pub struct McmcDiagnostics {
    /// Effective sample size for each parameter.
    ///
    /// For one chain, ESS is based on `n / (1 + 2 * tau)` where `tau` is the
    /// truncated integrated autocorrelation time. For multiple chains, this is
    /// the sum of each chain's ESS for the parameter.
    pub effective_sample_size: Vec<f64>,
    /// R-hat statistic for each parameter, or `None` for single-chain diagnostics.
    ///
    /// R-hat requires at least two chains with at least two samples each. The
    /// implementation returns `1.0` for identical constant chains and an error
    /// for constant chains with different means because the ratio is undefined.
    pub r_hat: Option<Vec<f64>>,
    /// Monte Carlo standard error for each parameter.
    ///
    /// MCSE is `sqrt(variance / ESS)`, with `0.0` returned for zero-variance
    /// chains.
    pub mc_se: Vec<f64>,
    /// Mean of each parameter.
    pub mean: Vec<f64>,
    /// Sample standard deviation of each parameter.
    pub std_dev: Vec<f64>,
    /// Empirical quantiles for each parameter: 2.5%, 25%, 50%, 75%, and 97.5%.
    pub quantiles: Vec<[f64; 5]>,
}

impl McmcDiagnostics {
    /// Create diagnostics from a single chain.
    ///
    /// Returns an error if `samples` is empty, if samples contain zero
    /// parameters, if dimensions differ across samples, if any value is
    /// non-finite, or if a computed diagnostic is non-finite. R-hat is not
    /// available for one chain and is set to `None`.
    pub fn from_single_chain(samples: &[DVector<f64>]) -> Result<Self> {
        validate_samples(samples)?;
        let n_params = samples[0].len();

        // Extract parameter samples
        let mut param_samples = vec![Vec::new(); n_params];
        for sample in samples {
            for (i, &value) in sample.iter().enumerate() {
                param_samples[i].push(value);
            }
        }

        let mut effective_sample_size = Vec::with_capacity(n_params);
        let mut mc_se = Vec::with_capacity(n_params);
        let mut mean = Vec::with_capacity(n_params);
        let mut std_dev = Vec::with_capacity(n_params);
        let mut quantiles = Vec::with_capacity(n_params);

        for param_chain in param_samples {
            let ess = calculate_ess(&param_chain);
            let mcse = calculate_mcse(&param_chain);
            let param_mean = calculate_mean(&param_chain);
            let param_std_dev = calculate_std_dev(&param_chain);
            let param_quantiles = calculate_quantiles(&param_chain);

            if !ess.is_finite()
                || !mcse.is_finite()
                || !param_mean.is_finite()
                || !param_std_dev.is_finite()
                || !param_quantiles.iter().all(|value| value.is_finite())
            {
                return Err(BayesError::numerical_error(
                    "Diagnostics produced non-finite values",
                ));
            }

            effective_sample_size.push(ess);
            mc_se.push(mcse);
            mean.push(param_mean);
            std_dev.push(param_std_dev);
            quantiles.push(param_quantiles);
        }

        Ok(Self {
            effective_sample_size,
            r_hat: None,
            mc_se,
            mean,
            std_dev,
            quantiles,
        })
    }

    /// Create diagnostics from multiple chains.
    ///
    /// Each chain must be non-empty, finite, have the same sample count, and use
    /// the same parameter dimension. R-hat additionally requires at least two
    /// samples per chain. Multi-chain ESS sums per-chain ESS estimates; MCSE is
    /// computed from pooled variance divided by that summed ESS.
    pub fn from_multiple_chains(chains: &[Vec<DVector<f64>>]) -> Result<Self> {
        if chains.is_empty() {
            return Err(BayesError::invalid_parameter("No chains provided"));
        }

        for chain in chains {
            validate_samples(chain)?;
        }

        let n_params = chains[0][0].len();
        let n_samples = chains[0].len();

        for chain in chains.iter().skip(1) {
            if chain.len() != n_samples {
                return Err(BayesError::invalid_parameter(
                    "All chains must have same length",
                ));
            }

            if chain[0].len() != n_params {
                return Err(BayesError::dimension_mismatch(n_params, chain[0].len()));
            }
        }

        let mut effective_sample_size = Vec::with_capacity(n_params);
        let mut mc_se = Vec::with_capacity(n_params);
        let mut mean = Vec::with_capacity(n_params);
        let mut std_dev = Vec::with_capacity(n_params);
        let mut quantiles = Vec::with_capacity(n_params);
        let mut r_hat = Vec::with_capacity(n_params);

        for param_idx in 0..n_params {
            let param_chains: Vec<Vec<f64>> = chains
                .iter()
                .map(|chain| chain.iter().map(|sample| sample[param_idx]).collect())
                .collect();
            let pooled_samples: Vec<f64> = param_chains.iter().flatten().copied().collect();

            let ess = calculate_multi_chain_ess(&param_chains);
            let pooled_variance = calculate_variance(&pooled_samples);
            // Use pooled variance with the summed per-chain ESS so chain boundaries do not
            // influence autocorrelation while between-chain variance still raises MCSE.
            let mcse = if pooled_variance == 0.0 {
                0.0
            } else {
                (pooled_variance / ess).sqrt()
            };
            let param_mean = calculate_mean(&pooled_samples);
            let param_std_dev = calculate_std_dev(&pooled_samples);
            let param_quantiles = calculate_quantiles(&pooled_samples);
            let param_r_hat = calculate_r_hat(&param_chains)?;

            if !ess.is_finite()
                || !mcse.is_finite()
                || !param_mean.is_finite()
                || !param_std_dev.is_finite()
                || !param_quantiles.iter().all(|value| value.is_finite())
                || !param_r_hat.is_finite()
            {
                return Err(BayesError::numerical_error(
                    "Diagnostics produced non-finite values",
                ));
            }

            effective_sample_size.push(ess);
            mc_se.push(mcse);
            mean.push(param_mean);
            std_dev.push(param_std_dev);
            quantiles.push(param_quantiles);
            r_hat.push(param_r_hat);
        }

        Ok(Self {
            effective_sample_size,
            r_hat: Some(r_hat),
            mc_se,
            mean,
            std_dev,
            quantiles,
        })
    }

    /// Check whether all parameters satisfy the R-hat convergence threshold.
    ///
    /// Returns `true` only when every available R-hat is strictly less than
    /// [`R_HAT_CONVERGENCE_THRESHOLD`]. Returns `false` for single-chain
    /// diagnostics because convergence cannot be assessed without multiple
    /// chains.
    pub fn has_converged(&self) -> bool {
        if let Some(ref r_hat) = self.r_hat {
            r_hat.iter().all(|&rhat| rhat < R_HAT_CONVERGENCE_THRESHOLD)
        } else {
            false // Cannot assess convergence without multiple chains
        }
    }

    /// Return parameter indices whose ESS is below [`LOW_ESS_THRESHOLD`].
    ///
    /// The comparison is strict: an ESS exactly equal to the threshold is not
    /// reported.
    pub fn low_ess_params(&self) -> Vec<usize> {
        self.effective_sample_size
            .iter()
            .enumerate()
            .filter(|(_, &ess)| ess < LOW_ESS_THRESHOLD)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return a compact per-parameter summary of R-hat, ESS, and MCSE.
    ///
    /// This is useful for reporting the core convergence diagnostics together
    /// while retaining the full [`McmcDiagnostics`] value for means, standard
    /// deviations, and quantiles.
    ///
    /// This summary assumes the diagnostics were produced by this crate's
    /// constructors, which populate ESS, MCSE, and any R-hat values with matching
    /// per-parameter lengths. Manually constructed inconsistent diagnostics may
    /// produce an incomplete summary.
    pub fn summary(&self) -> McmcDiagnosticSummary {
        McmcDiagnosticSummary::from(self)
    }
}

/// Compact diagnostic summary for one model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterDiagnosticSummary {
    /// Zero-based parameter index.
    pub parameter_index: usize,
    /// R-hat statistic for this parameter, or `None` for single-chain diagnostics.
    pub r_hat: Option<f64>,
    /// Effective sample size for this parameter.
    pub effective_sample_size: f64,
    /// Monte Carlo standard error for this parameter.
    pub mc_se: f64,
}

/// Compact diagnostics summary focused on convergence reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct McmcDiagnosticSummary {
    /// Per-parameter R-hat, ESS, and MCSE values.
    pub parameters: Vec<ParameterDiagnosticSummary>,
    /// Whether all available R-hat values satisfy the convergence threshold.
    pub has_converged: bool,
    /// Parameter indices whose ESS is below [`LOW_ESS_THRESHOLD`].
    pub low_ess_params: Vec<usize>,
}

impl From<&McmcDiagnostics> for McmcDiagnosticSummary {
    /// Build a compact summary from diagnostics produced by this crate's
    /// constructors.
    ///
    /// The conversion expects ESS, MCSE, and any R-hat vectors to use the same
    /// per-parameter indexing. Public struct literals can violate that invariant;
    /// prefer [`McmcDiagnostics::from_single_chain`] or
    /// [`McmcDiagnostics::from_multiple_chains`] when creating diagnostics for
    /// summarization.
    fn from(diagnostics: &McmcDiagnostics) -> Self {
        debug_assert_eq!(
            diagnostics.effective_sample_size.len(),
            diagnostics.mc_se.len(),
            "diagnostics ESS and MCSE lengths should match"
        );
        if let Some(r_hat) = diagnostics.r_hat.as_ref() {
            debug_assert_eq!(
                diagnostics.effective_sample_size.len(),
                r_hat.len(),
                "diagnostics ESS and R-hat lengths should match"
            );
        }

        let parameters = diagnostics
            .effective_sample_size
            .iter()
            .zip(diagnostics.mc_se.iter())
            .enumerate()
            .map(|(parameter_index, (&effective_sample_size, &mc_se))| {
                let r_hat = diagnostics
                    .r_hat
                    .as_ref()
                    .and_then(|values| values.get(parameter_index).copied());

                ParameterDiagnosticSummary {
                    parameter_index,
                    r_hat,
                    effective_sample_size,
                    mc_se,
                }
            })
            .collect();

        Self {
            parameters,
            has_converged: diagnostics.has_converged(),
            low_ess_params: diagnostics.low_ess_params(),
        }
    }
}

/// Validate shared single-chain sample requirements.
///
/// Samples must be non-empty, contain at least one parameter, have consistent
/// dimensions, and contain only finite values.
fn validate_samples(samples: &[DVector<f64>]) -> Result<()> {
    if samples.is_empty() {
        return Err(BayesError::invalid_parameter("No samples provided"));
    }

    let n_params = samples[0].len();
    if n_params == 0 {
        return Err(BayesError::invalid_parameter(
            "Samples must contain at least one parameter",
        ));
    }

    for sample in samples {
        if sample.len() != n_params {
            return Err(BayesError::dimension_mismatch(n_params, sample.len()));
        }

        if !sample.iter().all(|value| value.is_finite()) {
            return Err(BayesError::invalid_parameter(
                "Samples must contain only finite values",
            ));
        }
    }

    Ok(())
}

/// Calculate effective sample size using truncated autocorrelation.
///
/// For chains with fewer than ten samples, returns `n` because the
/// autocorrelation estimate is not considered meaningful. Constant chains also
/// return `n` via zero autocorrelation time.
fn calculate_ess(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    if n < 10.0 {
        return n; // Too few samples for meaningful ESS
    }

    let autocorr = calculate_autocorrelation(samples);
    let tau = calculate_integrated_autocorr_time(&autocorr);

    if tau <= 0.0 {
        n
    } else {
        n / (1.0 + 2.0 * tau)
    }
}

/// Calculate Monte Carlo standard error as `sqrt(sample_variance / ESS)`.
///
/// Returns `0.0` when the sample variance is zero.
fn calculate_mcse(samples: &[f64]) -> f64 {
    let variance = calculate_variance(samples);
    if variance == 0.0 {
        return 0.0;
    }

    let ess = calculate_ess(samples);

    (variance / ess).sqrt()
}

/// Estimate total ESS by summing per-chain ESS values.
///
/// Chains are not concatenated because that would treat unrelated chain
/// boundaries as adjacent Markov states and distort autocorrelation.
fn calculate_multi_chain_ess(chains: &[Vec<f64>]) -> f64 {
    chains.iter().map(|chain| calculate_ess(chain)).sum()
}

/// Calculate the Gelman-Rubin R-hat diagnostic.
///
/// Formula: `sqrt(var+ / W)`, where `W` is the mean within-chain sample
/// variance, `B` is the between-chain variance multiplied by chain length, and
/// `var+ = ((n - 1) / n) * W + B / n`. Requires at least two equal-length
/// chains with at least two samples each. Identical constant chains return
/// `1.0`; constant chains with different means return an error because `W = 0`
/// makes R-hat undefined.
fn calculate_r_hat(chains: &[Vec<f64>]) -> Result<f64> {
    if chains.len() < 2 {
        return Err(BayesError::invalid_parameter(
            "Need at least 2 chains for R-hat",
        ));
    }

    let m = chains.len() as f64; // number of chains
    if chains.iter().any(Vec::is_empty) {
        return Err(BayesError::invalid_parameter("Chains must not be empty"));
    }

    let n = chains[0].len() as f64; // samples per chain
    if n < 2.0 {
        return Err(BayesError::invalid_parameter(
            "Need at least 2 samples per chain for R-hat",
        ));
    }

    // Check all chains have same length
    if chains.iter().any(|chain| chain.len() != n as usize) {
        return Err(BayesError::invalid_parameter(
            "All chains must have same length",
        ));
    }

    // Calculate chain means
    let chain_means: Vec<f64> = chains.iter().map(|chain| calculate_mean(chain)).collect();

    // Calculate overall mean
    let overall_mean = chain_means.iter().sum::<f64>() / m;

    // Calculate within-chain variance
    let within_chain_var: f64 = chains
        .iter()
        .map(|chain| {
            let chain_mean = calculate_mean(chain);
            chain.iter().map(|&x| (x - chain_mean).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .sum::<f64>()
        / m;

    // Calculate between-chain variance
    let between_chain_var = n * chain_means
        .iter()
        .map(|&mean| (mean - overall_mean).powi(2))
        .sum::<f64>()
        / (m - 1.0);

    // Calculate R-hat
    let var_plus = ((n - 1.0) / n) * within_chain_var + (1.0 / n) * between_chain_var;
    if within_chain_var == 0.0 {
        return if between_chain_var == 0.0 {
            Ok(1.0)
        } else {
            Err(BayesError::numerical_error(
                "R-hat is undefined for constant chains with different means",
            ))
        };
    }

    let r_hat = (var_plus / within_chain_var).sqrt();

    Ok(r_hat)
}

/// Calculate lag autocorrelations up to `min(n / 4, 100)` lags.
///
/// Lag zero is included when variance is non-zero. Constant chains return a
/// single zero autocorrelation so downstream ESS falls back to the chain length.
fn calculate_autocorrelation(samples: &[f64]) -> Vec<f64> {
    let n = samples.len();
    let mean = calculate_mean(samples);
    let variance = calculate_variance(samples);
    if variance == 0.0 {
        return vec![0.0];
    }

    let max_lag = (n / 4).min(100); // Limit autocorrelation calculation
    let mut autocorr = Vec::with_capacity(max_lag);

    for lag in 0..max_lag {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..(n - lag) {
            sum += (samples[i] - mean) * (samples[i + lag] - mean);
            count += 1;
        }

        if count > 0 {
            autocorr.push(sum / (count as f64 * variance));
        } else {
            autocorr.push(0.0);
        }
    }

    autocorr
}

/// Calculate the truncated integrated autocorrelation time.
///
/// The lag-zero autocorrelation is skipped. Summation stops when the current
/// correlation is below `0.01` or the lag is greater than `6 * tau`, then the
/// result is clamped to be non-negative.
fn calculate_integrated_autocorr_time(autocorr: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut tau = 0.0;

    for (i, &corr) in autocorr.iter().enumerate() {
        if i == 0 {
            continue;
        }

        sum += corr;
        tau = sum;

        // Stop when autocorrelation becomes negligible
        if corr < 0.01 || i as f64 > 6.0 * tau {
            break;
        }
    }

    tau.max(0.0)
}

/// Calculate the arithmetic mean using an online update.
fn calculate_mean(samples: &[f64]) -> f64 {
    let mut mean = 0.0;
    for (idx, &sample) in samples.iter().enumerate() {
        mean += (sample - mean) / (idx + 1) as f64;
    }
    mean
}

/// Calculate unbiased sample variance with denominator `n - 1`.
///
/// Returns `0.0` for fewer than two samples.
fn calculate_variance(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mean = calculate_mean(samples);
    let sum_sq_diff = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    sum_sq_diff / (samples.len() - 1) as f64
}

/// Calculate sample standard deviation.
fn calculate_std_dev(samples: &[f64]) -> f64 {
    calculate_variance(samples).sqrt()
}

/// Calculate empirical quantiles at 2.5%, 25%, 50%, 75%, and 97.5%.
///
/// Quantile indices are selected by truncating `n * p` to an integer and
/// clamping to the final sample index.
fn calculate_quantiles(samples: &[f64]) -> [f64; 5] {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let indices = [
        (n as f64 * 0.025) as usize,
        (n as f64 * 0.25) as usize,
        (n as f64 * 0.5) as usize,
        (n as f64 * 0.75) as usize,
        (n as f64 * 0.975) as usize,
    ];

    [
        sorted[indices[0].min(n - 1)],
        sorted[indices[1].min(n - 1)],
        sorted[indices[2].min(n - 1)],
        sorted[indices[3].min(n - 1)],
        sorted[indices[4].min(n - 1)],
    ]
}

/// Simple trace plot data for visualization.
#[derive(Debug, Clone)]
pub struct TracePlot {
    /// Parameter index represented by this trace.
    pub parameter_index: usize,
    /// Parameter values across iterations.
    pub values: Vec<f64>,
    /// Zero-based iteration indices corresponding to [`Self::values`].
    pub iterations: Vec<usize>,
}

impl TracePlot {
    /// Create trace plot data for a parameter.
    ///
    /// Returns an error for invalid samples or if `parameter_index` is out of
    /// bounds for the sample dimension.
    pub fn new(samples: &[DVector<f64>], parameter_index: usize) -> Result<Self> {
        validate_samples(samples)?;

        if parameter_index >= samples[0].len() {
            return Err(BayesError::invalid_parameter(
                "Parameter index out of bounds",
            ));
        }

        let values: Vec<f64> = samples.iter().map(|s| s[parameter_index]).collect();
        let iterations: Vec<usize> = (0..samples.len()).collect();

        Ok(Self {
            parameter_index,
            values,
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_convergence_threshold_constants_are_public_contract() {
        assert_abs_diff_eq!(R_HAT_CONVERGENCE_THRESHOLD, 1.1, epsilon = 1e-12);
        assert_abs_diff_eq!(LOW_ESS_THRESHOLD, 400.0, epsilon = 1e-12);
    }

    #[test]
    fn test_convergence_and_low_ess_follow_public_threshold_contract() {
        let diagnostics = McmcDiagnostics {
            effective_sample_size: vec![LOW_ESS_THRESHOLD - 1.0, LOW_ESS_THRESHOLD],
            r_hat: Some(vec![R_HAT_CONVERGENCE_THRESHOLD - 0.01]),
            mc_se: vec![0.0, 0.0],
            mean: vec![0.0, 0.0],
            std_dev: vec![0.0, 0.0],
            quantiles: vec![[0.0; 5], [0.0; 5]],
        };

        assert!(diagnostics.has_converged());
        assert_eq!(diagnostics.low_ess_params(), vec![0]);

        let at_threshold = McmcDiagnostics {
            r_hat: Some(vec![R_HAT_CONVERGENCE_THRESHOLD]),
            ..diagnostics
        };
        assert!(!at_threshold.has_converged());
    }

    #[test]
    fn test_summary_groups_r_hat_ess_and_mcse_by_parameter() {
        let diagnostics = McmcDiagnostics {
            effective_sample_size: vec![100.0, LOW_ESS_THRESHOLD - 1.0],
            r_hat: Some(vec![1.01, 1.2]),
            mc_se: vec![0.1, 0.2],
            mean: vec![0.0, 1.0],
            std_dev: vec![1.0, 2.0],
            quantiles: vec![[0.0; 5], [1.0; 5]],
        };

        let summary = diagnostics.summary();

        assert_eq!(summary.parameters.len(), 2);
        assert_eq!(summary.parameters[0].parameter_index, 0);
        assert_eq!(summary.parameters[0].r_hat, Some(1.01));
        assert_eq!(summary.parameters[0].effective_sample_size, 100.0);
        assert_eq!(summary.parameters[0].mc_se, 0.1);
        assert_eq!(summary.parameters[1].parameter_index, 1);
        assert_eq!(summary.parameters[1].r_hat, Some(1.2));
        assert!(!summary.has_converged);
        assert_eq!(summary.low_ess_params, vec![0, 1]);
    }

    #[test]
    fn test_single_chain_summary_omits_r_hat() {
        let diagnostics = McmcDiagnostics {
            effective_sample_size: vec![LOW_ESS_THRESHOLD],
            r_hat: None,
            mc_se: vec![0.0],
            mean: vec![0.0],
            std_dev: vec![0.0],
            quantiles: vec![[0.0; 5]],
        };

        let summary = McmcDiagnosticSummary::from(&diagnostics);

        assert_eq!(summary.parameters[0].r_hat, None);
        assert!(!summary.has_converged);
        assert!(summary.low_ess_params.is_empty());
    }

    #[test]
    fn test_basic_statistics() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_abs_diff_eq!(calculate_mean(&samples), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(calculate_variance(&samples), 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(calculate_std_dev(&samples), 2.5_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_quantiles() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantiles = calculate_quantiles(&samples);

        assert_eq!(quantiles[2], 3.0); // median
        assert_eq!(quantiles[0], 1.0); // 2.5th percentile
        assert_eq!(quantiles[4], 5.0); // 97.5th percentile
    }

    #[test]
    fn test_single_chain_diagnostics() {
        let samples = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![1.1, 2.1]),
            DVector::from_vec(vec![0.9, 1.9]),
            DVector::from_vec(vec![1.2, 2.2]),
            DVector::from_vec(vec![0.8, 1.8]),
        ];

        let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

        assert_eq!(diagnostics.mean.len(), 2);
        assert_eq!(diagnostics.std_dev.len(), 2);
        assert_eq!(diagnostics.effective_sample_size.len(), 2);
        assert!(diagnostics.r_hat.is_none());
    }

    #[test]
    fn test_single_chain_rejects_inconsistent_dimensions() {
        let samples = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![1.0]),
        ];

        assert!(McmcDiagnostics::from_single_chain(&samples).is_err());
    }

    #[test]
    fn test_single_chain_rejects_zero_parameters() {
        let samples = vec![DVector::from_vec(Vec::new())];

        assert!(McmcDiagnostics::from_single_chain(&samples).is_err());
    }

    #[test]
    fn test_single_sample_chain_diagnostics_are_finite() {
        let samples = vec![DVector::from_vec(vec![2.0])];

        let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

        assert_abs_diff_eq!(diagnostics.std_dev[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diagnostics.mc_se[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diagnostics.effective_sample_size[0], 1.0, epsilon = 1e-10);
        assert!(diagnostics.mean[0].is_finite());
    }

    #[test]
    fn test_multiple_chains_rejects_single_sample_chains() {
        let chains = vec![
            vec![DVector::from_vec(vec![1.0])],
            vec![DVector::from_vec(vec![1.0])],
        ];

        assert!(McmcDiagnostics::from_multiple_chains(&chains).is_err());
    }

    #[test]
    fn test_multiple_chains_rejects_empty_chains() {
        let chains = vec![vec![DVector::from_vec(vec![1.0])], Vec::new()];

        assert!(McmcDiagnostics::from_multiple_chains(&chains).is_err());
    }

    #[test]
    fn test_multiple_chains_rejects_inconsistent_dimensions() {
        let chains = vec![
            vec![DVector::from_vec(vec![1.0, 2.0])],
            vec![DVector::from_vec(vec![1.0])],
        ];

        assert!(McmcDiagnostics::from_multiple_chains(&chains).is_err());
    }

    #[test]
    fn test_constant_chain_diagnostics_are_finite() {
        let samples = vec![DVector::from_vec(vec![2.0]); 20];

        let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

        assert_abs_diff_eq!(diagnostics.std_dev[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diagnostics.mc_se[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diagnostics.effective_sample_size[0], 20.0, epsilon = 1e-10);
        assert!(diagnostics.mean[0].is_finite());
    }

    #[test]
    fn test_constant_chains_with_different_means_reject_r_hat() {
        let chains = vec![vec![1.0, 1.0], vec![2.0, 2.0]];

        assert!(calculate_r_hat(&chains).is_err());
    }

    #[test]
    fn test_large_constant_chain_diagnostics_are_finite() {
        let samples = vec![DVector::from_vec(vec![1.0e308]); 4];

        let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

        assert!(diagnostics.mean[0].is_finite());
        assert_abs_diff_eq!(diagnostics.std_dev[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diagnostics.mc_se[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_overflowing_diagnostics_return_error() {
        let samples = vec![
            DVector::from_vec(vec![1.0e308]),
            DVector::from_vec(vec![-1.0e308]),
        ];

        assert!(McmcDiagnostics::from_single_chain(&samples).is_err());
    }

    #[test]
    fn test_r_hat_identical_chains_returns_raw_value() {
        let chain1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chain2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chains = vec![chain1, chain2];

        let r_hat = calculate_r_hat(&chains).unwrap();
        assert_abs_diff_eq!(r_hat, (4.0_f64 / 5.0).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_multiple_chain_ess_sums_per_chain_without_boundary_correlation() {
        let chain: Vec<DVector<f64>> = (1..=20)
            .map(|value| DVector::from_vec(vec![value as f64]))
            .collect();
        let chains = vec![chain.clone(), chain];

        let diagnostics = McmcDiagnostics::from_multiple_chains(&chains).unwrap();
        let param_chains: Vec<Vec<f64>> = chains
            .iter()
            .map(|chain| chain.iter().map(|sample| sample[0]).collect())
            .collect();
        let expected_ess = calculate_ess(&param_chains[0]) + calculate_ess(&param_chains[1]);

        assert_abs_diff_eq!(
            diagnostics.effective_sample_size[0],
            expected_ess,
            epsilon = 1e-10
        );
        assert!(diagnostics.effective_sample_size[0] < 40.0);
    }

    #[test]
    fn test_trace_plot_rejects_inconsistent_dimensions() {
        let samples = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![1.0]),
        ];

        assert!(TracePlot::new(&samples, 1).is_err());
    }

    #[test]
    fn test_trace_plot() {
        let samples = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![1.1, 2.1]),
            DVector::from_vec(vec![0.9, 1.9]),
        ];

        let trace_plot = TracePlot::new(&samples, 0).unwrap();
        assert_eq!(trace_plot.values, vec![1.0, 1.1, 0.9]);
        assert_eq!(trace_plot.iterations, vec![0, 1, 2]);
    }
}
