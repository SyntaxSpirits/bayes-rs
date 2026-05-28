//! MCMC diagnostics and convergence assessment

use crate::error::{BayesError, Result};
use nalgebra::DVector;

/// MCMC diagnostic statistics
#[derive(Debug, Clone)]
pub struct McmcDiagnostics {
    /// Effective sample size for each parameter
    pub effective_sample_size: Vec<f64>,
    /// R-hat statistic for each parameter (requires multiple chains)
    pub r_hat: Option<Vec<f64>>,
    /// Monte Carlo standard error for each parameter
    pub mc_se: Vec<f64>,
    /// Mean of each parameter
    pub mean: Vec<f64>,
    /// Standard deviation of each parameter
    pub std_dev: Vec<f64>,
    /// Quantiles (2.5%, 25%, 50%, 75%, 97.5%)
    pub quantiles: Vec<[f64; 5]>,
}

impl McmcDiagnostics {
    /// Create diagnostics from a single chain
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

    /// Create diagnostics from multiple chains
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

    /// Check if chains have converged (R-hat < 1.1)
    pub fn has_converged(&self) -> bool {
        if let Some(ref r_hat) = self.r_hat {
            r_hat.iter().all(|&rhat| rhat < 1.1)
        } else {
            false // Cannot assess convergence without multiple chains
        }
    }

    /// Get parameters with low effective sample size (< 400)
    pub fn low_ess_params(&self) -> Vec<usize> {
        self.effective_sample_size
            .iter()
            .enumerate()
            .filter(|(_, &ess)| ess < 400.0)
            .map(|(i, _)| i)
            .collect()
    }
}

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

/// Calculate effective sample size using autocorrelation
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

/// Calculate Monte Carlo standard error
fn calculate_mcse(samples: &[f64]) -> f64 {
    let variance = calculate_variance(samples);
    if variance == 0.0 {
        return 0.0;
    }

    let ess = calculate_ess(samples);

    (variance / ess).sqrt()
}

fn calculate_multi_chain_ess(chains: &[Vec<f64>]) -> f64 {
    chains.iter().map(|chain| calculate_ess(chain)).sum()
}

/// Calculate R-hat statistic (Gelman-Rubin diagnostic)
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

/// Calculate autocorrelation function
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

/// Calculate integrated autocorrelation time
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

/// Calculate mean of a sample
fn calculate_mean(samples: &[f64]) -> f64 {
    let mut mean = 0.0;
    for (idx, &sample) in samples.iter().enumerate() {
        mean += (sample - mean) / (idx + 1) as f64;
    }
    mean
}

/// Calculate variance of a sample
fn calculate_variance(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mean = calculate_mean(samples);
    let sum_sq_diff = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    sum_sq_diff / (samples.len() - 1) as f64
}

/// Calculate standard deviation of a sample
fn calculate_std_dev(samples: &[f64]) -> f64 {
    calculate_variance(samples).sqrt()
}

/// Calculate quantiles (2.5%, 25%, 50%, 75%, 97.5%)
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

/// Simple trace plot data for visualization
#[derive(Debug, Clone)]
pub struct TracePlot {
    pub parameter_index: usize,
    pub values: Vec<f64>,
    pub iterations: Vec<usize>,
}

impl TracePlot {
    /// Create trace plot data for a parameter
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
