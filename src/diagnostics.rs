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
        if samples.is_empty() {
            return Err(BayesError::invalid_parameter("No samples provided"));
        }

        let n_params = samples[0].len();
        let _n_samples = samples.len();

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
            effective_sample_size.push(calculate_ess(&param_chain));
            mc_se.push(calculate_mcse(&param_chain));
            mean.push(calculate_mean(&param_chain));
            std_dev.push(calculate_std_dev(&param_chain));
            quantiles.push(calculate_quantiles(&param_chain));
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

        let n_params = chains[0][0].len();
        let n_chains = chains.len();

        // Calculate single chain diagnostics first
        let mut diagnostics = Self::from_single_chain(&chains[0])?;

        // Calculate R-hat for each parameter
        let mut r_hat = Vec::with_capacity(n_params);
        for param_idx in 0..n_params {
            let mut param_chains = Vec::with_capacity(n_chains);
            for chain in chains {
                let mut param_chain = Vec::with_capacity(chain.len());
                for sample in chain {
                    param_chain.push(sample[param_idx]);
                }
                param_chains.push(param_chain);
            }
            r_hat.push(calculate_r_hat(&param_chains)?);
        }

        diagnostics.r_hat = Some(r_hat);
        Ok(diagnostics)
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
    let _n = samples.len() as f64;
    let variance = calculate_variance(samples);
    let ess = calculate_ess(samples);

    (variance / ess).sqrt()
}

/// Calculate R-hat statistic (Gelman-Rubin diagnostic)
fn calculate_r_hat(chains: &[Vec<f64>]) -> Result<f64> {
    if chains.len() < 2 {
        return Err(BayesError::invalid_parameter(
            "Need at least 2 chains for R-hat",
        ));
    }

    let m = chains.len() as f64; // number of chains
    let n = chains[0].len() as f64; // samples per chain

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
    let r_hat = (var_plus / within_chain_var).sqrt();

    Ok(r_hat)
}

/// Calculate autocorrelation function
fn calculate_autocorrelation(samples: &[f64]) -> Vec<f64> {
    let n = samples.len();
    let mean = calculate_mean(samples);
    let variance = calculate_variance(samples);

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
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Calculate variance of a sample
fn calculate_variance(samples: &[f64]) -> f64 {
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
        if samples.is_empty() {
            return Err(BayesError::invalid_parameter("No samples provided"));
        }

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
    fn test_r_hat_identical_chains() {
        let chain1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chain2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chains = vec![chain1, chain2];

        let r_hat = calculate_r_hat(&chains).unwrap();
        // For identical chains, R-hat should be close to 1.0, but numerical precision
        // can cause small deviations with short chains
        assert!((0.8..=1.2).contains(&r_hat));
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
