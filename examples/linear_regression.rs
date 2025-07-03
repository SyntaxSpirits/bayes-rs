//! Bayesian Linear Regression Example
//!
//! This example demonstrates how to perform Bayesian linear regression using
//! the bayes-rs library. We'll estimate the parameters of a simple linear
//! relationship with uncertainty quantification.

use bayes_rs::{
    diagnostics::McmcDiagnostics,
    distributions::{Distribution, Normal},
    prelude::*,
    samplers::{HamiltonianMonteCarlo, MetropolisHastings, Sampler},
};
use nalgebra::DVector;
use std::fs::File;
use std::io::Write;

/// Generate synthetic data for linear regression
fn generate_data(
    n: usize,
    true_slope: f64,
    true_intercept: f64,
    noise_std: f64,
) -> (Vec<f64>, Vec<f64>) {
    use rand::prelude::*;
    use rand_distr::Normal as RandNormal;

    let mut rng = thread_rng();
    let noise_dist = RandNormal::new(0.0, noise_std).unwrap();

    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..n {
        let x_val = i as f64 / n as f64 * 10.0; // x values from 0 to 10
        let y_val = true_intercept + true_slope * x_val + noise_dist.sample(&mut rng);

        x.push(x_val);
        y.push(y_val);
    }

    (x, y)
}

/// Bayesian linear regression with Metropolis-Hastings
fn bayesian_linear_regression_mh(x: &[f64], y: &[f64]) -> Result<Vec<DVector<f64>>> {
    let _n = x.len();

    // Define log posterior: p(β₀, β₁, log_σ | data)
    let log_posterior = |params: &DVector<f64>| -> f64 {
        let beta0 = params[0]; // intercept
        let beta1 = params[1]; // slope
        let log_sigma = params[2]; // log(noise standard deviation)
        let sigma = log_sigma.exp();

        // Prior distributions
        // β₀ ~ N(0, 10²)
        // β₁ ~ N(0, 10²)
        // log_σ ~ N(0, 1)
        let prior_beta0 = Normal::new(0.0, 10.0).unwrap();
        let prior_beta1 = Normal::new(0.0, 10.0).unwrap();
        let prior_log_sigma = Normal::new(0.0, 1.0).unwrap();

        let prior_log_prob = prior_beta0.log_pdf(beta0)
            + prior_beta1.log_pdf(beta1)
            + prior_log_sigma.log_pdf(log_sigma);

        // Likelihood: y_i ~ N(β₀ + β₁ * x_i, σ²)
        if !sigma.is_finite() || sigma <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let likelihood_dist = Normal::new(0.0, sigma).unwrap();
        let likelihood_log_prob: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&x_i, &y_i)| {
                let predicted = beta0 + beta1 * x_i;
                let residual = y_i - predicted;
                likelihood_dist.log_pdf(residual)
            })
            .sum();

        prior_log_prob + likelihood_log_prob
    };

    // The log_posterior function now returns f64 directly

    // Initial state: [intercept, slope, log_sigma]
    let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let proposal_std = DVector::from_vec(vec![0.5, 0.1, 0.1]);

    let mut sampler = MetropolisHastings::new(log_posterior, initial_state, proposal_std)?;

    // Warm-up phase
    let _ = sampler.sample(1000);

    // Main sampling
    let samples = sampler.sample(10000);

    println!(
        "Metropolis-Hastings Acceptance Rate: {:.3}",
        sampler.acceptance_rate().unwrap_or(0.0)
    );

    Ok(samples)
}

/// Bayesian linear regression with HMC
fn bayesian_linear_regression_hmc(x: &[f64], y: &[f64]) -> Result<Vec<DVector<f64>>> {
    let _n = x.len();

    // Define log posterior
    let log_posterior = |params: &DVector<f64>| -> f64 {
        let beta0 = params[0];
        let beta1 = params[1];
        let log_sigma = params[2];
        let sigma = log_sigma.exp();

        // Prior log probabilities
        let prior_beta0 = Normal::new(0.0, 10.0).unwrap();
        let prior_beta1 = Normal::new(0.0, 10.0).unwrap();
        let prior_log_sigma = Normal::new(0.0, 1.0).unwrap();

        let prior_log_prob = prior_beta0.log_pdf(beta0)
            + prior_beta1.log_pdf(beta1)
            + prior_log_sigma.log_pdf(log_sigma);

        // Likelihood
        if !sigma.is_finite() || sigma <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let likelihood_dist = Normal::new(0.0, sigma).unwrap();
        let likelihood_log_prob: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&x_i, &y_i)| {
                let predicted = beta0 + beta1 * x_i;
                let residual = y_i - predicted;
                likelihood_dist.log_pdf(residual)
            })
            .sum();

        prior_log_prob + likelihood_log_prob
    };

    // Gradient of log posterior
    let gradient = |params: &DVector<f64>| -> DVector<f64> {
        let beta0 = params[0];
        let beta1 = params[1];
        let log_sigma = params[2];
        let sigma = log_sigma.exp();

        // Gradient of prior terms
        let grad_beta0_prior = -beta0 / (10.0 * 10.0);
        let grad_beta1_prior = -beta1 / (10.0 * 10.0);
        let grad_log_sigma_prior = -log_sigma;

        // Gradient of likelihood terms
        let mut grad_beta0_likelihood = 0.0;
        let mut grad_beta1_likelihood = 0.0;
        let mut grad_log_sigma_likelihood = 0.0;

        for (&x_i, &y_i) in x.iter().zip(y.iter()) {
            let predicted = beta0 + beta1 * x_i;
            let residual = y_i - predicted;

            // Gradient w.r.t. beta0 and beta1
            let common_factor = residual / (sigma * sigma);
            grad_beta0_likelihood += common_factor;
            grad_beta1_likelihood += common_factor * x_i;

            // Gradient w.r.t. log_sigma
            grad_log_sigma_likelihood += -1.0 + residual * residual / (sigma * sigma);
        }

        DVector::from_vec(vec![
            grad_beta0_prior + grad_beta0_likelihood,
            grad_beta1_prior + grad_beta1_likelihood,
            grad_log_sigma_prior + grad_log_sigma_likelihood,
        ])
    };

    let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let step_size = 0.01;
    let n_leapfrog = 50;

    let mut sampler = HamiltonianMonteCarlo::new(
        log_posterior,
        gradient,
        initial_state,
        step_size,
        n_leapfrog,
    )?;

    // Warm-up phase
    let _ = sampler.sample(1000);

    // Main sampling
    let samples = sampler.sample(10000);

    println!(
        "HMC Acceptance Rate: {:.3}",
        sampler.acceptance_rate().unwrap_or(0.0)
    );

    Ok(samples)
}

/// Print diagnostic information
fn print_diagnostics(samples: &[DVector<f64>], method_name: &str) {
    let diagnostics = McmcDiagnostics::from_single_chain(samples).unwrap();

    println!("\n=== {method_name} Results ===");
    println!("Parameter estimates (posterior means):");
    println!(
        "  Intercept (β₀): {:.3} ± {:.3}",
        diagnostics.mean[0], diagnostics.std_dev[0]
    );
    println!(
        "  Slope (β₁):     {:.3} ± {:.3}",
        diagnostics.mean[1], diagnostics.std_dev[1]
    );
    println!(
        "  Noise std (σ):  {:.3} ± {:.3}",
        diagnostics.mean[2].exp(),
        diagnostics.std_dev[2] * diagnostics.mean[2].exp()
    );

    println!("\nDiagnostics:");
    println!(
        "  Effective sample sizes: {:?}",
        diagnostics
            .effective_sample_size
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
    );

    println!("  95% Credible Intervals:");
    for i in 0..3 {
        let param_name = match i {
            0 => "Intercept",
            1 => "Slope",
            2 => "Log(σ)",
            _ => "Unknown",
        };
        println!(
            "    {}: [{:.3}, {:.3}]",
            param_name, diagnostics.quantiles[i][0], diagnostics.quantiles[i][4]
        );
    }
}

/// Save samples to CSV file
fn save_samples_csv(samples: &[DVector<f64>], filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    writeln!(file, "iteration,intercept,slope,log_sigma")?;

    for (i, sample) in samples.iter().enumerate() {
        writeln!(file, "{},{},{},{}", i, sample[0], sample[1], sample[2])?;
    }

    println!("Samples saved to {filename}");
    Ok(())
}

/// Generate predictions with uncertainty
fn generate_predictions(samples: &[DVector<f64>], x_pred: &[f64]) -> Vec<(f64, f64, f64)> {
    let _n_samples = samples.len();
    let n_pred = x_pred.len();

    let mut predictions = vec![Vec::new(); n_pred];

    // Generate predictions for each sample
    for sample in samples {
        let beta0 = sample[0];
        let beta1 = sample[1];

        for (i, &x_val) in x_pred.iter().enumerate() {
            let y_pred = beta0 + beta1 * x_val;
            predictions[i].push(y_pred);
        }
    }

    // Calculate statistics for each prediction point
    predictions
        .into_iter()
        .map(|mut preds| {
            preds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = preds.len();
            let mean = preds.iter().sum::<f64>() / n as f64;
            let lower = preds[n / 40]; // 2.5th percentile
            let upper = preds[n * 39 / 40]; // 97.5th percentile
            (mean, lower, upper)
        })
        .collect()
}

fn main() -> Result<()> {
    println!("Bayesian Linear Regression Example");
    println!("=================================");

    // Generate synthetic data
    let true_intercept = 2.0;
    let true_slope = 1.5;
    let noise_std = 0.5;
    let n_data = 50;

    let (x_data, y_data) = generate_data(n_data, true_slope, true_intercept, noise_std);

    println!("Generated {n_data} data points");
    println!("True parameters: intercept = {true_intercept}, slope = {true_slope}, noise_std = {noise_std}");

    // Perform Bayesian inference with Metropolis-Hastings
    println!("\nRunning Metropolis-Hastings sampler...");
    let mh_samples = bayesian_linear_regression_mh(&x_data, &y_data)?;
    print_diagnostics(&mh_samples, "Metropolis-Hastings");

    // Perform Bayesian inference with HMC
    println!("\nRunning Hamiltonian Monte Carlo sampler...");
    let hmc_samples = bayesian_linear_regression_hmc(&x_data, &y_data)?;
    print_diagnostics(&hmc_samples, "HMC");

    // Save samples to files
    save_samples_csv(&mh_samples, "mh_samples.csv")
        .map_err(|e| BayesError::invalid_configuration(format!("IO error: {e}")))?;
    save_samples_csv(&hmc_samples, "hmc_samples.csv")
        .map_err(|e| BayesError::invalid_configuration(format!("IO error: {e}")))?;

    // Generate predictions
    let x_pred: Vec<f64> = (0..21).map(|i| i as f64 * 0.5).collect();
    let predictions = generate_predictions(&hmc_samples, &x_pred);

    println!("\nPredictions at selected points:");
    for (i, (mean, lower, upper)) in predictions.iter().enumerate() {
        if i % 5 == 0 {
            println!(
                "  x = {:.1}: y = {:.2} [{:.2}, {:.2}]",
                x_pred[i], mean, lower, upper
            );
        }
    }

    println!("\nExample completed successfully!");
    println!("Check the generated CSV files for detailed results.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() {
        let (x, y) = generate_data(10, 1.0, 0.0, 0.1);
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);

        // Check that x values are approximately correct
        assert!((x[0] - 0.0).abs() < 0.1);
        assert!((x[9] - 9.0).abs() < 0.1);
    }

    #[test]
    fn test_bayesian_regression_mh() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect linear relationship

        let samples = bayesian_linear_regression_mh(&x, &y).unwrap();
        assert!(samples.len() > 0);

        let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

        // Should recover approximately intercept ≈ 1, slope ≈ 1
        assert!((diagnostics.mean[0] - 1.0).abs() < 0.5);
        assert!((diagnostics.mean[1] - 1.0).abs() < 0.5);
    }
}
