use bayes_rs::{
    diagnostics::McmcDiagnostics,
    distributions::{Distribution, Normal},
    prelude::*,
    samplers::{MetropolisHastings, Sampler},
};
use nalgebra::DVector;

/// Hierarchical Bayesian Model Example
///
/// This example demonstrates a simple hierarchical model where we have:
/// - Multiple groups of observations
/// - Group-specific means that are drawn from a common population distribution
/// - Population-level hyperparameters to be estimated
///
/// Model structure:
/// - y_ij ~ Normal(theta_j, sigma²)  [observations in group j]
/// - theta_j ~ Normal(mu, tau²)      [group means]
/// - mu ~ Normal(0, 10²)             [population mean]
/// - tau ~ HalfNormal(0, 5²)         [population std dev]
/// - sigma ~ HalfNormal(0, 2²)       [observation noise]
fn main() -> Result<()> {
    println!("Hierarchical Bayesian Model Example");
    println!("===================================");

    // Generate synthetic hierarchical data
    let data = generate_hierarchical_data();
    println!("Generated data for {} groups", data.len());

    for (i, group) in data.iter().enumerate() {
        println!(
            "Group {}: {} observations, mean = {:.2}",
            i,
            group.len(),
            group.iter().sum::<f64>() / group.len() as f64
        );
    }

    // Perform hierarchical Bayesian inference
    println!("\nRunning hierarchical model inference...");
    let samples = hierarchical_inference(&data)?;

    // Analyze results
    analyze_results(&samples, &data);

    println!("\nHierarchical model inference completed successfully!");
    Ok(())
}

/// Generate synthetic hierarchical data
/// 3 groups with different means but shared population structure
fn generate_hierarchical_data() -> Vec<Vec<f64>> {
    use rand::prelude::*;
    use rand_distr::Normal as RandNormal;

    let mut rng = thread_rng();

    // True parameters (unknown to the inference)
    let true_population_mean = 5.0;
    let true_population_std = 2.0;
    let true_observation_std = 1.0;

    // Generate group means from population distribution
    let population_dist = RandNormal::new(true_population_mean, true_population_std).unwrap();
    let group_means: Vec<f64> = (0..3).map(|_| population_dist.sample(&mut rng)).collect();

    println!("True population mean: {true_population_mean:.2}, std: {true_population_std:.2}");
    println!(
        "True group means: {:?}",
        group_means
            .iter()
            .map(|x| format!("{x:.2}"))
            .collect::<Vec<_>>()
    );

    // Generate observations for each group
    let group_sizes = [15, 12, 18]; // Different group sizes
    let mut data = Vec::new();

    for (i, &group_mean) in group_means.iter().enumerate() {
        let obs_dist = RandNormal::new(group_mean, true_observation_std).unwrap();
        let group_data: Vec<f64> = (0..group_sizes[i])
            .map(|_| obs_dist.sample(&mut rng))
            .collect();
        data.push(group_data);
    }

    data
}

/// Perform hierarchical Bayesian inference
fn hierarchical_inference(data: &[Vec<f64>]) -> Result<Vec<DVector<f64>>> {
    let n_groups = data.len();

    // Parameter order: [mu, log_tau, log_sigma, theta_1, theta_2, theta_3]
    let n_params = 3 + n_groups;

    let log_posterior = |params: &DVector<f64>| -> f64 {
        let mu = params[0];
        let log_tau = params[1];
        let log_sigma = params[2];

        let tau = log_tau.exp();
        let sigma = log_sigma.exp();

        if !tau.is_finite() || tau <= 0.0 || !sigma.is_finite() || sigma <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let mut log_prob = 0.0;

        // Prior for mu: N(0, 10²)
        let mu_prior = Normal::new(0.0, 10.0).unwrap();
        log_prob += mu_prior.log_pdf(mu);

        // Prior for tau: HalfNormal(0, 5²) on log scale
        let tau_prior = Normal::new(0.0, 5.0).unwrap();
        log_prob += tau_prior.log_pdf(log_tau) + log_tau; // Jacobian for log transform

        // Prior for sigma: HalfNormal(0, 2²) on log scale
        let sigma_prior = Normal::new(0.0, 2.0).unwrap();
        log_prob += sigma_prior.log_pdf(log_sigma) + log_sigma; // Jacobian

        // Group means and likelihood
        for (j, group_data) in data.iter().enumerate() {
            let theta_j = params[3 + j];

            // Prior for theta_j: N(mu, tau²)
            let theta_prior = Normal::new(mu, tau).unwrap();
            log_prob += theta_prior.log_pdf(theta_j);

            // Likelihood for observations in group j
            let obs_likelihood = Normal::new(theta_j, sigma).unwrap();
            for &y_ij in group_data {
                log_prob += obs_likelihood.log_pdf(y_ij);
            }
        }

        log_prob
    };

    // Initial state: reasonable starting values
    let mut initial_values = vec![0.0, 0.0, 0.0]; // mu, log_tau, log_sigma

    // Initialize group means to empirical means
    for group_data in data {
        let group_mean = group_data.iter().sum::<f64>() / group_data.len() as f64;
        initial_values.push(group_mean);
    }

    let initial_state = DVector::from_vec(initial_values);
    let proposal_std = DVector::from_vec(vec![0.5; n_params]);

    let mut sampler = MetropolisHastings::new(log_posterior, initial_state, proposal_std)?;

    // Warm-up
    let _ = sampler.sample(2000);
    println!(
        "Warm-up completed. Acceptance rate: {:.3}",
        sampler.acceptance_rate().unwrap_or(0.0)
    );

    // Main sampling
    let samples = sampler.sample(8000);
    println!(
        "Sampling completed. Final acceptance rate: {:.3}",
        sampler.acceptance_rate().unwrap_or(0.0)
    );

    Ok(samples)
}

/// Analyze and display results
fn analyze_results(samples: &[DVector<f64>], data: &[Vec<f64>]) {
    let diagnostics = McmcDiagnostics::from_single_chain(samples).unwrap();

    println!("\n=== Hierarchical Model Results ===");

    // Population-level parameters
    println!("\nPopulation-level parameters:");
    println!(
        "  Population mean (μ):     {:.3} ± {:.3}",
        diagnostics.mean[0], diagnostics.std_dev[0]
    );
    println!(
        "  Population std (τ):      {:.3} ± {:.3}",
        diagnostics.mean[1].exp(),
        diagnostics.std_dev[1] * diagnostics.mean[1].exp()
    );
    println!(
        "  Observation noise (σ):   {:.3} ± {:.3}",
        diagnostics.mean[2].exp(),
        diagnostics.std_dev[2] * diagnostics.mean[2].exp()
    );

    // Group-level parameters
    println!("\nGroup-specific means:");
    for (j, group_data) in data.iter().enumerate() {
        let empirical_mean = group_data.iter().sum::<f64>() / group_data.len() as f64;
        println!(
            "  Group {} (θ_{}): {:.3} ± {:.3} [empirical: {:.3}]",
            j,
            j,
            diagnostics.mean[3 + j],
            diagnostics.std_dev[3 + j],
            empirical_mean
        );
    }

    // Shrinkage analysis
    println!("\nShrinkage analysis:");
    let population_mean_est = diagnostics.mean[0];
    for (j, group_data) in data.iter().enumerate() {
        let empirical_mean = group_data.iter().sum::<f64>() / group_data.len() as f64;
        let posterior_mean = diagnostics.mean[3 + j];
        let shrinkage = (empirical_mean - posterior_mean) / (empirical_mean - population_mean_est);
        println!(
            "  Group {}: {:.1}% shrinkage toward population mean",
            j,
            shrinkage * 100.0
        );
    }

    // Diagnostics
    println!("\nDiagnostics:");
    println!("  Effective sample sizes:");
    let param_names = ["μ", "log(τ)", "log(σ)", "θ_0", "θ_1", "θ_2"];
    for (i, name) in param_names.iter().enumerate() {
        println!("    {}: {:.0}", name, diagnostics.effective_sample_size[i]);
    }

    // Credible intervals for key parameters
    println!("\n95% Credible Intervals:");
    for (i, name) in param_names.iter().enumerate() {
        println!(
            "  {}: [{:.3}, {:.3}]",
            name,
            diagnostics.quantiles[i][0], // 2.5%
            diagnostics.quantiles[i][4]
        ); // 97.5%
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_data_generation() {
        let data = generate_hierarchical_data();

        assert_eq!(data.len(), 3); // 3 groups
        assert!(data.iter().all(|group| !group.is_empty()));
        assert!(data
            .iter()
            .all(|group| group.iter().all(|&x| x.is_finite())));
    }

    #[test]
    fn test_hierarchical_inference_runs() {
        // Small test with minimal data
        let test_data = vec![vec![1.0, 1.1, 0.9], vec![2.0, 2.1, 1.9]];

        let result = hierarchical_inference(&test_data);
        assert!(result.is_ok());

        let samples = result.unwrap();
        assert!(!samples.is_empty());
        assert_eq!(samples[0].len(), 5); // 3 population + 2 group params
    }
}
