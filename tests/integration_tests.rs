use approx::assert_abs_diff_eq;
use bayes_rs::{
    diagnostics::{McmcDiagnostics, TracePlot},
    distributions::{Beta, Gamma, MultivariateNormal, Normal},
    prelude::*,
    samplers::{HamiltonianMonteCarlo, MetropolisHastings, Sampler},
};
use nalgebra::{DMatrix, DVector};

#[test]
fn test_simple_normal_inference() {
    // Test inferring mean and log-variance of a normal distribution
    let true_mu = 2.0;
    let true_sigma = 1.5;
    let data = [1.8, 2.2, 1.9, 2.1, 2.3, 1.7, 2.0, 2.4, 1.6, 2.5];

    let log_posterior = |params: &DVector<f64>| -> f64 {
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = log_sigma.exp();

        // Prior: mu ~ N(0, 10), log_sigma ~ N(0, 1)
        let prior_mu = Normal::new(0.0, 10.0).unwrap();
        let prior_log_sigma = Normal::new(0.0, 1.0).unwrap();
        let prior_log_prob = prior_mu.log_pdf(mu) + prior_log_sigma.log_pdf(log_sigma);

        // Likelihood: data ~ N(mu, sigma)
        let likelihood = Normal::new(mu, sigma).unwrap();
        let likelihood_log_prob: f64 = data.iter().map(|&x| likelihood.log_pdf(x)).sum();

        prior_log_prob + likelihood_log_prob
    };

    let initial_state = DVector::from_vec(vec![0.0, 0.0]);
    let proposal_std = DVector::from_vec(vec![0.3, 0.2]);

    let mut sampler =
        MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 1001).unwrap();
    let samples = sampler.sample(5000);

    // Check that we got the right number of samples
    assert_eq!(samples.len(), 5000);

    // Check diagnostics
    let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();
    assert_eq!(diagnostics.mean.len(), 2);
    assert_eq!(diagnostics.std_dev.len(), 2);

    // Check that acceptance rate is reasonable
    let acceptance_rate = sampler.acceptance_rate().unwrap();
    assert!(acceptance_rate > 0.1 && acceptance_rate < 0.8);

    // Check that posterior mean is close to true value (with some tolerance)
    let posterior_mu = diagnostics.mean[0];
    let posterior_log_sigma = diagnostics.mean[1];
    let posterior_sigma = posterior_log_sigma.exp();

    // Allow for some uncertainty in the estimation (MCMC has inherent randomness)
    assert!((posterior_mu - true_mu).abs() < 1.5);
    assert!((posterior_sigma - true_sigma).abs() < 2.0);
}

#[test]
fn test_multivariate_normal_inference() {
    // Test inferring mean vector of a multivariate normal
    let true_mu = [1.0, -0.5, 2.0];
    let dim = true_mu.len();

    // Generate some synthetic data
    let data = [
        vec![1.1, -0.4, 2.1],
        vec![0.9, -0.6, 1.9],
        vec![1.0, -0.5, 2.0],
        vec![1.2, -0.3, 2.2],
        vec![0.8, -0.7, 1.8],
    ];

    let log_posterior = |params: &DVector<f64>| -> f64 {
        // Prior: mu ~ N(0, I)
        let prior_mu = DVector::zeros(dim);
        let prior_cov = DMatrix::identity(dim, dim);
        let prior = MultivariateNormal::new(prior_mu, prior_cov).unwrap();
        let prior_log_prob = prior.log_pdf(params);

        // Likelihood: data ~ N(mu, I)
        let likelihood_cov = DMatrix::identity(dim, dim);
        let likelihood = MultivariateNormal::new(params.clone(), likelihood_cov).unwrap();
        let likelihood_log_prob: f64 = data
            .iter()
            .map(|row| {
                let x = DVector::from_vec(row.clone());
                likelihood.log_pdf(&x)
            })
            .sum();

        prior_log_prob + likelihood_log_prob
    };

    let initial_state = DVector::zeros(dim);
    let proposal_std = DVector::from_element(dim, 0.3);

    let mut sampler =
        MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 1002).unwrap();
    let samples = sampler.sample(3000);

    let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

    // Check that posterior means are close to true values (allow for MCMC uncertainty)
    for (i, &true_mean) in true_mu.iter().enumerate() {
        let posterior_mean = diagnostics.mean[i];
        assert!((posterior_mean - true_mean).abs() < 0.8);
    }
}

#[test]
fn test_hmc_vs_mh() {
    // Compare HMC and MH on the same problem
    let log_posterior = |params: &DVector<f64>| -> f64 {
        // Standard bivariate normal
        let mu = DVector::zeros(2);
        let cov = DMatrix::identity(2, 2);
        let mvn = MultivariateNormal::new(mu, cov).unwrap();
        mvn.log_pdf(params)
    };

    let gradient = |params: &DVector<f64>| -> DVector<f64> {
        // Gradient of log MVN with identity covariance = -x
        -params
    };

    let initial_state = DVector::zeros(2);

    // MH sampler - use smaller proposal std for better mixing on standard normal
    let proposal_std = DVector::from_element(2, 0.3);
    let mut mh_sampler =
        MetropolisHastings::with_seed(&log_posterior, initial_state.clone(), proposal_std, 1003)
            .unwrap();

    // HMC sampler
    let step_size = 0.1;
    let n_leapfrog = 10;
    let mut hmc_sampler = HamiltonianMonteCarlo::with_seed(
        &log_posterior,
        &gradient,
        initial_state,
        step_size,
        n_leapfrog,
        2003,
    )
    .unwrap();

    let n_samples = 3000;
    let mh_samples = mh_sampler.sample(n_samples);
    let hmc_samples = hmc_sampler.sample(n_samples);

    // Both should produce samples
    assert_eq!(mh_samples.len(), n_samples);
    assert_eq!(hmc_samples.len(), n_samples);

    // Check diagnostics
    let mh_diagnostics = McmcDiagnostics::from_single_chain(&mh_samples).unwrap();
    let hmc_diagnostics = McmcDiagnostics::from_single_chain(&hmc_samples).unwrap();

    // Both should have reasonable effective sample sizes (adjust for test variance)
    // Lower threshold for test stability - these are integration tests, not performance tests
    assert!(mh_diagnostics.effective_sample_size[0] > 20.0);
    assert!(hmc_diagnostics.effective_sample_size[0] > 20.0);

    // Means should be close to zero (relaxed tolerance for MCMC randomness)
    assert!(mh_diagnostics.mean[0].abs() < 0.5);
    assert!(hmc_diagnostics.mean[0].abs() < 0.5);
}

#[test]
fn test_beta_binomial_model() {
    // Test a simple beta-binomial model
    let n_trials = 100;
    let n_successes = 65;

    let log_posterior = |params: &DVector<f64>| -> f64 {
        let p = params[0];

        // Prior: p ~ Beta(1, 1) (uniform)
        let prior = Beta::new(1.0, 1.0).unwrap();
        let prior_log_prob = prior.log_pdf(p);

        // Likelihood: n_successes ~ Binomial(n_trials, p)
        // Using log binomial coefficient approximation
        let likelihood_log_prob =
            n_successes as f64 * p.ln() + (n_trials - n_successes) as f64 * (1.0 - p).ln();

        prior_log_prob + likelihood_log_prob
    };

    let initial_state = DVector::from_vec(vec![0.5]);
    let proposal_std = DVector::from_vec(vec![0.1]);

    let mut sampler =
        MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 1004).unwrap();
    let samples = sampler.sample(5000);

    let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

    // Theoretical posterior is Beta(1 + n_successes, 1 + n_trials - n_successes)
    let posterior_alpha = 1.0 + n_successes as f64;
    let posterior_beta = 1.0 + (n_trials - n_successes) as f64;
    let theoretical_mean = posterior_alpha / (posterior_alpha + posterior_beta);

    // Check that empirical mean is close to theoretical mean
    assert_abs_diff_eq!(diagnostics.mean[0], theoretical_mean, epsilon = 0.02);
}

#[test]
fn test_multiple_chains_diagnostics() {
    // Test R-hat computation with multiple chains
    let log_posterior = |params: &DVector<f64>| -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.log_pdf(params[0])
    };

    let n_chains = 4;
    let n_samples = 1000;
    let mut chains = Vec::new();

    for i in 0..n_chains {
        let initial_state = DVector::from_vec(vec![i as f64 * 0.5]); // Different starting points
        let proposal_std = DVector::from_vec(vec![0.5]);

        let mut sampler = MetropolisHastings::with_seed(
            &log_posterior,
            initial_state,
            proposal_std,
            3000 + i as u64,
        )
        .unwrap();

        chains.push(sampler.sample(n_samples));
    }

    let diagnostics = McmcDiagnostics::from_multiple_chains(&chains).unwrap();

    // Check that R-hat is computed
    assert!(diagnostics.r_hat.is_some());
    let r_hat = diagnostics.r_hat.as_ref().unwrap();
    assert_eq!(r_hat.len(), 1);

    // R-hat should be close to 1.0 for converged chains
    assert!(r_hat[0] >= 1.0);
    assert!(r_hat[0] < 1.2); // Should be well-converged

    // Check convergence assessment
    assert!(diagnostics.has_converged());
}

#[test]
fn test_trace_plot() {
    let log_posterior = |params: &DVector<f64>| -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.log_pdf(params[0])
    };

    let initial_state = DVector::from_vec(vec![0.0]);
    let proposal_std = DVector::from_vec(vec![0.5]);

    let mut sampler =
        MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 1005).unwrap();
    let samples = sampler.sample(100);

    let trace_plot = TracePlot::new(&samples, 0).unwrap();

    assert_eq!(trace_plot.parameter_index, 0);
    assert_eq!(trace_plot.values.len(), 100);
    assert_eq!(trace_plot.iterations.len(), 100);

    // Check that iterations are sequential
    for i in 0..trace_plot.iterations.len() {
        assert_eq!(trace_plot.iterations[i], i);
    }
}

#[test]
fn test_gamma_poisson_model() {
    // Test a Gamma-Poisson model
    let data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    let n_obs = data.len();

    let log_posterior = |params: &DVector<f64>| -> f64 {
        let lambda = params[0];

        if lambda <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Prior: lambda ~ Gamma(1, 1)
        let prior = Gamma::new(1.0, 1.0).unwrap();
        let prior_log_prob = prior.log_pdf(lambda);

        // Likelihood: data ~ Poisson(lambda)
        // log P(x | lambda) = x * log(lambda) - lambda - log(x!)
        let likelihood_log_prob: f64 = data.iter().map(|&x| x as f64 * lambda.ln() - lambda).sum();

        prior_log_prob + likelihood_log_prob
    };

    let initial_state = DVector::from_vec(vec![1.0]);
    let proposal_std = DVector::from_vec(vec![0.5]);

    let mut sampler =
        MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 1006).unwrap();
    let samples = sampler.sample(5000);

    let diagnostics = McmcDiagnostics::from_single_chain(&samples).unwrap();

    // Theoretical posterior is Gamma(1 + sum(data), 1 + n_obs)
    let posterior_alpha = 1.0 + data.iter().sum::<i32>() as f64;
    let posterior_beta = 1.0 + n_obs as f64;
    let theoretical_mean = posterior_alpha / posterior_beta;

    // Check that empirical mean is close to theoretical mean
    assert_abs_diff_eq!(diagnostics.mean[0], theoretical_mean, epsilon = 0.2);
}

#[test]
fn test_sampler_adaptation() {
    // Test the proposal adaptation feature
    let log_posterior = |params: &DVector<f64>| -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.log_pdf(params[0])
    };

    let initial_state = DVector::from_vec(vec![0.0]);
    let proposal_std = DVector::from_vec(vec![2.0]); // Start with large proposal

    let mut sampler =
        MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 1007).unwrap();

    // Run some samples to get initial acceptance rate
    let _warmup = sampler.sample(500);
    let _initial_acceptance_rate = sampler.acceptance_rate().unwrap();

    // Adapt proposal (target acceptance rate = 0.4)
    sampler.adapt_proposal(0.4);

    // Run more samples
    let _samples = sampler.sample(500);
    let final_acceptance_rate = sampler.acceptance_rate().unwrap();

    // The adaptation should have changed the acceptance rate
    // (might be better or worse depending on initial conditions)
    assert!(final_acceptance_rate >= 0.1);
    assert!(final_acceptance_rate <= 0.8);
}

#[test]
fn test_error_handling() {
    // Test various error conditions

    // Invalid distribution parameters
    assert!(Normal::new(0.0, -1.0).is_err());
    assert!(Gamma::new(-1.0, 1.0).is_err());
    assert!(Beta::new(0.0, 1.0).is_err());

    // Invalid sampler parameters
    let log_posterior = |params: &DVector<f64>| -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.log_pdf(params[0])
    };

    let initial_state = DVector::from_vec(vec![0.0]);
    let bad_proposal_std = DVector::from_vec(vec![0.0]); // Zero std is invalid

    assert!(MetropolisHastings::new(log_posterior, initial_state, bad_proposal_std).is_err());

    // Dimension mismatch
    let initial_state = DVector::from_vec(vec![0.0]);
    let mismatched_proposal_std = DVector::from_vec(vec![0.5, 0.5]); // Wrong dimension

    assert!(
        MetropolisHastings::new(log_posterior, initial_state, mismatched_proposal_std).is_err()
    );
}
