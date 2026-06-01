//! Conjugate Bayesian model examples.
//!
//! These examples use closed-form posterior updates, then build the matching
//! bayes-rs distributions for posterior summaries and simple plug-in predictive
//! checks. The plug-in checks are intentionally lightweight examples; they are
//! not substitutes for the full posterior-predictive distributions that these
//! conjugate pairs also provide in closed form. They are deterministic and quick
//! enough to use as executable docs.

use bayes_rs::distributions::{
    Beta, Binomial, DiscreteDistribution, Distribution, Gamma, Normal, Poisson,
};

fn beta_binomial_posterior(
    prior_alpha: f64,
    prior_beta: f64,
    successes: u64,
    trials: u64,
) -> bayes_rs::Result<Beta> {
    let failures = trials
        .checked_sub(successes)
        .ok_or_else(|| bayes_rs::BayesError::invalid_parameter("successes cannot exceed trials"))?;

    Beta::new(prior_alpha + successes as f64, prior_beta + failures as f64)
}

fn gamma_poisson_posterior(
    prior_shape: f64,
    prior_rate: f64,
    counts: &[u64],
) -> bayes_rs::Result<Gamma> {
    let observed_events: u64 = counts.iter().sum();
    // bayes-rs parameterizes Gamma as shape/rate, so exposure increments rate.
    Gamma::new(
        prior_shape + observed_events as f64,
        prior_rate + counts.len() as f64,
    )
}

pub(crate) fn normal_normal_posterior(
    prior_mean: f64,
    prior_std_dev: f64,
    observation_std_dev: f64,
    observations: &[f64],
) -> bayes_rs::Result<Normal> {
    if observations.is_empty() {
        return Err(bayes_rs::BayesError::invalid_parameter(
            "observations cannot be empty",
        ));
    }
    if prior_std_dev <= 0.0 || observation_std_dev <= 0.0 {
        return Err(bayes_rs::BayesError::invalid_parameter(
            "standard deviations must be positive",
        ));
    }
    if !prior_mean.is_finite()
        || !prior_std_dev.is_finite()
        || !observation_std_dev.is_finite()
        || observations
            .iter()
            .any(|observation| !observation.is_finite())
    {
        return Err(bayes_rs::BayesError::invalid_parameter(
            "normal-normal inputs must be finite",
        ));
    }

    let prior_precision = 1.0 / prior_std_dev.powi(2);
    let observation_precision = 1.0 / observation_std_dev.powi(2);
    let posterior_precision = prior_precision + observations.len() as f64 * observation_precision;
    let observation_sum: f64 = observations.iter().sum();
    let posterior_mean = (prior_mean * prior_precision + observation_sum * observation_precision)
        / posterior_precision;
    let posterior_std_dev = (1.0 / posterior_precision).sqrt();

    Normal::new(posterior_mean, posterior_std_dev)
}

fn main() -> bayes_rs::Result<()> {
    let conversion_posterior = beta_binomial_posterior(2.0, 2.0, 42, 120)?;
    let posterior_mean = conversion_posterior.mean();
    let predictive_successes = Binomial::new(25, posterior_mean)?;

    println!("Beta-binomial conversion-rate update");
    println!("  Posterior mean success probability: {posterior_mean:.3}");
    println!(
        "  Plug-in predictive probability of at least 8 successes in 25 trials: {:.3}",
        (8..=25)
            .map(|successes| predictive_successes.pmf(successes))
            .sum::<f64>()
    );

    let daily_defects = [3, 4, 2, 5, 4, 1, 3];
    let defect_rate_posterior = gamma_poisson_posterior(1.5, 1.0, &daily_defects)?;
    let posterior_rate = defect_rate_posterior.mean();
    let next_day_defects = Poisson::new(posterior_rate)?;

    println!("Gamma-Poisson count-rate update");
    println!("  Posterior mean event rate: {posterior_rate:.3}");
    println!(
        "  Plug-in predictive probability of at most 2 events tomorrow: {:.3}",
        (0..=2)
            .map(|count| next_day_defects.pmf(count))
            .sum::<f64>()
    );
    println!(
        "  Posterior density at rate 3.0: {:.3}",
        defect_rate_posterior.pdf(3.0)
    );

    let measurements = [9.8, 10.3, 10.1, 9.9, 10.2];
    let mean_posterior = normal_normal_posterior(10.0, 2.0, 0.5, &measurements)?;

    println!("Normal-normal mean update with known observation variance");
    println!("  Posterior mean: {:.3}", mean_posterior.mean());
    println!(
        "  Posterior standard deviation: {:.3}",
        mean_posterior.std_dev()
    );
    println!(
        "  Posterior density at x = 10.0: {:.3}",
        mean_posterior.pdf(10.0)
    );

    Ok(())
}
