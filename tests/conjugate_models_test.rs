use approx::assert_relative_eq;
use bayes_rs::distributions::{Beta, Binomial, DiscreteDistribution, Gamma, Poisson};

#[test]
fn beta_binomial_update_matches_closed_form_posterior() {
    let prior_alpha = 2.0;
    let prior_beta = 2.0;
    let successes = 42;
    let trials = 120;
    let failures = trials - successes;

    let posterior = Beta::new(prior_alpha + successes as f64, prior_beta + failures as f64)
        .expect("posterior parameters should be valid");

    assert_relative_eq!(posterior.alpha(), 44.0);
    assert_relative_eq!(posterior.beta(), 80.0);
    assert_relative_eq!(posterior.mean(), 44.0 / 124.0);

    let plug_in_predictive =
        Binomial::new(25, posterior.mean()).expect("posterior mean is a probability");
    let probability_at_least_eight: f64 = (8..=25).map(|k| plug_in_predictive.pmf(k)).sum();

    assert!(probability_at_least_eight > 0.71);
    assert!(probability_at_least_eight < 0.72);
}

#[test]
fn gamma_poisson_update_matches_closed_form_posterior() {
    let prior_shape = 1.5;
    let prior_rate = 1.0;
    let counts = [3_u64, 4, 2, 5, 4, 1, 3];
    let observed_events: u64 = counts.iter().sum();

    let posterior = Gamma::new(
        prior_shape + observed_events as f64,
        prior_rate + counts.len() as f64,
    )
    .expect("posterior parameters should be valid");

    assert_relative_eq!(posterior.shape(), 23.5);
    assert_relative_eq!(posterior.rate(), 8.0);
    assert_relative_eq!(posterior.mean(), 23.5 / 8.0);

    let plug_in_predictive =
        Poisson::new(posterior.mean()).expect("posterior mean is a valid rate");
    let probability_at_most_two: f64 = (0..=2).map(|k| plug_in_predictive.pmf(k)).sum();

    assert!(probability_at_most_two > 0.42);
    assert!(probability_at_most_two < 0.44);
}
