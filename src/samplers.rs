//! MCMC samplers for Bayesian inference

use crate::error::{BayesError, Result};
use nalgebra::DVector;
use rand::prelude::*;
use rand::rngs::{StdRng, ThreadRng};
use rand::SeedableRng;
use rand_distr::{Distribution as RandDistribution, Normal as RandNormal};

/// Default finite-difference step size for gradient checks.
///
/// This is close to `f64::EPSILON.cbrt()`, a common central-difference
/// rule-of-thumb that balances truncation and floating-point roundoff error for
/// unit-scaled parameters.
pub const DEFAULT_FINITE_DIFFERENCE_STEP: f64 = 1e-5;

/// Estimate the gradient of a log-density with central finite differences.
///
/// This helper is useful when debugging gradients supplied to
/// [`HamiltonianMonteCarlo`]. It evaluates each coordinate independently using
/// `(f(x + h e_i) - f(x - h e_i)) / (2h)` and returns an error when the step is
/// not positive/finite or when a perturbed log-density is non-finite. Use a
/// step size appropriate to the parameter scale; widely different coordinate
/// scales may need separate checks after rescaling.
pub fn finite_difference_gradient<F>(
    log_density: F,
    point: &DVector<f64>,
    step_size: f64,
) -> Result<DVector<f64>>
where
    F: Fn(&DVector<f64>) -> f64,
{
    if step_size <= 0.0 || !step_size.is_finite() {
        return Err(BayesError::invalid_parameter(
            "Finite difference step size must be positive and finite",
        ));
    }

    let mut gradient = DVector::zeros(point.len());
    for coordinate in 0..point.len() {
        let mut forward = point.clone();
        forward[coordinate] += step_size;

        let mut backward = point.clone();
        backward[coordinate] -= step_size;

        let forward_value = log_density(&forward);
        let backward_value = log_density(&backward);
        if !forward_value.is_finite() || !backward_value.is_finite() {
            return Err(BayesError::numerical_error(format!(
                "Finite difference evaluation produced a non-finite log density at coordinate {coordinate}",
            )));
        }

        gradient[coordinate] = (forward_value - backward_value) / (2.0 * step_size);
    }

    Ok(gradient)
}

/// Compare an analytic HMC gradient with a finite-difference estimate.
///
/// Returns the maximum absolute coordinate-wise difference between the supplied
/// analytic gradient and the finite-difference estimate. A small value gives a
/// quick sanity check that the gradient sign and scale match the log-density,
/// but callers should scale tolerances to the expected gradient magnitude.
pub fn gradient_check<F, G>(
    log_density: F,
    gradient: G,
    point: &DVector<f64>,
    step_size: f64,
) -> Result<f64>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    if step_size <= 0.0 || !step_size.is_finite() {
        return Err(BayesError::invalid_parameter(
            "Finite difference step size must be positive and finite",
        ));
    }

    let analytic = gradient(point);
    if analytic.len() != point.len() {
        return Err(BayesError::dimension_mismatch(point.len(), analytic.len()));
    }
    if analytic.iter().any(|value| !value.is_finite()) {
        return Err(BayesError::numerical_error(
            "Analytic gradient produced a non-finite value",
        ));
    }

    let estimated = finite_difference_gradient(log_density, point, step_size)?;
    let max_error = analytic
        .iter()
        .zip(estimated.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);

    Ok(max_error)
}

/// Trait for MCMC samplers
pub trait Sampler {
    /// Sample from the posterior distribution
    fn sample(&mut self, n_samples: usize) -> Vec<DVector<f64>>;

    /// Run warmup iterations, discard those states, then collect posterior samples.
    ///
    /// Warmup iterations let a Markov chain move away from its initial state before
    /// collecting draws for posterior summaries. This method does not perform
    /// automatic adaptation; callers should tune sampler parameters separately when
    /// their workflow requires it. Statistics such as acceptance rate are reset
    /// after warmup, so they describe only the returned samples. Implementations
    /// that maintain running statistics must override [`Sampler::reset_statistics`]
    /// for this guarantee to hold.
    fn sample_with_warmup(&mut self, n_warmup: usize, n_samples: usize) -> Vec<DVector<f64>> {
        for _ in 0..n_warmup {
            self.step();
        }
        self.reset_statistics();

        self.sample(n_samples)
    }

    /// Get a single sample
    fn step(&mut self) -> DVector<f64>;

    /// Get the current state
    fn current_state(&self) -> &DVector<f64>;

    /// Reset sampler-level running statistics, such as acceptance counters.
    fn reset_statistics(&mut self) {}

    /// Get acceptance rate (if applicable)
    fn acceptance_rate(&self) -> Option<f64> {
        None
    }
}

/// Metropolis-Hastings sampler
pub struct MetropolisHastings<F, R = ThreadRng> {
    log_posterior: F,
    current_state: DVector<f64>,
    proposal_std: DVector<f64>,
    current_log_posterior: f64,
    n_accepted: usize,
    n_total: usize,
    rng: R,
}

impl<F> MetropolisHastings<F>
where
    F: Fn(&DVector<f64>) -> f64,
{
    /// Create a new Metropolis-Hastings sampler
    pub fn new(
        log_posterior: F,
        initial_state: DVector<f64>,
        proposal_std: DVector<f64>,
    ) -> Result<Self> {
        MetropolisHastings::with_rng(log_posterior, initial_state, proposal_std, thread_rng())
    }
}

impl<F> MetropolisHastings<F, StdRng>
where
    F: Fn(&DVector<f64>) -> f64,
{
    /// Create a new Metropolis-Hastings sampler with a reproducible seed.
    pub fn with_seed(
        log_posterior: F,
        initial_state: DVector<f64>,
        proposal_std: DVector<f64>,
        seed: u64,
    ) -> Result<Self> {
        MetropolisHastings::with_rng(
            log_posterior,
            initial_state,
            proposal_std,
            StdRng::seed_from_u64(seed),
        )
    }
}

impl<F, R> MetropolisHastings<F, R>
where
    F: Fn(&DVector<f64>) -> f64,
    R: Rng,
{
    /// Create a new Metropolis-Hastings sampler with a caller-provided RNG.
    pub fn with_rng(
        log_posterior: F,
        initial_state: DVector<f64>,
        proposal_std: DVector<f64>,
        rng: R,
    ) -> Result<Self> {
        if initial_state.len() != proposal_std.len() {
            return Err(BayesError::dimension_mismatch(
                initial_state.len(),
                proposal_std.len(),
            ));
        }

        if proposal_std.iter().any(|&std| std <= 0.0) {
            return Err(BayesError::invalid_parameter(
                "All proposal standard deviations must be positive",
            ));
        }

        let current_log_posterior = log_posterior(&initial_state);
        if !current_log_posterior.is_finite() {
            return Err(BayesError::invalid_parameter(
                "Initial state has non-finite log posterior",
            ));
        }

        Ok(Self {
            log_posterior,
            current_state: initial_state,
            proposal_std,
            current_log_posterior,
            n_accepted: 0,
            n_total: 0,
            rng,
        })
    }

    /// Set the proposal standard deviations
    pub fn set_proposal_std(&mut self, proposal_std: DVector<f64>) -> Result<()> {
        if proposal_std.len() != self.current_state.len() {
            return Err(BayesError::dimension_mismatch(
                self.current_state.len(),
                proposal_std.len(),
            ));
        }

        if proposal_std.iter().any(|&std| std <= 0.0) {
            return Err(BayesError::invalid_parameter(
                "All proposal standard deviations must be positive",
            ));
        }

        self.proposal_std = proposal_std;
        Ok(())
    }

    /// Adapt the proposal standard deviations based on acceptance rate
    pub fn adapt_proposal(&mut self, target_rate: f64) {
        if self.n_total == 0 {
            return;
        }

        let current_rate = self.n_accepted as f64 / self.n_total as f64;
        let factor = if current_rate > target_rate { 1.1 } else { 0.9 };

        self.proposal_std *= factor;
    }
}

impl<F, R> Sampler for MetropolisHastings<F, R>
where
    F: Fn(&DVector<f64>) -> f64,
    R: Rng,
{
    fn sample(&mut self, n_samples: usize) -> Vec<DVector<f64>> {
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            samples.push(self.step());
        }

        samples
    }

    fn step(&mut self) -> DVector<f64> {
        self.n_total += 1;

        // Generate proposal
        let mut proposal = self.current_state.clone();
        for i in 0..proposal.len() {
            let normal =
                RandNormal::new(0.0, self.proposal_std[i]).expect("Valid normal distribution");
            proposal[i] += normal.sample(&mut self.rng);
        }

        // Compute acceptance probability
        let proposal_log_posterior = (self.log_posterior)(&proposal);

        if !proposal_log_posterior.is_finite() {
            return self.current_state.clone();
        }

        let log_alpha = proposal_log_posterior - self.current_log_posterior;
        let alpha = log_alpha.exp().min(1.0);

        // Accept or reject
        if self.rng.gen::<f64>() < alpha {
            self.current_state = proposal;
            self.current_log_posterior = proposal_log_posterior;
            self.n_accepted += 1;
        }

        self.current_state.clone()
    }

    fn current_state(&self) -> &DVector<f64> {
        &self.current_state
    }

    fn reset_statistics(&mut self) {
        self.n_accepted = 0;
        self.n_total = 0;
    }

    fn acceptance_rate(&self) -> Option<f64> {
        if self.n_total > 0 {
            Some(self.n_accepted as f64 / self.n_total as f64)
        } else {
            None
        }
    }
}

/// Gibbs sampler for conditional distributions
pub struct GibbsSampler<F, R = ThreadRng> {
    conditional_samplers: Vec<F>,
    current_state: DVector<f64>,
    rng: R,
}

impl<F> GibbsSampler<F>
where
    F: Fn(&DVector<f64>, usize, &mut ThreadRng) -> f64,
{
    /// Create a new Gibbs sampler
    pub fn new(conditional_samplers: Vec<F>, initial_state: DVector<f64>) -> Result<Self> {
        GibbsSampler::with_rng(conditional_samplers, initial_state, thread_rng())
    }
}

impl<F> GibbsSampler<F, StdRng>
where
    F: Fn(&DVector<f64>, usize, &mut StdRng) -> f64,
{
    /// Create a new Gibbs sampler with a reproducible seed.
    pub fn with_seed(
        conditional_samplers: Vec<F>,
        initial_state: DVector<f64>,
        seed: u64,
    ) -> Result<Self> {
        GibbsSampler::with_rng(
            conditional_samplers,
            initial_state,
            StdRng::seed_from_u64(seed),
        )
    }
}

impl<F, R> GibbsSampler<F, R>
where
    F: Fn(&DVector<f64>, usize, &mut R) -> f64,
{
    /// Create a new Gibbs sampler with a caller-provided RNG.
    pub fn with_rng(
        conditional_samplers: Vec<F>,
        initial_state: DVector<f64>,
        rng: R,
    ) -> Result<Self> {
        if conditional_samplers.len() != initial_state.len() {
            return Err(BayesError::dimension_mismatch(
                conditional_samplers.len(),
                initial_state.len(),
            ));
        }

        Ok(Self {
            conditional_samplers,
            current_state: initial_state,
            rng,
        })
    }
}

impl<F, R> Sampler for GibbsSampler<F, R>
where
    F: Fn(&DVector<f64>, usize, &mut R) -> f64,
{
    fn sample(&mut self, n_samples: usize) -> Vec<DVector<f64>> {
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            samples.push(self.step());
        }

        samples
    }

    fn step(&mut self) -> DVector<f64> {
        // Sample each dimension conditionally
        for i in 0..self.current_state.len() {
            let new_value = (self.conditional_samplers[i])(&self.current_state, i, &mut self.rng);
            self.current_state[i] = new_value;
        }

        self.current_state.clone()
    }

    fn current_state(&self) -> &DVector<f64> {
        &self.current_state
    }
}

/// Simple Hamiltonian Monte Carlo sampler
pub struct HamiltonianMonteCarlo<F, G, R = ThreadRng> {
    log_posterior: F,
    gradient: G,
    current_state: DVector<f64>,
    step_size: f64,
    n_leapfrog: usize,
    mass_matrix: DVector<f64>,
    current_log_posterior: f64,
    n_accepted: usize,
    n_total: usize,
    rng: R,
}

impl<F, G> HamiltonianMonteCarlo<F, G>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    /// Create a new HMC sampler
    pub fn new(
        log_posterior: F,
        gradient: G,
        initial_state: DVector<f64>,
        step_size: f64,
        n_leapfrog: usize,
    ) -> Result<Self> {
        HamiltonianMonteCarlo::with_rng(
            log_posterior,
            gradient,
            initial_state,
            step_size,
            n_leapfrog,
            thread_rng(),
        )
    }
}

impl<F, G> HamiltonianMonteCarlo<F, G, StdRng>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    /// Create a new HMC sampler with a reproducible seed.
    pub fn with_seed(
        log_posterior: F,
        gradient: G,
        initial_state: DVector<f64>,
        step_size: f64,
        n_leapfrog: usize,
        seed: u64,
    ) -> Result<Self> {
        HamiltonianMonteCarlo::with_rng(
            log_posterior,
            gradient,
            initial_state,
            step_size,
            n_leapfrog,
            StdRng::seed_from_u64(seed),
        )
    }
}

impl<F, G, R> HamiltonianMonteCarlo<F, G, R>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
    R: Rng,
{
    /// Create a new HMC sampler with a caller-provided RNG.
    pub fn with_rng(
        log_posterior: F,
        gradient: G,
        initial_state: DVector<f64>,
        step_size: f64,
        n_leapfrog: usize,
        rng: R,
    ) -> Result<Self> {
        if step_size <= 0.0 {
            return Err(BayesError::invalid_parameter("Step size must be positive"));
        }

        if n_leapfrog == 0 {
            return Err(BayesError::invalid_parameter(
                "Number of leapfrog steps must be positive",
            ));
        }

        let current_log_posterior = log_posterior(&initial_state);
        if !current_log_posterior.is_finite() {
            return Err(BayesError::invalid_parameter(
                "Initial state has non-finite log posterior",
            ));
        }

        let dim = initial_state.len();
        let mass_matrix = DVector::from_element(dim, 1.0);

        Ok(Self {
            log_posterior,
            gradient,
            current_state: initial_state,
            step_size,
            n_leapfrog,
            mass_matrix,
            current_log_posterior,
            n_accepted: 0,
            n_total: 0,
            rng,
        })
    }

    /// Set the mass matrix (diagonal)
    pub fn set_mass_matrix(&mut self, mass_matrix: DVector<f64>) -> Result<()> {
        if mass_matrix.len() != self.current_state.len() {
            return Err(BayesError::dimension_mismatch(
                self.current_state.len(),
                mass_matrix.len(),
            ));
        }

        if mass_matrix.iter().any(|&m| m <= 0.0) {
            return Err(BayesError::invalid_parameter(
                "All mass matrix elements must be positive",
            ));
        }

        self.mass_matrix = mass_matrix;
        Ok(())
    }

    /// Leapfrog integrator step
    fn leapfrog(&self, mut q: DVector<f64>, mut p: DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        // Half step for momentum
        let grad = (self.gradient)(&q);
        p += &grad * (self.step_size / 2.0);

        // Full steps
        for _ in 0..self.n_leapfrog {
            // Full step for position
            for i in 0..q.len() {
                q[i] += self.step_size * p[i] / self.mass_matrix[i];
            }

            // Full step for momentum (except last step)
            let grad = (self.gradient)(&q);
            p += &grad * self.step_size;
        }

        // Final half step for momentum
        let grad = (self.gradient)(&q);
        p += &grad * (self.step_size / 2.0);

        (q, p)
    }

    /// Calculate kinetic energy
    fn kinetic_energy(&self, p: &DVector<f64>) -> f64 {
        let mut energy = 0.0;
        for i in 0..p.len() {
            energy += p[i] * p[i] / self.mass_matrix[i];
        }
        0.5 * energy
    }
}

impl<F, G, R> Sampler for HamiltonianMonteCarlo<F, G, R>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
    R: Rng,
{
    fn sample(&mut self, n_samples: usize) -> Vec<DVector<f64>> {
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            samples.push(self.step());
        }

        samples
    }

    fn step(&mut self) -> DVector<f64> {
        self.n_total += 1;

        // Sample momentum
        let mut p = DVector::zeros(self.current_state.len());
        for i in 0..p.len() {
            let normal = RandNormal::new(0.0, self.mass_matrix[i].sqrt())
                .expect("Valid normal distribution");
            p[i] = normal.sample(&mut self.rng);
        }

        // Current energy
        let current_kinetic = self.kinetic_energy(&p);
        let current_potential = -self.current_log_posterior;
        let current_energy = current_kinetic + current_potential;

        // Leapfrog integration
        let (proposal_q, proposal_p) = self.leapfrog(self.current_state.clone(), p);

        // Proposal energy
        let proposal_log_posterior = (self.log_posterior)(&proposal_q);

        if !proposal_log_posterior.is_finite() {
            return self.current_state.clone();
        }

        let proposal_kinetic = self.kinetic_energy(&proposal_p);
        let proposal_potential = -proposal_log_posterior;
        let proposal_energy = proposal_kinetic + proposal_potential;

        // Accept or reject
        let log_alpha = current_energy - proposal_energy;
        let alpha = log_alpha.exp().min(1.0);

        if self.rng.gen::<f64>() < alpha {
            self.current_state = proposal_q;
            self.current_log_posterior = proposal_log_posterior;
            self.n_accepted += 1;
        }

        self.current_state.clone()
    }

    fn current_state(&self) -> &DVector<f64> {
        &self.current_state
    }

    fn reset_statistics(&mut self) {
        self.n_accepted = 0;
        self.n_total = 0;
    }

    fn acceptance_rate(&self) -> Option<f64> {
        if self.n_total > 0 {
            Some(self.n_accepted as f64 / self.n_total as f64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Distribution, Normal};

    #[test]
    fn test_metropolis_hastings_creation() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        let sampler = MetropolisHastings::new(log_posterior, initial_state, proposal_std);
        assert!(sampler.is_ok());
    }

    #[test]
    fn test_metropolis_hastings_sampling() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        let mut sampler =
            MetropolisHastings::new(log_posterior, initial_state, proposal_std).unwrap();
        let samples = sampler.sample(100);

        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|s| s.len() == 1));
    }

    #[test]
    fn test_metropolis_hastings_seed_reproducibility() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        let mut first = MetropolisHastings::with_seed(
            log_posterior,
            initial_state.clone(),
            proposal_std.clone(),
            42,
        )
        .unwrap();
        let mut second =
            MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 42).unwrap();

        assert_eq!(first.sample(100), second.sample(100));
        assert_eq!(first.acceptance_rate(), second.acceptance_rate());
    }

    #[test]
    fn test_metropolis_hastings_acceptance_rate() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        let mut sampler =
            MetropolisHastings::new(log_posterior, initial_state, proposal_std).unwrap();

        // Initially no samples
        assert!(sampler.acceptance_rate().is_none());

        // After sampling
        let _ = sampler.sample(10);
        let acceptance_rate = sampler.acceptance_rate().unwrap();
        assert!((0.0..=1.0).contains(&acceptance_rate));
    }

    #[test]
    fn test_sample_with_warmup_discards_warmup_states() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        let mut warmup_sampler = MetropolisHastings::with_seed(
            log_posterior,
            initial_state.clone(),
            proposal_std.clone(),
            123,
        )
        .unwrap();
        let warmup_samples = warmup_sampler.sample_with_warmup(25, 50);

        let mut manual_sampler =
            MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 123).unwrap();
        manual_sampler.sample(25);
        manual_sampler.reset_statistics();
        let retained_samples = manual_sampler.sample(50);

        assert_eq!(warmup_samples.len(), 50);
        assert_eq!(warmup_samples, retained_samples);
        assert_eq!(
            warmup_sampler.current_state(),
            manual_sampler.current_state()
        );
        assert_eq!(
            warmup_sampler.acceptance_rate(),
            manual_sampler.acceptance_rate()
        );
    }

    #[test]
    fn test_sample_with_zero_warmup_matches_regular_sampling() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        let mut warmup_sampler = MetropolisHastings::with_seed(
            log_posterior,
            initial_state.clone(),
            proposal_std.clone(),
            456,
        )
        .unwrap();
        let mut regular_sampler =
            MetropolisHastings::with_seed(log_posterior, initial_state, proposal_std, 456).unwrap();

        assert_eq!(
            warmup_sampler.sample_with_warmup(0, 50),
            regular_sampler.sample(50)
        );
    }

    #[test]
    fn test_gibbs_sampler_creation() {
        let conditional_sampler = |_params: &DVector<f64>,
                                   _idx: usize,
                                   rng: &mut ThreadRng|
         -> f64 { rng.gen_range(-1.0..1.0) };

        let initial_state = DVector::from_vec(vec![0.0, 0.0]);
        let samplers = vec![conditional_sampler, conditional_sampler];

        let sampler = GibbsSampler::new(samplers, initial_state);
        assert!(sampler.is_ok());
    }

    #[test]
    fn test_finite_difference_gradient_matches_quadratic_gradient() {
        let log_density = |params: &DVector<f64>| -> f64 {
            -0.5 * (params[0] * params[0] + 4.0 * params[1] * params[1])
        };
        let point = DVector::from_vec(vec![1.5, -0.75]);

        let estimated = finite_difference_gradient(log_density, &point, 1e-6).unwrap();

        assert!((estimated[0] + 1.5).abs() < 1e-8);
        assert!((estimated[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_gradient_check_reports_small_error_for_matching_gradient() {
        let log_density = |params: &DVector<f64>| -> f64 {
            -0.5 * (params[0] * params[0] + 4.0 * params[1] * params[1])
        };
        let gradient = |params: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![-params[0], -4.0 * params[1]])
        };
        let point = DVector::from_vec(vec![1.5, -0.75]);

        let max_error = gradient_check(log_density, gradient, &point, 1e-6).unwrap();

        assert!(max_error < 1e-8);
    }

    #[test]
    fn test_gradient_check_reports_large_error_for_wrong_sign() {
        let log_density = |params: &DVector<f64>| -> f64 { -0.5 * params[0] * params[0] };
        let wrong_gradient =
            |params: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![params[0]]) };
        let point = DVector::from_vec(vec![2.0]);

        let max_error = gradient_check(log_density, wrong_gradient, &point, 1e-6).unwrap();

        assert!(max_error > 3.9);
    }

    #[test]
    fn test_gradient_check_rejects_invalid_inputs() {
        let log_density = |params: &DVector<f64>| -> f64 { -0.5 * params[0] * params[0] };
        let point = DVector::from_vec(vec![1.0]);

        assert!(finite_difference_gradient(log_density, &point, 0.0).is_err());

        let wrong_dimension_gradient =
            |_params: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![1.0, 2.0]) };
        assert!(gradient_check(log_density, wrong_dimension_gradient, &point, 1e-6).is_err());
    }

    #[test]
    fn test_gradient_check_rejects_non_finite_evaluations() {
        let non_finite_density = |params: &DVector<f64>| -> f64 {
            if params[0] > 1.0 {
                f64::NAN
            } else {
                -0.5 * params[0] * params[0]
            }
        };
        let point = DVector::from_vec(vec![1.0]);

        let finite_difference_error =
            finite_difference_gradient(non_finite_density, &point, 1e-6).unwrap_err();
        assert!(matches!(
            finite_difference_error,
            BayesError::NumericalError { .. }
        ));

        let non_finite_gradient =
            |_params: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![f64::INFINITY]) };
        let gradient_error =
            gradient_check(non_finite_density, non_finite_gradient, &point, 1e-6).unwrap_err();
        assert!(matches!(gradient_error, BayesError::NumericalError { .. }));
    }

    #[test]
    fn test_hmc_creation() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let gradient = |params: &DVector<f64>| -> DVector<f64> {
            // Gradient of log N(0,1) = -x
            DVector::from_vec(vec![-params[0]])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let step_size = 0.1;
        let n_leapfrog = 10;

        let sampler = HamiltonianMonteCarlo::new(
            log_posterior,
            gradient,
            initial_state,
            step_size,
            n_leapfrog,
        );
        assert!(sampler.is_ok());
    }

    #[test]
    fn test_hmc_sampling() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let gradient = |params: &DVector<f64>| -> DVector<f64> {
            // Gradient of log N(0,1) = -x
            DVector::from_vec(vec![-params[0]])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let step_size = 0.1;
        let n_leapfrog = 10;

        let mut sampler = HamiltonianMonteCarlo::new(
            log_posterior,
            gradient,
            initial_state,
            step_size,
            n_leapfrog,
        )
        .unwrap();

        let samples = sampler.sample(50);
        assert_eq!(samples.len(), 50);
        assert!(samples.iter().all(|s| s.len() == 1));
    }

    #[test]
    fn test_hmc_seed_reproducibility() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let gradient =
            |params: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![-params[0]]) };

        let initial_state = DVector::from_vec(vec![0.0]);

        let mut first = HamiltonianMonteCarlo::with_seed(
            log_posterior,
            gradient,
            initial_state.clone(),
            0.1,
            10,
            7,
        )
        .unwrap();
        let mut second =
            HamiltonianMonteCarlo::with_seed(log_posterior, gradient, initial_state, 0.1, 10, 7)
                .unwrap();

        assert_eq!(first.sample(50), second.sample(50));
        assert_eq!(first.acceptance_rate(), second.acceptance_rate());
    }

    #[test]
    fn test_hmc_sample_with_warmup_resets_acceptance_statistics() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let gradient =
            |params: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![-params[0]]) };

        let initial_state = DVector::from_vec(vec![0.0]);

        let mut warmup_sampler = HamiltonianMonteCarlo::with_seed(
            log_posterior,
            gradient,
            initial_state.clone(),
            0.1,
            10,
            99,
        )
        .unwrap();
        let warmup_samples = warmup_sampler.sample_with_warmup(10, 20);

        let mut manual_sampler =
            HamiltonianMonteCarlo::with_seed(log_posterior, gradient, initial_state, 0.1, 10, 99)
                .unwrap();
        manual_sampler.sample(10);
        manual_sampler.reset_statistics();
        let retained_samples = manual_sampler.sample(20);

        assert_eq!(warmup_samples, retained_samples);
        assert_eq!(
            warmup_sampler.acceptance_rate(),
            manual_sampler.acceptance_rate()
        );
    }

    #[test]
    fn test_invalid_parameters() {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let bad_proposal_std = DVector::from_vec(vec![0.0]); // Invalid: zero std

        let sampler = MetropolisHastings::new(log_posterior, initial_state, bad_proposal_std);
        assert!(sampler.is_err());
    }
}
