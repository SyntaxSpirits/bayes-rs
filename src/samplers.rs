//! MCMC samplers for Bayesian inference

use crate::error::{BayesError, Result};
use nalgebra::DVector;
use rand::prelude::*;
use rand_distr::{Distribution as RandDistribution, Normal as RandNormal};

/// Trait for MCMC samplers
pub trait Sampler {
    /// Sample from the posterior distribution
    fn sample(&mut self, n_samples: usize) -> Vec<DVector<f64>>;

    /// Get a single sample
    fn step(&mut self) -> DVector<f64>;

    /// Get the current state
    fn current_state(&self) -> &DVector<f64>;

    /// Get acceptance rate (if applicable)
    fn acceptance_rate(&self) -> Option<f64> {
        None
    }
}

/// Metropolis-Hastings sampler
pub struct MetropolisHastings<F> {
    log_posterior: F,
    current_state: DVector<f64>,
    proposal_std: DVector<f64>,
    current_log_posterior: f64,
    n_accepted: usize,
    n_total: usize,
    rng: ThreadRng,
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
            rng: thread_rng(),
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

impl<F> Sampler for MetropolisHastings<F>
where
    F: Fn(&DVector<f64>) -> f64,
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

    fn acceptance_rate(&self) -> Option<f64> {
        if self.n_total > 0 {
            Some(self.n_accepted as f64 / self.n_total as f64)
        } else {
            None
        }
    }
}

/// Gibbs sampler for conditional distributions
pub struct GibbsSampler<F> {
    conditional_samplers: Vec<F>,
    current_state: DVector<f64>,
    rng: ThreadRng,
}

impl<F> GibbsSampler<F>
where
    F: Fn(&DVector<f64>, usize, &mut ThreadRng) -> f64,
{
    /// Create a new Gibbs sampler
    pub fn new(conditional_samplers: Vec<F>, initial_state: DVector<f64>) -> Result<Self> {
        if conditional_samplers.len() != initial_state.len() {
            return Err(BayesError::dimension_mismatch(
                conditional_samplers.len(),
                initial_state.len(),
            ));
        }

        Ok(Self {
            conditional_samplers,
            current_state: initial_state,
            rng: thread_rng(),
        })
    }
}

impl<F> Sampler for GibbsSampler<F>
where
    F: Fn(&DVector<f64>, usize, &mut ThreadRng) -> f64,
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
pub struct HamiltonianMonteCarlo<F, G> {
    log_posterior: F,
    gradient: G,
    current_state: DVector<f64>,
    step_size: f64,
    n_leapfrog: usize,
    mass_matrix: DVector<f64>,
    current_log_posterior: f64,
    n_accepted: usize,
    n_total: usize,
    rng: ThreadRng,
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
            rng: thread_rng(),
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

impl<F, G> Sampler for HamiltonianMonteCarlo<F, G>
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
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
