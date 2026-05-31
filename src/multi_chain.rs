//! Helpers for running multiple MCMC chains and summarizing diagnostics.
//!
//! The primary entry point is [`run_multiple_chains`], which consumes a mutable
//! slice of already-configured samplers, runs the same warmup/sample schedule for
//! each chain, and returns both raw draws and [`McmcDiagnostics`]. Construct each
//! sampler with its own seed (for samplers that support seeded constructors) to
//! make multi-chain runs reproducible.
//! At least two samplers and at least two retained samples per sampler are
//! required because this helper always computes multi-chain diagnostics,
//! including R-hat; these preconditions are checked before any sampler is run.
//!
//! ```rust
//! use bayes_rs::{multi_chain::run_multiple_chains, samplers::Sampler};
//! use nalgebra::DVector;
//!
//! struct CounterSampler {
//!     state: DVector<f64>,
//! }
//!
//! impl CounterSampler {
//!     fn new(start: f64) -> Self {
//!         Self { state: DVector::from_vec(vec![start]) }
//!     }
//! }
//!
//! impl Sampler for CounterSampler {
//!     fn sample(&mut self, n_samples: usize) -> Vec<DVector<f64>> {
//!         (0..n_samples).map(|_| self.step()).collect()
//!     }
//!
//!     fn step(&mut self) -> DVector<f64> {
//!         self.state[0] += 1.0;
//!         self.state.clone()
//!     }
//!
//!     fn current_state(&self) -> &DVector<f64> {
//!         &self.state
//!     }
//! }
//!
//! let mut samplers = [CounterSampler::new(0.0), CounterSampler::new(1.0)];
//! let output = run_multiple_chains(&mut samplers, 5, 10).unwrap();
//!
//! assert_eq!(output.chains.len(), 2);
//! assert_eq!(output.chains[0].len(), 10);
//! assert!(output.diagnostics.r_hat.is_some());
//! assert_eq!(output.summary.parameters.len(), 1);
//! ```

use crate::diagnostics::{McmcDiagnosticSummary, McmcDiagnostics};
use crate::error::{BayesError, Result};
use crate::samplers::Sampler;
use nalgebra::DVector;

/// Raw multi-chain samples plus their diagnostics summary.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct MultiChainOutput {
    /// Samples for each chain, preserving sampler order.
    pub chains: Vec<Vec<DVector<f64>>>,
    /// Full diagnostics computed from [`Self::chains`].
    pub diagnostics: McmcDiagnostics,
    /// Compact per-parameter summary of R-hat, ESS, and MCSE.
    pub summary: McmcDiagnosticSummary,
}

/// Run multiple already-constructed samplers with a shared warmup/sample schedule.
///
/// Each sampler is run with [`Sampler::sample_with_warmup`], so sampler-level
/// statistics are reset after warmup and describe only the returned samples. The returned chains
/// are then passed to [`McmcDiagnostics::from_multiple_chains`], so the same
/// validation rules apply: at least two samplers and at least two retained
/// samples per chain are required for R-hat, and the generated chains must be
/// non-empty, equal length, finite, and have consistent dimensions. The sampler
/// count and retained sample count are checked before execution, so invalid
/// inputs do not advance or otherwise mutate the samplers.
///
/// For reproducible stochastic runs, construct each sampler with an explicit and
/// distinct seed before calling this helper.
///
/// # Errors
///
/// Returns [`BayesError::InvalidParameter`] if fewer than two samplers are
/// provided or if fewer than two posterior samples are requested. Other
/// validation or numerical errors are propagated from
/// [`McmcDiagnostics::from_multiple_chains`].
pub fn run_multiple_chains<S>(
    samplers: &mut [S],
    n_warmup: usize,
    n_samples: usize,
) -> Result<MultiChainOutput>
where
    S: Sampler,
{
    if samplers.len() < 2 {
        return Err(BayesError::invalid_parameter(
            "At least two samplers are required for multi-chain diagnostics",
        ));
    }

    if n_samples < 2 {
        return Err(BayesError::invalid_parameter(
            "At least two samples per chain are required for multi-chain diagnostics",
        ));
    }

    let chains: Vec<Vec<DVector<f64>>> = samplers
        .iter_mut()
        .map(|sampler| sampler.sample_with_warmup(n_warmup, n_samples))
        .collect();
    let diagnostics = McmcDiagnostics::from_multiple_chains(&chains)?;
    let summary = diagnostics.summary();

    Ok(MultiChainOutput {
        chains,
        diagnostics,
        summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct DeterministicSampler {
        state: DVector<f64>,
        steps: usize,
        resets: usize,
    }

    impl DeterministicSampler {
        fn new(start: f64) -> Self {
            Self {
                state: DVector::from_vec(vec![start]),
                steps: 0,
                resets: 0,
            }
        }
    }

    impl Sampler for DeterministicSampler {
        fn sample(&mut self, n_samples: usize) -> Vec<DVector<f64>> {
            (0..n_samples).map(|_| self.step()).collect()
        }

        fn step(&mut self) -> DVector<f64> {
            self.steps += 1;
            self.state[0] += 1.0;
            self.state.clone()
        }

        fn current_state(&self) -> &DVector<f64> {
            &self.state
        }

        fn reset_statistics(&mut self) {
            self.resets += 1;
        }
    }

    #[test]
    fn run_multiple_chains_returns_raw_chains_diagnostics_and_summary() {
        let mut samplers = [
            DeterministicSampler::new(0.0),
            DeterministicSampler::new(10.0),
        ];

        let output = run_multiple_chains(&mut samplers, 2, 4).unwrap();

        assert_eq!(
            output.chains,
            vec![
                vec![
                    DVector::from_vec(vec![3.0]),
                    DVector::from_vec(vec![4.0]),
                    DVector::from_vec(vec![5.0]),
                    DVector::from_vec(vec![6.0]),
                ],
                vec![
                    DVector::from_vec(vec![13.0]),
                    DVector::from_vec(vec![14.0]),
                    DVector::from_vec(vec![15.0]),
                    DVector::from_vec(vec![16.0]),
                ],
            ]
        );
        assert!(output.diagnostics.r_hat.is_some());
        assert_eq!(output.summary.parameters.len(), 1);
        assert_eq!(output.summary.parameters[0].parameter_index, 0);
        assert_eq!(
            output.summary.parameters[0].effective_sample_size,
            output.diagnostics.effective_sample_size[0]
        );
        assert_eq!(
            output.summary.parameters[0].mc_se,
            output.diagnostics.mc_se[0]
        );
        assert_eq!(
            output.summary.parameters[0].r_hat,
            Some(output.diagnostics.r_hat.as_ref().unwrap()[0])
        );
        assert_eq!(samplers[0].steps, 6);
        assert_eq!(samplers[1].steps, 6);
        assert_eq!(samplers[0].resets, 1);
        assert_eq!(samplers[1].resets, 1);
    }

    #[test]
    fn run_multiple_chains_rejects_empty_sampler_list() {
        let mut samplers: [DeterministicSampler; 0] = [];

        assert!(run_multiple_chains(&mut samplers, 0, 4).is_err());
    }

    #[test]
    fn run_multiple_chains_rejects_single_sampler_without_mutating_it() {
        let mut samplers = [DeterministicSampler::new(0.0)];

        assert!(run_multiple_chains(&mut samplers, 2, 4).is_err());
        assert_eq!(samplers[0].steps, 0);
        assert_eq!(samplers[0].resets, 0);
        assert_eq!(samplers[0].current_state(), &DVector::from_vec(vec![0.0]));
    }

    #[test]
    fn run_multiple_chains_rejects_too_few_samples_without_mutating_samplers() {
        let mut samplers = [
            DeterministicSampler::new(0.0),
            DeterministicSampler::new(10.0),
        ];

        assert!(run_multiple_chains(&mut samplers, 0, 1).is_err());
        assert_eq!(samplers[0].steps, 0);
        assert_eq!(samplers[1].steps, 0);
        assert_eq!(samplers[0].resets, 0);
        assert_eq!(samplers[1].resets, 0);
        assert_eq!(samplers[0].current_state(), &DVector::from_vec(vec![0.0]));
        assert_eq!(samplers[1].current_state(), &DVector::from_vec(vec![10.0]));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn multi_chain_output_serializes_to_json() {
        let mut samplers = [
            DeterministicSampler::new(0.0),
            DeterministicSampler::new(10.0),
        ];
        let output = run_multiple_chains(&mut samplers, 2, 4).unwrap();

        let json = serde_json::to_value(&output).unwrap();

        assert_eq!(json["chains"].as_array().unwrap().len(), 2);
        assert_eq!(json["summary"]["parameters"].as_array().unwrap().len(), 1);
        assert_eq!(json["diagnostics"]["mean"].as_array().unwrap().len(), 1);
    }
}
