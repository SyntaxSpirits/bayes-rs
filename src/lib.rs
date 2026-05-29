//! # bayes-rs
//!
//! A Rust library for Bayesian inference with MCMC samplers.
//!
//! This library provides implementations of various MCMC algorithms for Bayesian parameter estimation:
//! - Metropolis-Hastings sampler
//! - Gibbs sampler
//! - Hamiltonian Monte Carlo (HMC)
//!
//! ## Example
//!
//! ```rust
//! use bayes_rs::{
//!     distributions::{Normal, Distribution},
//!     samplers::{MetropolisHastings, Sampler},
//! };
//! use nalgebra::DVector;
//!
//! // Define a simple normal distribution likelihood
//! let likelihood = |params: &DVector<f64>, data: &[f64]| -> f64 {
//!     let mu = params[0];
//!     let sigma = params[1].exp(); // log-sigma for positivity
//!     data.iter().map(|&x| Normal::new(mu, sigma).unwrap().log_pdf(x)).sum()
//! };
//!
//! // Define prior
//! let prior = |params: &DVector<f64>| -> f64 {
//!     Normal::new(0.0, 10.0).unwrap().log_pdf(params[0]) +
//!     Normal::new(0.0, 1.0).unwrap().log_pdf(params[1])
//! };
//!
//! // Create data
//! let data = vec![1.0, 2.0, 3.0, 2.5, 1.8];
//!
//! // Initialize sampler with a deterministic seed for reproducibility
//! let mut sampler = MetropolisHastings::with_seed(
//!     move |params| likelihood(params, &data) + prior(params),
//!     DVector::from_vec(vec![0.0, 0.0]), // initial parameters
//!     DVector::from_vec(vec![0.5, 0.2]), // proposal standard deviations
//!     42,
//! ).unwrap();
//!
//! // Run MCMC
//! let samples = sampler.sample(1000);
//! ```

pub mod diagnostics;
pub mod distributions;
pub mod error;
pub mod samplers;

pub use error::{BayesError, Result};

/// Common traits and types used throughout the library
pub mod prelude {
    pub use crate::distributions::*;
    pub use crate::error::{BayesError, Result};
    pub use crate::samplers::*;
    pub use nalgebra::{DMatrix, DVector};
}
