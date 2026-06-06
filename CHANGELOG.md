# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation guidance for reproducible Criterion benchmark baseline capture,
  local comparisons, and non-flaky performance regression reporting.

## [0.3.0] - 2026-06-06

### Added
- Categorical discrete distribution with normalized category weights,
  PMF/log-PMF helpers, analytical moments, and seeded sampling support.

## [0.2.0] - 2026-06-06

### Added
- Discrete distributions module with discrete sampling primitives.
- Hamiltonian Monte Carlo gradient checker with `finite_difference_gradient` and
  `gradient_check` helpers for validating user-supplied gradients.
- Multi-chain runner (`multi_chain::run_multiple_chains` and `MultiChainOutput`)
  that drives several seeded samplers through a shared warmup/sample schedule
  and returns combined diagnostics.
- First-class warmup/burn-in workflow on the `Sampler` trait: `run_with_warmup`,
  `sample_with_warmup`, `WarmupMetadata`, and `WarmupRun`. Sampler statistics
  are reset between the warmup and retained phases.
- Optional `serde` support for `McmcDiagnostics`, `McmcDiagnosticSummary`,
  `ParameterDiagnosticSummary`, `TracePlot`, `WarmupMetadata`, `WarmupRun`, and
  `MultiChainOutput`, gated behind the `serde` feature.
- Normal-normal and additional conjugate-model examples
  (`examples/conjugate_models.rs`, `examples/serde_diagnostics.rs`,
  `examples/discrete_distributions.rs`) plus a conjugate-models test suite.
- MCSE summaries and richer per-parameter diagnostic output.

### Changed
- Hardened sampler input validation across Metropolis-Hastings, Gibbs, and HMC
  constructors to reject non-finite or mis-sized inputs earlier.
- Tightened gradient-check input validation (empty or non-finite points are
  rejected before evaluating the analytic gradient).

### Fixed
- Improved numerical accuracy of the Gamma log-probability computation; locked
  with a regression test.

[0.3.0]: https://github.com/SyntaxSpirits/bayes-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/SyntaxSpirits/bayes-rs/releases/tag/v0.2.0

## [0.1.0] - 2025-07-03

### Added
- Initial release of bayes-rs
- MCMC samplers:
  - Metropolis-Hastings with adaptive proposal tuning
  - Hamiltonian Monte Carlo with leapfrog integration
  - Gibbs sampler for conditional distributions
- Statistical distributions:
  - Univariate: Normal, Gamma, Beta, Exponential, Uniform, Student's t
  - Multivariate: MultivariateNormal with full covariance support
- MCMC diagnostics:
  - Effective sample size calculation
  - R-hat statistic for convergence assessment
  - Monte Carlo standard error computation
  - Autocorrelation analysis
  - Basic statistics and quantiles
  - Trace plot data structures
- Comprehensive test suite (27 unit + 9 integration + 1 doc test)
- Performance benchmarks for all major components
- Real-world example: Bayesian linear regression
- Complete documentation and README
- Error handling with structured error types
- Optional serde support for serialization

### Technical Details
- Uses nalgebra for efficient linear algebra
- Proper numerical stability and parameter validation
- Memory-efficient implementations
- Extensive documentation with examples

[0.1.0]: https://github.com/SyntaxSpirits/bayes-rs/releases/tag/v0.1.0 