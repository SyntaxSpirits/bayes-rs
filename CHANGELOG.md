# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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