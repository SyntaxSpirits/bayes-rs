# bayes-rs

A comprehensive Rust library for Bayesian inference with MCMC samplers, featuring robust statistical distributions and advanced diagnostic tools.

[![Crates.io](https://img.shields.io/crates/v/bayes-rs.svg)](https://crates.io/crates/bayes-rs)
[![Documentation](https://docs.rs/bayes-rs/badge.svg)](https://docs.rs/bayes-rs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **MCMC Samplers**: Metropolis-Hastings, Gibbs, and Hamiltonian Monte Carlo (HMC)
- **Statistical Distributions**: Normal, Multivariate Normal, Gamma, Beta, Exponential, Uniform, Student's t, Bernoulli, Binomial, Poisson
- **MCMC Diagnostics**: Effective sample size, R-hat statistic, MCSE summaries, autocorrelation analysis, trace plots
- **Multi-chain workflows**: Run multiple seeded chains with a shared warmup/sample schedule
- **Best Practices**: Comprehensive error handling, extensive testing, performance benchmarks
- **Easy to Use**: Clean API with extensive documentation and examples

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
bayes-rs = "0.1.0"
```

Enable the optional `serde` feature when you want to serialize user-facing MCMC output:

```toml
[dependencies]
bayes-rs = { version = "0.1.0", features = ["serde"] }
```

### Simple Example

```rust
use bayes_rs::{
    distributions::Normal,
    samplers::{MetropolisHastings, Sampler},
    prelude::*,
};
use nalgebra::DVector;

// Define a log posterior function
let log_posterior = |params: &DVector<f64>| -> f64 {
    let mu = params[0];
    let log_sigma = params[1];
    let sigma = log_sigma.exp();
    
    // Prior: mu ~ N(0, 10), log_sigma ~ N(0, 1)
    let prior_mu = Normal::new(0.0, 10.0).unwrap();
    let prior_log_sigma = Normal::new(0.0, 1.0).unwrap();
    
    prior_mu.log_pdf(mu) + prior_log_sigma.log_pdf(log_sigma)
};

// Set up sampler
let initial_state = DVector::from_vec(vec![0.0, 0.0]);
let proposal_std = DVector::from_vec(vec![0.5, 0.2]);

let mut sampler = MetropolisHastings::new(
    log_posterior,
    initial_state,
    proposal_std,
).unwrap();

// Generate samples after discarding warmup iterations
let samples = sampler.sample_with_warmup(1_000, 10_000);
println!("Generated {} posterior samples", samples.len());
```

`sample_with_warmup` discards initial iterations and resets sampler statistics before collecting the returned samples; it does not adapt proposal scales or other tuning parameters automatically.

## MCMC Samplers

### Metropolis-Hastings

The workhorse of MCMC sampling with adaptive proposal tuning:

```rust
use bayes_rs::{samplers::MetropolisHastings, prelude::*};

let mut sampler = MetropolisHastings::new(
    log_posterior_fn,
    initial_state,
    proposal_std,
)?;

// Run pilot adaptation before collecting samples
for _ in 0..1_000 {
    sampler.step();
    sampler.adapt_proposal(0.44); // Target 44% acceptance rate
}
sampler.reset_statistics();

let samples = sampler.sample(10_000);
println!("Acceptance rate: {:.3}", sampler.acceptance_rate().unwrap());
```

### Hamiltonian Monte Carlo (HMC)

Efficient sampling using gradient information:

```rust
use bayes_rs::{samplers::HamiltonianMonteCarlo, prelude::*};

let gradient_fn = |params: &DVector<f64>| -> DVector<f64> {
    // Compute gradient of log posterior
    // ...
};

let mut hmc_sampler = HamiltonianMonteCarlo::new(
    log_posterior_fn,
    gradient_fn,
    initial_state,
    step_size,      // e.g., 0.1
    n_leapfrog,     // e.g., 10
)?;

let samples = hmc_sampler.sample(5000);
```

### Gibbs Sampling

For models with known conditional distributions:

```rust
use bayes_rs::{samplers::GibbsSampler, prelude::*};

let conditional_samplers = vec![
    |params: &DVector<f64>, idx: usize, rng: &mut ThreadRng| -> f64 {
        // Sample parameter idx given all others
        // ...
    },
    // More conditional samplers...
];

let mut gibbs_sampler = GibbsSampler::new(
    conditional_samplers,
    initial_state,
)?;
```

## Statistical Distributions

### Univariate Distributions

```rust
use bayes_rs::distributions::*;

// Normal distribution
let normal = Normal::new(0.0, 1.0)?;
println!("PDF at x=1: {}", normal.pdf(1.0));
println!("Log PDF at x=1: {}", normal.log_pdf(1.0));

// Gamma distribution
let gamma = Gamma::new(2.0, 1.0)?;
println!("Mean: {}, Variance: {}", gamma.mean(), gamma.variance());

// Beta distribution
let beta = Beta::new(2.0, 3.0)?;
println!("PDF at x=0.5: {}", beta.pdf(0.5));

// Student's t-distribution
let t_dist = StudentT::new(3.0, 0.0, 1.0)?;
println!("Degrees of freedom: {}", t_dist.degrees_of_freedom());
```

### Multivariate Distributions

```rust
use bayes_rs::distributions::MultivariateNormal;
use nalgebra::{DVector, DMatrix};

let mu = DVector::from_vec(vec![0.0, 0.0]);
let cov = DMatrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]);

let mvn = MultivariateNormal::new(mu, cov)?;
let x = DVector::from_vec(vec![1.0, -1.0]);
println!("Log PDF: {}", mvn.log_pdf(&x));
```

### Discrete Distributions

Use `DiscreteDistribution` for count-valued distributions with probability mass functions and seeded sampling. `Poisson::new(lambda)` requires `lambda` to be finite, positive, and no greater than `1e12`. Add `rand = "0.8"` to your application dependencies when using the seeded sampling example.

```rust
use bayes_rs::distributions::{Bernoulli, Binomial, DiscreteDistribution, Poisson};
use rand::{rngs::StdRng, SeedableRng};

let bernoulli = Bernoulli::new(0.3)?;
let binomial = Binomial::new(10, 0.3)?;
let poisson = Poisson::new(2.5)?;

println!("Bernoulli P(X=1): {}", bernoulli.pmf(1));
println!("Binomial log P(X=3): {}", binomial.log_pmf(3));
println!("Poisson P(X=2): {}", poisson.pmf(2));

let mut rng = StdRng::seed_from_u64(42);
let draw = poisson.sample(&mut rng);
println!("Seeded Poisson draw: {}", draw);
```

## MCMC Diagnostics

Comprehensive diagnostic tools to assess convergence and sample quality:

```rust
use bayes_rs::diagnostics::{McmcDiagnostics, TracePlot};

// Single chain diagnostics
let diagnostics = McmcDiagnostics::from_single_chain(&samples)?;

println!("Effective sample sizes: {:?}", diagnostics.effective_sample_size);
println!("Parameter means: {:?}", diagnostics.mean);
println!("Parameter std devs: {:?}", diagnostics.std_dev);

// Multiple chain diagnostics (includes R-hat)
let diagnostics = McmcDiagnostics::from_multiple_chains(&chains)?;
let summary = diagnostics.summary();

if let Some(r_hat) = &diagnostics.r_hat {
    println!("R-hat values: {:?}", r_hat);
    println!("Converged: {}", diagnostics.has_converged());
}
println!("R-hat/ESS/MCSE summary: {:?}", summary.parameters);

// Run multiple pre-configured, independently seeded samplers and summarize in one step.
// The slice must contain at least two samplers of the same concrete sampler type,
// and at least two retained samples per chain are required for R-hat.
use bayes_rs::multi_chain::run_multiple_chains;
let mut seeded_chains = [
    build_sampler_with_seed(11)?,
    build_sampler_with_seed(22)?,
    build_sampler_with_seed(33)?,
    build_sampler_with_seed(44)?,
];
let output = run_multiple_chains(&mut seeded_chains, 1_000, 10_000)?;
println!("Converged: {}", output.summary.has_converged);

// Trace plots
let trace_plot = TracePlot::new(&samples, 0)?; // Parameter 0
// Use trace_plot.values and trace_plot.iterations for visualization
```

With the optional `serde` feature enabled, `McmcDiagnostics`, `McmcDiagnosticSummary`,
`ParameterDiagnosticSummary`, `TracePlot`, and `MultiChainOutput` derive `Serialize` for
JSON or other serde formats:

```rust
let summary_json = serde_json::to_string_pretty(&output.summary)?;
```

Add `serde_json` or another serde format crate to your application to emit a concrete format.
These structs use Rust field names in their serialized form. Treat that JSON shape as a
convenient interchange format for the current API, not as a long-term storage schema.
JSON serializers may reject non-finite diagnostics such as `NaN` or `Infinity` from
degenerate chains; handle those cases before persisting JSON output.

## Real-World Example: Bayesian Linear Regression

```rust
use bayes_rs::{
    distributions::Normal,
    samplers::{MetropolisHastings, Sampler},
    diagnostics::McmcDiagnostics,
    prelude::*,
};

fn bayesian_linear_regression(x_data: &[f64], y_data: &[f64]) -> Result<Vec<DVector<f64>>> {
    let log_posterior = |params: &DVector<f64>| -> f64 {
        let beta0 = params[0]; // intercept
        let beta1 = params[1]; // slope
        let log_sigma = params[2]; // log(noise std)
        let sigma = log_sigma.exp();
        
        // Priors
        let prior_beta0 = Normal::new(0.0, 10.0).unwrap();
        let prior_beta1 = Normal::new(0.0, 10.0).unwrap();
        let prior_log_sigma = Normal::new(0.0, 1.0).unwrap();
        
        let prior_log_prob = prior_beta0.log_pdf(beta0) + 
                            prior_beta1.log_pdf(beta1) + 
                            prior_log_sigma.log_pdf(log_sigma);
        
        // Likelihood
        if !sigma.is_finite() || sigma <= 0.0 {
            return f64::NEG_INFINITY;
        }
        
        let likelihood_dist = Normal::new(0.0, sigma).unwrap();
        let likelihood_log_prob: f64 = x_data.iter()
            .zip(y_data.iter())
            .map(|(&x_i, &y_i)| {
                let predicted = beta0 + beta1 * x_i;
                let residual = y_i - predicted;
                likelihood_dist.log_pdf(residual)
            })
            .sum();
        
        prior_log_prob + likelihood_log_prob
    };
    
    let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let proposal_std = DVector::from_vec(vec![0.5, 0.1, 0.1]);
    
    let mut sampler = MetropolisHastings::new(
        log_posterior,
        initial_state,
        proposal_std,
    )?;
    
    Ok(sampler.sample(10000))
}

// Usage
let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y_data = vec![2.1, 3.9, 6.1, 8.0, 9.9];

let samples = bayesian_linear_regression(&x_data, &y_data)?;
let diagnostics = McmcDiagnostics::from_single_chain(&samples)?;

println!("Intercept estimate: {:.3} ± {:.3}", 
         diagnostics.mean[0], diagnostics.std_dev[0]);
println!("Slope estimate: {:.3} ± {:.3}", 
         diagnostics.mean[1], diagnostics.std_dev[1]);
```

## Performance

Run benchmarks to see performance characteristics:

```bash
cargo bench
```

The library is optimized for:
- Efficient matrix operations using `nalgebra`
- Minimal memory allocations during sampling
- Fast distribution computations with precomputed constants

## Error Handling

The library uses comprehensive error handling with the `BayesError` enum:

```rust
use bayes_rs::error::{BayesError, Result};

match Normal::new(0.0, -1.0) {
    Ok(dist) => println!("Created distribution"),
    Err(BayesError::InvalidParameter { message }) => {
        println!("Invalid parameter: {}", message);
    },
    Err(e) => println!("Other error: {}", e),
}
```

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests

# All tests with output
cargo test -- --nocapture
```

## Examples

See the `examples/` directory for complete examples:

```bash
# Bayesian linear regression example
cargo run --example linear_regression

# Discrete Bernoulli, Binomial, and Poisson distributions
cargo run --example discrete_distributions

# Conjugate Bayesian model examples
cargo run --example conjugate_models

# Serde-enabled diagnostics serialization example
cargo run --features serde --example serde_diagnostics
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/SyntaxSpirits/bayes-rs.git
cd bayes-rs
cargo test
cargo doc --open
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{bayes_rs,
  title = {bayes-rs: A Rust Library for Bayesian Inference},
  author = {Alex Kholodniak},
  year = {2025},
  url = {https://github.com/SyntaxSpirits/bayes-rs}
}
```

## Related Projects

- [PyMC](https://www.pymc.io/) - Python library for Bayesian modeling
- [Stan](https://mc-stan.org/) - Platform for statistical modeling and high-performance statistical computation
- [Edward](http://edwardlib.org/) - Python library for probabilistic modeling
- [Turing.jl](https://turing.ml/) - Julia library for Bayesian inference 