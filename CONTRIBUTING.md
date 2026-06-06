# Contributing to bayes-rs

Thank you for your interest in contributing to bayes-rs! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/SyntaxSpirits/bayes-rs.git
   cd bayes-rs
   ```
3. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- Rust 1.70+ (stable, beta, and nightly are tested)
- Git

### Building and Testing
```bash
# Build the project
cargo build

# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run clippy (linting)
cargo clippy --all-targets --all-features

# Check formatting
cargo fmt --check

# Build documentation
cargo doc --all-features --no-deps

# Run benchmarks
cargo bench
```

### Benchmark Baseline Reporting

Benchmarks use Criterion and are intended to inform reviews without making CI
flaky. Do not add mandatory performance pass/fail gates unless the project later
adopts a dedicated, stable benchmarking runner.

For performance-sensitive changes, capture a reproducible local baseline before
editing and compare your branch against it:

```bash
# From the baseline branch or commit you want to compare against
cargo bench --all-features -- --save-baseline before-change

# From your feature branch
cargo bench --all-features -- --baseline before-change
```

Include the following details in the PR description when benchmark results are
relevant:

- Baseline and candidate git SHAs (`git rev-parse --short HEAD` for each)
- `rustc --version` and `cargo --version`
- CPU model/core count, operating system, and any relevant power or thermal
  constraints
- Exact benchmark command, including feature flags and any Criterion filters
- A short summary of the Criterion comparison, plus the HTML report location
  (`target/criterion/report/index.html`, or the equivalent path under
  `$CARGO_TARGET_DIR`) or attached artifacts when helpful

Prefer local Criterion comparisons and posted artifacts over CI performance
thresholds. CI should continue checking that benchmarks compile, for example with
`cargo bench --no-run --all-features`, without failing on noisy timing deltas.

## Contribution Guidelines

### Code Style
- Follow the standard Rust formatting (`cargo fmt`)
- Pass all clippy lints (`cargo clippy`)
- Write clear, self-documenting code
- Add comments for complex algorithms

### Testing
- Add unit tests for new functions and methods
- Add integration tests for new major features
- Ensure all tests pass before submitting
- Aim for high test coverage on new code

### Documentation
- Document all public APIs with doc comments
- Include examples in doc comments where helpful
- Update README.md if adding user-facing features
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format

### Pull Requests
1. **Write clear commit messages** following conventional commits format:
   ```
   feat: add new MCMC sampler
   fix: resolve numerical instability in HMC
   docs: improve documentation for diagnostics module
   ```

2. **Keep PRs focused** - one feature or fix per PR

3. **Write a good PR description** explaining:
   - What changes you made
   - Why you made them
   - How to test the changes
   - Any breaking changes

4. **Link to issues** if your PR addresses existing issues

5. **Request review** from maintainers

### Code Organization

#### Adding New Distributions
- Implement the `Distribution` or `MultivariateDistribution` trait
- Add comprehensive tests including edge cases
- Include parameter validation with proper error handling
- Add examples to documentation

#### Adding New Samplers
- Implement the `Sampler` trait
- Include acceptance rate tracking
- Add numerical stability considerations
- Test with various target distributions
- Document algorithm details and references

#### Adding New Diagnostics
- Ensure statistical correctness
- Include references to literature
- Test against known analytical results where possible
- Consider computational efficiency

## Issue Reporting

### Bug Reports
Please include:
- Rust version (`rustc --version`)
- bayes-rs version
- Minimal reproduction code
- Expected vs actual behavior
- Error messages (if any)

### Feature Requests
Please include:
- Clear description of the desired feature
- Use case or motivation
- Proposed API (if you have ideas)
- References to literature (for statistical methods)

## Areas for Contribution

### High Priority
- Additional MCMC samplers (NUTS, slice sampling, etc.)
- More probability distributions
- Improved numerical stability
- Performance optimizations
- Better error messages

### Medium Priority
- Visualization tools
- More real-world examples
- Python bindings (PyO3)
- WebAssembly support
- Parallel chain execution

### Documentation
- Tutorial content
- More examples
- Performance guides
- Mathematical background explanations

## Code of Conduct

Please note that this project follows a Code of Conduct. By participating, you agree to:
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different opinions and approaches

## Questions?

- Open an issue for discussion
- Check existing issues and documentation first
- Be specific about what you're trying to achieve

Thank you for contributing to bayes-rs! 🚀 