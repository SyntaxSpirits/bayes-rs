//! Statistical distributions for Bayesian inference

use crate::error::{BayesError, Result};
use nalgebra::DVector;
use rand_distr::Distribution as RandDistribution;
use std::f64::consts::PI;

// rand_distr::Poisson samples through f64, so keep accepted rates comfortably
// below extreme count ranges where precision and rejection-sampler behavior are
// less appropriate for this crate's infallible u64 sampling API.
const MAX_POISSON_SAMPLE_RATE: f64 = 1_000_000_000_000.0;

/// Trait for probability distributions
pub trait Distribution {
    /// Compute the log probability density function
    fn log_pdf(&self, x: f64) -> f64;

    /// Compute the probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }
}

/// Trait for scalar discrete probability distributions over non-negative integer support.
///
/// Constructors validate distribution parameters up front, so sampling is infallible.
/// The `u64` support type is intended for count-valued distributions such as
/// Bernoulli, Binomial, and Poisson.
pub trait DiscreteDistribution {
    /// Compute the log probability mass function at integer support value `k`
    fn log_pmf(&self, k: u64) -> f64;

    /// Compute the probability mass function at integer support value `k`
    fn pmf(&self, k: u64) -> f64 {
        self.log_pmf(k).exp()
    }

    /// Draw a sample from the distribution
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64;
}

/// Multivariate distribution trait
pub trait MultivariateDistribution {
    /// Compute the log probability density function for a vector
    fn log_pdf(&self, x: &DVector<f64>) -> f64;

    /// Compute the probability density function for a vector
    fn pdf(&self, x: &DVector<f64>) -> f64 {
        self.log_pdf(x).exp()
    }
}

/// Bernoulli distribution over {0, 1}
#[derive(Debug, Clone, PartialEq)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    /// Create a new Bernoulli distribution with success probability `p`.
    pub fn new(p: f64) -> Result<Self> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(BayesError::invalid_parameter(
                "probability must be finite and between 0 and 1",
            ));
        }

        Ok(Self { p })
    }

    /// Get the success probability.
    pub fn probability(&self) -> f64 {
        self.p
    }

    /// Get the mean.
    pub fn mean(&self) -> f64 {
        self.p
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
}

impl DiscreteDistribution for Bernoulli {
    fn log_pmf(&self, k: u64) -> f64 {
        match k {
            0 => {
                if self.p == 1.0 {
                    f64::NEG_INFINITY
                } else {
                    (-self.p).ln_1p()
                }
            }
            1 => {
                if self.p == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    self.p.ln()
                }
            }
            _ => f64::NEG_INFINITY,
        }
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        if rng.gen_bool(self.p) {
            1
        } else {
            0
        }
    }
}

/// Binomial distribution counting successes in `n` Bernoulli trials
#[derive(Debug, Clone, PartialEq)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// Create a new binomial distribution with `n` trials and success probability `p`.
    pub fn new(n: u64, p: f64) -> Result<Self> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(BayesError::invalid_parameter(
                "probability must be finite and between 0 and 1",
            ));
        }

        Ok(Self { n, p })
    }

    /// Get the number of trials.
    pub fn trials(&self) -> u64 {
        self.n
    }

    /// Get the success probability.
    pub fn probability(&self) -> f64 {
        self.p
    }

    /// Get the mean.
    pub fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

impl DiscreteDistribution for Binomial {
    fn log_pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return f64::NEG_INFINITY;
        }

        if self.p == 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if self.p == 1.0 {
            return if k == self.n { 0.0 } else { f64::NEG_INFINITY };
        }

        log_binomial_coefficient(self.n, k)
            + k as f64 * self.p.ln()
            + (self.n - k) as f64 * (-self.p).ln_1p()
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        if self.p == 0.0 {
            return 0;
        }
        if self.p == 1.0 {
            return self.n;
        }

        let binomial = rand_distr::Binomial::new(self.n, self.p)
            .expect("validated binomial parameters should be accepted by rand_distr");
        binomial.sample(rng)
    }
}

/// Poisson distribution over non-negative integer counts
#[derive(Debug, Clone, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Create a new Poisson distribution with rate `lambda`.
    ///
    /// `lambda` must be finite, positive, and no greater than `1e12`. The upper
    /// bound keeps sampling well inside the integer precision range of the
    /// underlying `rand_distr::Poisson<f64>` sampler.
    pub fn new(lambda: f64) -> Result<Self> {
        if !lambda.is_finite() || lambda <= 0.0 || lambda > MAX_POISSON_SAMPLE_RATE {
            return Err(BayesError::invalid_parameter(format!(
                "lambda must be finite, positive, and no greater than {MAX_POISSON_SAMPLE_RATE:e}",
            )));
        }

        Ok(Self { lambda })
    }

    /// Get the rate parameter.
    pub fn rate(&self) -> f64 {
        self.lambda
    }

    /// Get the mean.
    pub fn mean(&self) -> f64 {
        self.lambda
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.lambda
    }
}

impl DiscreteDistribution for Poisson {
    fn log_pmf(&self, k: u64) -> f64 {
        k as f64 * self.lambda.ln() - self.lambda - gamma_ln(k as f64 + 1.0)
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let poisson = rand_distr::Poisson::new(self.lambda)
            .expect("validated Poisson rate should be accepted by rand_distr");
        // rand_distr's Poisson sampler returns integer-valued f64 samples.
        poisson.sample(rng) as u64
    }
}

/// Categorical distribution over category indices `0..K`.
///
/// The constructor accepts non-negative, finite weights and stores them as
/// normalized probabilities. At least one category must have positive mass.
#[derive(Debug, Clone, PartialEq)]
pub struct Categorical {
    probabilities: Vec<f64>,
    cumulative_probabilities: Vec<f64>,
}

impl Categorical {
    /// Create a new categorical distribution from non-negative weights.
    ///
    /// The weights must be non-empty, finite, non-negative, and have positive
    /// finite total mass. They do not need to already sum to one.
    pub fn new(weights: Vec<f64>) -> Result<Self> {
        if weights.is_empty() {
            return Err(BayesError::invalid_parameter(
                "categorical weights must not be empty",
            ));
        }

        if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
            return Err(BayesError::invalid_parameter(
                "categorical weights must be finite and non-negative",
            ));
        }

        let total_mass: f64 = weights.iter().sum();
        if !total_mass.is_finite() || total_mass <= 0.0 {
            return Err(BayesError::invalid_parameter(
                "categorical weights must have positive finite total mass",
            ));
        }

        let probabilities: Vec<f64> = weights.into_iter().map(|w| w / total_mass).collect();
        let mut running_total = 0.0;
        let mut cumulative_probabilities: Vec<f64> = probabilities
            .iter()
            .map(|&p| {
                running_total += p;
                running_total
            })
            .collect();

        // Avoid roundoff leaving the final positive-mass category unreachable
        // for samples infinitesimally below 1.0. Clamp through trailing zero-mass
        // categories so they do not accidentally become sampleable.
        if let Some(last_positive) = probabilities.iter().rposition(|&p| p > 0.0) {
            for cumulative_probability in &mut cumulative_probabilities[last_positive..] {
                *cumulative_probability = 1.0;
            }
        }

        Ok(Self {
            probabilities,
            cumulative_probabilities,
        })
    }

    /// Get the normalized category probabilities.
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Get the number of categories.
    pub fn category_count(&self) -> usize {
        self.probabilities.len()
    }

    /// Get the mean category index.
    pub fn mean(&self) -> f64 {
        self.probabilities
            .iter()
            .enumerate()
            .map(|(category, &p)| category as f64 * p)
            .sum()
    }

    /// Get the variance of the category index.
    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        self.probabilities
            .iter()
            .enumerate()
            .map(|(category, &p)| {
                let diff = category as f64 - mean;
                diff * diff * p
            })
            .sum()
    }
}

impl DiscreteDistribution for Categorical {
    fn log_pmf(&self, k: u64) -> f64 {
        let Ok(index) = usize::try_from(k) else {
            return f64::NEG_INFINITY;
        };
        let Some(&probability) = self.probabilities.get(index) else {
            return f64::NEG_INFINITY;
        };

        if probability == 0.0 {
            f64::NEG_INFINITY
        } else {
            probability.ln()
        }
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let u: f64 = rng.gen();
        self.cumulative_probabilities
            .iter()
            .position(|&cdf| u < cdf)
            .unwrap_or(self.probabilities.len() - 1) as u64
    }
}

/// Normal (Gaussian) distribution
#[derive(Debug, Clone, PartialEq)]
pub struct Normal {
    mu: f64,
    sigma: f64,
    log_sigma: f64,
    inv_sigma: f64,
}

impl Normal {
    /// Create a new normal distribution
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(BayesError::invalid_parameter("sigma must be positive"));
        }
        if !mu.is_finite() || !sigma.is_finite() {
            return Err(BayesError::invalid_parameter("parameters must be finite"));
        }

        Ok(Self {
            mu,
            sigma,
            log_sigma: sigma.ln(),
            inv_sigma: 1.0 / sigma,
        })
    }

    /// Get the mean parameter
    pub fn mean(&self) -> f64 {
        self.mu
    }

    /// Get the standard deviation parameter
    pub fn std_dev(&self) -> f64 {
        self.sigma
    }

    /// Get the variance
    pub fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
}

impl Distribution for Normal {
    fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        let diff = x - self.mu;
        -0.5 * (2.0 * PI).ln()
            - self.log_sigma
            - 0.5 * diff * diff * self.inv_sigma * self.inv_sigma
    }
}

/// Multivariate normal distribution
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    mu: DVector<f64>,
    precision: nalgebra::DMatrix<f64>, // inverse covariance
    log_det_precision: f64,
    dim: usize,
}

impl MultivariateNormal {
    /// Create a new multivariate normal distribution
    pub fn new(mu: DVector<f64>, covariance: nalgebra::DMatrix<f64>) -> Result<Self> {
        if mu.len() != covariance.nrows() || covariance.nrows() != covariance.ncols() {
            return Err(BayesError::dimension_mismatch(mu.len(), covariance.nrows()));
        }

        let chol = covariance.clone().cholesky().ok_or_else(|| {
            BayesError::numerical_error("Covariance matrix is not positive definite")
        })?;

        let precision = chol.inverse();
        let log_det_precision = -2.0 * chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();

        Ok(Self {
            dim: mu.len(),
            mu,
            precision,
            log_det_precision,
        })
    }

    /// Create a multivariate normal with diagonal covariance
    pub fn new_diagonal(mu: DVector<f64>, variances: DVector<f64>) -> Result<Self> {
        if mu.len() != variances.len() {
            return Err(BayesError::dimension_mismatch(mu.len(), variances.len()));
        }

        if variances.iter().any(|&v| v <= 0.0) {
            return Err(BayesError::invalid_parameter(
                "All variances must be positive",
            ));
        }

        let dim = mu.len();
        let mut covariance = nalgebra::DMatrix::zeros(dim, dim);
        for i in 0..dim {
            covariance[(i, i)] = variances[i];
        }

        Self::new(mu, covariance)
    }

    /// Get the mean vector
    pub fn mean(&self) -> &DVector<f64> {
        &self.mu
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

impl MultivariateDistribution for MultivariateNormal {
    fn log_pdf(&self, x: &DVector<f64>) -> f64 {
        if x.len() != self.dim {
            return f64::NEG_INFINITY;
        }

        if !x.iter().all(|&val| val.is_finite()) {
            return f64::NEG_INFINITY;
        }

        let diff = x - &self.mu;
        let quadratic_form = diff.dot(&(self.precision.clone() * &diff));

        -0.5 * (self.dim as f64 * (2.0 * PI).ln() - self.log_det_precision + quadratic_form)
    }
}

/// Gamma distribution
#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    alpha: f64,
    beta: f64,
    log_gamma_alpha: f64,
}

impl Gamma {
    /// Create a new gamma distribution
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(BayesError::invalid_parameter(
                "alpha and beta must be positive",
            ));
        }
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(BayesError::invalid_parameter("parameters must be finite"));
        }

        Ok(Self {
            alpha,
            beta,
            log_gamma_alpha: gamma_ln(alpha),
        })
    }

    /// Get the shape parameter (alpha)
    pub fn shape(&self) -> f64 {
        self.alpha
    }

    /// Get the rate parameter (beta)
    pub fn rate(&self) -> f64 {
        self.beta
    }

    /// Get the mean
    pub fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Get the variance
    pub fn variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }
}

impl Distribution for Gamma {
    fn log_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        (self.alpha - 1.0) * x.ln() - self.beta * x + self.alpha * self.beta.ln()
            - self.log_gamma_alpha
    }
}

/// Beta distribution
#[derive(Debug, Clone, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
    log_beta_function: f64,
}

impl Beta {
    /// Create a new beta distribution
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(BayesError::invalid_parameter(
                "alpha and beta must be positive",
            ));
        }
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(BayesError::invalid_parameter("parameters must be finite"));
        }

        let log_beta_function = gamma_ln(alpha) + gamma_ln(beta) - gamma_ln(alpha + beta);

        Ok(Self {
            alpha,
            beta,
            log_beta_function,
        })
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the beta parameter
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Get the mean
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get the variance
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }
}

impl Distribution for Beta {
    fn log_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() - self.log_beta_function
    }
}

/// Exponential distribution
#[derive(Debug, Clone, PartialEq)]
pub struct Exponential {
    rate: f64,
    log_rate: f64,
}

impl Exponential {
    /// Create a new exponential distribution
    pub fn new(rate: f64) -> Result<Self> {
        if rate <= 0.0 {
            return Err(BayesError::invalid_parameter("rate must be positive"));
        }
        if !rate.is_finite() {
            return Err(BayesError::invalid_parameter("rate must be finite"));
        }

        Ok(Self {
            rate,
            log_rate: rate.ln(),
        })
    }

    /// Get the rate parameter
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Get the mean
    pub fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    /// Get the variance
    pub fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }
}

impl Distribution for Exponential {
    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        self.log_rate - self.rate * x
    }
}

/// Uniform distribution
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    a: f64,
    b: f64,
    log_density: f64,
}

impl Uniform {
    /// Create a new uniform distribution
    pub fn new(a: f64, b: f64) -> Result<Self> {
        if a >= b {
            return Err(BayesError::invalid_parameter("a must be less than b"));
        }
        if !a.is_finite() || !b.is_finite() {
            return Err(BayesError::invalid_parameter("parameters must be finite"));
        }

        Ok(Self {
            a,
            b,
            log_density: -(b - a).ln(),
        })
    }

    /// Get the lower bound
    pub fn lower_bound(&self) -> f64 {
        self.a
    }

    /// Get the upper bound
    pub fn upper_bound(&self) -> f64 {
        self.b
    }

    /// Get the mean
    pub fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    /// Get the variance
    pub fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
}

impl Distribution for Uniform {
    fn log_pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b || !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        self.log_density
    }
}

/// Student's t-distribution
#[derive(Debug, Clone, PartialEq)]
pub struct StudentT {
    nu: f64,
    mu: f64,
    sigma: f64,
    log_normalizer: f64,
}

impl StudentT {
    /// Create a new Student's t-distribution
    pub fn new(nu: f64, mu: f64, sigma: f64) -> Result<Self> {
        if nu <= 0.0 || sigma <= 0.0 {
            return Err(BayesError::invalid_parameter(
                "nu and sigma must be positive",
            ));
        }
        if !nu.is_finite() || !mu.is_finite() || !sigma.is_finite() {
            return Err(BayesError::invalid_parameter("parameters must be finite"));
        }

        let log_normalizer =
            gamma_ln((nu + 1.0) / 2.0) - gamma_ln(nu / 2.0) - 0.5 * (nu * PI).ln() - sigma.ln();

        Ok(Self {
            nu,
            mu,
            sigma,
            log_normalizer,
        })
    }

    /// Get the degrees of freedom
    pub fn degrees_of_freedom(&self) -> f64 {
        self.nu
    }

    /// Get the location parameter
    pub fn location(&self) -> f64 {
        self.mu
    }

    /// Get the scale parameter
    pub fn scale(&self) -> f64 {
        self.sigma
    }

    /// Get the mean (if nu > 1)
    pub fn mean(&self) -> Option<f64> {
        if self.nu > 1.0 {
            Some(self.mu)
        } else {
            None
        }
    }

    /// Get the variance (if nu > 2)
    pub fn variance(&self) -> Option<f64> {
        if self.nu > 2.0 {
            Some(self.sigma * self.sigma * self.nu / (self.nu - 2.0))
        } else {
            None
        }
    }
}

impl Distribution for StudentT {
    fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        let standardized = (x - self.mu) / self.sigma;
        self.log_normalizer
            - 0.5 * (self.nu + 1.0) * (1.0 + standardized * standardized / self.nu).ln()
    }
}

/// Natural logarithm of the binomial coefficient, `ln(n choose k)`.
fn log_binomial_coefficient(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }

    if k == 0 || k == n {
        return 0.0;
    }

    gamma_ln(n as f64 + 1.0) - gamma_ln(k as f64 + 1.0) - gamma_ln((n - k) as f64 + 1.0)
}

/// Approximation of the log gamma function using the Lanczos approximation.
///
/// This helper is defined for positive finite inputs, which covers the
/// distribution normalizers used in this crate.
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return f64::NEG_INFINITY;
    }

    // Lanczos coefficients for g=7, n=9 from Numerical Recipes.
    const LANCZOS_G: f64 = 7.0;
    const LANCZOS_COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        return PI.ln() - (PI * x).sin().ln() - gamma_ln(1.0 - x);
    }

    let z = x - 1.0;
    let mut a = LANCZOS_COEFFS[0];
    for (i, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        a += coeff / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;

    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;

    #[test]
    fn test_bernoulli_distribution() {
        let bernoulli = Bernoulli::new(0.25).unwrap();
        assert_eq!(bernoulli.probability(), 0.25);
        assert_eq!(bernoulli.mean(), 0.25);
        assert_abs_diff_eq!(bernoulli.variance(), 0.1875, epsilon = 1e-12);

        assert_abs_diff_eq!(bernoulli.pmf(0), 0.75, epsilon = 1e-12);
        assert_abs_diff_eq!(bernoulli.pmf(1), 0.25, epsilon = 1e-12);
        assert_eq!(bernoulli.pmf(2), 0.0);
        assert_abs_diff_eq!(bernoulli.log_pmf(1), 0.25_f64.ln(), epsilon = 1e-12);

        let rare_success = Bernoulli::new(1.0e-12).unwrap();
        assert_abs_diff_eq!(
            rare_success.log_pmf(0),
            (-1.0e-12_f64).ln_1p(),
            epsilon = 1e-24
        );
    }

    #[test]
    fn test_bernoulli_edge_cases() {
        let always_zero = Bernoulli::new(0.0).unwrap();
        assert_eq!(always_zero.pmf(0), 1.0);
        assert_eq!(always_zero.pmf(1), 0.0);
        assert_eq!(always_zero.log_pmf(1), f64::NEG_INFINITY);

        let always_one = Bernoulli::new(1.0).unwrap();
        assert_eq!(always_one.pmf(0), 0.0);
        assert_eq!(always_one.pmf(1), 1.0);
        assert_eq!(always_one.log_pmf(0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_bernoulli_invalid_params() {
        assert!(Bernoulli::new(-0.1).is_err());
        assert!(Bernoulli::new(1.1).is_err());
        assert!(Bernoulli::new(f64::NAN).is_err());
        assert!(Bernoulli::new(f64::INFINITY).is_err());
    }

    #[test]
    fn test_bernoulli_sampling_seeded() {
        let bernoulli = Bernoulli::new(0.5).unwrap();
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(42);
        let samples_a: Vec<_> = (0..16).map(|_| bernoulli.sample(&mut rng_a)).collect();
        let samples_b: Vec<_> = (0..16).map(|_| bernoulli.sample(&mut rng_b)).collect();

        assert_eq!(samples_a, samples_b);
        assert!(samples_a.iter().all(|&x| x <= 1));
    }

    #[test]
    fn test_binomial_distribution() {
        let binomial = Binomial::new(10, 0.5).unwrap();
        assert_eq!(binomial.trials(), 10);
        assert_eq!(binomial.probability(), 0.5);
        assert_abs_diff_eq!(binomial.mean(), 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(binomial.variance(), 2.5, epsilon = 1e-12);

        assert_abs_diff_eq!(binomial.pmf(0), 1.0 / 1024.0, epsilon = 1e-12);
        assert_abs_diff_eq!(binomial.pmf(5), 252.0 / 1024.0, epsilon = 1e-12);
        assert_eq!(binomial.pmf(11), 0.0);
        assert_eq!(binomial.log_pmf(11), f64::NEG_INFINITY);
        assert_abs_diff_eq!(
            binomial.log_pmf(5),
            (252.0_f64 / 1024.0).ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_binomial_edge_cases() {
        let zero_prob = Binomial::new(4, 0.0).unwrap();
        assert_eq!(zero_prob.pmf(0), 1.0);
        assert_eq!(zero_prob.pmf(1), 0.0);
        assert_eq!(zero_prob.sample(&mut rand::thread_rng()), 0);

        let one_prob = Binomial::new(4, 1.0).unwrap();
        assert_eq!(one_prob.pmf(3), 0.0);
        assert_eq!(one_prob.pmf(4), 1.0);
        assert_eq!(one_prob.sample(&mut rand::thread_rng()), 4);

        let zero_trials = Binomial::new(0, 0.75).unwrap();
        assert_eq!(zero_trials.pmf(0), 1.0);
        assert_eq!(zero_trials.pmf(1), 0.0);
        assert_eq!(zero_trials.sample(&mut rand::thread_rng()), 0);
    }

    #[test]
    fn test_binomial_invalid_params() {
        assert!(Binomial::new(10, -0.1).is_err());
        assert!(Binomial::new(10, 1.1).is_err());
        assert!(Binomial::new(10, f64::NAN).is_err());
        assert!(Binomial::new(10, f64::INFINITY).is_err());
    }

    #[test]
    fn test_binomial_sampling_seeded() {
        let binomial = Binomial::new(20, 0.25).unwrap();
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(7);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(7);
        let samples_a: Vec<_> = (0..8).map(|_| binomial.sample(&mut rng_a)).collect();
        let samples_b: Vec<_> = (0..8).map(|_| binomial.sample(&mut rng_b)).collect();

        assert_eq!(samples_a, samples_b);
        assert!(samples_a.iter().all(|&x| x <= binomial.trials()));

        let large_n = Binomial::new(1_000_000, 0.0001).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        assert!(large_n.sample(&mut rng) <= large_n.trials());
    }

    #[test]
    fn test_poisson_distribution() {
        let poisson = Poisson::new(3.0).unwrap();
        assert_eq!(poisson.rate(), 3.0);
        assert_eq!(poisson.mean(), 3.0);
        assert_eq!(poisson.variance(), 3.0);

        assert_abs_diff_eq!(poisson.pmf(0), (-3.0_f64).exp(), epsilon = 1e-12);
        assert_abs_diff_eq!(poisson.log_pmf(0), -3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(poisson.pmf(2), 4.5 * (-3.0_f64).exp(), epsilon = 1e-12);
        assert_abs_diff_eq!(
            poisson.log_pmf(2),
            (4.5_f64 * (-3.0_f64).exp()).ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_poisson_invalid_params() {
        assert!(Poisson::new(0.0).is_err());
        assert!(Poisson::new(-1.0).is_err());
        assert!(Poisson::new(f64::NAN).is_err());
        assert!(Poisson::new(f64::INFINITY).is_err());
        assert!(Poisson::new(MAX_POISSON_SAMPLE_RATE).is_ok());
        assert!(Poisson::new(MAX_POISSON_SAMPLE_RATE * 2.0).is_err());
    }

    #[test]
    fn test_poisson_sampling_seeded() {
        let poisson = Poisson::new(4.0).unwrap();
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(99);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(99);
        let samples_a: Vec<_> = (0..8).map(|_| poisson.sample(&mut rng_a)).collect();
        let samples_b: Vec<_> = (0..8).map(|_| poisson.sample(&mut rng_b)).collect();

        assert_eq!(samples_a, samples_b);
        assert!(samples_a.iter().all(|&x| x < 100));
    }

    #[test]
    fn test_categorical_distribution() {
        let categorical = Categorical::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(categorical.category_count(), 3);
        assert_abs_diff_eq!(categorical.probabilities()[0], 1.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(categorical.probabilities()[1], 2.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(categorical.probabilities()[2], 3.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(categorical.mean(), 4.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(categorical.variance(), 5.0 / 9.0, epsilon = 1e-12);

        assert_abs_diff_eq!(categorical.pmf(0), 1.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(categorical.pmf(1), 2.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(categorical.pmf(2), 3.0 / 6.0, epsilon = 1e-12);
        assert_eq!(categorical.pmf(3), 0.0);
        assert_eq!(categorical.log_pmf(3), f64::NEG_INFINITY);
        assert_abs_diff_eq!(categorical.log_pmf(2), (0.5_f64).ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_categorical_zero_probability_categories() {
        let categorical = Categorical::new(vec![0.0, 5.0, 0.0]).unwrap();
        assert_eq!(categorical.probabilities(), &[0.0, 1.0, 0.0]);
        assert_eq!(categorical.pmf(0), 0.0);
        assert_eq!(categorical.pmf(1), 1.0);
        assert_eq!(categorical.pmf(2), 0.0);
        assert_eq!(categorical.log_pmf(0), f64::NEG_INFINITY);
        assert_eq!(categorical.log_pmf(2), f64::NEG_INFINITY);
        assert_eq!(categorical.mean(), 1.0);
        assert_eq!(categorical.variance(), 0.0);
        assert_eq!(categorical.sample(&mut rand::thread_rng()), 1);
    }

    #[test]
    fn test_categorical_trailing_zero_weight_never_samples_zero_mass_category() {
        let categorical = Categorical::new(vec![1.0, 0.1, 0.0]).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(321);
        let samples: Vec<_> = (0..1_000).map(|_| categorical.sample(&mut rng)).collect();

        assert!(samples.iter().all(|&x| x < 2));
        assert_eq!(categorical.pmf(2), 0.0);
        assert_eq!(categorical.log_pmf(2), f64::NEG_INFINITY);
    }

    #[test]
    fn test_categorical_invalid_params() {
        assert!(Categorical::new(vec![]).is_err());
        assert!(Categorical::new(vec![0.0, 0.0]).is_err());
        assert!(Categorical::new(vec![1.0, -0.1]).is_err());
        assert!(Categorical::new(vec![1.0, f64::NAN]).is_err());
        assert!(Categorical::new(vec![1.0, f64::INFINITY]).is_err());
        assert!(Categorical::new(vec![f64::MAX, f64::MAX]).is_err());
    }

    #[test]
    fn test_categorical_sampling_seeded() {
        let categorical = Categorical::new(vec![0.2, 0.3, 0.5]).unwrap();
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(123);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(123);
        let samples_a: Vec<_> = (0..16).map(|_| categorical.sample(&mut rng_a)).collect();
        let samples_b: Vec<_> = (0..16).map(|_| categorical.sample(&mut rng_b)).collect();

        assert_eq!(samples_a, samples_b);
        assert!(samples_a
            .iter()
            .all(|&x| x < categorical.category_count() as u64));
        assert!(samples_a.contains(&0));
        assert!(samples_a.contains(&1));
        assert!(samples_a.contains(&2));
    }

    #[test]
    fn test_gamma_ln_known_values() {
        assert_abs_diff_eq!(gamma_ln(0.5), 0.5 * PI.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(gamma_ln(1.0e-8), 18.42068073818021, epsilon = 1e-12);
        assert_abs_diff_eq!(gamma_ln(1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(gamma_ln(5.0), 24.0_f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(gamma_ln(10.0), 362_880.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_ln_regression_absolute_log_probabilities() {
        // These analytical checks pin absolute normalizing constants used by
        // distributions that depend on gamma_ln, including non-integer and
        // larger combinatorial cases. Decimal expectations were computed from
        // the closed-form definitions with Python's math.lgamma at double
        // precision so future changes can be independently re-verified.
        let exponential_gamma = Gamma::new(1.0, 1.0).unwrap();
        assert_abs_diff_eq!(exponential_gamma.log_pdf(1.0), -1.0, epsilon = 1e-12);

        let non_integer_gamma = Gamma::new(2.5, 1.5).unwrap();
        assert_abs_diff_eq!(
            non_integer_gamma.log_pdf(1.2),
            -0.797_537_765_011_576_5,
            epsilon = 1e-12
        );

        let uniform_beta = Beta::new(1.0, 1.0).unwrap();
        assert_abs_diff_eq!(uniform_beta.log_pdf(0.5), 0.0, epsilon = 1e-12);

        let fractional_beta = Beta::new(2.5, 3.5).unwrap();
        assert_abs_diff_eq!(
            fractional_beta.log_pdf(0.4),
            0.650_335_112_735_843_9,
            epsilon = 1e-12
        );

        let cauchy = StudentT::new(1.0, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(cauchy.log_pdf(0.0), -PI.ln(), epsilon = 1e-12);

        let scaled_student_t = StudentT::new(5.0, 0.5, 1.25).unwrap();
        assert_abs_diff_eq!(
            scaled_student_t.log_pdf(1.75),
            -1.738_727_810_750_798_4,
            epsilon = 1e-12
        );

        let poisson = Poisson::new(3.0).unwrap();
        assert_abs_diff_eq!(poisson.log_pmf(0), -3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            poisson.log_pmf(4),
            -1.783_604_675_675_505_7,
            epsilon = 1e-12
        );

        let fair_coin = Binomial::new(4, 0.5).unwrap();
        assert_abs_diff_eq!(fair_coin.log_pmf(2), (6.0_f64 / 16.0).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(
            log_binomial_coefficient(5, 2),
            10.0_f64.ln(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            log_binomial_coefficient(10, 3),
            120.0_f64.ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_normal_creation() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        assert_eq!(normal.mean(), 0.0);
        assert_eq!(normal.std_dev(), 1.0);
        assert_eq!(normal.variance(), 1.0);
    }

    #[test]
    fn test_normal_invalid_params() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Normal::new(f64::NAN, 1.0).is_err());
        assert!(Normal::new(0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn test_normal_pdf() {
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Test at mean (should be maximum)
        let pdf_at_mean = normal.pdf(0.0);
        assert_abs_diff_eq!(pdf_at_mean, 1.0 / (2.0 * PI).sqrt(), epsilon = 1e-10);

        // Test symmetry
        assert_abs_diff_eq!(normal.pdf(1.0), normal.pdf(-1.0), epsilon = 1e-10);

        // Test log pdf
        assert_abs_diff_eq!(normal.log_pdf(0.0), pdf_at_mean.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_multivariate_normal() {
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let cov = nalgebra::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

        let mvn = MultivariateNormal::new(mu, cov).unwrap();
        assert_eq!(mvn.dimension(), 2);

        let x = DVector::from_vec(vec![0.0, 0.0]);
        let log_pdf = mvn.log_pdf(&x);
        assert!(log_pdf.is_finite());
    }

    #[test]
    fn test_gamma_distribution() {
        let gamma = Gamma::new(2.0, 1.0).unwrap();
        assert_eq!(gamma.shape(), 2.0);
        assert_eq!(gamma.rate(), 1.0);
        assert_eq!(gamma.mean(), 2.0);
        assert_eq!(gamma.variance(), 2.0);

        // Test PDF is finite for positive values
        assert!(gamma.pdf(1.0) > 0.0);
        assert!(gamma.pdf(1.0).is_finite());
        assert_abs_diff_eq!(gamma.log_pdf(1.0), -1.0, epsilon = 1e-12);

        // Test PDF is zero for non-positive values
        assert_eq!(gamma.pdf(0.0), 0.0);
        assert_eq!(gamma.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_gamma_invalid_params() {
        assert!(Gamma::new(0.0, 1.0).is_err());
        assert!(Gamma::new(1.0, 0.0).is_err());
        assert!(Gamma::new(-1.0, 1.0).is_err());
        assert!(Gamma::new(1.0, -1.0).is_err());
    }

    #[test]
    fn test_beta_distribution() {
        let beta = Beta::new(2.0, 3.0).unwrap();
        assert_eq!(beta.alpha(), 2.0);
        assert_eq!(beta.beta(), 3.0);
        assert_abs_diff_eq!(beta.mean(), 2.0 / 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(beta.variance(), 6.0 / 150.0, epsilon = 1e-10);

        // Test PDF for values in [0, 1]
        assert!(beta.pdf(0.5) > 0.0);
        assert!(beta.pdf(0.5).is_finite());
        assert_abs_diff_eq!(beta.pdf(0.5), 1.5, epsilon = 1e-12);

        // Test PDF is zero outside [0, 1]
        assert_eq!(beta.pdf(0.0), 0.0);
        assert_eq!(beta.pdf(1.0), 0.0);
        assert_eq!(beta.pdf(-0.1), 0.0);
        assert_eq!(beta.pdf(1.1), 0.0);
    }

    #[test]
    fn test_beta_invalid_params() {
        assert!(Beta::new(0.0, 1.0).is_err());
        assert!(Beta::new(1.0, 0.0).is_err());
        assert!(Beta::new(-1.0, 1.0).is_err());
        assert!(Beta::new(1.0, -1.0).is_err());
    }

    #[test]
    fn test_exponential_distribution() {
        let exp = Exponential::new(2.0).unwrap();
        assert_eq!(exp.rate(), 2.0);
        assert_abs_diff_eq!(exp.mean(), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(exp.variance(), 0.25, epsilon = 1e-10);

        // Test PDF for positive values
        assert!(exp.pdf(1.0) > 0.0);
        assert!(exp.pdf(1.0).is_finite());

        // Test PDF is zero for negative values
        assert_eq!(exp.pdf(-1.0), 0.0);

        // Test PDF at zero
        assert_eq!(exp.pdf(0.0), 2.0);
    }

    #[test]
    fn test_exponential_invalid_params() {
        assert!(Exponential::new(0.0).is_err());
        assert!(Exponential::new(-1.0).is_err());
        assert!(Exponential::new(f64::NAN).is_err());
        assert!(Exponential::new(f64::INFINITY).is_err());
    }

    #[test]
    fn test_uniform_distribution() {
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        assert_eq!(uniform.lower_bound(), 0.0);
        assert_eq!(uniform.upper_bound(), 1.0);
        assert_abs_diff_eq!(uniform.mean(), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(uniform.variance(), 1.0 / 12.0, epsilon = 1e-10);

        // Test PDF inside interval
        assert_abs_diff_eq!(uniform.pdf(0.5), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(uniform.pdf(0.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(uniform.pdf(1.0), 1.0, epsilon = 1e-10);

        // Test PDF outside interval
        assert_eq!(uniform.pdf(-0.1), 0.0);
        assert_eq!(uniform.pdf(1.1), 0.0);
    }

    #[test]
    fn test_uniform_invalid_params() {
        assert!(Uniform::new(1.0, 0.0).is_err()); // a >= b
        assert!(Uniform::new(1.0, 1.0).is_err()); // a == b
        assert!(Uniform::new(f64::NAN, 1.0).is_err());
        assert!(Uniform::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn test_student_t_distribution() {
        let t = StudentT::new(3.0, 0.0, 1.0).unwrap();
        assert_eq!(t.degrees_of_freedom(), 3.0);
        assert_eq!(t.location(), 0.0);
        assert_eq!(t.scale(), 1.0);
        assert_eq!(t.mean(), Some(0.0));
        assert!(t.variance().is_some());

        // Test PDF
        assert!(t.pdf(0.0) > 0.0);
        assert!(t.pdf(0.0).is_finite());

        // Test symmetry
        assert_abs_diff_eq!(t.pdf(1.0), t.pdf(-1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_invalid_params() {
        assert!(StudentT::new(0.0, 0.0, 1.0).is_err()); // nu <= 0
        assert!(StudentT::new(1.0, 0.0, 0.0).is_err()); // sigma <= 0
        assert!(StudentT::new(-1.0, 0.0, 1.0).is_err()); // nu < 0
        assert!(StudentT::new(1.0, 0.0, -1.0).is_err()); // sigma < 0
    }

    #[test]
    fn test_student_t_moments() {
        // Test that mean is undefined for nu <= 1
        let t1 = StudentT::new(0.5, 0.0, 1.0).unwrap();
        assert!(t1.mean().is_none());

        // Test that variance is undefined for nu <= 2
        let t2 = StudentT::new(1.5, 0.0, 1.0).unwrap();
        assert!(t2.variance().is_none());
    }
}
