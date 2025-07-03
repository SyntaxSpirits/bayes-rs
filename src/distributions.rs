//! Statistical distributions for Bayesian inference

use crate::error::{BayesError, Result};
use nalgebra::DVector;
use std::f64::consts::PI;

/// Trait for probability distributions
pub trait Distribution {
    /// Compute the log probability density function
    fn log_pdf(&self, x: f64) -> f64;

    /// Compute the probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }
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

/// Approximation of log gamma function using Stirling's approximation
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // For small values, use a simple approximation
    // For production use, consider using a more accurate implementation
    if x < 1.0 {
        gamma_ln(x + 1.0) - x.ln()
    } else {
        // Stirling's approximation
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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
