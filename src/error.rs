//! Error types for the bayes-rs library

use thiserror::Error;

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, BayesError>;

/// Main error type for the library
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BayesError {
    /// Invalid parameter values
    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },

    /// Numerical computation errors
    #[error("Numerical error: {message}")]
    NumericalError { message: String },

    /// Convergence failure
    #[error("Convergence failed: {message}")]
    ConvergenceError { message: String },

    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    /// Sampling errors
    #[error("Sampling error: {message}")]
    SamplingError { message: String },
}

impl BayesError {
    /// Create a new InvalidParameter error
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            message: message.into(),
        }
    }

    /// Create a new NumericalError
    pub fn numerical_error(message: impl Into<String>) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// Create a new ConvergenceError
    pub fn convergence_error(message: impl Into<String>) -> Self {
        Self::ConvergenceError {
            message: message.into(),
        }
    }

    /// Create a new DimensionMismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a new InvalidConfiguration error
    pub fn invalid_configuration(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Create a new SamplingError
    pub fn sampling_error(message: impl Into<String>) -> Self {
        Self::SamplingError {
            message: message.into(),
        }
    }
}
