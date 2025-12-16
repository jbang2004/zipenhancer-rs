//! Error Types - Simplified

use std::fmt;

/// Main error type
#[derive(Debug, Clone)]
pub enum ZipEnhancerError {
    Audio { message: String },
    Onnx { message: String },
    Config { message: String },
    Io { message: String },
    Processing { message: String },
}

impl fmt::Display for ZipEnhancerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Audio { message } => write!(f, "Audio error: {}", message),
            Self::Onnx { message } => write!(f, "ONNX error: {}", message),
            Self::Config { message } => write!(f, "Config error: {}", message),
            Self::Io { message } => write!(f, "IO error: {}", message),
            Self::Processing { message } => write!(f, "Processing error: {}", message),
        }
    }
}

impl std::error::Error for ZipEnhancerError {}

impl ZipEnhancerError {
    pub fn audio<S: Into<String>>(msg: S) -> Self { Self::Audio { message: msg.into() } }
    pub fn onnx<S: Into<String>>(msg: S) -> Self { Self::Onnx { message: msg.into() } }
    pub fn config<S: Into<String>>(msg: S) -> Self { Self::Config { message: msg.into() } }
    pub fn io<S: Into<String>>(msg: S) -> Self { Self::Io { message: msg.into() } }
    pub fn processing<S: Into<String>>(msg: S) -> Self { Self::Processing { message: msg.into() } }
}

pub type Result<T> = std::result::Result<T, ZipEnhancerError>;

impl From<std::io::Error> for ZipEnhancerError {
    fn from(err: std::io::Error) -> Self { Self::io(err.to_string()) }
}

impl From<onnxruntime::OrtError> for ZipEnhancerError {
    fn from(err: onnxruntime::OrtError) -> Self { Self::onnx(format!("ORT: {}", err)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = ZipEnhancerError::audio("test");
        assert!(e.to_string().contains("Audio"));
    }
}
