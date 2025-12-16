//! ZipEnhancer - High-Performance Audio Denoise Library
//!
//! Uses multi-session parallel ONNX inference for optimal performance.

pub mod audio;
pub mod config;
pub mod error;
pub mod onnx;
pub mod processing;

pub use config::{Config, Args};
pub use error::{ZipEnhancerError, Result};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

pub fn init_logging(verbose: bool) {
    unsafe {
        std::env::set_var("RUST_LOG", if verbose { "debug" } else { "info" });
    }
    env_logger::Builder::from_env("RUST_LOG")
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();
}

pub fn get_library_info() -> LibraryInfo {
    LibraryInfo {
        name: NAME.to_string(),
        version: VERSION.to_string(),
        description: DESCRIPTION.to_string(),
    }
}

#[derive(Debug, Clone)]
pub struct LibraryInfo {
    pub name: String,
    pub version: String,
    pub description: String,
}

impl std::fmt::Display for LibraryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} v{} - {}", self.name, self.version, self.description)
    }
}
