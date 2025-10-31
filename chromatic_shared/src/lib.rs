//! Chromatic Shared Library
//!
//! Shared types and utilities for the Chromatic Cognition System.
//!
//! This library provides:
//! - Chromatic Spiral Indicator (CSI) for real-time cognitive health monitoring
//! - Shared chromatic tensor types
//! - WGSL validation utilities
//! - Common type definitions

pub mod csi;

// Re-export commonly used types
pub use csi::{
    ChromaticSpiralIndicator, CSIMetrics, CSIObserver, DiagnosticAction, LogLevel, RGBState,
    SpiralPattern,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
