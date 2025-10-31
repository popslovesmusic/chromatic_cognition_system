//! CSI Integration for Chromatic Core Operations
//!
//! This module integrates the Chromatic Spiral Indicator (CSI) with core operations,
//! enabling real-time cognitive health monitoring.

use chromatic_shared::{ChromaticSpiralIndicator, RGBState};
use crate::tensor::{ChromaticTensor, TensorStatistics};
use std::sync::Mutex;

// Global CSI instance (thread-safe)
lazy_static::lazy_static! {
    static ref GLOBAL_CSI: Mutex<ChromaticSpiralIndicator> =
        Mutex::new(ChromaticSpiralIndicator::new(100));
}

/// Extract RGB state from tensor statistics
pub fn extract_rgb_state(stats: &TensorStatistics, operation_count: usize) -> RGBState {
    RGBState {
        r: stats.mean_rgb[0],
        g: stats.mean_rgb[1],
        b: stats.mean_rgb[2],
        timestamp: operation_count as f32,
        coherence: stats.mean_certainty,
    }
}

/// Observe operation in global CSI
pub fn observe_operation(stats: &TensorStatistics, operation_count: usize) {
    if let Ok(mut csi) = GLOBAL_CSI.lock() {
        let state = extract_rgb_state(stats, operation_count);
        csi.observe(state);
    }
}

/// Get current CSI metrics
pub fn get_csi_metrics() -> Option<chromatic_shared::CSIMetrics> {
    GLOBAL_CSI.lock().ok().map(|csi| csi.compute_metrics())
}

/// Diagnose current state
pub fn diagnose() -> Option<chromatic_shared::DiagnosticAction> {
    GLOBAL_CSI.lock().ok().map(|csi| csi.diagnose())
}

/// Clear CSI trajectory
pub fn reset_csi() {
    if let Ok(mut csi) = GLOBAL_CSI.lock() {
        csi.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csi_observation() {
        reset_csi();

        let stats = TensorStatistics {
            mean_rgb: [0.5, 0.6, 0.7],
            mean_certainty: 0.9,
            ..Default::default()
        };

        observe_operation(&stats, 0);
        let metrics = get_csi_metrics();

        assert!(metrics.is_some());
    }
}
