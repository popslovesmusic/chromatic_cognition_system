//! Chromatic Spiral Indicator (CSI) - Real-time Cognitive Health Monitor
//!
//! The CSI acts as an operational marker of active, balanced chromatic cognition.
//! It monitors RGB state trajectories and computes metrics to detect processing patterns.

pub mod metrics;
pub mod interpreter;

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// RGB state at a specific timestamp
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RGBState {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub timestamp: f32,
    pub coherence: f32,
}

/// CSI computed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CSIMetrics {
    /// Rotation rate: Δhue/Δt (rad/frame)
    /// Stable threshold: α > 0.05 rad/frame
    pub alpha: f32,

    /// Radial decay coefficient from S(t) = S₀e^(-βt)
    /// Stable threshold: β ∈ [0.01, 0.2]
    pub beta: f32,

    /// Energy variance (RGB vector magnitude stability)
    /// Stable threshold: < 3%
    pub energy_variance: f32,

    /// Classified pattern
    pub pattern: SpiralPattern,
}

/// Spiral pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum SpiralPattern {
    /// Clear inward spiral: α > 0.05, β ∈ [0.01, 0.2], σ² < 3%
    StableProcessing = 0,

    /// Oscillating loops: periodic equilibrium
    PeriodicResonance = 1,

    /// Expanding spiral: over-excitation
    OverExcitation = 2,

    /// Flat line or random walk: system fault
    SystemFault = 3,

    /// Insufficient data
    Indeterminate = 4,
}

/// Diagnostic action prescribed by CSI
#[derive(Debug, Clone)]
pub enum DiagnosticAction {
    /// Log metrics for analysis
    Log { message: String, level: LogLevel },

    /// Enable sonic coupling (APM)
    SonifySpiral { message: String },

    /// Trigger diagnostic check
    TriggerDiagnostic { message: String, check: String },

    /// Trigger error and system integrity check
    TriggerError { message: String, check: String },

    /// Continue normal operation
    Continue,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// Chromatic Spiral Indicator - Main structure
pub struct ChromaticSpiralIndicator {
    /// Time-series buffer of RGB states (ring buffer)
    trajectory: VecDeque<RGBState>,

    /// Maximum trajectory length
    max_trajectory_len: usize,

    /// Pattern interpreter
    interpreter: interpreter::CSIInterpreter,
}

impl ChromaticSpiralIndicator {
    /// Create new CSI with specified trajectory buffer size
    pub fn new(max_trajectory_len: usize) -> Self {
        Self {
            trajectory: VecDeque::with_capacity(max_trajectory_len),
            max_trajectory_len,
            interpreter: interpreter::CSIInterpreter::new(),
        }
    }

    /// Observe new RGB state
    pub fn observe(&mut self, state: RGBState) {
        if self.trajectory.len() >= self.max_trajectory_len {
            self.trajectory.pop_front();
        }
        self.trajectory.push_back(state);
    }

    /// Compute current metrics
    pub fn compute_metrics(&self) -> CSIMetrics {
        let alpha = metrics::compute_rotation_rate(&self.trajectory);
        let beta = metrics::compute_radial_decay(&self.trajectory);
        let energy_variance = metrics::compute_energy_variance(&self.trajectory);

        let mut metrics = CSIMetrics {
            alpha,
            beta,
            energy_variance,
            pattern: SpiralPattern::Indeterminate,
        };

        metrics.pattern = self.interpreter.classify_pattern(&metrics);
        metrics
    }

    /// Diagnose and prescribe action
    pub fn diagnose(&self) -> DiagnosticAction {
        let metrics = self.compute_metrics();
        self.interpreter.prescribe_action(metrics.pattern, &metrics)
    }

    /// Get trajectory buffer
    pub fn trajectory(&self) -> &VecDeque<RGBState> {
        &self.trajectory
    }

    /// Get trajectory length
    pub fn trajectory_len(&self) -> usize {
        self.trajectory.len()
    }

    /// Clear trajectory buffer
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }
}

/// Trait for CSI observation
pub trait CSIObserver {
    fn observe(&mut self, rgb: RGBState);
    fn compute_metrics(&self) -> CSIMetrics;
    fn diagnose(&self) -> DiagnosticAction;
}

impl CSIObserver for ChromaticSpiralIndicator {
    fn observe(&mut self, rgb: RGBState) {
        self.observe(rgb);
    }

    fn compute_metrics(&self) -> CSIMetrics {
        self.compute_metrics()
    }

    fn diagnose(&self) -> DiagnosticAction {
        self.diagnose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csi_creation() {
        let csi = ChromaticSpiralIndicator::new(100);
        assert_eq!(csi.trajectory_len(), 0);
    }

    #[test]
    fn test_observe() {
        let mut csi = ChromaticSpiralIndicator::new(10);

        for i in 0..5 {
            csi.observe(RGBState {
                r: 1.0,
                g: 0.5,
                b: 0.5,
                timestamp: i as f32,
                coherence: 0.9,
            });
        }

        assert_eq!(csi.trajectory_len(), 5);
    }

    #[test]
    fn test_ring_buffer() {
        let mut csi = ChromaticSpiralIndicator::new(3);

        for i in 0..5 {
            csi.observe(RGBState {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                timestamp: i as f32,
                coherence: 1.0,
            });
        }

        // Should only keep last 3
        assert_eq!(csi.trajectory_len(), 3);
        assert_eq!(csi.trajectory()[0].timestamp, 2.0);
    }
}
