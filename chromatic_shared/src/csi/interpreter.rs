//! CSI Pattern Interpreter
//!
//! Classifies spiral patterns and prescribes diagnostic actions

use super::{CSIMetrics, DiagnosticAction, LogLevel, RGBState, SpiralPattern};
use std::collections::VecDeque;

/// CSI pattern interpreter
pub struct CSIInterpreter {
    /// History for periodicity detection
    metric_history: VecDeque<(f32, f32, f32)>, // (alpha, beta, energy_variance)

    /// Max history length for pattern detection
    max_history: usize,
}

impl CSIInterpreter {
    pub fn new() -> Self {
        Self {
            metric_history: VecDeque::with_capacity(20),
            max_history: 20,
        }
    }

    /// Classify spiral pattern based on metrics
    pub fn classify_pattern(&self, metrics: &CSIMetrics) -> SpiralPattern {
        let alpha = metrics.alpha;
        let beta = metrics.beta;
        let variance = metrics.energy_variance;

        // 1. Stable Processing: Clear inward spiral
        //    α > 0.05, β ∈ [0.01, 0.2], σ² < 3%
        if alpha > 0.05 && (0.01..=0.2).contains(&beta) && variance < 3.0 {
            return SpiralPattern::StableProcessing;
        }

        // 2. Periodic Resonance: Oscillating loops
        //    Detected via periodicity in metrics + low variance
        if self.detect_periodicity() && variance < 5.0 && alpha > 0.03 {
            return SpiralPattern::PeriodicResonance;
        }

        // 3. Over-Excitation: Expanding spiral
        //    β < 0 (negative decay = expansion) OR high variance
        if beta < 0.0 || variance > 10.0 {
            return SpiralPattern::OverExcitation;
        }

        // 4. System Fault: Flat line or random walk
        //    α < 0.01 (no rotation) OR very high variance
        if alpha < 0.01 || variance > 20.0 {
            return SpiralPattern::SystemFault;
        }

        // 5. Indeterminate
        SpiralPattern::Indeterminate
    }

    /// Prescribe diagnostic action based on pattern
    pub fn prescribe_action(&self, pattern: SpiralPattern, metrics: &CSIMetrics) -> DiagnosticAction {
        match pattern {
            SpiralPattern::StableProcessing => DiagnosticAction::Log {
                message: format!(
                    "CSI: Stable processing | α={:.3} rad/frame, β={:.3}, σ²={:.2}%",
                    metrics.alpha, metrics.beta, metrics.energy_variance
                ),
                level: LogLevel::Info,
            },

            SpiralPattern::PeriodicResonance => DiagnosticAction::SonifySpiral {
                message: format!(
                    "CSI: Periodic equilibrium - ideal resonance | α={:.3} rad/frame",
                    metrics.alpha
                ),
            },

            SpiralPattern::OverExcitation => DiagnosticAction::TriggerDiagnostic {
                message: format!(
                    "CSI: Over-excitation | β={:.3}, σ²={:.2}%",
                    metrics.beta, metrics.energy_variance
                ),
                check: "UMS normalization gain".to_string(),
            },

            SpiralPattern::SystemFault => DiagnosticAction::TriggerError {
                message: format!(
                    "CSI: System fault - channel inactive or unbalanced | α={:.3} rad/frame, σ²={:.2}%",
                    metrics.alpha, metrics.energy_variance
                ),
                check: "System integrity".to_string(),
            },

            SpiralPattern::Indeterminate => DiagnosticAction::Continue,
        }
    }

    /// Record metrics for periodicity detection
    pub fn record_metrics(&mut self, metrics: &CSIMetrics) {
        if self.metric_history.len() >= self.max_history {
            self.metric_history.pop_front();
        }

        self.metric_history.push_back((
            metrics.alpha,
            metrics.beta,
            metrics.energy_variance,
        ));
    }

    /// Detect periodicity in metric history using autocorrelation
    fn detect_periodicity(&self) -> bool {
        if self.metric_history.len() < 10 {
            return false;
        }

        // Simple periodicity check: detect oscillations in alpha
        let alphas: Vec<f32> = self
            .metric_history
            .iter()
            .map(|(alpha, _, _)| *alpha)
            .collect();

        // Count zero-crossings (sign changes around mean)
        let mean_alpha: f32 = alphas.iter().sum::<f32>() / alphas.len() as f32;
        let mut zero_crossings = 0;

        for window in alphas.windows(2) {
            let prev_centered = window[0] - mean_alpha;
            let curr_centered = window[1] - mean_alpha;

            if prev_centered.signum() != curr_centered.signum() {
                zero_crossings += 1;
            }
        }

        // Periodic if we have multiple zero-crossings (oscillations)
        zero_crossings >= 4
    }

    /// Get metric history
    pub fn metric_history(&self) -> &VecDeque<(f32, f32, f32)> {
        &self.metric_history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.metric_history.clear();
    }
}

impl Default for CSIInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_processing_classification() {
        let interpreter = CSIInterpreter::new();

        let metrics = CSIMetrics {
            alpha: 0.08,   // > 0.05
            beta: 0.15,    // ∈ [0.01, 0.2]
            energy_variance: 2.0, // < 3%
            pattern: SpiralPattern::Indeterminate,
        };

        let pattern = interpreter.classify_pattern(&metrics);
        assert_eq!(pattern, SpiralPattern::StableProcessing);
    }

    #[test]
    fn test_over_excitation_classification() {
        let interpreter = CSIInterpreter::new();

        let metrics = CSIMetrics {
            alpha: 0.10,
            beta: -0.05,  // Negative = expansion!
            energy_variance: 8.0,
            pattern: SpiralPattern::Indeterminate,
        };

        let pattern = interpreter.classify_pattern(&metrics);
        assert_eq!(pattern, SpiralPattern::OverExcitation);
    }

    #[test]
    fn test_system_fault_classification() {
        let interpreter = CSIInterpreter::new();

        let metrics = CSIMetrics {
            alpha: 0.005,  // < 0.01 (no rotation)
            beta: 0.05,
            energy_variance: 25.0, // > 20%
            pattern: SpiralPattern::Indeterminate,
        };

        let pattern = interpreter.classify_pattern(&metrics);
        assert_eq!(pattern, SpiralPattern::SystemFault);
    }

    #[test]
    fn test_periodicity_detection() {
        let mut interpreter = CSIInterpreter::new();

        // Simulate oscillating alpha values
        for i in 0..15 {
            let alpha = 0.1 + 0.05 * ((i as f32 * 0.5).sin());
            let metrics = CSIMetrics {
                alpha,
                beta: 0.1,
                energy_variance: 2.0,
                pattern: SpiralPattern::Indeterminate,
            };
            interpreter.record_metrics(&metrics);
        }

        assert!(interpreter.detect_periodicity());
    }

    #[test]
    fn test_prescribe_action_stable() {
        let interpreter = CSIInterpreter::new();

        let metrics = CSIMetrics {
            alpha: 0.08,
            beta: 0.15,
            energy_variance: 2.0,
            pattern: SpiralPattern::StableProcessing,
        };

        let action = interpreter.prescribe_action(SpiralPattern::StableProcessing, &metrics);

        match action {
            DiagnosticAction::Log { message, level } => {
                assert!(message.contains("Stable processing"));
                assert!(matches!(level, LogLevel::Info));
            }
            _ => panic!("Expected Log action"),
        }
    }

    #[test]
    fn test_prescribe_action_resonance() {
        let interpreter = CSIInterpreter::new();

        let metrics = CSIMetrics {
            alpha: 0.12,
            beta: 0.08,
            energy_variance: 1.5,
            pattern: SpiralPattern::PeriodicResonance,
        };

        let action = interpreter.prescribe_action(SpiralPattern::PeriodicResonance, &metrics);

        match action {
            DiagnosticAction::SonifySpiral { message } => {
                assert!(message.contains("resonance"));
            }
            _ => panic!("Expected SonifySpiral action"),
        }
    }
}
