//! CSI CPU Metrics Computation
//!
//! Implements the three core metrics for Chromatic Spiral Indicator:
//! - α (alpha): Rotation rate (Δhue/Δt)
//! - β (beta): Radial decay coefficient
//! - σ² (energy_variance): RGB vector magnitude stability

use super::RGBState;
use std::collections::VecDeque;
use std::f32::consts::PI;

/// Compute rotation rate: Δhue/Δt (rad/frame)
///
/// **Stable Threshold:** α > 0.05 rad/frame
/// **Interpretation:** Processing is Active (Phase is changing)
pub fn compute_rotation_rate(trajectory: &VecDeque<RGBState>) -> f32 {
    if trajectory.len() < 2 {
        return 0.0;
    }

    let mut total_rotation = 0.0;
    let mut total_time = 0.0;

    for window in trajectory.iter().collect::<Vec<_>>().windows(2) {
        let prev = window[0];
        let curr = window[1];

        // Compute hue: H = atan2(b, r)
        let h_prev = prev.b.atan2(prev.r);
        let h_curr = curr.b.atan2(curr.r);

        // Handle wraparound: [-π, π]
        let mut delta_h = h_curr - h_prev;
        if delta_h > PI {
            delta_h -= 2.0 * PI;
        } else if delta_h < -PI {
            delta_h += 2.0 * PI;
        }

        total_rotation += delta_h.abs();
        total_time += curr.timestamp - prev.timestamp;
    }

    if total_time > 0.0 {
        total_rotation / total_time // rad/frame
    } else {
        0.0
    }
}

/// Compute radial decay coefficient from S(t) = S₀e^(-βt)
///
/// **Stable Threshold:** β ∈ [0.01, 0.2]
/// **Interpretation:** Processing is Stable (Energy is balancing/decaying)
pub fn compute_radial_decay(trajectory: &VecDeque<RGBState>) -> f32 {
    if trajectory.len() < 10 {
        return 0.0;
    }

    // Compute saturation: S = √(r² + b²)
    let saturations: Vec<(f64, f64)> = trajectory
        .iter()
        .map(|state| {
            let s = (state.r.powi(2) + state.b.powi(2)).sqrt() as f64;
            (state.timestamp as f64, s)
        })
        .collect();

    // Filter out zero/invalid saturations for log
    let valid_saturations: Vec<(f64, f64)> = saturations
        .into_iter()
        .filter(|(_, s)| *s > 1e-6)
        .collect();

    if valid_saturations.len() < 10 {
        return 0.0;
    }

    // Linear regression on log(S) vs t: log(S) = log(S₀) - βt
    let n = valid_saturations.len() as f64;
    let sum_t: f64 = valid_saturations.iter().map(|(t, _)| t).sum();
    let sum_log_s: f64 = valid_saturations.iter().map(|(_, s)| s.ln()).sum();
    let sum_t_log_s: f64 = valid_saturations
        .iter()
        .map(|(t, s)| t * s.ln())
        .sum();
    let sum_t_sq: f64 = valid_saturations.iter().map(|(t, _)| t.powi(2)).sum();

    // Least squares: β = -(slope)
    let denominator = n * sum_t_sq - sum_t.powi(2);
    if denominator.abs() < 1e-10 {
        return 0.0;
    }

    let slope = (n * sum_t_log_s - sum_t * sum_log_s) / denominator;

    (-slope as f32).max(0.0) // Negative slope → positive decay rate
}

/// Compute energy variance: variance of ||C(t)|| = √(r² + g² + b²)
///
/// **Stable Threshold:** < 3%
/// **Interpretation:** Coherence is maintained (Total energy is stable)
pub fn compute_energy_variance(trajectory: &VecDeque<RGBState>) -> f32 {
    if trajectory.is_empty() {
        return 0.0;
    }

    // Compute energy: E(t) = ||C(t)|| = √(r² + g² + b²)
    let energies: Vec<f32> = trajectory
        .iter()
        .map(|state| (state.r.powi(2) + state.g.powi(2) + state.b.powi(2)).sqrt())
        .collect();

    let mean = energies.iter().sum::<f32>() / energies.len() as f32;

    if mean < 1e-6 {
        return 100.0; // High variance if mean is near zero
    }

    let variance = energies
        .iter()
        .map(|e| (e - mean).powi(2))
        .sum::<f32>()
        / energies.len() as f32;

    // Return as percentage of mean
    (variance.sqrt() / mean) * 100.0
}

/// Helper: Compute saturation S = √(r² + b²)
pub fn compute_saturation(state: &RGBState) -> f32 {
    (state.r.powi(2) + state.b.powi(2)).sqrt()
}

/// Helper: Compute hue H = atan2(b, r)
pub fn compute_hue(state: &RGBState) -> f32 {
    state.b.atan2(state.r)
}

/// Helper: Compute energy E = √(r² + g² + b²)
pub fn compute_energy(state: &RGBState) -> f32 {
    (state.r.powi(2) + state.g.powi(2) + state.b.powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_spiral_trajectory(n: usize) -> VecDeque<RGBState> {
        let mut trajectory = VecDeque::new();

        for i in 0..n {
            let t = i as f32;
            let theta = t * 0.1; // Rotation
            let radius = (1.0 - t / n as f32) * 0.8; // Decay

            trajectory.push_back(RGBState {
                r: radius * theta.cos(),
                g: 0.5,
                b: radius * theta.sin(),
                timestamp: t,
                coherence: 0.9,
            });
        }

        trajectory
    }

    #[test]
    fn test_rotation_rate() {
        let trajectory = create_spiral_trajectory(50);
        let alpha = compute_rotation_rate(&trajectory);

        // Should detect rotation
        assert!(alpha > 0.05, "Expected α > 0.05, got {}", alpha);
    }

    #[test]
    fn test_radial_decay() {
        let trajectory = create_spiral_trajectory(50);
        let beta = compute_radial_decay(&trajectory);

        // Should detect decay
        assert!(
            beta >= 0.01 && beta <= 0.5,
            "Expected β ∈ [0.01, 0.5], got {}",
            beta
        );
    }

    #[test]
    fn test_energy_variance_stable() {
        let mut trajectory = VecDeque::new();

        // Stable energy
        for i in 0..50 {
            trajectory.push_back(RGBState {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                timestamp: i as f32,
                coherence: 1.0,
            });
        }

        let variance = compute_energy_variance(&trajectory);

        // Should be very low
        assert!(variance < 1.0, "Expected σ² < 1%, got {}%", variance);
    }

    #[test]
    fn test_energy_variance_unstable() {
        let mut trajectory = VecDeque::new();

        // Unstable energy
        for i in 0..50 {
            let amplitude = if i % 2 == 0 { 0.2 } else { 0.8 };
            trajectory.push_back(RGBState {
                r: amplitude,
                g: amplitude,
                b: amplitude,
                timestamp: i as f32,
                coherence: 0.5,
            });
        }

        let variance = compute_energy_variance(&trajectory);

        // Should be high
        assert!(variance > 10.0, "Expected σ² > 10%, got {}%", variance);
    }
}
