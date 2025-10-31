//! CSI Demo - Chromatic Spiral Indicator Demonstration
//!
//! This example demonstrates real-time CSI monitoring of chromatic operations.

use chromatic_core::{ChromaticTensor, csi_integration, mix};

fn main() {
    println!("🎨 Chromatic Spiral Indicator (CSI) Demo");
    println!("=========================================\n");

    // Reset CSI
    csi_integration::reset_csi();

    println!("Performing 50 mix operations with evolving colors...\n");

    // Create initial tensors
    let mut tensor_a = ChromaticTensor::from_seed(42, 12, 12, 3);
    let mut tensor_b = ChromaticTensor::from_seed(100, 12, 12, 3);

    for i in 0..50 {
        // Perform mix operation
        let result = mix(&tensor_a, &tensor_b);

        // Observe in CSI
        let stats = result.statistics();
        csi_integration::observe_operation(&stats, i);

        // Get metrics every 10 iterations
        if (i + 1) % 10 == 0 {
            if let Some(metrics) = csi_integration::get_csi_metrics() {
                println!("📊 Iteration {}/50:", i + 1);
                println!("   α (rotation):  {:.4} rad/frame", metrics.alpha);
                println!("   β (decay):     {:.4}", metrics.beta);
                println!("   σ² (variance): {:.2}%", metrics.energy_variance);
                println!("   Pattern:       {:?}\n", metrics.pattern);
            }

            // Check diagnostic action
            if let Some(action) = csi_integration::diagnose() {
                match action {
                    chromatic_shared::DiagnosticAction::Log { message, level } => {
                        println!("   ✅ {:?}: {}\n", level, message);
                    }
                    chromatic_shared::DiagnosticAction::SonifySpiral { message } => {
                        println!("   🔊 {}\n", message);
                    }
                    chromatic_shared::DiagnosticAction::TriggerDiagnostic { message, check } => {
                        println!("   ⚠️  {}", message);
                        println!("      Check: {}\n", check);
                    }
                    chromatic_shared::DiagnosticAction::TriggerError { message, check } => {
                        println!("   ❌ {}", message);
                        println!("      Check: {}\n", check);
                    }
                    chromatic_shared::DiagnosticAction::Continue => {}
                }
            }
        }

        // Evolve tensors for next iteration
        tensor_a = result.clone();
        tensor_b = result;
    }

    println!("═══════════════════════════════════════");
    println!("🎉 CSI Demo Complete!");
    println!("\nFinal Metrics:");

    if let Some(metrics) = csi_integration::get_csi_metrics() {
        println!("   α (rotation):  {:.4} rad/frame", metrics.alpha);
        println!("   β (decay):     {:.4}", metrics.beta);
        println!("   σ² (variance): {:.2}%", metrics.energy_variance);
        println!("   Pattern:       {:?}", metrics.pattern);
    }
}
