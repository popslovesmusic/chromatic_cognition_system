//! Training pipeline for WGSL code generation models

use crate::config::TrainingConfig;
use crate::dataset::WGSLDataset;
use crate::model::CodeGenerationModel;
use crate::tokenizer::WGSLTokenizer;
use std::time::Instant;

/// Training orchestrator
pub struct Trainer {
    pub config: TrainingConfig,
}

impl Trainer {
    /// Create a new trainer with the given configuration
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train a model
    pub fn train(
        &mut self,
        model: &mut CodeGenerationModel,
        train_dataset: &WGSLDataset,
        val_dataset: &WGSLDataset,
        tokenizer: &WGSLTokenizer,
    ) -> crate::Result<TrainingResults> {
        tracing::info!("Starting training for {} epochs", self.config.num_epochs);
        tracing::info!("Training examples: {}", train_dataset.len());
        tracing::info!("Validation examples: {}", val_dataset.len());
        tracing::info!("Model parameters: {}", model.num_parameters());

        let start_time = Instant::now();
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.num_epochs {
            let epoch_start = Instant::now();

            // Training phase
            let train_loss = self.train_epoch(model, train_dataset, tokenizer)?;

            // Validation phase
            let val_loss = self.validate(model, val_dataset, tokenizer)?;

            let epoch_time = epoch_start.elapsed().as_secs_f64();

            tracing::info!(
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}, time={:.2}s",
                epoch + 1,
                self.config.num_epochs,
                train_loss,
                val_loss,
                epoch_time
            );

            // Early stopping check
            if val_loss < best_loss {
                best_loss = val_loss;
                patience_counter = 0;
                tracing::info!("New best validation loss: {:.4}", best_loss);
            } else if self.config.early_stopping {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    tracing::info!("Early stopping triggered after {} epochs", epoch + 1);
                    break;
                }
            }

            // Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 {
                tracing::info!("Checkpoint would be saved at epoch {}", epoch + 1);
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        tracing::info!("Training complete! Total time: {:.2}s", total_time);

        Ok(TrainingResults {
            final_loss: best_loss,
            best_loss,
            epochs_completed: self.config.num_epochs,
            training_time_secs: total_time,
        })
    }

    fn train_epoch(
        &self,
        model: &mut CodeGenerationModel,
        dataset: &WGSLDataset,
        tokenizer: &WGSLTokenizer,
    ) -> crate::Result<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // Simple sequential processing (no batching for now)
        for example in &dataset.examples {
            // Tokenize input and target
            let input_tokens = tokenizer.encode_text(&example.natural_language);
            let target_tokens = tokenizer.encode_text(&example.wgsl_code);

            if target_tokens.is_empty() {
                continue;
            }

            // Forward pass
            let logits = model.forward(&input_tokens);

            // Compute simple cross-entropy loss (simplified)
            let target_id = target_tokens[0]; // Use first token as target
            let loss = self.compute_loss(&logits, target_id);

            total_loss += loss;
            num_batches += 1;
        }

        if num_batches == 0 {
            return Ok(1.0);
        }

        Ok(total_loss / num_batches as f32)
    }

    fn validate(
        &self,
        model: &CodeGenerationModel,
        dataset: &WGSLDataset,
        tokenizer: &WGSLTokenizer,
    ) -> crate::Result<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for example in &dataset.examples {
            let input_tokens = tokenizer.encode_text(&example.natural_language);
            let target_tokens = tokenizer.encode_text(&example.wgsl_code);

            if target_tokens.is_empty() {
                continue;
            }

            let logits = model.forward(&input_tokens);
            let target_id = target_tokens[0];
            let loss = self.compute_loss(&logits, target_id);

            total_loss += loss;
            num_batches += 1;
        }

        if num_batches == 0 {
            return Ok(1.0);
        }

        Ok(total_loss / num_batches as f32)
    }

    fn compute_loss(&self, logits: &[f32], target_id: usize) -> f32 {
        // Simple cross-entropy loss
        if target_id >= logits.len() {
            return 1.0;
        }

        // Softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_prob = logits[target_id] - max_logit - exp_sum.ln();

        -log_prob
    }
}

/// Training results summary
#[derive(Debug, Clone)]
pub struct TrainingResults {
    pub final_loss: f32,
    pub best_loss: f32,
    pub epochs_completed: usize,
    pub training_time_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingConfig;

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig {
            num_epochs: 10,
            batch_size: 16,
            learning_rate: 0.001,
            optimizer: "adamw".to_string(),
            early_stopping: false,
            early_stopping_patience: 10,
            gradient_clip_norm: 1.0,
            save_every: 5,
        };

        let trainer = Trainer::new(config);
        assert_eq!(trainer.config.num_epochs, 10);
    }
}
