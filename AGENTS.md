# Chromatic Cognition System - Master Agent Guide

**Version**: 2.0.0
**Last Updated**: 2025-10-31
**Current Status**: Phase 3 Complete - CSI Integration Achieved

---

## 🎯 **Mission Overview**

This document provides the complete roadmap for developing the Chromatic Cognition System - a self-generating cognitive architecture with real-time health monitoring. Each phase is designed to be executed by specialized AI agents (or human developers) following a structured progression.

### **System Goals**

1. **Chromatic Semantic Archive (CSA)** - Store knowledge as RGB tensors with ΔE₉₄ ≤ 1.0×10⁻³ fidelity
2. **WGSL Shader Generation** - Generate operational code that the system needs
3. **Real-Time Health Monitoring** - CSI (Chromatic Spiral Indicator) diagnostic system
4. **Self-Generation Loop** - System produces its own computational tools
5. **Medical Imaging Application** - Anomaly detection via spectral dissonance

---

## 📊 **Phase Completion Status**

| Phase | Focus | Status | Tests | Notes |
|-------|-------|--------|-------|-------|
| **Phase 1** | Workspace Setup + CSI Foundation | ✅ Complete | 18/18 | CSI module implemented |
| **Phase 2** | Project Migration + Integration | ✅ Complete | - | Both projects in workspace |
| **Phase 3** | CSI Integration into Core | ✅ Complete | 233/236 | Global CSI + demo |
| **Phase 4** | Training Loop + GPU Renderer | 🚧 Next | - | CSI adaptive control |
| **Phase 5** | Dream Pool + CSI Filtering | 📋 Planned | - | Selective archival |
| **Phase 6** | Self-Generation Loop | 📋 Planned | - | Core generates shaders |
| **Phase 7** | Backpropagation Implementation | 📋 Planned | - | Full training capability |
| **Phase 8** | Medical Imaging Application | 📋 Planned | - | Spectral dissonance |
| **Phase 9** | Production Hardening | 📋 Planned | - | Error handling + logging |
| **Phase 10** | Documentation + Deployment | 📋 Planned | - | Release preparation |

---

## 🚀 **Phase 4: Training Loop Integration + GPU Renderer**

**Goal**: Integrate CSI into the wgsl_trainer training loop with adaptive learning rate control, and implement real-time GPU visualization.

**Estimated Effort**: 1-2 sessions
**Prerequisites**: Phase 3 complete, 233+ tests passing

### **Subphase 4.A: CSI Training Integration**

**Focus**: Add CSI observation to the training loop with metric mapping.

**Agent Instructions**:

1. **Create Training CSI Module** (`wgsl_trainer/src/training/csi_monitor.rs`)
   ```rust
   use chromatic_shared::{ChromaticSpiralIndicator, RGBState, DiagnosticAction};

   pub struct TrainingMonitor {
       csi: ChromaticSpiralIndicator,
       iteration: usize,
   }

   impl TrainingMonitor {
       pub fn observe_step(&mut self, loss: f32, accuracy: f32, grad_norm: f32) {
           let state = RGBState {
               r: loss.min(1.0),  // Normalize to [0,1]
               g: accuracy,
               b: (grad_norm / 10.0).min(1.0),  // Scale appropriately
               timestamp: self.iteration as f32,
               coherence: 1.0 - loss,  // Higher coherence when loss is low
           };
           self.csi.observe(state);
           self.iteration += 1;
       }

       pub fn check_health(&self) -> DiagnosticAction {
           self.csi.diagnose()
       }
   }
   ```

2. **Integrate into Training Loop** (`wgsl_trainer/src/training/mod.rs`)
   - Add `TrainingMonitor` field to `Trainer` struct
   - Call `monitor.observe_step()` after each training iteration
   - Implement adaptive learning rate based on `DiagnosticAction`
   - Log CSI metrics every N iterations

3. **Adaptive Learning Rate Control**
   ```rust
   match self.monitor.check_health() {
       DiagnosticAction::TriggerDiagnostic { message, check } => {
           // OverExcitation detected - reduce learning rate
           self.config.learning_rate *= 0.9;
           warn!("CSI: {} - Damping LR to {:.6}", message, self.config.learning_rate);
       },
       DiagnosticAction::TriggerError { message, check } => {
           // SystemFault - critical issue
           error!("CSI: {} - Check: {}", message, check);
           return Err(TrainingError::SystemFault(message));
       },
       _ => {}
   }
   ```

4. **Add Training Metrics Display**
   - Show α (rotation rate) - indicates training activity
   - Show β (decay) - indicates convergence stability
   - Show σ² (variance) - indicates consistency
   - Display pattern classification (StableProcessing, OverExcitation, etc.)

**Success Criteria**:
- CSI observes every training step
- Adaptive LR control prevents divergence
- Training metrics logged with CSI status
- Pattern classification matches training behavior

---

### **Subphase 4.B: GPU Renderer Implementation**

**Focus**: Create real-time CSI visualization using the extracted WGSL shader.

**Agent Instructions**:

1. **Create Renderer Module** (`chromatic_shared/src/csi/renderer.rs`)
   ```rust
   pub struct CSIRenderer {
       device: wgpu::Device,
       queue: wgpu::Queue,
       pipeline: wgpu::RenderPipeline,
       uniform_buffer: wgpu::Buffer,
       bind_group: wgpu::BindGroup,
   }

   impl CSIRenderer {
       pub fn new(device: &wgpu::Device) -> Self {
           // Load csi_spiral.wgsl shader
           // Create render pipeline
           // Setup uniform buffer for ChromaticSpiralUniform
       }

       pub fn render(&mut self, metrics: &CSIMetrics, encoder: &mut wgpu::CommandEncoder) {
           // Update uniform buffer with latest metrics
           // Execute render pass
       }
   }
   ```

2. **Integrate WGSL Shader**
   - Use existing `chromatic_shared/shaders/csi_spiral.wgsl`
   - Create `ChromaticSpiralUniform` struct matching shader layout:
     ```rust
     #[repr(C)]
     #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
     pub struct ChromaticSpiralUniform {
         pub position_x: f32,
         pub position_y: f32,
         pub rotation_rate: f32,      // α
         pub decay_factor: f32,        // β
         pub coherence_score: f32,     // 1.0 - σ²/100
         pub pattern_id: u32,
         pub _padding: [f32; 2],
     }
     ```

3. **Add Window Integration** (Optional - for standalone demo)
   - Use `winit` for window creation
   - Target 60 fps rendering
   - Add keyboard controls (R=reset, P=pause, Q=quit)

4. **Create Renderer Example** (`chromatic_shared/examples/csi_visualizer.rs`)
   - Demonstrates real-time CSI rendering
   - Simulates varying RGB states
   - Shows all 5 pattern types

**Success Criteria**:
- Shader compiles and renders correctly
- Metrics update in real-time (60 fps)
- Visual patterns match diagnostic classifications
- Low overhead (<5ms per frame)

---

### **Subphase 4.C: Training Visualization**

**Focus**: Connect training CSI to GPU renderer for live monitoring.

**Agent Instructions**:

1. **Add Visualization Flag to Training Config**
   ```toml
   [visualization]
   enabled = true
   update_interval = 10  # Update every N iterations
   window_size = [800, 600]
   ```

2. **Integrate Renderer into Training Loop**
   ```rust
   if self.config.visualization.enabled {
       if iteration % self.config.visualization.update_interval == 0 {
           let metrics = self.monitor.get_metrics();
           self.renderer.render(&metrics, &mut encoder);
       }
   }
   ```

3. **Add Metrics Overlay** (Optional)
   - Display current loss, accuracy, LR
   - Show CSI metrics (α, β, σ²)
   - Display pattern classification

**Success Criteria**:
- Training window shows live CSI visualization
- Visual feedback corresponds to training progress
- No significant performance impact (<1% slowdown)

---

## 🗄️ **Phase 5: Dream Pool + CSI Filtering**

**Goal**: Integrate CSI pattern classification into the Dream Pool to selectively archive only high-quality cognitive states.

**Estimated Effort**: 1 session
**Prerequisites**: Phase 4 complete

### **Subphase 5.A: Dream Pool CSI Integration**

**Focus**: Filter dream entries based on CSI pattern before archival.

**Agent Instructions**:

1. **Add CSI Field to DreamEntry** (`chromatic_core/src/dream/simple_pool.rs`)
   ```rust
   pub struct DreamEntry {
       // ... existing fields
       pub csi_pattern: Option<SpiralPattern>,
       pub csi_metrics: Option<CSIMetrics>,
   }
   ```

2. **Implement Archival Filter**
   ```rust
   impl SimpleDreamPool {
       pub fn should_archive(&self, pattern: SpiralPattern) -> bool {
           matches!(pattern,
               SpiralPattern::StableProcessing |
               SpiralPattern::PeriodicResonance
           )
       }

       pub fn add_with_csi(&mut self, tensor: ChromaticTensor, result: SolverResult, metrics: CSIMetrics) {
           if self.should_archive(metrics.pattern) {
               let entry = DreamEntry {
                   // ... populate fields
                   csi_pattern: Some(metrics.pattern),
                   csi_metrics: Some(metrics),
               };
               self.entries.push_back(entry);
           } else {
               debug!("Rejecting dream with pattern {:?}", metrics.pattern);
           }
       }
   }
   ```

3. **Add CSI-Based Retrieval**
   ```rust
   pub fn retrieve_by_pattern(&self, pattern: SpiralPattern) -> Vec<EntryId> {
       self.entries.iter()
           .filter(|e| e.csi_pattern == Some(pattern))
           .map(|e| e.id)
           .collect()
   }
   ```

**Success Criteria**:
- Only StableProcessing/PeriodicResonance dreams archived
- OverExcitation/SystemFault states rejected
- Dream Pool quality improves (higher coherence)
- New tests pass for CSI filtering

---

### **Subphase 5.B: CSI Persistence**

**Focus**: Save/load CSI trajectory with checkpoints.

**Agent Instructions**:

1. **Implement CSI Serialization**
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct CSICheckpoint {
       trajectory: Vec<RGBState>,
       max_len: usize,
       timestamp: u64,
   }

   impl ChromaticSpiralIndicator {
       pub fn save(&self) -> CSICheckpoint { ... }
       pub fn load(checkpoint: CSICheckpoint) -> Self { ... }
   }
   ```

2. **Integrate with Checkpointable Trait**
   - Add CSI checkpoint to model checkpoint files
   - Restore CSI trajectory when loading model

**Success Criteria**:
- CSI trajectory persists across restarts
- Checkpoint roundtrip preserves exact state
- Minimal storage overhead (<10KB per checkpoint)

---

## 🔄 **Phase 6: Self-Generation Loop**

**Goal**: Enable the Core to request shaders from the Trainer, creating a self-generating cognitive loop.

**Estimated Effort**: 2 sessions
**Prerequisites**: Phase 5 complete, backpropagation working

### **Subphase 6.A: Shader Request Interface**

**Focus**: Create API for Core to request WGSL shaders from Trainer.

**Agent Instructions**:

1. **Define Request Protocol** (`chromatic_shared/src/shader_request.rs`)
   ```rust
   pub struct ShaderRequest {
       pub operation: ShaderOperation,
       pub constraints: ShaderConstraints,
       pub priority: RequestPriority,
   }

   pub enum ShaderOperation {
       ChromaticMix,
       ChromaticFilter,
       SpectralBridge,
       Custom(String),
   }

   pub struct ShaderConstraints {
       pub max_lines: usize,
       pub target_fps: f32,
       pub memory_budget_mb: usize,
   }
   ```

2. **Implement Request Handler** (`wgsl_trainer/src/inference/shader_generator.rs`)
   ```rust
   pub struct ShaderGenerator {
       model: Transformer,
       tokenizer: WGSLTokenizer,
   }

   impl ShaderGenerator {
       pub fn generate_from_request(&mut self, request: &ShaderRequest) -> Result<String> {
           let prompt = self.request_to_prompt(request);
           let tokens = self.model.generate(&prompt, 512);
           let shader = self.tokenizer.decode(&tokens);
           self.validate_shader(&shader)?;
           Ok(shader)
       }

       fn validate_shader(&self, shader: &str) -> Result<()> {
           // Use naga to validate WGSL
           naga::front::wgsl::parse_str(shader)?;
           Ok(())
       }
   }
   ```

3. **Create IPC Mechanism**
   - Option A: Shared memory queue (fast, local)
   - Option B: Unix socket / Named pipe (flexible)
   - Option C: In-process (same binary, simplest)

**Success Criteria**:
- Core can request shaders programmatically
- Trainer generates valid WGSL within 100ms
- Generated shaders compile successfully
- Self-generation loop demonstrated end-to-end

---

### **Subphase 6.B: Adaptive Shader Optimization**

**Focus**: Trainer learns to generate shaders optimized for Core's current workload.

**Agent Instructions**:

1. **Implement Performance Feedback Loop**
   ```rust
   pub struct ShaderPerformance {
       pub compile_time_ms: f32,
       pub execution_time_ms: f32,
       pub memory_usage_mb: f32,
       pub correctness_score: f32,  // From Core's validation
   }

   // Core reports performance back to Trainer
   pub fn report_performance(&mut self, shader_id: u64, perf: ShaderPerformance) {
       self.performance_history.insert(shader_id, perf);
       // Use for fine-tuning
   }
   ```

2. **Add Reinforcement Signal**
   - Positive reward for fast, correct shaders
   - Negative reward for slow, incorrect, or non-compiling shaders
   - Update training data with successful examples

**Success Criteria**:
- Generated shaders improve over time
- Performance metrics tracked and logged
- Training data enriched with real-world examples

---

## 🔙 **Phase 7: Backpropagation Implementation**

**Goal**: Implement full gradient-based training for the wgsl_trainer Transformer.

**Estimated Effort**: 2-3 sessions
**Prerequisites**: Phase 6 underway

### **Subphase 7.A: Manual Backpropagation**

**Focus**: Implement backpropagation from scratch in pure Rust.

**Agent Instructions**:

See detailed plan in `wgsl_trainer/docs/BACKPROPAGATION_EXPLANATION.md`.

**Key Steps**:
1. Implement `backward()` for each layer (attention, FFN, embeddings)
2. Compute gradients via chain rule
3. Store activations during forward pass
4. Update weights using Adam optimizer

**Alternative**: Use `tch-rs` (LibTorch bindings) for automatic differentiation.

**Success Criteria**:
- Loss decreases over training
- Gradient flow verified (no vanishing/exploding)
- Model converges on training set
- Generated shaders improve qualitatively

---

## 🏥 **Phase 8: Medical Imaging Application**

**Goal**: Apply spectral dissonance detection to medical imaging for anomaly detection.

**Estimated Effort**: 2-3 sessions
**Prerequisites**: Phase 7 complete, Core fully functional

### **Subphase 8.A: Spectral Bridge for Medical Data**

**Focus**: Adapt spectral bridge to process medical imaging modalities.

**Agent Instructions**:

1. **Support Medical Imaging Formats**
   - DICOM reader integration
   - MRI/CT/PET preprocessing
   - Multi-modal fusion (RGB-like representation)

2. **Define Dissonance Metrics**
   ```rust
   pub struct SpectralDissonance {
       pub delta_e94: f32,           // Color fidelity error
       pub frequency_coherence: f32, // Spectral consistency
       pub anomaly_score: f32,       // Combined metric
   }

   pub fn detect_anomaly(tensor: &ChromaticTensor) -> SpectralDissonance {
       // Compare to healthy tissue baseline
       // Flag regions with high dissonance
   }
   ```

3. **Implement Tissue Classification**
   - Normal tissue → StableProcessing pattern
   - Anomalous tissue → OverExcitation / SystemFault
   - Archive high-confidence examples in Dream Pool

**Success Criteria**:
- Medical images successfully converted to ChromaticTensors
- Anomaly detection demonstrates sensitivity
- False positive rate acceptable (<10%)
- Visualization shows dissonance heatmap

---

## 🛡️ **Phase 9: Production Hardening**

**Goal**: Stabilize system for long-term deployment with robust error handling and logging.

**Estimated Effort**: 2 sessions
**Prerequisites**: Phase 8 complete

### **Subphase 9.A: Error Handling**

**Focus**: Comprehensive error handling and graceful degradation.

**Agent Instructions**:

1. **Define Error Taxonomy**
   ```rust
   #[derive(Error, Debug)]
   pub enum SystemError {
       #[error("CSI detected system fault: {0}")]
       CSIFault(String),

       #[error("Dream Pool capacity exceeded")]
       CapacityError,

       #[error("Shader generation failed: {0}")]
       ShaderGenerationError(String),

       #[error("Spectral bridge conversion error")]
       SpectralError,
   }
   ```

2. **Implement Fallback Strategies**
   - If shader generation fails → use pre-compiled fallback
   - If CSI detects SystemFault → reset to safe state
   - If Dream Pool full → evict lowest-utility entries

3. **Add Health Checks**
   ```rust
   pub struct SystemHealth {
       pub csi_status: DiagnosticAction,
       pub dream_pool_utilization: f32,
       pub trainer_available: bool,
       pub last_check: Instant,
   }
   ```

**Success Criteria**:
- System recovers from transient failures
- No unhandled panics in release mode
- Graceful degradation under load
- Health checks pass continuously

---

### **Subphase 9.B: Logging and Telemetry**

**Focus**: Comprehensive logging for monitoring and debugging.

**Agent Instructions**:

1. **Structured Logging**
   ```rust
   info!(
       csi_pattern = ?metrics.pattern,
       alpha = metrics.alpha,
       beta = metrics.beta,
       variance = metrics.energy_variance,
       "CSI observation recorded"
   );
   ```

2. **Metrics Export**
   - Prometheus-compatible metrics endpoint
   - Track: training loss, generation latency, CSI metrics
   - Alert on: SystemFault, capacity warnings, performance degradation

3. **Distributed Tracing** (Optional)
   - OpenTelemetry integration
   - Trace shader requests end-to-end

**Success Criteria**:
- All critical operations logged
- Metrics dashboards functional
- Performance overhead <2%
- Debugging simplified by rich logs

---

## 📚 **Phase 10: Documentation + Deployment**

**Goal**: Finalize documentation and prepare for release.

**Estimated Effort**: 1-2 sessions
**Prerequisites**: Phase 9 complete

### **Subphase 10.A: Documentation**

**Focus**: Comprehensive user and developer documentation.

**Agent Instructions**:

1. **User Guide**
   - Installation instructions (Linux, macOS, Windows)
   - Quick start tutorial
   - Example workflows
   - Configuration reference
   - Troubleshooting guide

2. **Developer Guide**
   - Architecture overview
   - API reference (rustdoc)
   - Contributing guidelines
   - Testing strategy
   - Release process

3. **Research Documentation**
   - CSI methodology paper
   - Spectral bridge theory
   - Performance benchmarks
   - Medical imaging case studies

**Success Criteria**:
- All modules have rustdoc comments
- User guide covers common workflows
- Examples run on fresh install
- Research results reproducible

---

### **Subphase 10.B: Packaging and Release**

**Focus**: Prepare distributable packages.

**Agent Instructions**:

1. **Create Release Builds**
   ```bash
   cargo build --release --workspace
   strip target/release/chromatic_core
   strip target/release/wgsl-trainer
   ```

2. **Package for Distribution**
   - Linux: `.tar.gz` with install script
   - Windows: `.zip` with `.exe` binaries
   - macOS: `.dmg` or Homebrew formula
   - Docker image for containerized deployment

3. **Continuous Integration**
   - GitHub Actions workflow
   - Run tests on push
   - Generate release artifacts
   - Publish to crates.io (optional)

4. **Versioning and Changelog**
   - Follow SemVer 2.0
   - Update CHANGELOG.md
   - Tag releases in git

**Success Criteria**:
- Release packages install cleanly
- CI pipeline passes
- Version 2.0.0 tagged and published
- Deployment guide verified on multiple platforms

---

## 🔧 **Development Guidelines**

### **Testing Requirements**

Each phase must maintain or improve test coverage:

- **Unit Tests**: ≥80% coverage for new code
- **Integration Tests**: Key workflows end-to-end
- **Performance Tests**: Benchmark critical paths
- **CSI Tests**: Validate diagnostic accuracy

### **Code Quality**

- Run `cargo clippy` before commit
- Format with `cargo fmt`
- No warnings in release build
- Update rustdoc for public APIs

### **Git Workflow**

```bash
# Feature branch for each subphase
git checkout -b phase-4-training-integration

# Commit frequently with descriptive messages
git commit -m "Phase 4A: Add CSI observation to training loop

- Created TrainingMonitor struct
- Integrated CSI into Trainer
- Added adaptive LR control based on diagnostics
- Tests: 235/236 passing"

# Push and create PR when subphase complete
git push origin phase-4-training-integration
```

### **Communication Protocol**

When working with AI agents:

1. **Provide Context**: Reference this AGENTS.md and relevant phase
2. **State Current Status**: "Phase 3 complete, starting 4.A"
3. **Define Success Criteria**: Clear, testable goals
4. **Request Clarification**: If ambiguous, ask before implementing
5. **Report Progress**: Show test results and metrics

---

## 📞 **Support and Troubleshooting**

### **Common Issues**

**Problem**: Tests failing after CSI integration
**Solution**: Ensure CSI is properly initialized with `reset_csi()` in test setup

**Problem**: Training diverges (OverExcitation pattern)
**Solution**: Reduce learning rate, check gradient clipping

**Problem**: Shader generation fails validation
**Solution**: Check tokenizer vocabulary, ensure templates in training data

**Problem**: Performance degradation
**Solution**: Profile with `cargo flamegraph`, optimize hot paths

### **Getting Help**

- **Documentation**: Check relevant phase in this file
- **Code Examples**: See `examples/` directory
- **Tests**: Unit tests demonstrate expected behavior
- **Issues**: File bug reports with CSI metrics and logs

---

## 🎉 **Project Completion Checklist**

When all phases are complete, verify:

- [ ] All workspace tests passing (≥236 tests)
- [ ] CSI integration functional across all components
- [ ] Self-generation loop demonstrated (Core requests shader, Trainer generates, Core uses)
- [ ] Backpropagation implemented and training converges
- [ ] Medical imaging application validated on test dataset
- [ ] Production deployment successful on target platforms
- [ ] Documentation complete and published
- [ ] Performance benchmarks meet targets:
  - CSI observation: <1ms
  - Shader generation: <100ms
  - Dream Pool retrieval: <10ms
  - Training throughput: >100 examples/sec

---

## 📖 **References**

- **CSI Specification**: `PHASE_3_COMPLETE.md`
- **Workspace Setup**: `SESSION_1_COMPLETE.md`
- **Merge Plan**: `MERGE_STATUS.md`
- **Backpropagation Guide**: `wgsl_trainer/docs/BACKPROPAGATION_EXPLANATION.md`
- **Architecture Overview**: `chromatic_core/docs/ARCHITECTURE.md`
- **API Reference**: `chromatic_core/docs/API.md`

---

**Generated**: 2025-10-31
**Authors**: Chromatic Cognition Team + Claude Code
**License**: MIT OR Apache-2.0

🎨 **Chromatic Cognition System - Self-Generating Architecture with Real-Time Health Monitoring**
