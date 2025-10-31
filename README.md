# Chromatic Cognition System

**A Self-Generating Cognitive Architecture with Real-Time Health Monitoring**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

---

## 🎯 **Core Mission**

The Chromatic Cognition System is a unified framework that:

1. **Stores semantic knowledge** as 3×12×12×3 RGB tensors with 512D UMS vectors (ΔE₉₄ ≤ 1.0×10⁻³ fidelity)
2. **Generates operational code** (WGSL shaders) that the system needs to run
3. **Monitors cognitive health** in real-time via the Chromatic Spiral Indicator (CSI)
4. **Self-generates** its own computational tools through a training loop
5. **Detects anomalies** in multi-modal data via spectral dissonance (medical imaging application)

---

## 🏗️ **Architecture**

This is a Cargo workspace consisting of three interconnected crates:

```
chromatic_cognition_system/
├── chromatic_core/        # Chromatic Semantic Archive (CSA)
│   ├── Dream Pool          # Semantic memory (HNSW-indexed)
│   ├── Spectral Bridge     # Color ↔ Frequency conversion
│   ├── Meta-Awareness      # Self-monitoring and adaptive control
│   └── Neural Networks     # Color-space computation
│
├── wgsl_trainer/          # WGSL Shader Generator
│   ├── Training Loop       # Gradient descent + backpropagation
│   ├── Tokenizer           # WGSL-specific tokenization
│   └── Transformer Model   # 44M parameter encoder-decoder
│
└── chromatic_shared/      # Shared Types & CSI
    ├── CSI Module          # Chromatic Spiral Indicator
    ├── RGB Tensor Types    # ChromaticTensor, UMSVector
    └── WGSL Validation     # Naga integration
```

---

## 🎨 **Chromatic Spiral Indicator (CSI)**

The CSI is a real-time visualization and diagnostic system that monitors the "cognitive health" of the system by tracking RGB state trajectories and computing three key metrics:

### Metrics

| Metric | Calculation | Stable Threshold | Interpretation |
|--------|-------------|------------------|----------------|
| **Rotation Rate (α)** | Δhue/Δt | α > 0.05 rad/frame | Processing is Active |
| **Radial Decay (β)** | Fit S(t) = S₀e^(-βt) | β ∈ [0.01, 0.2] | Energy is Balancing |
| **Energy Variance (σ²)** | Var(‖RGB‖) | < 3% | Coherence Maintained |

### Pattern Classification

| Visual Pattern | Operational State | Action |
|----------------|-------------------|--------|
| **Clear Inward Spiral** | Stable Processing | Log metrics for analysis |
| **Oscillating Loops** | Periodic Resonance | Sonify (enable APM) |
| **Expanding Spiral** | Over-Excitation | Check UMS normalization |
| **Flat Line / Random Walk** | System Fault | Integrity check |

---

## 🚀 **Quick Start**

### Build the Workspace

```bash
cd chromatic_cognition_system
cargo build --release
```

### Run CSI Demo

```bash
cargo run --example csi_demo --release
```

### Train WGSL Generator

```bash
cd wgsl_trainer
cargo run --release -- train --config config/wgsl_generation.toml --epochs 100
```

### Generate WGSL Shader

```bash
cd wgsl_trainer
cargo run --release -- generate \
    --model checkpoints/model.bin \
    --prompt "Generate chromatic mix shader with additive coherence" \
    --output output.wgsl
```

### Validate WGSL

```bash
cd wgsl_trainer
cargo run --release -- validate output.wgsl
```

---

## 📊 **System Status**

### Chromatic Core
- ✅ Dream Pool with HNSW retrieval (< 10ms latency)
- ✅ Spectral Bridge (ΔE₉₄ ≤ 0.001 reversibility)
- ✅ Meta-Awareness with adaptive control
- ✅ MLP Classifier (100% accuracy on test set)
- ✅ 223 tests passing

### WGSL Trainer
- ✅ Training infrastructure complete
- ✅ 290-token vocabulary
- ✅ 44.4M parameter transformer
- ⚠️ Backpropagation pending (see docs/)
- ✅ WGSL validation (naga)

### Chromatic Shared
- ✅ CSI metrics implementation (α, β, σ²)
- ✅ Pattern classifier (4 patterns + indeterminate)
- ✅ RGB state tracking
- 🚧 GPU renderer (in progress)

---

## 📖 **Documentation**

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
- **[CSI_SPECIFICATION.md](docs/CSI_SPECIFICATION.md)** - CSI technical spec
- **[SELF_GENERATION_GUIDE.md](docs/SELF_GENERATION_GUIDE.md)** - Self-generation loop
- **[BACKPROPAGATION_EXPLANATION.md](wgsl_trainer/docs/BACKPROPAGATION_EXPLANATION.md)** - Why training needs backprop

---

## 🧪 **Testing**

Run all tests across the workspace:

```bash
cargo test --workspace
```

Run benchmarks:

```bash
cargo bench
```

---

## 📜 **License**

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## 🤝 **Contributing**

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🔬 **Research Applications**

### Medical Imaging (Future)

The system is designed for **steganographic anomaly detection** in multi-channel medical scans (MRI/CT):

1. Encode tissue structure as ChromaticTensor
2. Apply Spectral Bridge (FFT analysis)
3. Detect dissonance patterns (anomalies = frequency spikes)
4. Sonify via Auditory Processing Module (APM)
5. CSI monitors diagnostic quality

---

**Built with Rust 🦀 | Powered by Color 🎨 | Monitored by CSI 📊**
