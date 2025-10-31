# Chromatic Cognition Core

A deterministic Rust engine that represents cognition as an RGB tensor field. Each cell contains an (r,g,b) triple with scalar certainty, enabling novel approaches to neural computation through color-space operations.

## Features

- **Chromatic Tensor**: 4D RGB tensor field with certainty weights
- **Parallel Operations**: CPU-optimized operations using `rayon` for parallelization
- **Primitive Operations**: mix, filter, complement, and saturate operations
- **Gradient Projection**: Weighted layer blending for visualization
- **Training Support**: MSE loss computation and metrics tracking
- **Configuration**: TOML-based engine configuration
- **Logging**: JSON line-delimited operation and training logs
- **Meta Awareness**: Cycle-level observation and autoregressive prediction of coherence signals

## Quick Start

### Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo

### Installation

```bash
git clone <repository-url>
cd chromatic_cognition_core
cargo build --release
```

### Running the Demo

```bash
cargo run --example demo
```

This will:
1. Load configuration from `config/engine.toml`
2. Create two random chromatic tensors
3. Apply operations: mix → filter → complement → saturate
4. Generate a PNG visualization in `out/frame_0001.png`
5. Log operations to `logs/operations.jsonl`
6. Compute and log MSE loss to `logs/run.jsonl`

### Running Tests

```bash
cargo test
```

See [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) for the latest recorded results, environment details, and reproduction steps.

### Release Packaging

To build a distributable archive with the statically linked binary, finalized configuration
files, and the `FINAL_BASELINE` performance report, run:

```bash
./scripts/build_release.sh
```

The script creates `release/chromatic_cognition_core-<version>.tar.gz`, which can be
shared directly with end users.

### Hybrid Semantic Retrieval

The Chromatic Semantic Archive exposes low-latency retrieval via `SimpleDreamPool::retrieve_hybrid`. This API combines the hue
category filter with UMS similarity ranking and relies on an HNSW index for production workloads.

```rust
use chromatic_cognition_core::dream::{EmbeddingMapper, PoolConfig, SimpleDreamPool};
use chromatic_cognition_core::tensor::ChromaticTensor;
use chromatic_cognition_core::solver::SolverResult;
use serde_json::json;

// Configure the pool to enable HNSW search and store up to 1,024 dreams
let config = PoolConfig {
    max_size: 1_024,
    coherence_threshold: 0.75,
    retrieval_limit: 8,
    use_hnsw: true,
    memory_budget_mb: None,
};

let mut pool = SimpleDreamPool::new(config);

// Insert a coherent dream and build the semantic index
let tensor = ChromaticTensor::from_seed(42, 8, 8, 3);
let result = SolverResult {
    energy: 0.1,
    coherence: 0.9,
    violation: 0.02,
    grad: None,
    mask: None,
    meta: json!({}),
};
pool.add_if_coherent(tensor, result);

let mapper = EmbeddingMapper::new(64);
pool.rebuild_soft_index(&mapper, None);

// Issue a hybrid retrieval request for the top 5 category-aligned entries
let query = ChromaticTensor::from_seed(99, 8, 8, 3);
let results = pool.retrieve_hybrid(&query, 5);
assert!(results.len() <= 5);
```

The pool automatically routes semantic queries through the HNSW index (with a linear fallback) and filters inactive entries,
making `retrieve_hybrid` the preferred interface for CSA retrieval workflows.

## Architecture

### Core Types

#### ChromaticTensor

The fundamental data structure representing a 4D tensor field:

```rust
pub struct ChromaticTensor {
    pub colors: Array4<f32>,      // [rows, cols, layers, 3] RGB values
    pub certainty: Array3<f32>,   // [rows, cols, layers] certainty weights
}
```

**Key Methods:**
- `new(rows, cols, layers)` - Create zero-filled tensor
- `from_seed(seed, rows, cols, layers)` - Create deterministic random tensor
- `normalize()` - Normalize color values to [0.0, 1.0]
- `clamp(min, max)` - Clamp all color values to range
- `statistics()` - Compute mean RGB, variance, and certainty

### Operations

All operations are pure functions that return new tensors:

#### mix(a, b)
Additive coherence - combines two tensors and normalizes:
```rust
let result = mix(&tensor_a, &tensor_b);
```

#### filter(a, b)
Subtractive distinction - computes clamped difference:
```rust
let result = filter(&tensor_a, &tensor_b);
```

#### complement(a)
Rotates hue 180° by inverting green and blue channels:
```rust
let result = complement(&tensor_a);
```

#### saturate(a, alpha)
Adjusts color saturation by scaling distance from mean:
```rust
let result = saturate(&tensor_a, 1.25);
```

### Gradient Layer

Projects 3D layered tensor to 2D image using certainty-weighted averaging:

```rust
let gradient = GradientLayer::from_tensor(&tensor);
gradient.to_png("output/frame.png")?;
```

### Awareness & Prediction

Phase 5A introduces deterministic telemetry and short-term forecasting:

- `Awareness::observe` captures coherence, entropy, gradient energy, and spectral bands each cycle.
- `Predictor::new` creates an AR(2) forecaster for configured features and horizon.
- `Predictor::predict` returns bounded forecasts suitable for scheduling dream pool refreshes.

### Configuration

Configure engine parameters via `config/engine.toml`:

```toml
[engine]
rows = 64
cols = 64
layers = 8
seed = 42
device = "cpu"

[p5a]
predict_horizon = 2
features = ["coherence", "entropy", "grad_energy"]
```

Load configuration in your code:

```rust
use chromatic_cognition_core::{BridgeConfig, EngineConfig};

let config = EngineConfig::load_from_file("config/engine.toml")?;
let tensor = ChromaticTensor::from_seed(config.seed, config.rows, config.cols, config.layers);

let bridge = BridgeConfig::load_from_file("config/bridge.toml")?;
println!("Bridge octaves: {}", bridge.base.octaves);
```

Hue ↔ frequency bridge parameters live in `config/bridge.toml`:

```toml
[bridge]
f_min = 110.0
octaves = 7
gamma = 1.0
sample_rate = 44100

[bridge.spectral]
fft_size = 4096
accum_format = "Q16.48"
reduction_mode = "pairwise_neumaier"
categorical_count = 12

[bridge.reversibility]
delta_e_tolerance = 0.001
```

### Logging

#### Operations Log
Each operation logs its statistics to `logs/operations.jsonl`:

```json
{"operation":"mix","timestamp_ms":1761535834354,"mean_rgb":[0.53,0.53,0.53],"variance":0.065,"certainty_mean":0.55}
```

#### Training Log
Training iterations log to `logs/run.jsonl`:

```json
{"iteration":0,"loss":0.32,"mean_rgb":[0.054,0.946,0.946],"variance":0.022,"timestamp_ms":1761535816560}
```

## Project Structure

```
chromatic_cognition_core/
├── config/
│   ├── bridge.toml          # Spectral bridge configuration
│   └── engine.toml          # Engine configuration
├── examples/
│   └── demo.rs              # Demo application
├── src/
│   ├── lib.rs               # Library exports
│   ├── config.rs            # Configuration parsing
│   ├── meta/                # Awareness and prediction utilities
│   ├── logging.rs           # JSON logging utilities
│   ├── training.rs          # Loss computation and metrics
│   └── tensor/
│       ├── mod.rs           # Tensor module exports
│       ├── chromatic_tensor.rs  # Core tensor type
│       ├── operations.rs    # Primitive operations
│       └── gradient.rs      # Gradient layer projection
├── tests/
│   └── operations.rs        # Unit tests
├── out/                     # Generated PNG outputs
├── logs/                    # JSON logs
├── Cargo.toml              # Dependencies and metadata
├── CHANGELOG.md            # Version history
└── README.md               # This file
```

## Dependencies

- `ndarray` (0.15) - N-dimensional arrays with rayon support
- `rayon` (1.8) - Data parallelism
- `serde` (1.0) - Serialization framework
- `serde_json` (1.0) - JSON serialization
- `toml` (0.8) - TOML configuration parsing
- `plotters` (0.3) - PNG visualization

## Development

### Adding New Operations

1. Implement the operation in `src/tensor/operations.rs`
2. Add logging via `log_operation()`
3. Export from `src/lib.rs`
4. Add unit tests in `tests/operations.rs`

Example:

```rust
pub fn my_operation(a: &ChromaticTensor) -> ChromaticTensor {
    // Your operation logic
    let result = /* ... */;

    log_operation("my_operation", &result.statistics());
    result
}
```

### Running Benchmarks

```bash
cargo bench
```

### Building Documentation

```bash
cargo doc --no-deps --open
```

## Roadmap

### Milestone 1: Chromatic Tensor Core ✓
- [x] ChromaticTensor struct with RGB 4D array
- [x] Basic operations (mix, filter, complement, saturate)
- [x] Unit tests for all operations
- [x] Demo example with PNG output

### Milestone 2: Gradient Projection + Logger
- [ ] Enhanced gradient computation
- [ ] Advanced logging features
- [ ] Performance profiling
- [ ] Additional visualization options

### Milestone 3: Training Loop
- [ ] Gradient descent implementation
- [ ] Multiple loss functions
- [ ] Training callbacks
- [ ] Checkpoint saving/loading

### Future: GPU Support
- [ ] Port to Candle framework
- [ ] CUDA/Metal backends
- [ ] Performance comparison CPU vs GPU

## Performance

Current performance on 64×64×8 tensor (130,560 cells):

- Random initialization: ~5ms
- Mix operation: ~2ms
- Filter operation: ~2ms
- Complement operation: ~15ms
- Saturate operation: ~25ms
- Gradient projection: ~50ms
- PNG export: ~10ms

*Benchmarked on CPU using rayon parallel iterators*

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `cargo test`
5. Update CHANGELOG.md
6. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

Built with inspiration from color-based cognitive models and chromatic information processing theory.
