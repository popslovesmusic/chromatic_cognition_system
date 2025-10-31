# Understanding Backpropagation: Why the Model Doesn't Learn

## The Problem: Static Weights

Our current training loop looks like this:

```rust
for epoch in 0..100 {
    for example in &dataset {
        // 1. Forward pass: compute predictions
        let logits = model.forward(&input_tokens);

        // 2. Compute loss: how wrong are we?
        let loss = cross_entropy(logits, target);

        // 3. ❌ MISSING: Update weights based on loss
        // The weights stay exactly the same!
    }
}
```

This is like a student taking the same test 100 times without studying between attempts. They'll get the same score every time because they haven't learned anything.

---

## What is Backpropagation?

**Backpropagation** (backward propagation of errors) is the algorithm that tells us **how to change each weight** to reduce the loss.

### The Core Idea

Given:
- Current weights: `W`
- Loss function: `L(W)`
- Goal: Find new weights `W'` such that `L(W') < L(W)`

Backpropagation computes: `∂L/∂W` (the gradient) for every weight in the network.

Then we update: `W_new = W - learning_rate × ∂L/∂W`

---

## Analogy: Hiking Down a Mountain

Imagine you're on a foggy mountain and want to reach the valley (minimum loss):

1. **Current position**: Your current weights
2. **Height**: The loss value (higher = worse)
3. **Gradient**: The slope of the ground under your feet
4. **Taking a step**: Updating the weights

**Forward pass** = Checking how high you are (computing loss)
**Backward pass** = Feeling which direction slopes down (computing gradients)
**Optimizer step** = Actually walking in that direction (updating weights)

Our current implementation only does step 1 (checking height) but never walks anywhere!

---

## Mathematical Breakdown

### Forward Pass (Already Implemented)
```
Input → Embedding → Encoder → Decoder → Output → Loss
  x   →    e      →    h    →    y    →  logits → L
```

For our transformer with 44M parameters, this computes:
```rust
// Simplified forward pass
let embedded = token_embedding[token_id] + positional_encoding[pos];
let hidden = encoder_layers.forward(embedded);
let output = decoder_layers.forward(hidden);
let logits = final_linear.forward(output);
let loss = cross_entropy(logits, target);
```

### Backward Pass (NOT Implemented)
```
∂L/∂logits ← ∂L/∂output ← ∂L/∂hidden ← ∂L/∂embedded ← ∂L/∂weights
```

For each operation in the forward pass, we need its **gradient**:

#### 1. Cross-Entropy Gradient
```rust
// Forward
let probs = softmax(logits);
let loss = -log(probs[target]);

// Backward
let grad_logits = probs.clone();
grad_logits[target] -= 1.0;  // Derivative of cross-entropy
```

#### 2. Linear Layer Gradient
```rust
// Forward
let output = input.dot(&weight) + bias;

// Backward
let grad_input = grad_output.dot(&weight.transpose());
let grad_weight = input.transpose().dot(&grad_output);
let grad_bias = grad_output.sum_axis(0);
```

#### 3. Attention Gradient (Complex!)
```rust
// Forward
let scores = query.dot(&key.transpose()) / sqrt(d_k);
let attention = softmax(scores);
let output = attention.dot(&value);

// Backward (simplified)
let grad_value = attention.transpose().dot(&grad_output);
let grad_attention = grad_output.dot(&value.transpose());
let grad_scores = softmax_backward(attention, grad_attention);
let grad_query = grad_scores.dot(&key) / sqrt(d_k);
let grad_key = grad_scores.transpose().dot(&query) / sqrt(d_k);
```

---

## Why Is This Hard in Rust?

### Challenge 1: Computation Graph
In PyTorch/TensorFlow, operations automatically build a graph:

```python
# Python (PyTorch)
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y.sum()
z.backward()  # Automatically computes gradients!
print(x.grad)  # [2.0, 2.0]
```

In Rust with ndarray, we have to track this manually:
```rust
// Rust (current)
let x = Array1::from(vec![1.0, 2.0]);
let y = x * 2.0;
let z = y.sum();
// ❌ No .backward() method!
// We have to manually compute and apply gradients
```

### Challenge 2: Memory Management
For our 44M parameter model, we need to store:

1. **Weights**: 44M floats = 176 MB
2. **Gradients**: 44M floats = 176 MB
3. **Optimizer state** (Adam momentum): 44M × 2 = 352 MB
4. **Intermediate activations** (for backprop): ~100 MB

**Total**: ~800 MB per batch

In Rust, we have to manually allocate and manage all this memory.

### Challenge 3: Layer-Specific Gradients
Each layer type needs custom gradient code:

```rust
trait Layer {
    fn forward(&self, input: &Array2<f32>) -> Array2<f32>;

    // ❌ Not implemented
    fn backward(&self, grad_output: &Array2<f32>) -> (Array2<f32>, Vec<Gradient>);
    fn update(&mut self, gradients: &[Gradient], learning_rate: f32);
}
```

Our current implementation only has `forward()`.

---

## Implementation Options

### Option 1: Manual Backpropagation
**Implement gradients for each layer from scratch**

#### Pros:
- No external dependencies
- Full control
- Educational

#### Cons:
- 2-3 weeks of work
- Error-prone (easy to get gradients wrong)
- Numerical stability issues
- No GPU acceleration

#### Code Example:
```rust
struct Linear {
    weight: Array2<f32>,
    bias: Array1<f32>,
    // Cache for backward pass
    last_input: Option<Array2<f32>>,
}

impl Linear {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());  // Cache for backward
        input.dot(&self.weight) + &self.bias
    }

    fn backward(&self, grad_output: &Array2<f32>) -> (Array2<f32>, LinearGradients) {
        let input = self.last_input.as_ref().unwrap();

        // Compute gradients
        let grad_input = grad_output.dot(&self.weight.t());
        let grad_weight = input.t().dot(grad_output);
        let grad_bias = grad_output.sum_axis(Axis(0));

        (grad_input, LinearGradients {
            weight: grad_weight,
            bias: grad_bias,
        })
    }

    fn update(&mut self, grads: &LinearGradients, lr: f32) {
        self.weight -= &(grads.weight * lr);
        self.bias -= &(grads.bias * lr);
    }
}
```

For our 6-layer transformer with attention, this needs to be done for:
- ✅ Linear layers (simple)
- ❌ Layer normalization (medium complexity)
- ❌ Multi-head attention (complex)
- ❌ Softmax with masking (tricky)
- ❌ Embedding layers (simple but memory-intensive)

---

### Option 2: Use tch-rs (PyTorch Bindings)
**Leverage existing PyTorch autograd in Rust**

#### Pros:
- Autograd already implemented
- GPU support (CUDA)
- Proven optimizer implementations
- Much faster development (1 week vs 3 weeks)

#### Cons:
- External dependency (libtorch)
- Larger binary size (~500MB with libtorch)
- Platform-specific builds
- Less educational

#### Code Example:
```rust
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

// Convert our model to tch
let vs = nn::VarStore::new(Device::Cpu);
let model = TransformerModel::new(&vs.root(), vocab_size, d_model);

let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

for epoch in 0..100 {
    for (input, target) in dataloader {
        let logits = model.forward(&input);
        let loss = logits.cross_entropy_for_logits(&target);

        opt.zero_grad();      // Clear gradients
        loss.backward();       // 🎉 Automatic backprop!
        opt.step();           // Update weights
    }
}
```

**Installation**:
```toml
[dependencies]
tch = "0.13"  # Requires libtorch installation
```

---

### Option 3: Hybrid Approach
**Train in Python, Inference in Rust**

#### Workflow:
1. Export preprocessed data from Rust → JSON/Parquet
2. Train in Python with PyTorch
3. Export weights to binary format
4. Load weights in Rust for inference

#### Pros:
- Best of both worlds
- Rapid experimentation (Python)
- Fast inference (Rust)
- No autograd overhead in production

#### Cons:
- Two-language system
- More complex deployment
- Weight format compatibility

#### Code Example (Python Training):
```python
import torch
import torch.nn as nn

# Load Rust-preprocessed data
data = load_from_rust_json("preprocessed_data.json")

# Define same architecture
model = Transformer(vocab_size=290, d_model=512, nhead=8, num_layers=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    for input_ids, target_ids in dataloader:
        logits = model(input_ids)
        loss = F.cross_entropy(logits, target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Export weights for Rust
torch.save(model.state_dict(), "model_weights.pt")
```

#### Code Example (Rust Inference):
```rust
// Load pre-trained weights
let weights = load_weights_from_pytorch("model_weights.pt")?;
let model = CodeGenerationModel::from_pretrained(weights);

// Run inference (no gradients needed)
let logits = model.forward(&input_tokens);
let predicted = argmax(&logits);
```

---

## Comparison Table

| Feature | Manual Backprop | tch-rs | Hybrid |
|---------|----------------|--------|--------|
| Development Time | 2-3 weeks | 1 week | 3-4 days |
| Dependencies | None | libtorch | Python + Rust |
| Binary Size | ~5 MB | ~500 MB | ~5 MB (inference) |
| GPU Support | No | Yes | Yes (training) |
| Flexibility | Full | Medium | High |
| Educational Value | High | Low | Medium |
| Production-Ready | Eventually | Immediately | Immediately |

---

## What We'd Need to Implement (Option 1: Manual)

### 1. Gradient Storage
```rust
struct ModelGradients {
    token_embedding_grad: Array2<f32>,  // vocab_size × d_model
    encoder_grads: Vec<EncoderLayerGradients>,
    decoder_grads: Vec<DecoderLayerGradients>,
    final_linear_weight_grad: Array2<f32>,
    final_linear_bias_grad: Array1<f32>,
}
```

### 2. Backward Pass for Each Layer
```rust
impl EncoderLayer {
    fn backward(
        &self,
        grad_output: &Array2<f32>,
        cached_forward: &ForwardCache,
    ) -> (Array2<f32>, EncoderLayerGradients) {
        // 1. Backprop through feedforward
        let (grad_after_ffn, ffn_grads) =
            self.feedforward.backward(grad_output, &cached_forward.ffn_input);

        // 2. Backprop through attention
        let (grad_input, attn_grads) =
            self.attention.backward(&grad_after_ffn, &cached_forward.attn_input);

        (grad_input, EncoderLayerGradients { ffn_grads, attn_grads })
    }
}
```

### 3. Optimizer Implementation
```rust
struct AdamOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,

    // Momentum storage (same size as model weights)
    first_moment: ModelGradients,   // 176 MB
    second_moment: ModelGradients,  // 176 MB
}

impl AdamOptimizer {
    fn step(&mut self, model: &mut CodeGenerationModel, grads: &ModelGradients) {
        // Update each parameter
        for (param, grad, m, v) in zip_all_params(model, grads, &mut self.first_moment, &mut self.second_moment) {
            // Momentum update
            *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
            *v = self.beta2 * *v + (1.0 - self.beta2) * grad.powi(2);

            // Bias correction
            let m_hat = m / (1.0 - self.beta1.powi(self.t));
            let v_hat = v / (1.0 - self.beta2.powi(self.t));

            // Parameter update
            *param -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        self.t += 1;
    }
}
```

### 4. Training Loop (Complete)
```rust
for epoch in 0..num_epochs {
    for example in &train_dataset {
        // Forward pass (already implemented)
        let (logits, forward_cache) = model.forward_with_cache(&input_tokens);
        let loss = cross_entropy(&logits, target);

        // Backward pass (NEW)
        let grad_logits = cross_entropy_backward(&logits, target);
        let grads = model.backward(&grad_logits, &forward_cache);

        // Optimizer step (NEW)
        optimizer.step(&mut model, &grads);
    }
}
```

---

## Why Loss Stays Constant: A Detailed Example

### Epoch 1
```
Weights: [random initialization]
Forward: input → embedding → ... → logits
Loss: 7.0398

❌ No backward pass
❌ Weights unchanged
```

### Epoch 2
```
Weights: [EXACTLY THE SAME as Epoch 1]
Forward: input → embedding → ... → logits
Loss: 7.0398 (identical!)

❌ No backward pass
❌ Weights unchanged
```

### What Should Happen (with backprop)
```
Epoch 1:
  Weights: [random]
  Loss: 7.0398
  Gradients: [computed]
  Weights updated: W = W - 0.0001 × grad

Epoch 2:
  Weights: [slightly different]
  Loss: 6.8234 (improved!)
  Gradients: [computed]
  Weights updated: W = W - 0.0001 × grad

Epoch 3:
  Weights: [even better]
  Loss: 6.4102
  ...

Epoch 50:
  Loss: 0.1234 ✅ Converged!
```

---

## Recommendation

For this project, I recommend **Option 2 (tch-rs)** because:

1. ✅ **Fastest path to working training** (1 week vs 3 weeks)
2. ✅ **GPU acceleration** for free
3. ✅ **Proven optimizer implementations** (Adam, AdamW, SGD)
4. ✅ **Focus on architecture**, not gradient plumbing
5. ✅ **Production-ready** immediately

### Migration Path
```rust
// Add to Cargo.toml
[dependencies]
tch = "0.13"

// Update model to use tch::Tensor instead of ndarray
// Update training loop to use optimizer.step()
// Profit! 🎉
```

The gradient computation is the hardest part of deep learning infrastructure. Unless the goal is educational (learning how backprop works), leveraging existing solutions is the pragmatic choice.

---

## Summary

**What we have**: A car with an engine and wheels
**What's missing**: Fuel and ignition

The model architecture is complete and correct. It just needs the gradient computation mechanism to actually learn from the training data.

**Current state**: `loss = compute_loss(model(input), target)` → observe, repeat
**Needed state**: `loss = compute_loss(model(input), target)` → compute gradients → update weights → improve

---

**Next Steps**:
1. Choose implementation option (recommend tch-rs)
2. Implement backward pass / integrate autograd
3. Add optimizer step to training loop
4. Watch loss decrease 📉
5. Celebrate convergence 🎉

**Estimated time to working training**:
- Manual: 2-3 weeks
- tch-rs: 1 week
- Hybrid: 3-4 days (Python training)
