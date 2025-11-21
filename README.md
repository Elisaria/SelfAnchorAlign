# SelfAnchorAlign

**Anchor-Mediated Vision-Language Alignment without Negative Sampling**

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)

## Overview

SelfAnchorAlign (SAA) introduces a novel paradigm for vision-language alignment that completely eliminates the need for negative sampling. Traditional contrastive learning methods suffer from:

- **Quadratic complexity O(n²)** due to pairwise comparisons
- **Semantic limitations**: inadequate diversity leads to trivial solutions, while pseudo-negatives push apart semantically consistent representations

Our key insight: if a visual anchor feature is trained to generate captions autoregressively, it must maintain semantic structure—collapse would immediately break the generation process. This replaces the role of negative samples in preventing representation collapse.

**Key Benefits:**
- O(n) complexity through autoregressive generation
- Small batch training (512 vs 4096+ for baselines)
- Star-topology framework with unidirectional alignment
- Strong performance on fine-tuning tasks (+13-14% over CoCa/CLIP)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SelfAnchorAlign                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Image I ──► ViT Encoder (8 layers) ──► P_CLS (anchor)  │
│                      │                      │            │
│                      │              stop_gradient        │
│                      │                      ▼            │
│  Image I_aug ──► ViT Encoder ──► P_CLS_aug ──► L_aug    │
│                                                          │
│  Text ──► Text Encoder (8 layers) ──► E_CLS ──► L_txt   │
│                                                          │
│  [P_CLS, T_1...T_n] ──► Causal Decoder (4 layers) ──► y │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Components:**
- **Visual Encoder (ViT)**: 8 transformer blocks, processes 224×224 images into 49 patches
- **Text Encoder**: 8 transformer blocks, encodes captions up to 30 tokens
- **Causal Decoder**: 4 transformer blocks with causal masking for caption generation
- **Latent Dimension**: 512
- **Attention Heads**: 4

## Installation

### Requirements

- Python 3.8
- TensorFlow 2.18.1 / Keras 3
- NumPy



## Usage

### Training

```python
import tensorflow as tf
from SelfAnchorAlign import SAA

# Initialize model
model = SAA(vocab_size=10002)

# Prepare dataset
# Format: ((in_text, patches, aug_patches), out_text)
train_dataset = tf.data.Dataset.from_tensor_slices(
    ((input_text, patches, aug_patches), output_text)
).batch(512)

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=256)
```

### Inference with Pre-trained Weights

```python
from forward import forward_SAA

# Load model with pre-trained weights
model = forward_SAA(vocab_size=10002)

# Run inference
output = model([text_input, image_patches, image_patches])
```

## Implementation Details

### 1. Gamma Sharing for RMSNorm

During experiments, we observed that after training, the gamma parameters of all RMSNorm layers approximate a vector of 1s. To reduce model parameter redundancy:

- **All RMSNorm layers within the same AttentionBlock share a single gamma parameter**
- Gamma parameters are **not shared across different AttentionBlocks**

```python
def RMSNorm(self, inputs):
    x = inputs / ops.sqrt(
        ops.var(inputs, axis=-1, keepdims=True) + 1e-3)
    x = self.gamma * x  # Same gamma for all RMSNorm in this block
    return x
```

### 2. Weight Initialization in build() Method

Following TensorFlow's implementation convention for network layers (e.g., Dense), where weight parameters (kernel, bias) are initialized in the `build()` method:

- **All weight parameters of SelfAnchorAlign (SAA) are initialized in the `build()` method**
- Officially implemented layers (e.g., TensorFlow's native Dense) are defined in `__init__`, as their internal weight parameters are inherently created in their own `build()` methods

**Advantage:** This binds weight parameters to the layer instance, facilitating debugging.

**Important Note:** To use these weight parameters, the `build()` method must be called explicitly in advance—this is critical for loading pre-trained parameters into the model later.

```python
def build(self, input_shape):
    in_dim = input_shape[-1]
    self.query_kernel = self.add_weight(shape=[in_dim, self.half_dim], ...)
    self.key_kernel = self.add_weight(shape=[in_dim, self.half_dim], ...)
    self.value_kernel = self.add_weight(shape=[in_dim, self.dim], ...)
    self.W1 = self.add_weight(shape=[self.dim, 2 * self.dim], ...)
    self.W2 = self.add_weight(shape=[self.dim, 2 * self.dim], ...)
    self.W3 = self.add_weight(shape=[2 * self.dim, self.dim], ...)
    self.gamma = self.add_weight(shape=[self.dim], initializer='ones', ...)
```

### 3. Gamma Initialization for RMSNorm

The gamma parameter of RMSNorm **must be initialized as a vector of 1s**. This ensures that:

- In the early stages of training, RMSNorm approximately only performs standard deviation scaling
- Training is more stable and converges faster

**Warning:** Random initialization of gamma will disrupt the original data distribution, leading to unstable training and slow convergence.

```python
self.gamma = self.add_weight(
    shape=[self.dim],
    initializer='ones',  # Critical: must be ones
    trainable=True,
    name='gamma'
)
```

### 4. Pre-trained Weight File

We provide the weight file `TEm_TDe_IEn.npy`, which stores:

- **Text Embedding (TEm)** weights
- **Text Decoder (TDe)** weights
- **Image Encoder (IEn)** weights

**Format:** Stored as a Python dictionary.

**Weight loading code** is provided in `parameters.py`:

```python
import numpy as np

# Load weights
weights = np.load("TEm_TDe_IEn.npy", allow_pickle=True).item()

# Access specific weights
text_embedding = weights['saa/embedding/embeddings']
vit_CLS = weights['saa/vit/CLS']

# ViT Attention Blocks (8 layers)
for i in range(8):
    query = weights[f'saa/vit/AB{i}/query_kernel']
    key = weights[f'saa/vit/AB{i}/key_kernel']
    value = weights[f'saa/vit/AB{i}/value_kernel']
    gamma = weights[f'saa/vit/AB{i}/gamma']
    # ...

# Text Decoder (4 layers)
for i in range(4):
    # Similar structure
    pass
```

### 5. Input/Output Specifications

#### Data Dimensions

| Component | Specification |
|-----------|---------------|
| Input image resolution | 224×224 |
| Patch configuration | 49 patches (7×7 grid), each 32×32 pixels |
| Patch dimension | 3072 (32×32×3) |
| Maximum text length | 30 tokens + [CLS] |
| Latent dimension | 512 |

#### Training Data Format

**Tuple structure:** `((in_text, patches, aug_patches), out_text)`

**Example of `in_text`** (token indices):
```python
[2, 272, 2170, 136, 991, 8, 7, 333, 1494, 177, 339, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Corresponding `out_text`** (shifted target tokens):
```python
[272, 2170, 136, 991, 8, 7, 333, 1494, 177, 339, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

- Padding tokens are represented by `0`
- `in_text` starts with [CLS] token (index 2)
- `out_text` ends with [END] token (index 3)

### 6. Vocabulary Details

**Vocabulary file:** `vocabulary.txt`

**Special token constraints:**

| Token | Index | Description |
|-------|-------|-------------|
| `[PAD]` | 0 | Padding token |
| `[cls]` | 2 | Classification/start token |
| `[END]` | 3 | Caption prediction end signal |

**Vocabulary size:** 10,002 tokens

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 512 |
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| Learning rate schedule | Cosine decay |
| Decay steps | 100,000 |
| Weight decay | 1e-6 |
| Pretraining epochs | 256 |

## Loss Functions

```python
# Total loss
L_total = L_IC + λ₁ * L_txt→anchor + λ₂ * L_aug→anchor

# Where:
# L_IC: Image captioning loss (cross-entropy)
# L_txt→anchor: Text-to-anchor consistency (MSE + MAE)
# L_aug→anchor: Augmented-to-anchor consistency

# Loss weights
λ₁ = 0.4 * sqrt(epoch)  # Increases over training
λ₂ = 0.01              # Fixed, small value
```

## Results

### Image-Text Retrieval

| Model | Flickr30K I2T R@1 | Flickr30K T2I R@1 | MSCOCO I2T R@1 | MSCOCO T2I R@1 |
|-------|-------------------|-------------------|----------------|----------------|
| CLIP/16 | 71.7 | 45.7 | 51.4 | 31.2 |
| CoCa/16 | 75.6 | 56.6 | 56.5 | 34.3 |
| **SAA/16** | **78.3** | **61.0** | **59.8** | **36.8** |

### Fine-tuned Classification

| Model | DTD | Pets | Food101 | Flowers | Average |
|-------|-----|------|---------|---------|---------|
| CLIP/16 | 63.0 | 72.8 | 70.6 | 73.7 | 69.78 |
| CoCa/16 | 65.0 | 71.8 | 72.7 | 75.1 | 71.15 |
| **SAA/16** | **74.2** | **84.4** | **82.3** | **92.9** | **84.20** |

## File Structure

```
SelfAnchorAlign/
├── components.py        # Core components (Attention_Block, ViT, Encoders)
├── SelfAnchorAlign.py   # Main SAA model class
├── forward.py           # Inference with pre-trained weights
├── parameters.py        # Weight loading utilities
├── train.py             # Training script
├── vocabulary.txt       # Vocabulary file (10,002 tokens)
├── TEm_TDe_IEn.npy      # Pre-trained weights
└── README.md            # This file
```

## Citation

```bibtex
@inproceedings{selfanchoralign2026,
  title={SelfAnchorAlign: Anchor-Mediated Vision-Language Alignment without Negative Sampling},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

None

## Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grant 62176039 and Grant 62293541.

