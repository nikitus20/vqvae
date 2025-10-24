# VQ-VAE Research Framework

A modular research codebase for studying **Vector Quantized Variational Autoencoders (VQ-VAE)** and discrete representation learning.

## Research Scope

This framework enables systematic investigation of:
- **Quantization methods**: Vector Quantization (VQ), Finite Scalar Quantization (FSQ), Gumbel-Softmax
- **Gradient estimators**: Straight-Through Estimator (STE), rotation-based gradients
- **Initialization strategies**: Uniform, k-means, rate-distortion optimal
- **Architecture variations**: Linear encoders, convolutional networks, transformers

## Design Philosophy

**Composable Components**: Mix and match different implementations
```
data → encoder → quantizer → decoder → reconstruction
```

Each component is:
- **Independent**: Follows clear interface contracts
- **Swappable**: Easy to replace or extend
- **Controllable**: Can be frozen or trained selectively

## Project Structure

```
vqvae/
├── src/
│   ├── data/              # Dataset implementations
│   ├── encoders/          # Encoder modules (PCA, learned linear, etc.)
│   ├── decoders/          # Decoder modules
│   ├── quantizers/        # Quantization methods (VQ, rotation, etc.)
│   ├── initialization/    # Codebook initialization strategies
│   ├── models/            # Composable VQ-VAE models
│   ├── training/          # Training utilities and loss functions
│   └── evaluation/        # Metrics and visualization
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/nikitus20/vqvae.git
cd vqvae

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from src.data import GaussianDataset
from src.encoders import PCAEncoder
from src.decoders import PCADecoder
from src.quantizers import VectorQuantizer
from src.models import VQVAE
from src.initialization import rd_gaussian_init

# 1. Create dataset
dataset = GaussianDataset(d=64, k=8, n_samples=10000)

# 2. Create encoder & decoder
encoder = PCAEncoder(dataset.U_k, trainable=False)
decoder = PCADecoder(dataset.U_k, trainable=False)

# 3. Initialize codebook
init_data = dataset.get_initialization_batch(1000)
codebook = rd_gaussian_init(codebook_size=256, dim=8, data=init_data)

# 4. Create quantizer
quantizer = VectorQuantizer(dim=8, codebook_size=256, init_codebook=codebook)

# 5. Compose model
model = VQVAE(encoder, quantizer, decoder)

# 6. Use model
x = torch.randn(32, 64)  # batch of data
outputs = model(x)
# outputs: {'x_recon', 'z', 'z_q', 'indices', ...}
```

## Extending the Framework

### Adding a New Quantizer

```python
from src.quantizers.base import BaseQuantizer

class MyQuantizer(BaseQuantizer):
    def forward(self, z):
        # Your quantization logic
        z_q, indices, info = ...
        return z_q, indices, info
```

### Adding a New Dataset

```python
from src.data.base import BaseDataset

class MyDataset(BaseDataset):
    def get_dataloader(self, batch_size, shuffle=True):
        # Return PyTorch DataLoader
        ...
```

### Adding a New Encoder/Decoder

```python
from src.encoders.base import BaseEncoder

class MyEncoder(BaseEncoder):
    def forward(self, x):
        # x: (batch, d) -> z: (batch, k)
        z = ...
        return z
```

## Core Features

- **Modular Design**: Swap components without changing other code
- **Type Safety**: Abstract base classes ensure correct interfaces
- **Flexibility**: Train or freeze any component
- **Extensibility**: Add new methods by inheriting from base classes
- **Clean Data Flow**: Explicit transformations at each stage

## Current Implementations

### Data
- `GaussianDataset`: Linear Gaussian model (X = AY + W)

### Encoders
- `PCAEncoder`: Fixed PCA projection
- `LearnedLinearEncoder`: Trainable linear encoder

### Decoders
- `PCADecoder`: Fixed PCA reconstruction
- `LearnedLinearDecoder`: Trainable linear decoder

### Quantizers
- `VectorQuantizer`: Standard VQ with straight-through estimator
- `RotationVectorQuantizer`: VQ with rotation-based gradients

### Initialization
- `uniform_init`: Uniform random initialization
- `kmeans_init`: K-means clustering on data
- `rd_gaussian_init`: Rate-distortion optimal (Gaussian sources)

## Research Goals

This framework is designed for:
1. **Theoretical validation**: Test rate-distortion optimal initialization
2. **Method comparison**: Benchmark different quantization approaches
3. **Scalability studies**: Understand performance across codebook sizes
4. **Real data experiments**: Extend to images, audio, etc.

## Contributing

This is an active research project. Contributions are welcome:
1. Fork the repository
2. Create a feature branch
3. Implement your changes following the base class interfaces
4. Submit a pull request

## License

MIT License - see LICENSE file for details

---

**Author**: Nikita Karagodin
**Status**: Active Development
**Contact**: https://github.com/nikitus20/vqvae
