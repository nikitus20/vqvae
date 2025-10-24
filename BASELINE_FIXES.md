# VQ-VAE Baseline Bug Fixes

## Overview
Critical bugs preventing correct VQ-VAE training. Fix these before running experiments.

---

## CRITICAL BUG #1: Incorrect Codebook Loss Gradient Flow

**Location**: `src/training/losses.py`, line 40

**Problem**: Codebook loss uses `z_q` (which includes STE trick), preventing correct gradient flow to codebook.

**Current code**:
```python
def vqvae_loss(x, x_recon, z, z_q, beta=0.25):
    recon_loss = F.mse_loss(x_recon, x)
    codebook_loss = F.mse_loss(z.detach(), z_q)  # ❌ WRONG
    commitment_loss = F.mse_loss(z, z_q.detach())
    total_loss = recon_loss + codebook_loss + beta * commitment_loss
    return total_loss, loss_dict
```

**Issue**: `z_q = z + (c_k - z).detach()` from STE, so gradients don't flow to codebook correctly.

**Fix**:

1. **Modify quantizer to return hard quantization**:

```python
# src/quantizers/vq.py - forward method
def forward(self, z):
    distances = torch.cdist(z, self.codebook)
    indices = distances.argmin(dim=1)
    z_q_hard = self.codebook[indices]  # Hard quantization
    z_q = z + (z_q_hard - z).detach()  # STE
    
    return z_q, indices, {'z_q_hard': z_q_hard}  # Return both
```

2. **Update loss function to use hard quantization**:

```python
# src/training/losses.py
def vqvae_loss(x, x_recon, z, z_q, z_q_hard, beta=0.25):
    recon_loss = F.mse_loss(x_recon, x)
    codebook_loss = F.mse_loss(z.detach(), z_q_hard)  # ✅ Use hard quantization
    commitment_loss = F.mse_loss(z, z_q_hard.detach())
    total_loss = recon_loss + codebook_loss + beta * commitment_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'codebook': codebook_loss.item(),
        'commitment': commitment_loss.item()
    }
    return total_loss, loss_dict
```

3. **Update model forward pass**:

```python
# src/models/vqvae.py - forward method
def forward(self, x):
    z = self.encoder(x)
    z_q, indices, quantizer_info = self.quantizer(z)
    x_recon = self.decoder(z_q)
    
    outputs = {
        'x_recon': x_recon,
        'z': z,
        'z_q': z_q,
        'z_q_hard': quantizer_info['z_q_hard'],  # Add this
        'indices': indices,
    }
    return outputs
```

4. **Update trainer to pass z_q_hard**:

```python
# src/training/trainer.py - train_step method
def train_step(self, batch, beta=0.25):
    # ... existing code ...
    outputs = self.model(x)
    
    loss, loss_dict = vqvae_loss(
        x=x,
        x_recon=outputs['x_recon'],
        z=outputs['z'],
        z_q=outputs['z_q'],
        z_q_hard=outputs['z_q_hard'],  # Add this
        beta=beta
    )
    # ... rest of training ...
```

---

## CRITICAL BUG #2: Inconsistent STE Application

**Location**: `src/quantizers/vq.py`, lines 63-71

**Problem**: Conditional STE breaks expected gradient flow.

**Current code**:
```python
if z.requires_grad:
    z_q = z + (z_q_hard - z).detach()
else:
    z_q = z_q_hard  # ❌ Inconsistent behavior
```

**Fix**: Always apply STE trick consistently:

```python
# src/quantizers/vq.py - forward method
def forward(self, z):
    distances = torch.cdist(z, self.codebook)
    indices = distances.argmin(dim=1)
    z_q_hard = self.codebook[indices]
    
    # Always use STE - gradients naturally zero if z frozen
    z_q = z + (z_q_hard - z).detach()  # ✅ Consistent
    
    return z_q, indices, {'z_q_hard': z_q_hard}
```

---

## CRITICAL BUG #3: Backwards Naming in Legacy Code

**Location**: `src/models/legacy.py`, lines 47-50, 68, 78

**Problem**: `encoder_weight` and `decoder_weight` are swapped in usage.

**Current code**:
```python
self.register_buffer('encoder_weight', U_k.T)  # (k, d)
self.register_buffer('decoder_weight', U_k)    # (d, k)

# But usage is backwards:
z = x @ self.decoder_weight  # ❌ Using decoder_weight to encode!
x_recon = z @ self.encoder_weight  # ❌ Using encoder_weight to decode!
```

**Fix**: Swap buffer names to match usage:

```python
# src/models/legacy.py - __init__ method
self.register_buffer('encoder_weight', U_k)    # (d, k) for encoding
self.register_buffer('decoder_weight', U_k.T)  # (k, d) for decoding

def encode(self, x):
    z = x @ self.encoder_weight  # ✅ Correct: (B,d) @ (d,k) -> (B,k)
    return z

def decode(self, z):
    x_recon = z @ self.decoder_weight  # ✅ Correct: (B,k) @ (k,d) -> (B,d)
    return x_recon
```

**Note**: Add deprecation warning at top of file recommending composable VQVAE instead.

---

## MINOR FIX #4: Simplify PCA Encoder/Decoder

**Location**: `src/encoders/linear.py` (line 34) and `src/decoders/linear.py` (line 33)

**Problem**: Unnecessary double transpose in encoder.

**Current code**:
```python
# Encoder
self.register_buffer('encoder_weight', U_k.T)  # (k, d)
z = x @ self.encoder_weight.T  # Transpose again: (B,d) @ (d,k)

# Decoder
self.register_buffer('decoder_weight', U_k)  # (d, k)
x_recon = z @ self.decoder_weight.T  # (B,k) @ (k,d)
```

**Fix**: Store in natural orientation:

```python
# src/encoders/linear.py - PCAEncoder
def __init__(self, U_k, trainable=False):
    d, k = U_k.shape
    super().__init__(input_dim=d, latent_dim=k, trainable=trainable)
    self.register_buffer('encoder_weight', U_k)  # ✅ (d, k)
    if not trainable:
        self.freeze()

def forward(self, x):
    z = x @ self.encoder_weight  # ✅ (B,d) @ (d,k) -> (B,k)
    return z
```

```python
# src/decoders/linear.py - PCADecoder
def __init__(self, U_k, trainable=False):
    d, k = U_k.shape
    super().__init__(latent_dim=k, output_dim=d, trainable=trainable)
    self.register_buffer('decoder_weight', U_k)  # ✅ (d, k)
    if not trainable:
        self.freeze()

def forward(self, z):
    x_recon = z @ self.decoder_weight.T  # ✅ (B,k) @ (k,d) -> (B,d)
    return x_recon
```

---

## MINOR FIX #5: Update RotationVectorQuantizer

**Location**: `src/quantizers/rotation_vq.py`

**Problem**: Also needs to return `z_q_hard` for correct loss computation.

**Fix**: Match the VectorQuantizer interface:

```python
# src/quantizers/rotation_vq.py - forward method
def forward(self, z):
    # ... existing distance computation ...
    indices = distances_sq.argmin(dim=1)
    q = self.codebook[indices]  # This is z_q_hard
    
    z_q = RotationEstimator.apply(z, q)
    
    distances = (z - q).norm(dim=1)
    commitment_loss = distances.pow(2).mean()
    
    info = {
        'z_q_hard': q,  # ✅ Add this
        'distances': distances,
        'commitment_loss': commitment_loss
    }
    
    return z_q, indices, info
```

---

## Summary of Changes

### Files to modify:
1. `src/quantizers/vq.py` - Fix STE, return z_q_hard
2. `src/quantizers/rotation_vq.py` - Return z_q_hard in info dict
3. `src/training/losses.py` - Use z_q_hard for codebook loss
4. `src/models/vqvae.py` - Pass z_q_hard in outputs
5. `src/training/trainer.py` - Pass z_q_hard to loss function
6. `src/encoders/linear.py` - Simplify PCAEncoder (remove double transpose)
7. `src/decoders/linear.py` - Store decoder_weight correctly
8. `src/models/legacy.py` - Fix encoder/decoder weight naming

### Test after fixing:
```python
# Quick sanity check
import torch
from src.data import GaussianDataset
from src.encoders import PCAEncoder
from src.decoders import PCADecoder
from src.quantizers import VectorQuantizer
from src.models import VQVAE
from src.initialization import rd_gaussian_init

# Generate data
dataset = GaussianDataset(d=64, k=8, n_samples=1000)

# Create model
encoder = PCAEncoder(dataset.U_k, trainable=False)
init_data = dataset.get_initialization_batch(500)
codebook = rd_gaussian_init(256, 8, init_data)
quantizer = VectorQuantizer(8, 256, init_codebook=codebook)
decoder = PCADecoder(dataset.U_k, trainable=False)
model = VQVAE(encoder, quantizer, decoder)

# Forward pass
x = dataset.X[:32]
outputs = model(x)

# Check outputs
print("Keys:", outputs.keys())
assert 'z_q_hard' in outputs, "Missing z_q_hard!"
assert outputs['z_q_hard'].shape == (32, 8), "Wrong shape!"

# Check gradient flow
from src.training.losses import vqvae_loss
loss, _ = vqvae_loss(
    x, outputs['x_recon'], outputs['z'], 
    outputs['z_q'], outputs['z_q_hard']
)
loss.backward()

# Verify codebook has gradients
assert quantizer.codebook.grad is not None, "No gradient to codebook!"
print("✅ Gradient flow working!")
```

---

## After Fixes

Once these bugs are fixed, the baseline should:
- ✅ Have correct gradient flow to codebook
- ✅ Use consistent STE implementation
- ✅ Have clean, non-confusing variable names
- ✅ Work with both VectorQuantizer and RotationVectorQuantizer

Then ready for experiments on Idea 7 (R(D) initialization).
