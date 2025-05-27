# DDPM Diffusion Model for CIFAR-10

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on CIFAR-10 dataset.

## Project Structure

```
diffusion_models/
├── model/
│   └── Blocks.py          # U-Net architecture components
├── data/
│   └── process_cifar.py   # CIFAR-10 data loading
├── diffusion.py           # DDPM model implementation
├── train.py              # Training script
├── test_imports.py       # Import testing
└── README.md
```

## Features

- Complete U-Net architecture with ResNet blocks
- Time embedding for conditioning on noise level
- Self-attention mechanisms for capturing long-range dependencies
- Group normalization for stable training
- Proper skip connections for information preservation
- CIFAR-10 optimized architecture (32x32 RGB images)

## Requirements

```bash
pip install torch torchvision tqdm tensorboard
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd diffusion_models
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision tqdm tensorboard
   ```

3. **Start training**
   ```bash
   python train.py
   ```

4. **Monitor training (optional)**
   ```bash
   tensorboard --logdir runs
   ```

## Model Architecture

### U-Net Components:
- **ResNetBlock**: Residual blocks with time embedding integration
- **SelfAttention**: Multi-head attention for spatial dependencies
- **DownBlock**: Downsampling with optional attention
- **UpBlock**: Upsampling with skip connections
- **MiddleBlock**: Bottleneck processing with attention

### Architecture Details:
- Input/Output: 3x32x32 (RGB CIFAR-10 images)
- Channel progression: [64, 128, 256]
- Time embedding dimension: 128
- Number of ResNet blocks per level: 2
- Attention in deeper layers for better feature modeling

## Training Configuration

- **Epochs**: 100 (configurable)
- **Batch size**: 128
- **Learning rate**: 2e-4
- **Timesteps**: 1000
- **Beta schedule**: Linear (1e-4 to 0.02)
- **Optimizer**: Adam

## Key Features

1. **Automatic Data Download**: CIFAR-10 dataset downloads automatically
2. **Checkpointing**: Model saves every 10 epochs
3. **Sample Generation**: Generates samples every 20 epochs
4. **TensorBoard Logging**: Real-time loss monitoring
5. **Device Detection**: Automatic GPU/CPU selection

## File Descriptions

- `model/Blocks.py`: Contains all neural network components
- `data/process_cifar.py`: CIFAR-10 data loading and preprocessing
- `diffusion.py`: DDPM model with forward/reverse processes
- `train.py`: Complete training pipeline with logging
- `test_imports.py`: Verify all dependencies are working

## Usage Examples

### Basic Training
```python
python train.py
```

### Custom Configuration
Modify `train.py` parameters:
```python
train_ddpm(
    num_epochs=50,      # Train for 50 epochs
    batch_size=64,      # Smaller batch size
    lr=1e-4,           # Lower learning rate
    device='cuda'       # Force GPU usage
)
```

### Generate Samples
```python
from diffusion import create_cifar10_ddpm
import torch

model = create_cifar10_ddpm()
model.load_state_dict(torch.load('checkpoints/ddpm_epoch_100.pth')['model_state_dict'])
samples = model.sample(16, device='cuda')
```

## Expected Output

- **Training Loss**: Should decrease from ~0.2 to ~0.02
- **Sample Quality**: Improves significantly after 50+ epochs
- **Training Time**: ~3-4 hours on modern GPU for 100 epochs

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch_size in `train.py`
2. **Slow Training**: Enable GPU or reduce model size
3. **Import Errors**: Run `python test_imports.py` to verify setup

## Architecture Comparison

This implementation uses:
- Group normalization instead of batch normalization
- SiLU activation functions
- Efficient attention mechanisms
- Proper channel progression for 32x32 images

## License

This project is open source and available under the MIT License. 