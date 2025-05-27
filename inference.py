import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from diffusion import create_cifar10_ddpm
import os

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""

    model = create_cifar10_ddpm()
    

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    

    model = model.to(device)
    model.eval()
    

    
    return model

def generate_samples(model, num_samples=16, device='cuda', save_path='generated_samples.png'):

    
    with torch.no_grad():

        samples = model.sample(num_samples, device)
        

        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
 
        vutils.save_image(samples, save_path, nrow=4, padding=2)
        print(f"💾 Samples saved to {save_path}")
        
        return samples

def display_samples(samples, title="Generated CIFAR-10 Images"):

    grid = vutils.make_grid(samples, nrow=4, padding=2)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    

    checkpoint_path = '../Downloads/ddpm_epoch_70.pth'  
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists('checkpoints/'):
            for f in os.listdir('checkpoints/'):
                if f.endswith('.pth'):
                    print(f"  - checkpoints/{f}")
        return
    

    model = load_model_from_checkpoint(checkpoint_path, device)
    

    samples = generate_samples(model, num_samples=16, device=device)

    display_samples(samples)

if __name__ == "__main__":
    main()