import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from data.process_cifar import get_cifar10_data
from diffusion import create_cifar10_ddpm


def train_ddpm(num_epochs=100, batch_size=128, lr=2e-4, device='cuda'):
  
    

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    writer = SummaryWriter('runs/ddpm_cifar10')
    

    train_loader, _ = get_cifar10_data(batch_size=batch_size)
    

    model = create_cifar10_ddpm().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
 
    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
       
            loss = model(data)
            
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
         
            epoch_loss += loss.item()
            writer.add_scalar('Loss/Train', loss.item(), step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            step += 1
            
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
        
    
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/ddpm_epoch_{epoch+1}.pth')
            

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample(16, device)
                samples = (samples + 1) / 2  
                samples = torch.clamp(samples, 0, 1)
                
             
                torch.save(samples, f'samples/samples_epoch_{epoch+1}.pt')
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_ddpm(device=device)