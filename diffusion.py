import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from model.Blocks import UNet
import numpy as np

class DDPM(nn.Module):
    def __init__(self,unet,num_timesteps=1000,beta_start=1e-4,beta_end=0.02):
        super().__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps
        #cuanto noise se agrega a los datos en cada timestep
        self.betas = torch.linspace(beta_start,beta_end,num_timesteps)
        #cuanta informacion de la imagen orgia=nal queda en cada timestep 
        self.alphas = 1 - self.betas
        #producto cumulativo del tensor alphas
        self.alpha_cumprod = torch.cumprod(self.alphas,dim=0)
        #tensor cumprod de alphas, pero movido un timestep, esto ayuda en el proceso de reconstruccion 
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1-self.alpha_cumprod) 

#usado en el proceso de difusion reversan para escalar el noise predicho
        self.sqrt_recip_alpha = torch.sqrt(1/self.alphas)
        #ayuda a identificar cuanta informacion queda despues de n pasos de difusion 
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1/self.alpha_cumprod)
        #ayuda a ajustar las predicciones, basado en la cantidad de noise añadido en forward
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1 / self.alpha_cumprod - 1)

    def forward_process(self,x0,t,noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1,1,1,1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1,1,1,1)
        return sqrt_alpha_cumprod_t * x0 +  sqrt_one_minus_alpha_cumprod_t * noise,noise
    

    def forward(self,x0):
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.randint(0,self.num_timesteps, (batch_size,), device=device)
        xt,noise = self.forward_process(x0,t)

        predicted_noise = self.unet(xt,t)

        loss = F.mse_loss(predicted_noise,noise)
        return loss 
    
    @torch.no_grad()
    def sample(self,batch_size,device,image_shape=(3,32,32)):
        x = torch.randn(batch_size,*image_shape,device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,),t,device=device,dtype=torch.long)
            predicted_noise = self.unet(x,t_batch)   
            beta_t = self.betas[t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t]
            sqrt_recip_alpha_t = self.sqrt_recip_alpha[t]

            x = sqrt_recip_alpha_t * (x-beta_t/sqrt_one_minus_alpha_cumprod_t * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) *noise
        return x 

def create_cifar10_ddpm():
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_embed_dim=128,
        down_channels=[64, 128, 256],
        num_res_blocks=2
    )
    model = DDPM(unet, num_timesteps=1000)
    return model






