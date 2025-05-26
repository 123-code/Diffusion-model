import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math


def get_time_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal time embeddings for DDPM.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbedding(nn.Module):
    """
    Time embedding network to process timestep information.
    """
    def __init__(self, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.lin1 = nn.Linear(time_embed_dim, time_embed_dim * 4)
        self.lin2 = nn.Linear(time_embed_dim * 4, time_embed_dim * 4)
        
    def forward(self, time_embed):
        time_embed = self.lin1(time_embed)
        time_embed = F.silu(time_embed)
        time_embed = self.lin2(time_embed)
        return time_embed


class ResNetBlock(nn.Module):
    """
    Residual block with time embedding and group normalization.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First convolution block
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_embed_dim * 4, out_channels)
        
        # Second convolution block
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, time_embed):
        skip = self.skip(x)
        
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Add time embedding
        time_embed = self.time_proj(time_embed)
        x = x + time_embed[:, :, None, None]
        
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + skip


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for capturing long-range dependencies.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        batch, channels, height, width = x.shape
        skip = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # Reshape for attention
        qkv = qkv.view(batch, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, heads, hw, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch, channels, height, width)
        
        out = self.proj(out)
        return out + skip


class DownBlock(nn.Module):
    """
    Downsampling block with ResNet blocks and optional attention.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, num_layers=2, 
                 downsample=True, attention=False):
        super().__init__()
        self.downsample = downsample
        
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(in_channels if i == 0 else out_channels, out_channels, time_embed_dim)
            for i in range(num_layers)
        ])
        
        if attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = None
            
        if downsample:
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x, time_embed):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embed)
            
        if self.attention:
            x = self.attention(x)
            
        if self.downsample:
            x = self.downsample_conv(x)
            
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with ResNet blocks and optional attention.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, num_layers=2, 
                 upsample=True, attention=False):
        super().__init__()
        self.upsample = upsample
        
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(in_channels + out_channels if i == 0 else out_channels, 
                       out_channels, time_embed_dim)
            for i in range(num_layers)
        ])
        
        if attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = None
            
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x, skip_x, time_embed):
        # Concatenate skip connection
        x = torch.cat([x, skip_x], dim=1)
        
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embed)
            
        if self.attention:
            x = self.attention(x)
            
        if self.upsample:
            x = self.upsample_conv(x)
            
        return x


class MiddleBlock(nn.Module):
    """
    Middle block with attention mechanism.
    """
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.resnet1 = ResNetBlock(channels, channels, time_embed_dim)
        self.attention = SelfAttention(channels)
        self.resnet2 = ResNetBlock(channels, channels, time_embed_dim)
        
    def forward(self, x, time_embed):
        x = self.resnet1(x, time_embed)
        x = self.attention(x)
        x = self.resnet2(x, time_embed)
        return x


class UNet(nn.Module):
    """
    Complete U-Net model for DDPM with CIFAR-10 specifications.
    """
    def __init__(self, in_channels=3, out_channels=3, time_embed_dim=128,
                 down_channels=[64, 128, 256], num_res_blocks=2):
        super().__init__()
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        ch_in = down_channels[0]
        for i, ch_out in enumerate(down_channels):
            is_last = (i == len(down_channels) - 1)
            attention = (i >= len(down_channels) - 2)  # attention in last 2 levels
            
            self.down_blocks.append(
                DownBlock(ch_in, ch_out, time_embed_dim, num_res_blocks, 
                         downsample=not is_last, attention=attention)
            )
            ch_in = ch_out
            
        # Middle block
        self.middle_block = MiddleBlock(down_channels[-1], time_embed_dim)
        
        # Up blocks - fix channel calculations for 3-level architecture
        self.up_blocks = nn.ModuleList()
        up_channels = list(reversed(down_channels))  # [256, 128, 64]
        
        # First up block: 256 (from middle) + 128 (skip) -> 128, upsample 8->16
        self.up_blocks.append(
            UpBlock(up_channels[0], up_channels[1], time_embed_dim, num_res_blocks,
                   upsample=True, attention=True)
        )
        
        # Second up block: 128 (from prev) + 64 (skip) -> 64, upsample 16->32
        self.up_blocks.append(
            UpBlock(up_channels[1], up_channels[2], time_embed_dim, num_res_blocks,
                   upsample=True, attention=False)
        )
            
        # Final convolution
        self.norm_out = nn.GroupNorm(8, down_channels[0])
        self.conv_out = nn.Conv2d(down_channels[0], out_channels, 3, padding=1)
        
    def forward(self, x, timesteps):
        # Time embedding
        time_embed = get_time_embedding(timesteps, self.time_embed_dim)
        time_embed = self.time_embedding(time_embed)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Store skip connections from down blocks
        skip_connections = []
        
        # Down path - store outputs for skip connections
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x, time_embed)
            # Store skip connections for all but the last down block
            if i < len(self.down_blocks) - 1:
                skip_connections.append(x)
        
        # Middle block
        x = self.middle_block(x, time_embed)
        
        # Up path - use skip connections in reverse order
        for up_block in self.up_blocks:
            if len(skip_connections) > 0:
                skip_x = skip_connections.pop()
                x = up_block(x, skip_x, time_embed)
            else:
                # Should not happen if architecture is correct
                raise RuntimeError("No skip connection available for up block")
        
        # Final output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x





            