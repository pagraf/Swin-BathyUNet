'''
Initial Pytorch Implementation: Panagiotis Agrafiotis (https://github.com/pagraf/Swin-BathyUNet)
Email: agrafiotis.panagiotis@gmail.com

Description:  Swin-BathyUNet, a deep learning model that combines U-Net with Swin Transformer self-attention 
layers and a cross-attention mechanism, tailored specifically for SDB. Swin-BathyUNet is designed to improve 
bathymetric accuracy by capturing long-range spatial relationships and can also function as a standalone solution 
for standard bathymetric mapping with various training depth data, independent of SfM-MVS output.
It outputs continuous values.

If you use this code please cite our paper: "Panagiotis Agrafiotis, Begüm Demir,
Deep learning-based bathymetry retrieval without in-situ depths using remote sensing imagery and SfM-MVS DSMs with data gaps,
ISPRS Journal of Photogrammetry and Remote Sensing,
Volume 225,
2025,
Pages 341-361,
ISSN 0924-2716,
https://doi.org/10.1016/j.isprsjprs.2025.04.020."



Attribution-NonCommercial-ShareAlike 4.0 International License

Copyright (c) 2025 Panagiotis Agrafiotis

This license requires that reusers give credit to the creator. It allows reusers 
to distribute, remix, adapt, and build upon the material in any medium or format,
for noncommercial purposes only. If others modify or adapt the material, they 
must license the modified material under identical terms.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


This work is part of MagicBathy project funded by the European Union’s HORIZON Europe research and innovation 
programme under the Marie Skłodowska-Curie GA 101063294. Work has been carried out at the Remote Sensing Image 
Analysis group. For more information about the project visit https://www.magicbathy.eu/.
'''



#parameters optimized for images of 720x720 pixels

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten spatial dimensions
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * (dim // num_heads) ** -0.5)

    def forward(self, query, key, value):
        B, N, C = query.shape
        qkv = self.qkv(query).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, dropout=0.1):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * (dim // num_heads) ** -0.5)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=32, shift_size=0, mlp_ratio=4., dropout=0.1):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout=dropout)
        self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout)  # Cross-attention layer
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            #nn.BatchNorm1d(mlp_hidden_dim),  # Add BatchNorm here
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, dim),
            #nn.BatchNorm1d(dim),  # Add BatchNorm here
            nn.Dropout(dropout)
        )
        


    def forward(self, x, H, W, cross_input=None):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x  # Save the input for skip connection

        # Layer 1: Self-Attention
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x  # Residual connection for attention layer

        # Cross-Attention (if cross_input is provided)
        if cross_input is not None:
            x = self.cross_attn(query=x, key=cross_input, value=cross_input)

        # Layer 2: MLP
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Residual connection for MLP layer

        return x


# Helper Functions for Window Partition and Reverse

def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    H_padded = H + pad_h
    W_padded = W + pad_w

    B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
    x = windows.view(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
    return x[:, :H, :W, :].contiguous()

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, depth, num_heads, window_size=32, mlp_ratio=4., dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=32)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))  # Initialize for 1024 patches
        self.pos_drop = nn.Dropout(p=dropout)
        self.transformer_blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, dropout=dropout) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        num_patches = (H // self.patch_embed.patch_size) * (W // self.patch_embed.patch_size)
        
        # Ensure positional embedding size matches number of patches
        if self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, x.shape[-1]).to(x.device))

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x, H // self.patch_embed.patch_size, W // self.patch_embed.patch_size)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H // self.patch_embed.patch_size, w=W // self.patch_embed.patch_size)
        return x


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, num_heads=8, dropout=0.1):
        super(UNetWithAttention, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.center = self.conv_block(512, 1024)
        
        self.vit1 = VisionTransformer(in_channels=512, embed_dim=512, depth=depth, num_heads=num_heads, window_size=64, dropout=dropout)
        self.vit2 = VisionTransformer(in_channels=256, embed_dim=256, depth=depth, num_heads=num_heads, window_size=64, dropout=dropout)
        self.vit3 = VisionTransformer(in_channels=128, embed_dim=128, depth=depth, num_heads=num_heads, window_size=64, dropout=dropout)

        self.upconv4 = self.upconv(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.upconv1 = self.upconv(128, 64)
        
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
    	return nn.Sequential(
        	nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        	#nn.BatchNorm2d(out_channels),  # Add BatchNorm here
        	nn.ReLU(inplace=True),
        	nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        	#nn.BatchNorm2d(out_channels),  # Add BatchNorm here
        	nn.ReLU(inplace=True)
    	)
	
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        # Center
        center = self.center(F.max_pool2d(e4, 2))
        
        
        # Decoder with ViT Attention
        d4 = self.upconv4(center)
        e4_vit = self.vit1(e4)
        e4_vit = F.interpolate(e4_vit, size=d4.shape[2:])
        d4 = torch.cat([d4, e4_vit], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        e3_vit = self.vit2(e3)
        e3_vit = F.interpolate(e3_vit, size=d3.shape[2:])
        d3 = torch.cat([d3, e3_vit], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        e2_vit = self.vit3(e2)
        e2_vit = F.interpolate(e2_vit, size=d2.shape[2:])
        d2 = torch.cat([d2, e2_vit], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return out   
