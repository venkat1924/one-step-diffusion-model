import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torchvision

# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# Sinusoidal position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

# Adaptive Group Normalization
class AdaGN(nn.Module):
    def __init__(self, groups, in_channels, time_emb_dim):
        super().__init__()
        self.gn = nn.GroupNorm(groups, in_channels, affine=False)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels * 2)
        )

    def forward(self, x, t_emb):
        params = self.time_mlp(t_emb)
        scale, shift = params.chunk(2, dim=1)
        x = self.gn(x)
        return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

# Residual Block with AdaGN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.ada_gn1 = AdaGN(groups, out_channels, time_emb_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.ada_gn2 = AdaGN(groups, out_channels, time_emb_dim)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.ada_gn1(self.conv1(x), t_emb))
        h = self.ada_gn2(self.conv2(h), t_emb)
        return h + self.res_conv(x)

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C//self.num_heads, H*W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, C//self.num_heads, H*W)
        v = v.view(B, self.num_heads, C//self.num_heads, H*W).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        h = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(h)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_channels=64, 
                 time_emb_dim=256, channel_mult=(1,2,4,8), num_res_blocks=2,
                 num_heads=4, attn_resolutions=(16,)):
        super().__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Downsampling path
        self.init_conv = nn.Conv2d(in_channels, init_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.down_ch = []  # Stores channels from skip connections
        
        current_ch = init_channels
        for i, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                target_ch = mult * init_channels
                self.down_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(current_ch, target_ch, time_emb_dim),
                        AttentionBlock(target_ch, num_heads) if mult in attn_resolutions else nn.Identity()
                    ])
                )
                self.down_ch.append(target_ch)
                current_ch = target_ch
            
            if i != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(current_ch, current_ch, 3, stride=2, padding=1))
        
        # Middle blocks
        self.middle_block1 = ResidualBlock(current_ch, current_ch, time_emb_dim)
        self.middle_attn = AttentionBlock(current_ch, num_heads)
        self.middle_block2 = ResidualBlock(current_ch, current_ch, time_emb_dim)
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            target_ch = mult * init_channels
            for _ in range(num_res_blocks):
                prev_ch = self.down_ch.pop() if self.down_ch else current_ch
                self.up_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(current_ch + prev_ch, target_ch, time_emb_dim),
                        AttentionBlock(target_ch, num_heads) if mult in attn_resolutions else nn.Identity(),
                        nn.ConvTranspose2d(target_ch, target_ch, 4, stride=2, padding=1) if i != 0 else nn.Identity()
                    ])
                )
                current_ch = target_ch
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, current_ch),
            nn.SiLU(),
            nn.Conv2d(current_ch, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        skips = []
        
        x = self.init_conv(x)
        
        for layer in self.down_blocks:
            if isinstance(layer, nn.ModuleList):
                res, attn = layer
                x = res(x, t_emb)
                x = attn(x)
                skips.append(x)
            else:
                x = layer(x)
        
        x = self.middle_block1(x, t_emb)
        x = self.middle_attn(x)
        x = self.middle_block2(x, t_emb)
        
        for layer in self.up_blocks:
            res, attn, upsample = layer
            skip = skips.pop()
            if x.size()[-2:] != skip.size()[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = res(x, t_emb)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x)

class RectifiedFlow(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

    def p_losses(self, x_start, noise, t):
        t_scaled = t * (self.timesteps - 1)
        x_t = (1 - t.view(-1, 1, 1, 1)) * x_start + t.view(-1, 1, 1, 1) * noise
        predicted_velocity = self.model(x_t, t_scaled)
        loss = F.mse_loss(predicted_velocity, (noise - x_start))
        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w = x.shape
        t = torch.rand((b,), device=x.device, dtype=torch.float32)
        noise = torch.randn_like(x)
        return self.p_losses(x, noise, t)

    @torch.no_grad()
    def sample(self, noise=None, batch_size=16, image_size=(3, 32, 32), num_steps=100):
        device = next(self.model.parameters()).device
        if noise is None:
            noise = torch.randn((batch_size, *image_size), device=device)
        img = noise.clone()
        times = torch.linspace(1, 0, num_steps, device=device)
        for i in range(num_steps):
            t = times[i] * torch.ones(img.shape[0], device=device)
            t_scaled = t * (self.timesteps - 1)
            velocity = self.model(img, t_scaled)
            img = img - velocity * (1.0 / num_steps)
        return img

def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=64)

    # Create directories
    os.makedirs('output_images_100', exist_ok=True)
    os.makedirs('output_images_4', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    model = UNet(
        in_channels=3,
        out_channels=3,
        init_channels=32,
        time_emb_dim=256,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        num_heads=4,
        attn_resolutions=(16,)
    ).to(device)

    rectified_flow = RectifiedFlow(model, timesteps=1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Check for existing checkpoints
    start_epoch = 0
    checkpoint_path = 'latest_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    num_epochs = 4000
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        model.train()
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, _ = batch
            images = images.to(device)
            optimizer.zero_grad()
            loss = rectified_flow(images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        #torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, 'latest_checkpoint.pth')

        # Generate and save samples
        model.eval()
        # samples = rectified_flow.sample(batch_size=16, image_size=(3, 32, 32), num_steps=1000)
        # grid = torchvision.utils.make_grid(samples.cpu(), nrow=4, normalize=True)
        # plt.figure(figsize=(8,8))
        # plt.imshow(grid.permute(1, 2, 0))
        # plt.axis('off')
        # plt.title(f'Epoch {epoch+1}')
        # plt.savefig(f'output_images_1000/epoch_{epoch+1:02d}.png')
        # plt.close()

        samples = rectified_flow.sample(batch_size=16, image_size=(3, 32, 32), num_steps=100)
        grid = torchvision.utils.make_grid(samples.cpu(), nrow=4, normalize=True)
        plt.figure(figsize=(8,8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Epoch {epoch+1}')
        plt.savefig(f'output_images_100/epoch_{epoch+1:02d}.png')
        plt.close()

        samples = rectified_flow.sample(batch_size=16, image_size=(3, 32, 32), num_steps=4)
        grid = torchvision.utils.make_grid(samples.cpu(), nrow=4, normalize=True)
        plt.figure(figsize=(8,8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Epoch {epoch+1}')
        plt.savefig(f'output_images_4/epoch_{epoch+1:02d}.png')
        plt.close()

    # Save final model
    torch.save(model.state_dict(), "rectified_flow_model_final.pth")
    print("Training complete! Final model saved.")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
