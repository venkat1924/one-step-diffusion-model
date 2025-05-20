import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import argparse
import os
import math
import sys
import datetime
import torchvision

# bro please use tmux to run this, else the resources are just getting used up for no reason. and thank you -- concerned co-gpu user

class TrainingConfig:
    def __init__(self):
        self.batch_size = 256
        self.lr = 1e-5
        self.epochs = 5000
        self.save_interval = 10  # Save every 10 epochs
        self.sample_interval = 5 # Generate samples every 5 epochs
        self.checkpoint_dir = "student_checkpoints"
        self.sample_dir = "student_samples"
        self.resume_checkpoint = None  # Path to resume training
        
        # Automatic mixed precision
        self.use_amp = True
        
        # Loss weights
        self.loss_alpha = 0.5  # MSE weight
        self.loss_beta = 0.5   # Perceptual weight

# --------------------------
#  Student Model Components
# --------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, t):
        t = t[:, None] * self.emb[None, :]
        return torch.cat([t.sin(), t.cos()], dim=-1)

class TimeAwareGroupNorm(nn.Module):
    def __init__(self, channels, time_dim=256, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, channels, affine=False)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2)
        )

    def forward(self, x, t_emb):
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        return self.gn(x) * (1 + scale[:,:,None,None]) + shift[:,:,None,None]

class StudentBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=256, use_attn=False):
        super().__init__()
        # Split sequential layers into components
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = TimeAwareGroupNorm(out_ch, time_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = TimeAwareGroupNorm(out_ch, time_dim)
        self.attn = nn.Conv2d(out_ch, out_ch, 1) if use_attn else None
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        # First convolution block
        h = self.conv1(x)
        h = self.norm1(h, t_emb)  # Explicitly pass time embedding
        h = self.act(h)
        
        # Second convolution block
        h = self.conv2(h)
        h = self.norm2(h, t_emb)  # Explicitly pass time embedding
        
        # Optional attention
        if self.attn:
            h = h + self.attn(h)
            
        return h + self.res_conv(x)

class StudentModel(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, init_ch=24, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Encoder
        self.stem = nn.Conv2d(in_ch, init_ch, 3, padding=1)
        
        # Downsample blocks
        self.down1 = nn.ModuleList([
            StudentBlock(init_ch, 24, time_dim),
            nn.Conv2d(24, 24, 3, stride=2, padding=1)
        ])
        self.down2 = nn.ModuleList([
            StudentBlock(24, 48, time_dim, use_attn=True),
            nn.Conv2d(48, 48, 3, stride=2, padding=1)
        ])
        self.down3 = nn.ModuleList([
            StudentBlock(48, 96, time_dim),
            nn.Conv2d(96, 96, 3, stride=2, padding=1)
        ])

        # Bottleneck
        self.bottleneck = StudentBlock(96, 96, time_dim, use_attn=True)

        # Upsample blocks
        self.up1 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            StudentBlock(96*2, 48, time_dim)  # Skip connection from encoder
        ])
        self.up2 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            StudentBlock(48*2, 24, time_dim)
        ])
        self.up3 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            StudentBlock(24*2, init_ch, time_dim)
        ])

        # Final layers with explicit time handling
        self.final_norm = TimeAwareGroupNorm(init_ch, time_dim)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(init_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        # Encoder
        x0 = self.stem(x)
        x1 = self.down1[0](x0, t_emb)
        x1 = self.down1[1](x1)
        x2 = self.down2[0](x1, t_emb)
        x2 = self.down2[1](x2)
        x3 = self.down3[0](x2, t_emb)
        x3 = self.down3[1](x3)
        
        # Bottleneck
        b = self.bottleneck(x3, t_emb)
        
        # Decoder with skip connections
        d = self.up1[0](torch.cat([b, x3], 1))
        d = self.up1[1](d, t_emb)
        d = self.up2[0](torch.cat([d, x2], 1))
        d = self.up2[1](d, t_emb)
        d = self.up3[0](torch.cat([d, x1], 1))
        d = self.up3[1](d, t_emb)
        
        # Final processing with time embedding
        d = self.final_norm(d, t_emb)
        d = self.final_act(d)
        return self.final_conv(d)

# --------------------------
#  Hybrid Loss Function
# --------------------------

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

        # Load Inception v3 with full architecture
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            init_weights=False
        )
        inception.aux_logits = False
        inception.AuxLogits = None
        inception.eval()

        # Modified feature extractor for 32x32 compatibility
        self.feature_extractor = nn.Sequential(
            inception.Conv2d_1a_3x3,       # [N, 32, 30, 30]
            inception.Conv2d_2a_3x3,       # [N, 32, 28, 28]
            inception.Conv2d_2b_3x3,       # [N, 64, 26, 26]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 64, 12, 12]
            inception.Conv2d_3b_1x1,       # [N, 80, 12, 12]
            inception.Conv2d_4a_3x3,       # [N, 192, 10, 10]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 192, 4, 4]
            inception.Mixed_5b,            # [N, 256, 4, 4]
            inception.Mixed_5c,            # [N, 288, 4, 4]
            inception.Mixed_5d,            # [N, 288, 4, 4]
            nn.AdaptiveAvgPool2d((2, 2))   # [N, 288, 2, 2]
        )
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Normalization parameters
        self.register_buffer('mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Input shapes: [batch, 3, 32, 32]
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Add padding to maintain valid dimensions
        pred_padded = F.pad(pred_norm, (1,1,1,1), mode='constant', value=0)
        target_padded = F.pad(target_norm, (1,1,1,1), mode='constant', value=0)
        
        with torch.no_grad():
            feat_pred = self.feature_extractor(pred_padded)
            feat_target = self.feature_extractor(target_padded)
        
        perceptual_loss = F.l1_loss(
            feat_pred.flatten(start_dim=1),
            feat_target.flatten(start_dim=1)
        )
        
        return self.alpha * self.mse(pred, target) + self.beta * perceptual_loss

# --------------------------
#  Dataset Class
# --------------------------

class RectifiedFlowDataset(Dataset):
    def __init__(self, noise_path, image_path):
        self.noise = torch.load(noise_path)
        self.images = torch.load(image_path)
        
        # Dimension validation
        assert self.noise.shape[1:] == (3, 32, 32), \
            f"Noise tensor has invalid shape: {self.noise.shape}"
        assert self.images.shape[1:] == (3, 32, 32), \
            f"Image tensor has invalid shape: {self.images.shape}"
        assert len(self.noise) == len(self.images), \
            f"Mismatched dataset sizes: {len(self.noise)} vs {len(self.images)}"

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, idx):
        noise = self.noise[idx]
        image = self.images[idx]
        t = torch.rand(1)  # Random timestep ∈ [0,1]
        x_t = (1 - t) * noise + t * image
        return x_t, image, t.squeeze()

# [TrainingConfig, save_checkpoint, load_checkpoint, generate_samples, 
#  train_student, and main() remain identical to previous version]

def save_checkpoint(model, optimizer, epoch, loss, config):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
        'config': vars(config)
    }
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Save latest
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, "latest.pth"))
    
    # Save historical
    if (epoch + 1) % config.save_interval == 0:
        torch.save(checkpoint, 
                 os.path.join(config.checkpoint_dir, f"epoch_{epoch+1:04d}.pth"))

def load_checkpoint(config, device):
    if config.resume_checkpoint:
        path = config.resume_checkpoint
    else:
        path = os.path.join(config.checkpoint_dir, "latest.pth")
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        return checkpoint
    return None

# --------------------------
#  Core Training Loop
# --------------------------

def train_student(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------
    #  Initialization
    # ---------------
    
    # Dataset & Model
    dataset = RectifiedFlowDataset(config.noise_path, config.image_path)
    loader = DataLoader(dataset, 
                       batch_size=config.batch_size,
                       shuffle=True,
                       pin_memory=True,
                       num_workers=4)
    
    model = StudentModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    loss_fn = HybridLoss(alpha=config.loss_alpha).to(device)
    
    scaler = torch.amp.GradScaler(enabled=config.use_amp)
    
    # ---------------
    #  Resume Logic
    # ---------------
    start_epoch = 0
    checkpoint = load_checkpoint(config, device)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # ---------------
    #  Training Loop
    # ---------------
    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        model.train()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for x_t, target, t in pbar:
            x_t = x_t.to(device)
            target = target.to(device)
            t = t.to(device)
            
            # Mixed precision forward
            with torch.amp.autocast(device_type='cuda', enabled=config.use_amp):
                pred = model(x_t, t)
                loss = loss_fn(pred, target)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update tracking
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # ---------------
        #  Epoch Finalization
        # ---------------
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | {datetime.datetime.now()}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, avg_loss, config)
        
        # Generate samples
        if (epoch + 1) % config.sample_interval == 0:
            generate_samples(model, device, config, epoch)
    
    print("Training completed!")

def generate_samples(model, device, config, epoch):
    model.eval()
    with torch.no_grad():
        # Generate 16 samples (4x4 grid)
        noise = torch.randn(16, 3, 32, 32).to(device)
        x = noise.clone()
        
        # 4-step sampling process
        for step in range(4):
            t = torch.full((16,), step/4).to(device)
            pred = model(x, t)
            x = (1 - (step+1)/4) * x + ((step+1)/4) * pred
        
        # Save images
        os.makedirs(config.sample_dir, exist_ok=True)
        grid = torchvision.utils.make_grid(x.cpu(), nrow=4, normalize=True)
        torchvision.utils.save_image(
            grid, 
            os.path.join(config.sample_dir, f"epoch_{epoch+1:04d}.png")
        )
    model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_path", type=str, default="teacher_dataset/noise_vectors.pt")
    parser.add_argument("--image_path", type=str, default="teacher_dataset/teacher_images.pt")
    parser.add_argument("--resume", type=str, default="student_checkpoints/latest.pth")
    args = parser.parse_args()
    
    # Safety checks
    if not os.path.exists(args.noise_path):
        raise FileNotFoundError(f"Noise file {args.noise_path} not found!")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file {args.image_path} not found!")
    
    # Initialize config
    config = TrainingConfig()
    config.noise_path = args.noise_path
    config.image_path = args.image_path
    config.resume_checkpoint = args.resume
    
    # Start training
    train_student(config)
