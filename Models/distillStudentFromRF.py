import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm import tqdm
import argparse
import os
import torchvision

# -------------------------
#  Student Model Components
# -------------------------

class StaticGroupNorm(nn.Module):
    """Simplified normalization without time conditioning"""
    def __init__(self, channels, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, channels)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        return self.gn(x) * self.gamma + self.beta

class DepthwiseConv(nn.Module):
    """More efficient than standard convolutions"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        return self.pw(self.dw(x))

class EfficientAttention(nn.Module):
    """From 'Rethinking Attention in Knowledge Distillation' (ICCV 2023)"""
    def __init__(self, dim, heads=4, reduction=4):
        super().__init__()
        self.heads = heads
        self.reduction = reduction
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, 3*(dim//reduction), 1)
        self.proj = nn.Conv2d(dim//reduction, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(B, -1, H*W), qkv)
        
        attn = (q.transpose(-2,-1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (v @ attn.transpose(-2,-1)).view(B, -1, H, W)
        return self.proj(x)

class StudentBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseConv(in_ch, out_ch),
            StaticGroupNorm(out_ch),
            nn.SiLU(),
            DepthwiseConv(out_ch, out_ch),
            StaticGroupNorm(out_ch)
        )
        self.attn = EfficientAttention(out_ch) if use_attn else None
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv(x)
        if self.attn: h = self.attn(h)
        return h + self.res(x)

# --------------------------
#  Student Model Architecture
# --------------------------

class StudentModel(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, init_ch=24):
        super().__init__()
        
        # Encoder
        self.stem = nn.Conv2d(in_ch, init_ch, 3, padding=1)
        self.down1 = nn.Sequential(
            StudentBlock(init_ch, 24),
            nn.Conv2d(24, 24, 3, stride=2, padding=1)  # 32→16
        )
        self.down2 = nn.Sequential(
            StudentBlock(24, 48, use_attn=True),
            nn.Conv2d(48, 48, 3, stride=2, padding=1)  # 16→8
        )
        self.down3 = nn.Sequential(
            StudentBlock(48, 96),
            nn.Conv2d(96, 96, 3, stride=2, padding=1)  # 8→4
        )

        # Bottleneck
        self.bottleneck = StudentBlock(96, 96, use_attn=True)

        # Decoder (fixed channel dimensions)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 4→8
            StudentBlock(96*2, 48)  # 96 (bottleneck) + 96 (skip) = 192
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8→16
            StudentBlock(48*2, 24)  # 48 + 48 = 96
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 16→32
            StudentBlock(24*2, init_ch)  # 24 + 24 = 48
        )

        self.final = nn.Sequential(
            StaticGroupNorm(init_ch),
            nn.SiLU(),
            nn.Conv2d(init_ch, out_ch, 3, padding=1)
        )
    def forward(self, x):
        # Encoder
        x0 = self.stem(x)        # 32x32 → 32x32
        x1 = self.down1(x0)       # 32x32 → 16x16
        x2 = self.down2(x1)       # 16x16 → 8x8
        x3 = self.down3(x2)       # 8x8 → 4x4
        
        # Bottleneck
        b = self.bottleneck(x3)   # 4x4
        
        # Decoder with skip connections
        d = self.up1(torch.cat([b, x3], dim=1))  # 96+96=192ch → 48ch
        d = self.up2(torch.cat([d, x2], dim=1))  # 48+48=96ch → 24ch
        d = self.up3(torch.cat([d, x1], dim=1))  # 24+24=48ch → 24ch
        
        return self.final(d)

# --------------------------
#  Dataset & Loss Functions
# --------------------------

class TeacherDataset(Dataset):
    def __init__(self, noise_path, image_path):
        self.noise = torch.load(noise_path)
        self.images = torch.load(image_path)
        
        assert len(self.noise) == len(self.images), \
            f"Mismatched dataset sizes: {len(self.noise)} vs {len(self.images)}"
        assert self.noise.shape[1:] == (3,32,32), \
            f"Unexpected noise shape: {self.noise.shape}"
        assert self.images.shape[1:] == (3,32,32), \
            f"Unexpected image shape: {self.images.shape}"

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, idx):
        return self.noise[idx], self.images[idx]

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights = 'DEFAULT').features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, input, target):
        # Normalize for VGG
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Use multiple feature layers
        losses = []
        x, y = input, target
        for layer in self.vgg:
            x = layer(x)
            y = layer(y)
            if isinstance(layer, nn.MaxPool2d):
                losses.append(F.mse_loss(x, y))
        return sum(losses) / len(losses)

# --------------------------
#  Training Utilities
# --------------------------

def validate(model, device, save_path="student_samples.png"):
    """Generate validation samples"""
    model.eval()
    with torch.no_grad():
        noise = torch.randn(16, 3, 32, 32).to(device)
        samples = model(noise).cpu()
        torchvision.utils.save_image(samples, save_path, nrow=4, normalize=True)
    model.train()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    dataset = TeacherDataset(args.noise_path, args.image_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # Model & Loss
    model = StudentModel().to(device)
    mse_loss = nn.MSELoss()
    percep_loss = PerceptualLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Mixed Precision
    scaler = torch.amp.GradScaler()

    start_epoch = 0
    checkpoint_path = 'distillationCheckpoints/latest_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (noise, target) in enumerate(pbar):
            noise = noise.to(device)
            target = target.to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                pred = model(noise)
                loss_mse = mse_loss(pred, target)
                loss_percep = percep_loss(pred, target)
                loss = 0.8*loss_mse + 0.2*loss_percep
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation & Checkpoint
        if (epoch+1) % args.save_freq == 0:
            validate(model, device, f"student_samples/epoch{epoch+1}.png")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"distillationCheckpoints/latest_checkpoint.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_path", type=str, default="teacher_dataset/noise_vectors.pt")
    parser.add_argument("--image_path", type=str, default="teacher_dataset/teacher_images.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=10)
    args = parser.parse_args()
    
    # Safety Checks
    assert os.path.exists(args.noise_path), f"Noise file {args.noise_path} not found!"
    assert os.path.exists(args.image_path), f"Image file {args.image_path} not found!"
    
    main(args)
