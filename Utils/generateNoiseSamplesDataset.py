import torch
import os
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

# Import your model definitions from original code
from teacherRectifiedFlow import UNet, RectifiedFlow  # Assuming your original file is named this

class NoiseDataset(Dataset):
    def __init__(self, num_samples, image_size=(3, 32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.noise = torch.randn(num_samples, *image_size)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.noise[idx]

def generate_teacher_pairs(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
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
    
    rf = RectifiedFlow(model, timesteps=1000).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save noise vectors for reproducibility
    dataset = NoiseDataset(args.num_samples)
    torch.save(dataset.noise, os.path.join(args.save_dir, "noise_vectors.pt"))
    
    # Generate images in batches
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    all_images = []
    for batch in tqdm(dataloader, desc="Generating teacher samples"):
        noise_batch = batch.to(device)
        with torch.no_grad():
            images = rf.sample(
                noise=noise_batch,
                num_steps=args.num_steps
            )
        all_images.append(images.cpu())
    
    # Save final images
    teacher_images = torch.cat(all_images, dim=0)
    torch.save(teacher_images, os.path.join(args.save_dir, "teacher_images.pt"))
    
    print(f"Saved {len(teacher_images)} teacher pairs to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="teacher_dataset")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=1000)
    args = parser.parse_args()
    
    generate_teacher_pairs(args)
