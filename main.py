import os
import glob
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self attention layer for the generator."""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Reshape for matrix multiplication
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma * out + x

class Generator(nn.Module):
    """Generator network for image inpainting.
    
    A lightweight U-Net style architecture with skip connections and attention.
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        # Reduced number of filters
        nf = 16  # base number of filters
        
        # Encoder (with downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, nf, 4, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, 4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Attention layer at the bottleneck (16x16)
        self.attention = SelfAttention(nf*4)
        
        # Bottleneck (at 16x16)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(nf*4, nf*4, 3, padding=1),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder (with upsampling)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(nf, 3, 4, stride=2, padding=1),  # 64 -> 128
            nn.Tanh()
        )
        
    def forward(self, x, mask=None):
        if mask is None:
            # During training, extract mask from the 4th channel
            mask = x[:, 3:4]  # Get the mask
            x = x[:, :3]      # Get the RGB channels
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Apply attention at 16x16 resolution
        e3 = self.attention(e3)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder
        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        # Only output the hole region
        return d1 * (1 - mask[:, :3])

class Discriminator(nn.Module):
    """Discriminator network.
    
    A lightweight PatchGAN discriminator.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Reduced number of filters
        nf = 32  # base number of filters
        
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(nf, nf*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(nf*2, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class CelebDataset(Dataset):
    """Custom Dataset for loading CelebA face images with masks for inpainting."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create a random hole mask (32x32)
        mask = torch.zeros((4, 128, 128))  # 4 channels: RGB mask + hole position
        hole_h = random.randint(0, 128-32)
        hole_w = random.randint(0, 128-32)
        
        # Set the RGB mask channels
        mask[:3, hole_h:hole_h+32, hole_w:hole_w+32] = 1
        
        # Set the hole position channel
        mask[3, hole_h:hole_h+32, hole_w:hole_w+32] = 1
        
        # Apply the mask to create the masked image
        masked_image = image * (1 - mask[:3])
        
        # Add the mask as the 4th channel
        masked_image = torch.cat([masked_image, mask[3:]], dim=0)
        
        return image, masked_image, mask

def calculate_ssim(img1, img2, pos_mask):
    """Calculate Structural Similarity Index between two images.
    Only calculates SSIM for the hole region plus padding."""
    # Convert to numpy arrays
    img1 = ((img1.detach().cpu().numpy() + 1) / 2).transpose(1, 2, 0)
    img2 = ((img2.detach().cpu().numpy() + 1) / 2).transpose(1, 2, 0)
    pos_mask = pos_mask.detach().cpu().numpy()[3]  # Get hole position from alpha channel
    
    # Find hole boundaries
    y_indices, x_indices = np.where(pos_mask > 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Add padding (10% of hole size)
    pad = int(0.1 * (y_max - y_min))
    y_min = max(0, y_min - pad)
    y_max = min(img1.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(img1.shape[1], x_max + pad)
    
    # Crop both images to hole region plus padding
    img1_hole = img1[y_min:y_max, x_min:x_max, :]
    img2_hole = img2[y_min:y_max, x_min:x_max, :]
    
    return ssim(img1_hole, img2_hole, channel_axis=2, data_range=1.0)

def load_and_split_dataset(data_dir, max_files=None, batch_size=32):
    """Load and split the dataset into train, validation, and test sets.
    Ensures each split has at least one batch."""
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if max_files:
        # Ensure max_files is a multiple of batch_size * 3 (for train/val/test)
        max_files = (max_files // (batch_size * 3)) * (batch_size * 3)
        if max_files < batch_size * 3:
            max_files = batch_size * 3  # Minimum size to ensure one batch per split
        image_paths = image_paths[:max_files]
    
    total_size = len(image_paths)
    if total_size < batch_size * 3:
        raise ValueError(f"Need at least {batch_size * 3} images for training (got {total_size})")
    
    # Ensure each split gets at least one batch
    min_split_size = batch_size
    remaining_size = total_size - (min_split_size * 3)  # Reserve one batch for each split
    
    # Distribute remaining samples proportionally (60/20/20)
    extra_train = (remaining_size * 60 // 100) // batch_size * batch_size
    extra_val = (remaining_size * 20 // 100) // batch_size * batch_size
    extra_test = (remaining_size * 20 // 100) // batch_size * batch_size
    
    train_size = min_split_size + extra_train
    val_size = min_split_size + extra_val
    test_size = min_split_size + extra_test
    
    print(f"\nDataset split sizes (batch_size={batch_size}):")
    print(f"Total images: {total_size}")
    print(f"Train size: {train_size} ({train_size//batch_size} batches)")
    print(f"Val size: {val_size} ({val_size//batch_size} batches)")
    print(f"Test size: {test_size} ({test_size//batch_size} batches)\n")
    
    # Split paths
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:train_size + val_size]
    test_paths = image_paths[train_size + val_size:train_size + val_size + test_size]
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CelebDataset(train_paths, transform=transform)
    val_dataset = CelebDataset(val_paths, transform=transform)
    test_dataset = CelebDataset(test_paths, transform=transform)
    
    return train_dataset, val_dataset, test_dataset

def train_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion_gan, criterion_pixel, device):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_pixel_loss = 0
    total_ssim = 0
    num_batches = 0
    
    for batch_idx, (real_imgs, masked_imgs, masks) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        masked_imgs = masked_imgs.to(device)
        masks = masks.to(device)
        batch_size = real_imgs.size(0)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        # Generate inpainted image
        gen_imgs = generator(masked_imgs)
        
        # Composite the generated hole with the original image
        hole_mask = masks[:, 3].unsqueeze(1)  # Get the hole position
        composited_imgs = masked_imgs[:, :3] + gen_imgs * (1 - masks[:, :3])
        
        # Get discriminator outputs
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(composited_imgs.detach())
        
        # Discriminator loss
        d_real_loss = criterion_gan(real_validity, torch.ones_like(real_validity))
        d_fake_loss = criterion_gan(fake_validity, torch.zeros_like(fake_validity))
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        
        # Generator adversarial loss
        fake_validity = discriminator(composited_imgs)
        g_adv_loss = criterion_gan(fake_validity, torch.ones_like(fake_validity))
        
        # Pixel-wise loss (only for the hole region)
        hole_region_real = real_imgs * (1 - masks[:, :3])
        hole_region_fake = gen_imgs * (1 - masks[:, :3])
        pixel_loss = criterion_pixel(hole_region_fake, hole_region_real)
        
        # Total generator loss
        g_loss = g_adv_loss + 100 * pixel_loss
        
        g_loss.backward()
        g_optimizer.step()
        
        # Calculate SSIM for the hole region
        ssim_val = calculate_ssim(
            real_imgs[0].cpu(),
            composited_imgs[0].detach().cpu(),
            masks[0].cpu()
        )
        
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_pixel_loss += pixel_loss.item()
        total_ssim += ssim_val
        num_batches += 1
    
    return (total_g_loss / num_batches, 
            total_d_loss / num_batches,
            total_pixel_loss / num_batches,
            total_ssim / num_batches)

def validate(generator, discriminator, val_loader, criterion_gan, criterion_pixel, device):
    """Validate the model."""
    generator.eval()
    discriminator.eval()
    
    total_g_loss = 0
    total_pixel_loss = 0
    total_ssim = 0
    num_batches = 0
    
    with torch.no_grad():
        for real_imgs, masked_imgs, masks in val_loader:
            real_imgs = real_imgs.to(device)
            masked_imgs = masked_imgs.to(device)
            masks = masks.to(device)
            
            # Generate inpainted image
            gen_imgs = generator(masked_imgs)
            
            # Composite the generated hole with the original image
            composited_imgs = masked_imgs[:, :3] + gen_imgs * (1 - masks[:, :3])
            
            # Generator adversarial loss
            fake_validity = discriminator(composited_imgs)
            g_adv_loss = criterion_gan(fake_validity, torch.ones_like(fake_validity))
            
            # Pixel-wise loss (only for the hole region)
            hole_region_real = real_imgs * (1 - masks[:, :3])
            hole_region_fake = gen_imgs * (1 - masks[:, :3])
            pixel_loss = criterion_pixel(hole_region_fake, hole_region_real)
            
            # Total generator loss
            g_loss = g_adv_loss + 100 * pixel_loss
            
            # Calculate SSIM for the hole region
            ssim_val = calculate_ssim(
                real_imgs[0].cpu(),
                composited_imgs[0].detach().cpu(),
                masks[0].cpu()
            )
            
            total_g_loss += g_loss.item()
            total_pixel_loss += pixel_loss.item()
            total_ssim += ssim_val
            num_batches += 1
    
    return (total_g_loss / num_batches,
            total_pixel_loss / num_batches,
            total_ssim / num_batches)

def evaluate_and_display(generator, test_loader, device, num_images=3):
    """Display original, masked, and inpainted images side by side."""
    generator.eval()
    
    # Get a batch of test images
    real_images, masked_images, pos_masks = next(iter(test_loader))
    real_images = real_images[:num_images].to(device)
    masked_images = masked_images[:num_images].to(device)
    masks = pos_masks[:, :3][:num_images].to(device)  # RGB channels contain the mask
    
    with torch.no_grad():
        generated_images = generator(masked_images)
        completed_images = masked_images[:, :3] + generated_images * (1 - masks)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4*num_images))
    
    for i in range(num_images):
        # Convert images back to display format
        real_img = ((real_images[i].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        masked_img = ((masked_images[i, :3].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        completed_img = ((completed_images[i].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        
        # Display images
        axes[i, 0].imshow(real_img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masked_img)
        axes[i, 1].set_title('Masked')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(completed_img)
        axes[i, 2].set_title('Inpainted')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('inpainting_results.png')
    plt.close()

def print_model_summary(model):
    """Print a summary of the model architecture and parameters."""
    print(f"\nModel Summary for {model.__class__.__name__}:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Output Shape':<20} {'Param #'}")
    print("-" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_children():
        # Get parameters for this layer
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Get output shape
        if hasattr(module, 'weight'):
            shape = list(module.weight.shape)
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                shape = [None, shape[0], None, None]  # Batch size and spatial dims are dynamic
            shape_str = str(shape)
        else:
            shape_str = "multiple"
        
        print(f"{name:<40} {shape_str:<20} {params:,}")
        
        total_params += params
        trainable_params += trainable
    
    print("-" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 80)

if __name__ == "__main__":
    # Configuration
    data_dir = "data_celeb"
    batch_size = 16
    max_files = batch_size * 6  # Ensure 2 batches per split
    epochs = 5
    
    # Check for AMD GPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        device = torch.device("cuda")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load and split the dataset
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(data_dir, max_files=max_files, batch_size=batch_size)
    
    print(f"Dataset splits (using {max_files} files):")
    print(f"Train: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    # Create dataloaders with more workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)  # Disabled pin_memory for CPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Print model summaries
    print_model_summary(generator)
    print_model_summary(discriminator)
    
    # Setup optimizers and losses
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()  # L1 loss for pixel-wise reconstruction
    
    # Training history
    train_g_losses = []
    train_d_losses = []
    train_pixel_losses = []
    train_ssims = []
    val_g_losses = []
    val_pixel_losses = []
    val_ssims = []
    
    # Training loop
    start_time = time.time()
    best_val_ssim = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_g_loss, train_d_loss, train_pixel_loss, train_ssim = train_epoch(
            generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion_gan, criterion_pixel, device
        )
        train_g_losses.append(train_g_loss)
        train_d_losses.append(train_d_loss)
        train_pixel_losses.append(train_pixel_loss)
        train_ssims.append(train_ssim)
        
        # Validation
        val_g_loss, val_pixel_loss, val_ssim = validate(
            generator, discriminator, val_loader, criterion_gan, criterion_pixel, device
        )
        val_g_losses.append(val_g_loss)
        val_pixel_losses.append(val_pixel_loss)
        val_ssims.append(val_ssim)
        
        # Save best model
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            print(f"Saving best model with SSIM: {val_ssim:.4f}")
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch,
                'val_ssim': val_ssim,
            }, 'best_model.pth')
        
        epoch_time = time.time() - epoch_start
        total_time = epoch_time * epoch
        
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Train - G_loss: {train_g_loss:.4f}, D_loss: {train_d_loss:.4f}, Pixel_loss: {train_pixel_loss:.4f}, SSIM: {train_ssim:.4f}")
        print(f"Val - G_loss: {val_g_loss:.4f}, Pixel_loss: {val_pixel_loss:.4f}, SSIM: {val_ssim:.4f}")
        print(f"Time - Epoch: {epoch_time:.1f}s, Total: {total_time:.1f}s\n")
    
    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'epoch': epochs,
        'val_ssim': val_ssim,
    }, 'final_model.pth')
    
    print("\nGenerating inpainting results...")
    evaluate_and_display(generator, test_loader, device)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_g_losses, label='Train G Loss')
    plt.plot(train_d_losses, label='Train D Loss')
    plt.plot(val_g_losses, label='Val G Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Adversarial Loss')
    plt.legend()
    plt.title('Generator and Discriminator Losses')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_pixel_losses, label='Train Pixel Loss')
    plt.plot(val_pixel_losses, label='Val Pixel Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Loss')
    plt.legend()
    plt.title('Pixel-wise Reconstruction Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(train_ssims, label='Train SSIM')
    plt.plot(val_ssims, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('Structural Similarity Index')
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.close()

    evaluate_and_display(generator, test_loader, device)