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
    A U-Net style architecture with skip connections and attention."""
    def __init__(self, nf=16):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, nf, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf*2, nf*2, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf*4, nf*4, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Attention
        self.attention = SelfAttention(in_dim=nf*4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(nf*4, nf*4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf*4, nf*4, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(nf*8, nf*4, 3, padding=1),  # nf*8 because of skip connection
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf*4, nf*4, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(nf*6, nf*2, 3, padding=1),  # nf*6 because of skip connection
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf*2, nf*2, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(nf*3, nf, 3, padding=1),    # nf*3 because of skip connection
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = x[:, 3:4]  # Get the mask
            x = x[:, :3]      # Get the RGB channels
            x = torch.cat([x, mask], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Attention
        att = self.attention(p3)
        
        # Bottleneck
        b = self.bottleneck(att)
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1)
        
        # Only output the hole region
        return d1 * mask[:, :3]

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
            nn.Conv2d(nf*2, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

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

def edge_aware_loss(real_imgs, fake_imgs, mask):
    """Calculate edge-aware loss between real and fake images in the hole region."""
    # Simple Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=real_imgs.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sobel_y = sobel_x.transpose(2, 3)
    
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    
    # Detect edges in both images
    real_edges_x = F.conv2d(real_imgs, sobel_x, padding=1, groups=3)
    real_edges_y = F.conv2d(real_imgs, sobel_y, padding=1, groups=3)
    fake_edges_x = F.conv2d(fake_imgs, sobel_x, padding=1, groups=3)
    fake_edges_y = F.conv2d(fake_imgs, sobel_y, padding=1, groups=3)
    
    # Combine edges with numerical stability
    real_edges = torch.sqrt(real_edges_x.pow(2) + real_edges_y.pow(2) + eps)
    fake_edges = torch.sqrt(fake_edges_x.pow(2) + fake_edges_y.pow(2) + eps)
    
    # Normalize edge responses to [0, 1] range
    real_edges = real_edges / (torch.max(real_edges) + eps)
    fake_edges = fake_edges / (torch.max(fake_edges) + eps)
    
    # Only compare edges in the hole region
    edge_loss = F.l1_loss(real_edges * mask[:, :3], fake_edges * mask[:, :3])
    
    # Check for NaN and return zero if found
    if torch.isnan(edge_loss):
        return torch.tensor(0.0, device=real_imgs.device)
    
    return edge_loss

def load_and_split_dataset(data_dir, max_files=None, batch_size=32):
    """Load and split the dataset into train, validation, and test sets (80/10/10 split)."""
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if max_files:
        image_paths = image_paths[:max_files]
    
    total_size = len(image_paths)
    
    # Calculate split sizes (80/10/10)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size  # Remainder to ensure we use all images
    
    print(f"\nDataset split sizes:")
    print(f"Total images: {total_size}")
    print(f"Train size: {train_size} ({train_size//batch_size} batches)")
    print(f"Val size: {val_size} ({val_size//batch_size} batches)")
    print(f"Test size: {test_size} ({test_size//batch_size} batches)\n")
    
    # Split paths
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:train_size + val_size]
    test_paths = image_paths[train_size + val_size:]
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CelebDataset(train_paths, transform=transform)
    val_dataset = CelebDataset(val_paths, transform=transform)
    test_dataset = CelebDataset(test_paths, transform=transform)
    
    return train_dataset, val_dataset, test_dataset

class CelebDataset(Dataset):
    """Custom Dataset for loading CelebA face images with masks for inpainting."""
    def __init__(self, image_paths, transform=None, fixed_hole=None, margin=16):
        self.image_paths = image_paths
        self.transform = transform
        self.fixed_hole = fixed_hole
        self.margin = margin
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create a hole mask (32x32)
        mask = torch.zeros((4, 128, 128))  # 4 channels: RGB mask + hole position
        
        if self.fixed_hole is not None:
            # Use fixed hole position
            hole_h, hole_w = self.fixed_hole
        else:
            # Random hole position with margin
            hole_h = random.randint(self.margin, 128-32-self.margin)
            hole_w = random.randint(self.margin, 128-32-self.margin)
        
        # Set the RGB mask channels
        mask[:3, hole_h:hole_h+32, hole_w:hole_w+32] = 1
        
        # Set the hole position channel
        mask[3, hole_h:hole_h+32, hole_w:hole_w+32] = 1
        
        # Create dilated mask by expanding hole region by x pixels in each direction
        context_size = 4
        h_start = max(0, hole_h - context_size)
        h_end = min(128, hole_h + 32 + context_size)
        w_start = max(0, hole_w - context_size)
        w_end = min(128, hole_w + 32 + context_size)
        
        dilated_mask = mask.clone()
        dilated_mask[:3, h_start:h_end, w_start:w_end] = 1
        dilated_mask[3, h_start:h_end, w_start:w_end] = 1
        
        # Apply the mask to create the masked image
        masked_image = image * (1 - mask[:3])
        
        # Add the dilated mask as the 4th channel
        masked_image = torch.cat([masked_image, dilated_mask[3:]], dim=0)
        
        return image, masked_image, mask, dilated_mask

def validate(generator, discriminator, val_loader, criterion_gan, criterion_pixel, device):
    """Validate the model."""
    generator.eval()
    discriminator.eval()
    
    total_g_loss = 0
    total_pixel_loss = 0
    total_edge_loss = 0
    total_ssim = 0
    num_batches = 0
    
    def normalize_image(img):
        """Normalize image to [0,1] range"""
        img = img.clamp(-1, 1)  # First clamp to [-1,1]
        img = (img + 1) / 2     # Then scale to [0,1]
        return img
    
    with torch.no_grad():
        for real_imgs, masked_imgs, masks, dilated_masks in val_loader:
            real_imgs = real_imgs.to(device)
            masked_imgs = masked_imgs.to(device)
            masks = masks.to(device)
            dilated_masks = dilated_masks.to(device)
            
            # Generate inpainted image
            gen_imgs = generator(masked_imgs)
            
            # Normalize images
            real_imgs = normalize_image(real_imgs)
            gen_imgs = normalize_image(gen_imgs)
            
            # Composite the generated hole with the original image using original mask
            composited_imgs = real_imgs * (1 - masks[:, :3]) + gen_imgs * masks[:, :3]
            
            # Generator adversarial loss
            fake_validity = discriminator(composited_imgs)
            g_adv_loss = criterion_gan(fake_validity, torch.ones_like(fake_validity))
            
            # Pixel-wise loss (use dilated mask for learning context)
            hole_region_real = real_imgs * dilated_masks[:, :3]
            hole_region_fake = gen_imgs * dilated_masks[:, :3]
            pixel_loss = criterion_pixel(hole_region_fake, hole_region_real)
            
            # Edge-aware loss with dilated context
            edge_loss = edge_aware_loss(real_imgs, composited_imgs, dilated_masks)
            
            # Total generator loss
            g_loss = g_adv_loss + 100 * pixel_loss + 25 * edge_loss
            
            # Calculate SSIM for the hole region (use original mask)
            ssim_val = calculate_ssim(
                real_imgs[0].cpu(),
                composited_imgs[0].detach().cpu(),
                masks[0].cpu()
            )
            
            total_g_loss += g_loss.item()
            total_pixel_loss += pixel_loss.item()
            total_edge_loss += edge_loss.item()
            total_ssim += ssim_val
            num_batches += 1
    
    return (total_g_loss / num_batches,
            total_pixel_loss / num_batches,
            total_edge_loss / num_batches,
            total_ssim / num_batches)

def train_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion_gan, criterion_pixel, device):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_pixel_loss = 0
    total_edge_loss = 0
    total_ssim = 0
    num_batches = 0
    
    # Set gradient clipping threshold
    max_grad_norm = 1.0
    
    # Track discriminator accuracy for dynamic training
    d_real_acc = []
    d_fake_acc = []
    
    for batch_idx, (real_imgs, masked_imgs, masks, dilated_masks) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        masked_imgs = masked_imgs.to(device)
        masks = masks.to(device)
        dilated_masks = dilated_masks.to(device)
        batch_size = real_imgs.size(0)
        
        # Only train discriminator if its accuracy is below threshold
        train_disc = True
        if len(d_real_acc) > 0 and len(d_fake_acc) > 0:
            avg_acc = (np.mean(d_real_acc[-50:]) + np.mean(d_fake_acc[-50:])) / 2
            if avg_acc > 0.8:  # If discriminator is too strong, skip training it
                train_disc = False
        
        # Train Discriminator
        if train_disc:
            d_optimizer.zero_grad()
            
            # Generate inpainted image
            gen_imgs = generator(masked_imgs)
            
            # Ensure images are in [-1, 1] range for discriminator
            real_imgs = torch.clamp(real_imgs, -1, 1)
            gen_imgs = torch.clamp(gen_imgs, -1, 1)
            
            # Composite the generated hole with the original image using original mask
            composited_imgs = real_imgs * (1 - masks[:, :3]) + gen_imgs * masks[:, :3]
            composited_imgs = torch.clamp(composited_imgs, -1, 1)
            
            # Get discriminator outputs
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(composited_imgs.detach())
            
            # Create labels with label smoothing for more stable training
            valid = torch.ones_like(real_validity, device=device) * 0.9
            fake = torch.zeros_like(fake_validity, device=device) * 0.1
            
            # Discriminator loss
            d_real_loss = criterion_gan(real_validity, valid)
            d_fake_loss = criterion_gan(fake_validity, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            if not torch.isnan(d_loss):
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
                d_optimizer.step()
            
            # Track discriminator accuracy
            d_real_acc.append((real_validity > 0.5).float().mean().item())
            d_fake_acc.append((fake_validity < 0.5).float().mean().item())
        
        # Train Generator
        g_optimizer.zero_grad()
        
        # Generate inpainted image using dilated mask for context
        gen_imgs = generator(masked_imgs)  # masked_imgs already contains dilated mask
        
        # Generator adversarial loss
        fake_validity = discriminator(composited_imgs)
        g_adv_loss = criterion_gan(fake_validity, valid)
        
        # Pixel-wise loss (use dilated mask for learning context)
        hole_region_real = real_imgs * dilated_masks[:, :3]
        hole_region_fake = gen_imgs * dilated_masks[:, :3]
        pixel_loss = criterion_pixel(hole_region_fake, hole_region_real)
        
        # Edge-aware loss with dilated context
        edge_loss = edge_aware_loss(real_imgs, composited_imgs, dilated_masks)
        
        # Total generator loss (weighted sum)
        g_loss = g_adv_loss + 100 * pixel_loss + 25 * edge_loss
        
        if not torch.isnan(g_loss):
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
            g_optimizer.step()
        
        # Calculate SSIM for the hole region (use original mask for evaluation)
        with torch.no_grad():
            ssim_val = calculate_ssim(
                real_imgs[0].cpu(),
                composited_imgs[0].detach().cpu(),
                masks[0].cpu()  # Use original mask for SSIM
            )
        
        # Update totals only if values are not NaN
        if not torch.isnan(g_loss):
            total_g_loss += g_loss.item()
        if not torch.isnan(d_loss) and train_disc:
            total_d_loss += d_loss.item()
        if not torch.isnan(pixel_loss):
            total_pixel_loss += pixel_loss.item()
        if not torch.isnan(edge_loss):
            total_edge_loss += edge_loss.item()
        if not np.isnan(ssim_val):
            total_ssim += ssim_val
        num_batches += 1
    
    # Safely compute averages
    if num_batches > 0:
        return (
            total_g_loss / num_batches if total_g_loss > 0 else 0,
            total_d_loss / num_batches if total_d_loss > 0 else 0,
            total_pixel_loss / num_batches if total_pixel_loss > 0 else 0,
            total_edge_loss / num_batches if total_edge_loss > 0 else 0,
            total_ssim / num_batches if total_ssim > 0 else 0
        )
    return 0, 0, 0, 0, 0

def evaluate_and_display(generator, test_loader, device, num_images=10):
    """Display original, masked, and inpainted images side by side."""
    generator.eval()
    
    # Get a batch of test images
    real_images, masked_images, masks, dilated_masks = next(iter(test_loader))
    real_images = real_images[:num_images].to(device)
    masked_images = masked_images[:num_images].to(device)
    masks = masks[:num_images].to(device)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    def normalize_image(img):
        """Normalize image to [0,1] range"""
        img = img.clamp(-1, 1)  # First clamp to [-1,1]
        img = (img + 1) / 2     # Then scale to [0,1]
        return img
    
    with torch.no_grad():
        # Generate inpainted images
        # Measure inference time
        start_time = time.time()
        generated_images = generator(masked_images)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.6f} seconds")
        generated_images = normalize_image(generated_images)
        
        # Normalize other images
        real_images = normalize_image(real_images)
        masked_images = normalize_image(masked_images[:, :3])  # Only RGB channels
        
        # Composite the generated hole with the original image using original mask
        completed_images = real_images * (1 - masks[:, :3]) + generated_images * masks[:, :3]
        
        for i in range(num_images):
            # Convert images to numpy and transpose
            real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
            masked_img = masked_images[i].cpu().numpy().transpose(1, 2, 0)
            generated_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
            completed_img = completed_images[i].cpu().numpy().transpose(1, 2, 0)
            
            # Display images
            axes[i, 0].imshow(real_img)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masked_img)
            axes[i, 1].set_title('Masked')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(generated_img)
            axes[i, 2].set_title('Generated (hole only)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(completed_img)
            axes[i, 3].set_title('Inpainted')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    os.makedirs('out', exist_ok=True)
    plt.savefig(os.path.join('out', 'inpainting_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_and_display_fixed(generator, test_dataset, device, fixed_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fixed_holes=[(48, 48), (32, 32), (32, 64), (64, 32), (64, 64), (48, 32), (48, 64), (32, 48), (64, 48), (40, 40)]):
    """Display results for fixed images and hole positions."""
    generator.eval()
    
    num_images = len(fixed_indices)
    assert len(fixed_holes) == num_images, "Number of fixed holes must match number of indices"
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    def normalize_image(img):
        """Normalize image to [0,1] range"""
        img = img.clamp(-1, 1)  # First clamp to [-1,1]
        img = (img + 1) / 2     # Then scale to [0,1]
        return img
    
    with torch.no_grad():
        for i, (idx, hole_pos) in enumerate(zip(fixed_indices, fixed_holes)):
            # Get the image
            test_dataset.fixed_hole = hole_pos
            image, masked_image, mask, dilated_mask = test_dataset[idx]
            
            # Add batch dimension and move to device
            image = image.unsqueeze(0).to(device)
            masked_image = masked_image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Generate inpainted image
            generated_image = generator(masked_image)
            
            # Normalize images
            image = normalize_image(image)
            masked_image = normalize_image(masked_image[:, :3])
            generated_image = normalize_image(generated_image)
            
            # Composite the generated hole with the original image
            completed_image = image * (1 - mask[:, :3]) + generated_image * mask[:, :3]
            
            # Convert to numpy for display
            image = image[0].cpu().numpy().transpose(1, 2, 0)
            masked_image = masked_image[0].cpu().numpy().transpose(1, 2, 0)
            generated_image = generated_image[0].cpu().numpy().transpose(1, 2, 0)
            completed_image = completed_image[0].cpu().numpy().transpose(1, 2, 0)
            
            # Display images
            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masked_image)
            axes[i, 1].set_title('Masked')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(generated_image)
            axes[i, 2].set_title('Generated (hole only)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(completed_image)
            axes[i, 3].set_title('Inpainted')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    os.makedirs('out', exist_ok=True)
    plt.savefig(os.path.join('out', 'fixed_inpainting_results.png'), dpi=150, bbox_inches='tight')
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
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    
    # Configuration
    data_dir = "data_celeb"
    batch_size = 10
    max_files = 200
    epochs = 20
    
    print(f"Starting training with edge-aware loss...")
    
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
    
    print(f"Dataset splits:")
    print(f"Train: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
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
    
    # Use BCEWithLogitsLoss instead of BCELoss since we removed sigmoid from discriminator
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()
    
    # Training history
    train_g_losses = []
    train_d_losses = []
    train_pixel_losses = []
    train_edge_losses = []
    train_ssims = []
    val_g_losses = []
    val_pixel_losses = []
    val_edge_losses = []
    val_ssims = []
    
    # Training loop
    start_time = time.time()
    best_val_ssim = 0
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_g_loss, train_d_loss, train_pixel_loss, train_edge_loss, train_ssim = train_epoch(
            generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion_gan, criterion_pixel, device
        )
        train_g_losses.append(train_g_loss)
        train_d_losses.append(train_d_loss)
        train_pixel_losses.append(train_pixel_loss)
        train_edge_losses.append(train_edge_loss)
        train_ssims.append(train_ssim)
        
        # Validation
        val_g_loss, val_pixel_loss, val_edge_loss, val_ssim = validate(
            generator, discriminator, val_loader, criterion_gan, criterion_pixel, device
        )
        val_g_losses.append(val_g_loss)
        val_pixel_losses.append(val_pixel_loss)
        val_edge_losses.append(val_edge_loss)
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
            }, os.path.join(out_dir, 'best_model.pth'))
        
        epoch_time = time.time() - epoch_start
        total_time = epoch_time * epoch
        
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Train - G_loss: {train_g_loss:.4f}, D_loss: {train_d_loss:.4f}, Pixel_loss: {train_pixel_loss:.4f}, Edge_loss: {train_edge_loss:.4f}, SSIM: {train_ssim:.4f}")
        print(f"Val - G_loss: {val_g_loss:.4f}, Pixel_loss: {val_pixel_loss:.4f}, Edge_loss: {val_edge_loss:.4f}, SSIM: {val_ssim:.4f}")
        print(f"Time - Epoch: {epoch_time:.1f}s, Total: {total_time:.1f}s\n")
    
    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'epoch': epochs,
        'val_ssim': val_ssim,
    }, os.path.join(out_dir, 'final_model.pth'))
    
    print("\nGenerating inpainting results...")
    evaluate_and_display(generator, test_loader, device)
    evaluate_and_display_fixed(generator, test_dataset, device)
    
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
    plt.plot(train_edge_losses, label='Train Edge Loss')
    plt.plot(val_edge_losses, label='Val Edge Loss')
    plt.plot(train_ssims, label='Train SSIM')
    plt.plot(val_ssims, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Edge Loss and SSIM')
    plt.legend()
    plt.title('Edge-aware Loss and Structural Similarity Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_plots.png'))
    plt.close()

    evaluate_and_display(generator, test_loader, device)