import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from main import Generator, CelebDataset  # Import our model classes

def load_model(model_path):
    """Load the trained generator model."""
    device = torch.device("cpu")
    
    # Initialize model
    generator = Generator().to(device)
    
    # Load weights
    checkpoint = torch.load(model_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation SSIM: {checkpoint['val_ssim']:.4f}")
    return generator

def inpaint_image(generator, image_path, save_path=None):
    """Inpaint a single image and display/save results."""
    device = torch.device("cpu")
    
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset for single image to reuse masking logic
    dataset = CelebDataset([image_path], transform=transform)
    original_image, masked_image, mask = dataset[0]
    
    # Add batch dimension
    masked_image = masked_image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    
    # Generate inpainting
    with torch.no_grad():
        # Generator now takes both image and mask
        generated = generator(masked_image, mask)
        # Composite with original image
        composited = masked_image[:, :3] + generated * (1 - mask[:, :3])
    
    # Convert tensors to display format
    def to_display(img):
        return (img.cpu().squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    
    original = to_display(original_image)
    masked = to_display(masked_image[:, :3])  # Only use RGB channels
    inpainted = to_display(composited)
    hole_only = to_display(generated)  # Now only shows the generated hole content
    
    # Display results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(masked)
    axes[1].set_title('Masked')
    axes[1].axis('off')
    
    axes[2].imshow(inpainted)
    axes[2].set_title('Inpainted')
    axes[2].axis('off')
    
    axes[3].imshow(hole_only)
    axes[3].set_title('Generated Hole Only')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved result to {save_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # Load the model (use 'best_model.pth' or 'final_model.pth')
    generator = load_model('best_model.pth')
    
    # Process a single image or all images in a directory
    input_path = "data_celeb"  # Can be a single image or directory
    
    if os.path.isfile(input_path):
        # Process single image
        inpaint_image(generator, input_path, "result.png")
    else:
        # Process first 5 images in directory
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
        
        for i, image_file in enumerate(image_files):
            print(f"\nProcessing image {i+1}/{len(image_files)}: {image_file}")
            inpaint_image(generator, image_file, f"result_{i+1}.png")
