import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from main import Generator, load_and_split_dataset, evaluate_and_display, evaluate_and_display_fixed

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    generator = Generator(nf=32).to(device)
    
    # Load the best checkpoint
    checkpoint_path = os.path.join('out', 'final_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Load the dataset
    data_dir = 'data_celeb'  # Updated path
    _, _, test_dataset = load_and_split_dataset(data_dir, batch_size=10)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # Generate regular results
    print("Generating inpainting results...")
    evaluate_and_display(generator, test_loader, device, num_images=10)
    
    # Generate fixed position results with 10 images
    print("Generating fixed position results...")
    fixed_indices = list(range(10))  # Use first 10 images
    fixed_holes = [
        (48, 48),   # Center
        (32, 32),   # Top-left
        (32, 64),   # Top-right
        (64, 32),   # Bottom-left
        (64, 64),   # Bottom-right
        (48, 32),   # Top-center
        (48, 64),   # Bottom-center
        (32, 48),   # Left-center
        (64, 48),   # Right-center
        (40, 40),   # Off-center
    ]
    evaluate_and_display_fixed(generator, test_dataset, device, fixed_indices, fixed_holes)
    
    print("Results saved in 'out' directory")

if __name__ == "__main__":
    main()
