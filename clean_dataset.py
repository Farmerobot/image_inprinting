import os
import shutil

def keep_first_n_images(data_dir, n=1024):
    """Keep only the first n images in the directory."""
    # Get all image files
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort to ensure consistent order
    
    # Create list of files to remove (everything after first n)
    files_to_remove = image_files[n:]
    
    # Remove extra files
    for file in files_to_remove:
        file_path = os.path.join(data_dir, file)
        os.remove(file_path)
        print(f"Removed: {file}")
    
    print(f"\nKept first {n} images.")
    print(f"Removed {len(files_to_remove)} images.")

if __name__ == "__main__":
    data_dir = "data_celeb"
    keep_first_n_images(data_dir)
