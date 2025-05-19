import os
import sys
import shutil
from datetime import datetime
from app.image_loader import load_image
from app.image_processor import create_bread_mask, apply_mask, split_into_tiles, compute_tile_averages
from scripts.visualize_bread_heatmap import visualize_heatmap, save_heatmap_with_percentage
from PIL import Image

def process_bread_image(image_name):
    """
    Process a bread image through the full pipeline.
    Args:
        image_name: Name of the image file in tests/sample_data/
    """
    # Create timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_base = os.path.splitext(image_name)[0]
    output_dir = f'state/temp/{image_base}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {image_name}...")
    print(f"Output directory: {output_dir}")
    
    # Load image
    input_path = os.path.join('tests', 'sample_data', image_name)
    rgb_image = load_image(input_path)
    
    # Save original
    original_path = os.path.join(output_dir, 'original.png')
    Image.fromarray(rgb_image).save(original_path)
    print("✓ Saved original image")
    
    # Create and save mask
    mask = create_bread_mask(rgb_image)
    mask_path = os.path.join(output_dir, 'mask.png')
    Image.fromarray(mask).save(mask_path)
    print("✓ Created and saved mask")
    
    # Apply mask and save
    masked_image = apply_mask(rgb_image, mask)
    masked_path = os.path.join(output_dir, 'masked.png')
    Image.fromarray(masked_image).save(masked_path)
    print("✓ Applied mask and saved result")
    
    # Create heatmap with default settings
    heatmap_path = os.path.join(output_dir, 'heatmap.png')
    pct, _, heatmap_img = visualize_heatmap(
        masked_image=masked_image,
        mask=mask,
        output_path=heatmap_path,
        red_threshold=0.0,  # Use your default
        red_intensity=60,   # Use your default
        enable_flood_fill=False,  # Use your default
        flood_fill_kernel=1       # Use your default
    )
    print("✓ Created heatmap")
    
    # Save combined heatmap with percentage
    combined_path = os.path.join(output_dir, 'heatmap_with_percentage.png')
    save_heatmap_with_percentage(heatmap_img, pct, combined_path)
    print("✓ Saved combined heatmap with percentage")
    
    print("\nProcessing complete! Results saved in:", output_dir)
    return output_dir

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python process_bread_image.py <image_name>")
        print("Example: python process_bread_image.py test1.jpeg")
        sys.exit(1)
    
    image_name = sys.argv[1]
    process_bread_image(image_name) 