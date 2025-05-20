import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.image_processor import split_into_tiles, compute_tile_averages
from scipy.ndimage import binary_dilation
import cv2

def visualize_heatmap(masked_image, mask, output_path, 
                     # Clustering parameters
                     k_means_attempts=10,        # Number of times to run K-means
                     k_means_iterations=100,     # Max iterations per attempt
                     k_means_epsilon=0.2,        # Stop if change is less than this
                     
                     # Dark region detection
                     dark_brightness_threshold=1,  # Threshold for dark regions (0-1)
                     dark_region_bias=1.2,          # Bias towards dark regions (>1 = more aggressive)
                     
                     # Crust detection
                     crust_edge_threshold=0.1,      # How far from edge to look for crust (0-1)
                     crust_brown_threshold=0.6,     # How brown a color needs to be to be crust
                     
                     # Visualization parameters
                     red_intensity=120,          # Opacity of red overlay
                     green_intensity=120,        # Opacity of green overlay
                     blue_intensity=120,         # Opacity of blue overlay
                     
                     # Post-processing
                     enable_flood_fill=False,    # Whether to fill gaps
                     flood_fill_kernel=3,        # Size of flood fill kernel
                     
                     # Grid parameters
                     grid_size=100):            # Number of tiles per side
    """
    Create a heatmap visualization of the bread image using K-means clustering to find
    crust (blue), darker parts (red), and lighter parts (green).
    
    Args:
        masked_image: The input image
        mask: Binary mask of the bread
        output_path: Where to save the output
        k_means_attempts: Number of times to run K-means (higher = more stable)
        k_means_iterations: Max iterations per K-means attempt
        k_means_epsilon: Stop K-means if change is less than this
        dark_brightness_threshold: Threshold for dark regions (0-1)
        dark_region_bias: Bias towards dark regions (>1 = more aggressive)
        crust_edge_threshold: How far from edge to look for crust (0-1)
        crust_brown_threshold: How brown a color needs to be to be crust
        red_intensity: Opacity of red overlay (0-255)
        green_intensity: Opacity of green overlay (0-255)
        blue_intensity: Opacity of blue overlay (0-255)
        enable_flood_fill: Whether to fill gaps in regions
        flood_fill_kernel: Size of flood fill kernel
        grid_size: Number of tiles per side
    
    Returns: (green_tile_percentage, red_mask, heatmap_img)
    """
    img_np = np.array(masked_image)
    
    # Get tiles
    tiles = split_into_tiles(img_np, mask, grid_size=grid_size)
    tile_avgs = compute_tile_averages(tiles)
    
    if not tile_avgs:
        raise ValueError('No valid tiles found!')
    
    # Find bread bounding box for position-based analysis
    ys, xs = np.where(mask == 255)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    
    # Create a grid to store tile colors
    grid = np.zeros((grid_size, grid_size), dtype=float)
    red_mask = np.zeros((grid_size, grid_size), dtype=bool)
    blue_mask = np.zeros((grid_size, grid_size), dtype=bool)
    
    # First pass: Identify crust based on position and color
    for (x, y, avg_rgb) in tile_avgs:
        # Calculate normalized position (0-1)
        norm_x = (x - x_min) / box_w
        norm_y = (y - y_min) / box_h
        
        # Check if tile is near the edge
        is_near_edge = (norm_x < crust_edge_threshold or 
                       norm_x > (1 - crust_edge_threshold) or
                       norm_y < crust_edge_threshold or 
                       norm_y > (1 - crust_edge_threshold))
        
        # Calculate how "brown" the color is
        r, g, b = avg_rgb
        brightness = np.mean(avg_rgb) / 255.0
        is_brown = (r > g and r > b and  # Red component is dominant
                   abs(r - g) > 30 and    # Significant difference between R and G
                   brightness < 0.8)       # Not too bright
        
        if is_near_edge and is_brown:
            blue_mask[y, x] = True
            grid[y, x] = 2  # Mark as crust
    
    # Second pass: Identify dark regions (excluding crust)
    for (x, y, avg_rgb) in tile_avgs:
        if grid[y, x] == 2:  # Skip if already marked as crust
            continue
            
        tile_brightness = np.mean(avg_rgb) / 255.0
        
        # Apply dark region bias
        if tile_brightness < dark_brightness_threshold * dark_region_bias:
            red_mask[y, x] = True
            grid[y, x] = -1  # Mark as darker region
        else:
            grid[y, x] = 1   # Mark as lighter region
    
    # Apply flood filling if enabled
    if enable_flood_fill:
        red_mask = binary_dilation(red_mask, iterations=flood_fill_kernel)
        blue_mask = binary_dilation(blue_mask, iterations=flood_fill_kernel)
    
    # Calculate tile dimensions
    tile_w = box_w / grid_size
    tile_h = box_h / grid_size
    
    # Prepare overlay
    if isinstance(masked_image, np.ndarray):
        overlay_size = (masked_image.shape[1], masked_image.shape[0])
        base_img = Image.fromarray(masked_image)
    else:
        overlay_size = masked_image.size
        base_img = masked_image
    overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    green_count = 0
    total_count = 0
    for y in range(grid_size):
        for x in range(grid_size):
            if grid[y, x] == 0:  # Skip empty tiles
                continue
            total_count += 1
            # Tile coordinates
            x0 = int(x_min + x * tile_w)
            y0 = int(y_min + y * tile_h)
            x1 = int(x_min + (x + 1) * tile_w)
            y1 = int(y_min + (y + 1) * tile_h)
            if blue_mask[y, x]:
                color = (0, 0, 255, blue_intensity)  # Blue for crust
            elif red_mask[y, x]:
                color = (255, 0, 0, red_intensity)  # Red for darker regions
            else:
                green_count += 1
                color = (0, 255, 0, green_intensity)  # Green for lighter regions
            draw.rectangle([x0, y0, x1, y1], fill=color)
    
    green_tile_percentage = 0.0 if total_count == 0 else 100.0 * green_count / total_count
    heatmap_img = Image.alpha_composite(base_img.convert('RGBA'), overlay)
    heatmap_img.save(output_path)
    print(f"Heatmap saved to {output_path}")
    return green_tile_percentage, red_mask, heatmap_img

def save_heatmap_with_percentage(heatmap_img, percentage, output_path):
    """
    Save a combined image with the heatmap on top and the percentage below (just the percent, large, black, centered).
    """
    width, height = heatmap_img.size
    # Create a new image with extra space for the percentage
    extra_height = int(height * 0.18)
    combined = Image.new('RGBA', (width, height + extra_height), (255, 255, 255, 255))
    combined.paste(heatmap_img, (0, 0))
    # Draw the percentage text (just the percent, large, black, centered)
    draw = ImageDraw.Draw(combined)
    font_path = "assets/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, int(extra_height * 0.9))
    except Exception as e:
        print(f"Warning: Could not load {font_path} ({e}), falling back to default font.")
        font = ImageFont.load_default()
    text = f"{percentage:.1f}%"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (width - text_w) // 2
    text_y = height + (extra_height - text_h) // 2
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    combined.save(output_path)
    print(f"Saved combined heatmap and percentage to {output_path}")

if __name__ == '__main__':
    img_path = 'state/temp/masked_test2.png'
    mask_path = 'state/temp/mask_test2.png'
    output_path = 'state/temp/bread_heatmap.png'
    img = Image.open(img_path).convert('RGB')
    mask = np.array(Image.open(mask_path).convert('L'))
    pct, _, heatmap_img = visualize_heatmap(
        masked_image=img,
        mask=mask,
        output_path=output_path,
        red_intensity=60,
        green_intensity=120,
        blue_intensity=120,
        enable_flood_fill=False,
        flood_fill_kernel=1
    )
    save_heatmap_with_percentage(heatmap_img, pct, 'state/temp/bread_heatmap_with_pct.png') 