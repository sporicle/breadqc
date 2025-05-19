import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.image_processor import split_into_tiles, compute_tile_averages
from scipy.ndimage import binary_dilation

def visualize_heatmap(masked_image, mask, output_path, red_threshold=0.0, red_intensity=120, 
                     enable_flood_fill=False, flood_fill_kernel=3, grid_size=100):
    """
    Create a heatmap visualization of the bread image.
    Returns: (green_tile_percentage, red_mask, heatmap_img)
    """
    img_np = np.array(masked_image)
    
    # Compute bread-wide average RGB (only bread pixels)
    bread_pixels = img_np[mask == 255]
    if len(bread_pixels) == 0:
        raise ValueError('No bread pixels found!')
    avg_bread_rgb = np.mean(bread_pixels, axis=0)
    avg_bread_brightness = np.mean(avg_bread_rgb)

    # Get tiles
    tiles = split_into_tiles(img_np, mask, grid_size=grid_size)
    tile_avgs = compute_tile_averages(tiles)

    # Find bread bounding box
    ys, xs = np.where(mask == 255)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    tile_w = box_w / grid_size
    tile_h = box_h / grid_size

    # Create a grid to store tile colors
    grid = np.zeros((grid_size, grid_size), dtype=float)
    for x, y, avg_rgb in tile_avgs:
        if np.mean(avg_rgb) < 10:  # Skip black tiles
            continue
        tile_brightness = np.mean(avg_rgb)
        diff = tile_brightness - avg_bread_brightness
        grid[y, x] = diff

    # Create a mask for red regions (darker than threshold)
    red_mask = grid < red_threshold

    # Apply flood filling if enabled
    if enable_flood_fill:
        red_mask = binary_dilation(red_mask, iterations=flood_fill_kernel)

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
            if red_mask[y, x]:
                color = (255, 0, 0, red_intensity)
            else:
                green_count += 1
                diff = grid[y, x]
                norm = max(abs(diff) / 50, 0)
                norm = min(norm, 1.0)
                color = (0, int(255 * norm), 0, int(120 * norm))
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
        red_threshold=0.0,
        red_intensity=60,
        enable_flood_fill=False,
        flood_fill_kernel=1
    )
    save_heatmap_with_percentage(heatmap_img, pct, 'state/temp/bread_heatmap_with_pct.png') 