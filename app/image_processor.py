import cv2
import numpy as np

def create_bread_mask(rgb_image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask isolating the largest contour in the image (assumed to be bread).
    Uses HSV color-based segmentation to better handle off-white backgrounds.
    
    Args:
        rgb_image (np.ndarray): Input RGB image as numpy array
        
    Returns:
        np.ndarray: Binary mask where 1 indicates bread and 0 indicates background
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Define ranges for off-white background in HSV
    # H: 0-180, S: 0-30 (low saturation for white/off-white), V: 200-255 (high value for white)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Create mask for off-white background
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Invert the mask (now 1 for bread, 0 for background)
    mask = cv2.bitwise_not(mask_white)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a blank mask
    final_mask = np.zeros_like(mask)
    
    # Draw the largest contour on the mask
    cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
    
    return final_mask

def apply_mask(rgb_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an RGB image, zeroing out the background.
    Args:
        rgb_image (np.ndarray): Input RGB image
        mask (np.ndarray): Binary mask (same height/width as image, 0/255)
    Returns:
        np.ndarray: Masked RGB image (background is black)
    """
    # Ensure mask is boolean
    mask_bool = (mask == 255)
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask_bool]*3, axis=-1)
    # Zero out background
    masked_img = np.zeros_like(rgb_image)
    masked_img[mask_3ch] = rgb_image[mask_3ch]
    return masked_img 

def split_into_tiles(masked_image: np.ndarray, mask: np.ndarray, grid_size: int = 100) -> list[tuple[int, int, np.ndarray]]:
    """
    Split the bread bounding box into a grid_size x grid_size grid of tiles.
    Only tiles containing bread pixels are included.
    Args:
        masked_image (np.ndarray): RGB image with background zeroed out
        mask (np.ndarray): Binary mask (same height/width as image, 0/255)
        grid_size (int): Number of tiles per side (default: 100)
    Returns:
        list[tuple[int, int, np.ndarray]]: List of (x, y, tile_image) tuples
    """
    # Find bounding box of bread in the mask
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        return []
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    tile_w = box_w // grid_size
    tile_h = box_h // grid_size
    tiles = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            x0 = x_min + gx * tile_w
            y0 = y_min + gy * tile_h
            # Last tile should go to the edge
            x1 = x_min + (gx + 1) * tile_w if gx < grid_size - 1 else x_max + 1
            y1 = y_min + (gy + 1) * tile_h if gy < grid_size - 1 else y_max + 1
            tile = masked_image[y0:y1, x0:x1]
            tile_mask = mask[y0:y1, x0:x1]
            if np.any(tile_mask == 255):
                tiles.append((gx, gy, tile))
    return tiles 

def compute_tile_averages(tiles: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, int, tuple[float, float, float]]]:
    """
    Compute the average RGB value for each tile.
    Args:
        tiles (list[tuple[int, int, np.ndarray]]): List of (x, y, tile_image) tuples
    Returns:
        list[tuple[int, int, tuple[float, float, float]]]: List of (x, y, avg_rgb) tuples
    """
    averages = []
    for x, y, tile in tiles:
        # Only consider non-zero pixels (bread pixels)
        mask = np.any(tile > 0, axis=2)
        if not np.any(mask):
            continue
            
        # Calculate mean RGB values for non-zero pixels
        avg_r = np.mean(tile[mask, 0])
        avg_g = np.mean(tile[mask, 1])
        avg_b = np.mean(tile[mask, 2])
        
        averages.append((x, y, (avg_r, avg_g, avg_b)))
    
    return averages 