import cv2
import numpy as np

def create_bread_mask(rgb_image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask isolating the bread using K-means color clustering.
    This approach is more robust than simple HSV thresholding as it can handle
    varying lighting conditions and different background colors.
    
    Args:
        rgb_image (np.ndarray): Input RGB image as numpy array
        
    Returns:
        np.ndarray: Binary mask where 255 indicates bread and 0 indicates background
    """
    # Reshape the image to be a list of pixels
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)
    
    # Define criteria for K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2  # We want to separate bread from background
    
    # Perform K-means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    
    # Flatten the labels array
    labels = labels.flatten()
    
    # Create a mask where the darker cluster is considered bread
    # (assuming bread is generally darker than the background)
    bread_cluster = np.argmin([np.mean(center) for center in centers])
    mask = (labels == bread_cluster).reshape(rgb_image.shape[:2])
    
    # Convert to uint8 and scale to 0/255
    mask = np.uint8(mask * 255)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    # Find the largest contour (assumed to be the bread)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Create a blank mask
    final_mask = np.zeros_like(mask)
    
    # Draw the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
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