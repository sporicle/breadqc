import cv2
import numpy as np

def create_bread_mask(rgb_image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask isolating the largest contour in the image (assumed to be bread).
    
    Args:
        rgb_image (np.ndarray): Input RGB image as numpy array
        
    Returns:
        np.ndarray: Binary mask where 1 indicates bread and 0 indicates background
    """
    # Convert RGB to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding instead of Otsu's
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # block size
        2    # constant subtracted from mean
    )
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a blank mask
    mask = np.zeros_like(gray)
    
    # Draw the largest contour on the mask
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    return mask 