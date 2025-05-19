import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """
    Load an image from the given path and return it as an RGB numpy array.
    
    Args:
        path (str): Path to the image file
        
    Returns:
        np.ndarray: RGB image as numpy array with shape (height, width, 3)
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded
    """
    # Read image using OpenCV
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
        
    # Convert from BGR to RGB (OpenCV loads as BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb 