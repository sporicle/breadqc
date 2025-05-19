import pytest
import numpy as np
import cv2
import os
from app.image_loader import load_image
from app.image_processor import create_bread_mask

def test_create_bread_mask():
    # Load test image
    test_image_path = os.path.join('tests', 'sample_data', 'test1.jpeg')
    rgb_image = load_image(test_image_path)
    
    # Create mask
    mask = create_bread_mask(rgb_image)
    
    # Verify mask properties
    assert isinstance(mask, np.ndarray)
    assert len(mask.shape) == 2  # Should be 2D (height, width)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})  # Should be binary
    
    # Test with an empty image (should return all zeros or all 255 mask)
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    empty_mask = create_bread_mask(empty_image)
    unique_vals = np.unique(empty_mask)
    assert (len(unique_vals) == 1) and (unique_vals[0] in [0, 255]) 