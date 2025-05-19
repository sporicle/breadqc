import pytest
import numpy as np
from app.image_loader import load_image
import os

def test_load_image():
    # Get the path to our test image
    test_image_path = os.path.join('tests', 'sample_data', 'test1.jpeg')
    
    # Test loading the image
    img = load_image(test_image_path)
    
    # Verify the image was loaded correctly
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3  # Should be a 3D array (height, width, channels)
    assert img.shape[2] == 3    # Should have 3 color channels (RGB)
    
    # Test that it raises ValueError for non-existent file
    with pytest.raises(ValueError):
        load_image("nonexistent.jpg") 