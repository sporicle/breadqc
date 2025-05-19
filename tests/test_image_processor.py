import pytest
import numpy as np
import cv2
import os
from app.image_loader import load_image
from app.image_processor import create_bread_mask, apply_mask, split_into_tiles, compute_tile_averages

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

def test_apply_mask():
    test_image_path = os.path.join('tests', 'sample_data', 'test1.jpeg')
    rgb_image = load_image(test_image_path)
    mask = create_bread_mask(rgb_image)
    masked_img = apply_mask(rgb_image, mask)
    # Check shape and type
    assert masked_img.shape == rgb_image.shape
    assert masked_img.dtype == rgb_image.dtype
    # Check that background is zeroed out
    background = masked_img[mask == 0]
    assert np.all(background == 0)

def test_split_into_tiles():
    # Create a test image with some bread and empty areas
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    # Add some "bread" in the middle (spanning 4 tiles, but also partially in others)
    test_image[50:250, 50:250] = [255, 200, 150]  # Bread-colored pixels
    
    # Create a mask for the bread area
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[50:250, 50:250] = 255
    
    # Split into tiles
    tiles = split_into_tiles(test_image, mask)
    
    # Check that we got tiles
    assert len(tiles) > 0
    
    # Check that all tiles have the correct shape
    for x, y, tile in tiles:
        assert tile.shape[2] == 3  # RGB channels
        assert np.any(tile > 0)  # Should have some non-zero pixels
    
    # Test with empty image
    empty_image = np.zeros((300, 300, 3), dtype=np.uint8)
    empty_mask = np.zeros((300, 300), dtype=np.uint8)
    empty_tiles = split_into_tiles(empty_image, empty_mask)
    assert len(empty_tiles) == 0  # Should have no tiles

def test_compute_tile_averages():
    # Create a test tile with known RGB values
    # 2x2 tile with red, green, blue, and white pixels
    tile = np.array([
        [[255, 0, 0], [0, 255, 0]],  # Red and Green
        [[0, 0, 255], [255, 255, 255]]  # Blue and White
    ], dtype=np.uint8)
    
    # Create test tiles list
    tiles = [(0, 0, tile)]
    
    # Compute averages
    averages = compute_tile_averages(tiles)
    
    # Check that we got one result
    assert len(averages) == 1
    
    # Check coordinates
    x, y, avg_rgb = averages[0]
    assert x == 0
    assert y == 0
    
    # Check average RGB values
    # Expected: (255+0+0+255)/4, (0+255+0+255)/4, (0+0+255+255)/4
    expected_avg = (127.5, 127.5, 127.5)
    assert np.allclose(avg_rgb, expected_avg, atol=0.1)
    
    # Test with empty tile
    empty_tile = np.zeros((2, 2, 3), dtype=np.uint8)
    empty_tiles = [(1, 1, empty_tile)]
    empty_averages = compute_tile_averages(empty_tiles)
    assert len(empty_averages) == 0  # Should skip empty tiles 