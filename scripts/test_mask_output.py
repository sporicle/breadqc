import os
from app.image_loader import load_image
from app.image_processor import create_bread_mask, apply_mask
from PIL import Image
import numpy as np

# Paths
input_path = 'tests/sample_data/test2.jpg'
output_path = 'state/temp/masked_test2.png'

# 1. Load image
img = load_image(input_path)

# 2. Create bread mask
mask = create_bread_mask(img)

# 3. Apply mask
masked_img = apply_mask(img, mask)

# 4. Save result (convert numpy array to PIL Image)
Image.fromarray(masked_img).save(output_path)
print(f"Masked image saved to {output_path}")

# Optionally, also save the mask for inspection
Image.fromarray(mask).save('state/temp/mask_test2.png')
print("Mask image saved to state/temp/mask_test2.png") 