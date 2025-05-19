import os
from PIL import Image, ImageDraw
import numpy as np

# Load the masked image and mask
img_path = 'state/temp/masked_test2.png'
mask_path = 'state/temp/mask_test2.png'
img = Image.open(img_path).convert('RGB')
mask = np.array(Image.open(mask_path).convert('L'))
draw = ImageDraw.Draw(img)

grid_size = 100

# Find bread bounding box in the mask
ys, xs = np.where(mask == 255)
if len(xs) == 0 or len(ys) == 0:
    raise ValueError("No bread found in mask!")
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

box_w = x_max - x_min + 1
box_h = y_max - y_min + 1

# Calculate tile dimensions to ensure even spacing
tile_w = box_w / grid_size
tile_h = box_h / grid_size

# Draw grid lines over the bounding box
for gx in range(grid_size + 1):
    x = int(x_min + gx * tile_w)
    draw.line([(x, y_min), (x, y_max + 1)], fill=(255, 0, 0), width=1)

for gy in range(grid_size + 1):
    y = int(y_min + gy * tile_h)
    draw.line([(x_min, y), (x_max + 1, y)], fill=(255, 0, 0), width=1)

# Save the overlay image
output_path = 'state/temp/tile_grid_overlay.png'
img.save(output_path)
print(f"Tile grid overlay saved to {output_path}") 