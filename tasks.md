## 🏗 STAGE 1: setup + scaffolding

### ✅ task 1: initialize repo + folders
- **start**: empty directory
- **end**: structure matches this:
bread_whiteness_detector/
├── app/
├── bot/
├── state/temp/
├── tests/sample_data/
└── main.py
---

### ✅ task 2: set up `requirements.txt`
- **start**: blank file
- **end**: has working versions of:
- `opencv-python`
- `numpy`
- `pillow`
- `flask` (placeholder for webhook)
- `line-bot-sdk`
- `scikit-image`
- `pytest`

---

### ✅ task 3: implement bread image loader
- **start**: raw file path
- **end**: function `load_image(path)` returns RGB numpy array
- **test**: use a test image from `tests/sample_data/`

---

## 🎨 STAGE 2: image processing pipeline

### ✅ task 4: naive bread mask via contour
- **start**: RGB image
- **end**: binary mask isolating largest contour (assumed to be bread)
- **test**: visualize mask over input image and save result

---

### ✅ task 5: apply mask to isolate bread
- **start**: original image + binary mask
- **end**: return new image with background zeroed out
- **test**: visually confirm only bread is visible

---

### ✅ task 6: split masked image into 100x100 tiles
- **start**: masked image
- **end**: list of `(x, y, tile_image)` tuples (ignore empty/non-bread tiles)
- **test**: count and visualize tile grid overlay

---

### ✅ task 7: compute avg RGB per tile
- **start**: list of tiles
- **end**: return list of `(x, y, avg_rgb)`
- **test**: print one row of tile averages

---

## 💡 STAGE 3: whiteness analysis

### ✅ task 8: implement whiteness function
- **start**: rgb value
- **end**: `is_too_white(rgb)` → bool
- **test**: verify some edge cases (pure white, tan, brown)

---

### ✅ task 9: analyze percent whiteness
- **start**: list of `(x, y, avg_rgb)`
- **end**: return `total_tiles`, `white_tiles`, `percent_white`
- **test**: unit test with mocked tiles

---

### ✅ task 10: visualize white tiles
- **start**: original image + white tile positions
- **end**: overlay red boxes on white tiles and save result
- **test**: verify accuracy visually

---

## 💬 STAGE 4: CLI + dev interface

### ✅ task 11: create `main.py` pipeline
- **start**: image path
- **end**: prints `% whiteness`, saves overlay image
- **test**: run on sample_data image

---

## 🤖 STAGE 5: line bot basics

### ✅ task 12: implement LINE webhook stub
- **start**: empty `handler.py`
- **end**: flask app that returns `200 OK` on POST
- **test**: `curl` to webhook returns OK

---

### ✅ task 13: download image from line
- **start**: line image event
- **end**: save image to `/state/temp/`
- **test**: manual trigger with sample event

---

### ✅ task 14: call core pipeline from image
- **start**: saved image file
- **end**: return % whiteness from line handler
- **test**: confirm log output

---

### ✅ task 15: send result message to user
- **start**: output string
- **end**: line bot replies with `XX% of your bread is too white`
- **test**: live round-trip through LINE UI

---

## ✅ STAGE 6: testing

### ✅ task 16: write unit test for tile averaging
- input: 3x3 pixel array of known RGB values
- assert: avg_rgb is correct

---

### ✅ task 17: write unit test for whiteness check
- input: white, tan, brown RGBs
- assert: thresholding logic consistent

---

### ✅ task 18: write integration test (e2e)
- input: test image
- output: check percent whiteness within expected range

---


## 🖼 STAGE 7: labeled output image

### ✅ task 19: implement chunk color labeling
- **start**: `(x, y, avg_rgb)` list + `is_too_white()` fn
- **end**: return new list of `(x, y, label)` where label is `"white"` or `"dark"`
- **test**: print list of labels

---

### ✅ task 20: draw labeled tile overlays
- **start**: image + labeled tile positions
- **end**: draw:
  - semi-transparent **green** rectangles for white tiles
  - semi-transparent **red** rectangles for dark tiles
- **test**: output `labeled_output.jpg` and visually confirm

---

### ✅ task 21: add output image writer
- **start**: overlayed image (as numpy array or PIL image)
- **end**: write to `/state/temp/labeled_<uuid>.jpg`
- **test**: confirm saved file has expected highlights

---

## 💬 STAGE 8: line bot image reply

### ✅ task 22: upload labeled image to line server
- **start**: local path to labeled image
- **end**: upload using `line-bot-sdk` and return a `MessageContent` object
- **test**: confirm upload via response metadata

---

### ✅ task 23: send labeled image in LINE reply
- **start**: path to image + reply token
- **end**: LINE reply includes:
  1. the labeled image
  2. a text caption: “🥖 X% of your bread is too white”
- **test**: send a test image through LINE and verify visual + text reply

---

## 🧪 STAGE 9: test full LINE flow w/ labeled image

### ✅ task 24: end-to-end integration test
- **start**: LINE user sends bread photo
- **end**: bot replies with:
  - processed image (green/red overlay)
  - text with % whiteness
- **test**: use real phone + LINE UI to confirm round-trip

---

## ✅ BONUS (optional polish)

### 🟨 task 25: cleanup temp files after each request
- **start**: handler finishes request
- **end**: delete any `state/temp/*.jpg` generated during run
- **test**: rerun request, check `/state/temp/` is empty after

---