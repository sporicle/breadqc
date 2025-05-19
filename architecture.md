# 🍞 Bread Whiteness Detector – Full Architecture (Python + Line Bot)

## Overview

this is a system that takes an image of bread, isolates the bread, splits it into 100x100 chunks, computes the average color per chunk, and calculates what percent of the bread is "too white" based on a threshold. it's later wrapped in a line bot interface.

---

## 🗂 File & Folder Structure

bread_whiteness_detector/
├── app/
│ ├── init.py
│ ├── image_processing.py # bread detection, segmentation, chunking, color averaging
│ ├── whiteness_analysis.py # computes percent whiteness
│ ├── utils.py # small helpers (e.g., color utils, thresholding)
│ └── constants.py # tunable params like whiteness threshold, chunk size
│
├── bot/
│ ├── init.py
│ ├── handler.py # main entrypoint for line webhook
│ ├── line_interface.py # receives image, returns result
│ └── formatter.py # formats results for messaging
│
├── state/
│ └── temp/ # optional: stores intermediate images, logs etc.
│
├── tests/
│ ├── test_image_processing.py
│ ├── test_whiteness_analysis.py
│ └── sample_data/ # test images
│
├── main.py # dev CLI runner
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🧠 What Each Part Does

### `app/` – core logic

- **`image_processing.py`**
  - load image
  - segment bread (mask out background)
  - resize to canonical size (optional)
  - split into 100x100 px tiles *of bread only*
  - get avg RGB per tile
  - returns list of `(tile_coords, avg_color)` for bread tiles only

- **`whiteness_analysis.py`**
  - uses color space logic to determine if a tile is "too white"
  - counts how many tiles exceed whiteness threshold
  - returns: `total_tiles`, `white_tiles`, `percent_white`

- **`utils.py`**
  - rgb to hsv/lab converters
  - whiteness threshold logic
  - image masks, chunking logic, etc

- **`constants.py`**
  - tweakables like:
    - `CHUNK_SIZE = 100`
    - `WHITENESS_THRESHOLD = 0.85` (can be L value in LAB or just lightness in HSV)

---

### `bot/` – line bot logic

- **`handler.py`**
  - webhook handler for line events
  - verifies signature, routes messages

- **`line_interface.py`**
  - gets image from line server
  - saves locally to `/state/temp`
  - invokes pipeline:
    1. `process_image(path)` → bread mask + chunks
    2. `analyze_whiteness(...)`
    3. `format_response(...)`
  - sends result back to user via line SDK

- **`formatter.py`**
  - builds user-readable response
  - can optionally include visualization (highlight white chunks)

---

### `state/`

- **`temp/`**
  - ephemeral storage of:
    - uploaded image
    - bread mask
    - chunk overlays
    - optionally a report image
  - can be cleaned up per request

---

### `tests/`

- end-to-end and unit tests
- golden path test images and their expected percent whiteness

---

### `main.py`

- cli for dev use:

```bash
$ python main.py bread.jpg
Bread is 18% too white.
🧬 Where State Lives
image state: state/temp/

bread masks and intermediate data never persist outside one request

config lives in constants.py

if later needed: could write % whiteness + timestamp to a db for history

🔌 How Services Connect
pgsql
Copy
Edit
LINE → handler.py → line_interface.py
                         ↓
                 image_processing.py
                         ↓
               whiteness_analysis.py
                         ↓
                    formatter.py
                         ↓
              reply to user via LINE SDK
🔮 Notes for Later
might need opencv or segment-anything (if classical contouring fails on messy backgrounds)

can export chunk mask image with white tiles highlighted to show user

for metrics: can keep track of trends over time for a given user

consider caching or compressing images pre-analysis

🔁 Example Flow
user sends bread pic

bot downloads → saves temp

image_processing creates mask

mask used to split bread into 100x100 px tiles

average color per tile computed

tiles over whiteness threshold are counted

percent whiteness = (# white tiles / total bread tiles) × 100

bot returns: “⚠️ 23% of this bread is too white”