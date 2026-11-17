# Circuit Component Identification

Finds electronic components in a circuit image and saves an annotated copy.
It looks for: resistor, capacitor, LED, transistor, IC chip, diode,
battery and breadboard.

There is no custom training. Two pre-trained models do the work:

1. **YOLO-World** (`models/yolov8s-worldv2.pt`) is an open-vocabulary detector.
   `set_classes()` gives it the component names, and it proposes bounding boxes.
2. **CLIP ViT-B/32** (`models/ViT-B-32.pt`) re-labels each box crop. YOLO-World
   finds good boxes on electronics, but its labels are poor: it calls almost
   everything "battery". CLIP names the crop much more reliably.
   Use `--yolo-only` to see YOLO-World's raw labels.

## Download (full project with models)

The model files are too large for a normal git commit (GitHub's limit is 100 MB per file),
so the complete project, **including both models**, is attached to the
[latest release](https://github.com/Phantom2006-dot/cird/releases/latest):

**[circuit_identification.zip (353 MB)](https://github.com/Phantom2006-dot/cird/releases/latest/download/circuit_identification.zip)**
SHA-256: `4ef5e88f3d91b18a6010f15fd850a0f899510b6d426d88c0b23affbae8d39d11`

If you clone this repo instead, put `yolov8s-worldv2.pt` and `ViT-B-32.pt` into `models/`.
Both are also attached to the release as separate files.

## Setup

```bash
pip install -r requirements.txt
```

Both model files are already in `models/`, so no download is needed.

If you need to download CLIP yourself, note that the old
`openaipublic.azureedge.net` link is dead. Use:
https://openaipublic.blob.core.windows.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
The script also points the `clip` package at this URL.

## Run

```bash
python identify_circuit.py                    # uses sample_circuit.jpg
python identify_circuit.py my_photo.jpg -o result.jpg
python identify_circuit.py real_test/real3.jpg   # a real photo
python identify_circuit.py --test             # self-test
python make_sample.py                         # redraw the sample image
```

The output lists counts per component type. The annotated image is saved as
`<name>_detected.jpg`.

## Accuracy, honestly

I tested the tool on three real photos from Wikimedia Commons, in
`real_test/` (see `ATTRIBUTION.md`). The annotated results are next to them.

| Photo | What it got right | What it got wrong |
|-------|-------------------|-------------------|
| `real3.jpg`: loose LEDs with resistors soldered on | 10 resistors found, mostly with high scores (up to 0.97) | Clear LEDs are usually labelled by their attached resistor; one is called "capacitor" |
| `real4.jpg`: a small parts kit | Breadboard, DIP IC chips, the box of resistors | Some ICs are called "transistor"; the empty tin lid is called "battery" |
| `real1.jpg`: Arduino blink circuit | Breadboard (0.80) | Missed the single lit LED; a desk sticker is called "IC chip"; the Arduino board is not a class |
| `sample_circuit.jpg`: computer-drawn | 3 resistors, breadboard, IC chip, both LEDs | Battery and capacitor mislabelled; transistor missed |

Bottom line: it works well for **resistors, breadboards and IC chips**. It is
unreliable for small, ambiguous parts such as transistors, diodes and
capacitors. This is the limit of zero-shot models; for high accuracy, train
YOLO on a labelled electronic-components dataset. For best results use
clear, well-lit, top-down photos where parts do not overlap.

Post-processing: any box that encloses two or more other detections is
dropped, because YOLO-World tends to draw a box around a whole pile of parts.
Breadboards are exempt from this rule.

Tuning:
- `--conf` sets the YOLO-World box threshold (default 0.005, deliberately low,
  because CLIP filters the boxes afterwards).
- `CLIP_PROMPTS` and `TEMPLATES` in `identify_circuit.py` control how CLIP
  names crops. To add a new component, add a line there and to `COMPONENTS`.
