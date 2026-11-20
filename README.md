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

The model files are too big for GitHub's 100 MB file limit, so they are not in
`models/` here. Get the complete, ready-to-run project in one of these ways:

- **Full zip, one file (353 MB, includes both models):**
  **[Download circuit_identification.zip](https://github.com/Phantom2006-dot/cird/releases/latest/download/circuit_identification.zip)**
  (from the [Releases](https://github.com/Phantom2006-dot/cird/releases) page).
- **Same zip in parts:** it is in [`full_zip/`](full_zip),
  split into 4 parts. Download them and double-click `JOIN_PARTS_WINDOWS.bat`
  (or run `python join_parts.py`). The joined file is the exact original zip.
- **Models only:** if you cloned this repo, download `yolov8s-worldv2.pt` from
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt
  and `ViT-B-32.pt` from the CLIP link below, and put both in `models/`.

## Setup

```bash
pip install -r requirements.txt
```

Both model files are already in `models/`, so no download is needed.

If you need to download CLIP yourself, note that the old
`openaipublic.azureedge.net` link is dead. Use:
https://openaipublic.blob.core.windows.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
The script also points the `clip` package at this URL.

## Desktop app (GUI)

```bash
python gui.py                      # open the app
python gui.py real_test/real3.jpg  # open it with a photo already loaded
```

1. Wait until the status bar says **Ready**. The AI models load once at
   start, which takes 20–60 s.
2. Click **Open Image...** and choose a circuit photo.
3. Click **Identify**. Boxes appear on the photo, with a summary table and a
   list of every detection on the right. Click a row to highlight that part
   in pink.
4. Drag **Min score** to hide low-confidence guesses. This is instant and
   doesn't re-run the detection.
5. Click **Save Result...** to save the labelled image plus a `.csv` list of the parts.

In PyCharm, right-click `gui.py` and choose **Run 'gui'**.
Tkinter is included with Python on Windows and macOS. On Linux, install it
with `sudo apt install python3-tk`.

## Troubleshooting: find faults (cut wires, missing parts)

`troubleshoot.py` compares a photo of the circuit with a photo of the same
circuit **when it was working** (a "known-good" or "golden sample" photo, the
method used in factory circuit inspection):

1. It finds every spot where the two photos differ.
2. If a part from the good photo is gone, it reports **"Missing component: LED"**
   (the AI detector names the part).
3. Any other change is reported as a **"Wire or connection fault (possible cut or loose wire)"**.
4. It saves the checked photo with red boxes around each fault.

```bash
python troubleshoot.py --demo                             # check the two example faults
python troubleshoot.py good.jpg checked.jpg               # your own photos
python troubleshoot.py good.jpg checked.jpg --no-ai       # faster: find changes only
python troubleshoot.py --test                             # self-test
```

In the app window: open the photo to check, click **Troubleshoot...**, then
choose the photo of the working circuit.

**Take both photos from the same position** (for example with the phone on a
stand) and in similar light. Otherwise the whole picture counts as "changed".

Example fault images, made by `make_sample.py`, are in `fault_test/`:
`fault_cut_wire.jpg` (the battery's red wire is cut) and `fault_missing_led.jpg`
(the green LED is missing). `sample_circuit.jpg` is the working circuit.

## Command line

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
