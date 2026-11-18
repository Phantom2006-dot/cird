"""Identify electronic components in a circuit image.

Two stages, no custom training needed:
  1. YOLO-World (open-vocabulary detector, yolov8s-worldv2) is given the
     component names with set_classes() and proposes boxes.
  2. Each box is cropped and re-labelled by CLIP ViT-B/32 (zero-shot image
     classification). YOLO-World's own labels are unreliable for small
     electronic parts (it tends to call everything "battery"), but its
     boxes are good; CLIP is much better at naming what is inside a box.

Usage:
    python identify_circuit.py                       # sample_circuit.jpg
    python identify_circuit.py photo.jpg -o out.jpg
    python identify_circuit.py photo.jpg --yolo-only # skip the CLIP stage
    python identify_circuit.py --test                # self-test
"""
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "yolov8s-worldv2.pt")
CLIP_PATH = os.path.join(HERE, "models", "ViT-B-32.pt")
# The old CLIP URL (openaipublic.azureedge.net) is dead; this is the working one.
CLIP_URL = ("https://openaipublic.blob.core.windows.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")

COMPONENTS = [
    "resistor", "capacitor", "LED", "transistor", "IC chip",
    "diode", "battery", "breadboard",
]
# Descriptions CLIP uses to name a crop; each is combined with every template
# below and the text embeddings are averaged (a "prompt ensemble").
# "wire" lets CLIP reject boxes that only contain jumper wires.
CLIP_PROMPTS = {
    "resistor": ["resistor", "resistor with colored bands"],
    "capacitor": ["electrolytic capacitor", "capacitor"],
    "LED": ["LED", "light emitting diode", "red LED lamp"],
    "transistor": ["transistor", "black transistor with three legs"],
    "IC chip": ["integrated circuit chip", "IC chip", "microchip with pins"],
    "diode": ["diode", "rectifier diode"],
    "battery": ["9 volt battery", "battery"],
    "breadboard": ["breadboard", "solderless breadboard"],
    "wire": ["jumper wires", "wire"],
}
TEMPLATES = ["a photo of a {}.", "an illustration of a {}.",
             "a close-up photo of a {}, an electronic component."]
COLORS = {
    "resistor": (230, 126, 34), "capacitor": (52, 152, 219), "LED": (231, 76, 60),
    "transistor": (155, 89, 182), "IC chip": (26, 188, 156), "diode": (241, 196, 15),
    "battery": (46, 204, 113), "breadboard": (149, 165, 166),
}


def prepare_clip_cache():
    """Make every `clip.load("ViT-B/32")` call, including the one Ultralytics
    makes inside set_classes(), use the bundled models/ViT-B-32.pt directly
    (no copying). If the file is missing, download it once into models/
    from the working URL (the old azureedge one is dead)."""
    import clip as clip_pkg
    from clip import clip  # the submodule holds _MODELS
    clip._MODELS["ViT-B/32"] = CLIP_URL
    if not os.path.exists(CLIP_PATH):
        # Download ONCE into models/ (clip._download checks the SHA-256).
        # Otherwise Ultralytics and the relabelling step would each download
        # their own copy into two different cache folders.
        print("CLIP model not found in models/, downloading it once (338 MB)...")
        clip._download(CLIP_URL, os.path.dirname(CLIP_PATH))
    if not getattr(clip.load, "_bundled", False):
        original = clip.load

        def load(name, *args, **kwargs):
            if name == "ViT-B/32":
                name = CLIP_PATH
            return original(name, *args, **kwargs)

        load._bundled = True
        clip.load = clip_pkg.load = load
    return CLIP_PATH


class Identifier:
    def __init__(self, classes=COMPONENTS):
        clip_file = prepare_clip_cache()
        from ultralytics import YOLOWorld
        self.yolo = YOLOWorld(MODEL_PATH)
        self.yolo.set_classes(list(classes))
        self._clip_file = clip_file
        self._clip = None

    def _load_clip(self):
        if self._clip is None:
            import torch
            from clip import clip
            model, preprocess = clip.load(self._clip_file, device="cpu")
            names = list(CLIP_PROMPTS)
            rows = []
            with torch.no_grad():
                for n in names:
                    prompts = [t.format(d) for d in CLIP_PROMPTS[n] for t in TEMPLATES]
                    e = model.encode_text(clip.tokenize(prompts))
                    e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
                    rows.append(e / e.norm())
            self._clip = (model, preprocess, names, torch.stack(rows))
        return self._clip

    def classify_crop(self, crop):
        """Return (label, probability) for a PIL image crop using CLIP."""
        import torch
        model, preprocess, names, text = self._load_clip()
        with torch.no_grad():
            img = model.encode_image(preprocess(crop).unsqueeze(0))
            img = img / img.norm(dim=-1, keepdim=True)
            probs = (100.0 * img @ text.T).softmax(dim=-1)[0]
        i = int(probs.argmax())
        return names[i], float(probs[i])

    def detect(self, image_path, conf=0.005, min_clip=0.2, use_clip=True):
        """Return a list of dicts: label, score, box (x1, y1, x2, y2)."""
        from PIL import Image
        r = self.yolo.predict(image_path, conf=conf, iou=0.5, agnostic_nms=True,
                              max_det=100, verbose=False)[0]
        img = Image.open(image_path).convert("RGB")
        dets = []
        for cls, score, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            box = [int(v) for v in box]
            label = r.names[int(cls)]
            if use_clip:
                x1, y1, x2, y2 = box
                pad = int(0.1 * max(x2 - x1, y2 - y1))  # a bit of context helps CLIP
                crop = img.crop((max(0, x1 - pad), max(0, y1 - pad),
                                 min(img.width, x2 + pad), min(img.height, y2 + pad)))
                label, score = self.classify_crop(crop)
                if label not in COMPONENTS or score < min_clip:
                    continue
            dets.append({"label": label, "score": round(score, 3), "box": box})
        return drop_group_boxes(dets)


def _contains(outer, inner, slack=0.9):
    """True if most of `inner` lies inside `outer`."""
    ix1, iy1 = max(outer[0], inner[0]), max(outer[1], inner[1])
    ix2, iy2 = min(outer[2], inner[2]), min(outer[3], inner[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return area > 0 and inter / area >= slack


def drop_group_boxes(dets):
    """YOLO-World often adds one big box around a pile of parts. Drop any
    box (except a breadboard, which holds parts by design) that encloses two
    or more other detections."""
    keep = []
    for d in dets:
        inside = sum(1 for o in dets if o is not d and _contains(d["box"], o["box"]))
        if d["label"] == "breadboard" or inside < 2:
            keep.append(d)
    return keep


def summarize(dets):
    return Counter(d["label"] for d in dets)


def draw(image_path, dets, output):
    from PIL import Image, ImageDraw
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    # draw big boxes first so labels of small parts stay on top
    for det in sorted(dets, key=lambda x: -(x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1])):
        c = COLORS.get(det["label"], (255, 0, 0))
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        text = f"{det['label']} {det['score']:.2f}"
        tw = d.textlength(text) + 8
        d.rectangle([x1, y1 - 16, x1 + tw, y1], fill=c)
        d.text((x1 + 4, y1 - 15), text, fill=(0, 0, 0))
    img.save(output)
    return output


def identify(image, output=None, conf=0.005, use_clip=True, identifier=None):
    identifier = identifier or Identifier()
    dets = identifier.detect(image, conf=conf, use_clip=use_clip)
    output = output or os.path.splitext(image)[0] + "_detected.jpg"
    draw(image, dets, output)
    return dets, summarize(dets), output


def self_test():
    fake = [{"label": "resistor", "score": .9, "box": [0, 0, 5, 5]},
            {"label": "resistor", "score": .4, "box": [5, 5, 9, 9]},
            {"label": "LED", "score": .7, "box": [1, 1, 3, 3]}]
    assert summarize(fake) == {"resistor": 2, "LED": 1}
    group = {"label": "resistor", "score": .5, "box": [0, 0, 100, 100]}
    a = {"label": "LED", "score": .9, "box": [10, 10, 20, 20]}
    b = {"label": "diode", "score": .9, "box": [50, 50, 60, 60]}
    board = dict(group, label="breadboard")
    assert drop_group_boxes([group, a, b]) == [a, b]
    assert drop_group_boxes([board, a, b]) == [board, a, b]
    assert set(COMPONENTS) <= set(CLIP_PROMPTS) and set(COMPONENTS) <= set(COLORS)
    assert os.path.exists(MODEL_PATH), "YOLO-World weights missing (models/yolov8s-worldv2.pt)"
    sample = os.path.join(HERE, "sample_circuit.jpg")
    out = os.path.join(HERE, "_selftest_out.jpg")
    dets, counts, path = identify(sample, out)
    assert os.path.getsize(path) > 0
    os.remove(path)
    assert all(d["label"] in COMPONENTS for d in dets)
    assert len(counts) >= 3, f"expected >=3 component types on the sample, got {dict(counts)}"
    assert counts["resistor"] >= 3, f"expected the 3 resistors, got {dict(counts)}"
    for must in ("breadboard", "IC chip", "LED"):
        assert must in counts, f"{must} not detected on the sample"
    print(f"All circuit_identification tests passed: {dict(counts)}")


def main():
    p = argparse.ArgumentParser(description="Identify circuit components (YOLO-World + CLIP)")
    p.add_argument("image", nargs="?", default=os.path.join(HERE, "sample_circuit.jpg"))
    p.add_argument("-o", "--output", help="annotated output image path")
    p.add_argument("--conf", type=float, default=0.005, help="YOLO-World box threshold")
    p.add_argument("--yolo-only", action="store_true", help="use YOLO-World labels, skip CLIP")
    p.add_argument("--test", action="store_true", help="run self-test")
    a = p.parse_args()
    if a.test:
        self_test()
        return
    dets, counts, out = identify(a.image, a.output, a.conf, use_clip=not a.yolo_only)
    print(f"Found {len(dets)} component(s) in {a.image}:")
    for name, n in counts.most_common():
        best = max(d["score"] for d in dets if d["label"] == name)
        print(f"  {name:<12} x{n}  (best score {best:.2f})")
    print(f"Annotated image saved to {out}")


if __name__ == "__main__":
    main()
