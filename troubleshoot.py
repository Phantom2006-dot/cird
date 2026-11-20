"""Circuit troubleshooting: compare a circuit photo with a known-good photo.

How it works (the "golden sample" method used in factory circuit inspection):
  1. Put the two photos side by side at the same size.
  2. Find every area where they differ (a "difference picture").
  3. For each changed area, decide what happened:
       - a part that is in the good photo is gone   ->  MISSING COMPONENT (e.g. LED)
       - otherwise                                   ->  WIRE / CONNECTION FAULT (e.g. cut wire)
     The AI detector from identify_circuit.py names the parts.
  4. Draw red boxes on the checked photo and print a fault report.

Important: take both photos from the SAME position (e.g. phone on a stand),
otherwise everything looks "different".

Usage:
    python troubleshoot.py good.jpg checked.jpg           # compare two photos
    python troubleshoot.py good.jpg checked.jpg --no-ai   # faster, without naming parts
    python troubleshoot.py --demo                         # uses the fault_test images
    python troubleshoot.py --test                         # self-test
"""
import argparse
import os
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIFF_LEVEL = 40        # how much a pixel must change (0-255) to count as different
MIN_AREA = 0.0003      # ignore changed spots smaller than this part of the photo (noise)
GONE_COVER = 0.6       # a part counts as "gone" if this much of it changed


def load_pair(good_path, checked_path):
    """Read both photos; resize the checked one to the good one's size."""
    good = cv2.imread(good_path)
    checked = cv2.imread(checked_path)
    if good is None or checked is None:
        raise SystemExit("Could not open one of the photos.")
    if checked.shape != good.shape:
        checked = cv2.resize(checked, (good.shape[1], good.shape[0]))
    return good, checked


def find_changes(good, checked):
    """Return (list of changed areas as boxes, the change mask)."""
    a = cv2.GaussianBlur(good, (5, 5), 0)                # soften camera noise
    b = cv2.GaussianBlur(checked, (5, 5), 0)
    diff = cv2.absdiff(a, b).max(axis=2)                  # biggest change in any colour
    _, mask = cv2.threshold(diff, DIFF_LEVEL, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)   # join nearby spots
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA * good.shape[0] * good.shape[1]
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
    return sorted(boxes), mask


def changed_share(box, mask):
    """How much of a box (0 to 1) lies in the changed area."""
    x1, y1, x2, y2 = box
    part = mask[y1:y2, x1:x2]
    return float((part > 0).mean()) if part.size else 0.0


def still_there(part, checked_parts):
    """Is the same kind of part still found at the same place in the checked photo?"""
    x1, y1, x2, y2 = part["box"]
    for other in checked_parts:
        ox1, oy1, ox2, oy2 = other["box"]
        cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        if other["label"] == part["label"] and x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def diagnose(changes, mask, good_parts, checked_parts):
    """Turn each changed area into a fault with a plain-English description."""
    faults = []
    for area in changes:
        ax1, ay1, ax2, ay2 = area
        gone = [p for p in good_parts
                if changed_share(p["box"], mask) >= GONE_COVER
                and ax1 <= (p["box"][0] + p["box"][2]) / 2 <= ax2
                and ay1 <= (p["box"][1] + p["box"][3]) / 2 <= ay2
                and not still_there(p, checked_parts)]
        if gone:
            names = ", ".join(sorted({p["label"] for p in gone}))
            faults.append({"box": area, "kind": "MISSING COMPONENT",
                           "text": f"Missing component: {names}"})
        else:
            faults.append({"box": area, "kind": "WIRE / CONNECTION",
                           "text": "Wire or connection fault (possible cut or loose wire)"})
    return faults


def draw_faults(checked, faults, path):
    out = checked.copy()
    for i, f in enumerate(faults, 1):
        x1, y1, x2, y2 = f["box"]
        cv2.rectangle(out, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 0, 255), 3)
        label = f"FAULT {i}: {f['kind']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 10, th + 6)
        cv2.rectangle(out, (x1 - 6, ty - th - 6), (x1 - 6 + tw + 8, ty + 4), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 - 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    banner = "NO FAULTS FOUND" if not faults else f"{len(faults)} FAULT(S) FOUND"
    color = (40, 160, 40) if not faults else (0, 0, 255)
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), color, -1)
    cv2.putText(out, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(path, out)
    return path


def troubleshoot(good_path, checked_path, identifier=None, output=None, use_ai=True):
    """Compare the photos and return (faults, good part counts, checked part counts, output path)."""
    good, checked = load_pair(good_path, checked_path)
    changes, mask = find_changes(good, checked)
    good_parts, checked_parts = [], []
    if use_ai and changes:                    # only run the AI if something changed
        if identifier is None:
            from identify_circuit import Identifier
            identifier = Identifier()
        good_parts = identifier.detect(good_path)
        checked_parts = identifier.detect(checked_path)
    faults = diagnose(changes, mask, good_parts, checked_parts)
    output = output or os.path.splitext(checked_path)[0] + "_faults.jpg"
    draw_faults(checked, faults, output)
    return faults, Counter(p["label"] for p in good_parts), \
        Counter(p["label"] for p in checked_parts), output


def print_report(good_path, checked_path, faults, good_counts, checked_counts, output):
    print("TROUBLESHOOTING REPORT")
    print(f"Good photo:     {good_path}")
    print(f"Checked photo:  {checked_path}")
    if not faults:
        print("Result:         NO FAULTS FOUND - the circuit matches the good photo.")
    else:
        print(f"Result:         {len(faults)} fault(s) found")
        for i, f in enumerate(faults, 1):
            x1, y1, x2, y2 = f["box"]
            print(f"  {i}. {f['text']}   (area x {x1}-{x2}, y {y1}-{y2})")
    # count only the parts reported missing (the AI's labels elsewhere can vary a little)
    missing = sorted({name.strip() for f in faults if f["kind"] == "MISSING COMPONENT"
                      for name in f["text"].split(":", 1)[1].split(",")})
    if missing:
        print("Parts counted:  " + ", ".join(f"{k} {good_counts[k]} -> {checked_counts[k]}"
                                               for k in missing))
    print(f"Marked photo:   {output}")


def self_test():
    good = os.path.join(HERE, "sample_circuit.jpg")
    cut = os.path.join(HERE, "fault_test", "fault_cut_wire.jpg")
    led = os.path.join(HERE, "fault_test", "fault_missing_led.jpg")
    tmp = os.path.join(HERE, "_troubleshoot_test.jpg")

    # the good photo compared with itself: no faults
    faults, *_ = troubleshoot(good, good, output=tmp, use_ai=False)
    assert faults == []

    # cut wire: one wire fault, where the red wire was cut (x 190-235, y 360)
    faults, *_ = troubleshoot(good, cut, output=tmp, use_ai=False)
    assert len(faults) == 1 and faults[0]["kind"] == "WIRE / CONNECTION", faults
    x1, y1, x2, y2 = faults[0]["box"]
    assert x1 <= 200 and x2 >= 225 and y1 <= 360 <= y2

    # missing LED: one fault where the green LED was (x about 966-1014, y 210-350)
    faults, *_ = troubleshoot(good, led, output=tmp, use_ai=False)
    assert len(faults) == 1
    x1, y1, x2, y2 = faults[0]["box"]
    assert x1 <= 970 and x2 >= 1010 and y1 <= 220 and y2 >= 300

    # with the AI: the missing LED is named
    faults, before, after, _ = troubleshoot(good, led, output=tmp)
    assert faults[0]["kind"] == "MISSING COMPONENT", faults
    assert "LED" in faults[0]["text"] and before["LED"] > after["LED"]
    os.remove(tmp)
    print("All troubleshooting tests passed.")


def main():
    p = argparse.ArgumentParser(description="Find faults by comparing with a known-good photo")
    p.add_argument("good", nargs="?", help="photo of the working circuit")
    p.add_argument("checked", nargs="?", help="photo of the circuit to check")
    p.add_argument("-o", "--output", help="where to save the marked photo")
    p.add_argument("--no-ai", action="store_true", help="find changes only, don't name parts")
    p.add_argument("--demo", action="store_true", help="check the two fault_test images")
    p.add_argument("--test", action="store_true", help="run the self-test")
    a = p.parse_args()

    if a.test:
        self_test()
        return
    if a.demo:
        good = os.path.join(HERE, "sample_circuit.jpg")
        identifier = None
        if not a.no_ai:
            from identify_circuit import Identifier
            identifier = Identifier()
        for name in ("fault_cut_wire.jpg", "fault_missing_led.jpg"):
            checked = os.path.join(HERE, "fault_test", name)
            result = troubleshoot(good, checked, identifier, use_ai=not a.no_ai)
            print_report(good, checked, *result)
            print()
        return
    if not (a.good and a.checked):
        p.error("give two photos: the good one, then the one to check (or use --demo)")
    result = troubleshoot(a.good, a.checked, output=a.output, use_ai=not a.no_ai)
    print_report(a.good, a.checked, *result)


if __name__ == "__main__":
    main()
