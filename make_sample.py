"""Draw synthetic circuit images.

    sample_circuit.jpg               the working circuit (also the "known-good" photo)
    fault_test/fault_cut_wire.jpg    same circuit, but the battery's red wire is cut
    fault_test/fault_missing_led.jpg same circuit, but the green LED is missing

The detector works best on real photos; these images exist so the project
can be demonstrated out of the box. Replace them with your own photos.
"""
import os
import random
from PIL import Image, ImageDraw, ImageFilter

W, H = 1280, 860


def breadboard(d, x0, y0, x1, y1):
    d.rounded_rectangle([x0, y0, x1, y1], 18, fill=(236, 233, 224), outline=(190, 186, 176), width=3)
    # power rails
    for ry in (y0 + 25, y1 - 55):
        d.line([x0 + 30, ry, x1 - 30, ry], fill=(200, 40, 40), width=3)
        d.line([x0 + 30, ry + 30, x1 - 30, ry + 30], fill=(40, 60, 200), width=3)
        for x in range(x0 + 45, x1 - 40, 22):
            for dy in (10, 18):
                d.rectangle([x, ry + dy - 2, x + 5, ry + dy + 3], fill=(70, 70, 70))
    # center gap
    cy = (y0 + y1) // 2
    d.rectangle([x0 + 20, cy - 8, x1 - 20, cy + 8], fill=(215, 211, 200))
    for x in range(x0 + 45, x1 - 40, 22):
        for y in list(range(y0 + 95, cy - 20, 22)) + list(range(cy + 25, y1 - 80, 22)):
            d.rectangle([x, y, x + 6, y + 6], fill=(60, 60, 60))


def resistor(d, x, y, length=150, vertical=False):
    bands = [(150, 75, 0), (0, 0, 0), (220, 0, 0), (212, 175, 55)]
    if vertical:
        d.line([x, y, x, y + length], fill=(170, 170, 170), width=4)
        bx, by = x - 14, y + length // 2 - 38
        d.rounded_rectangle([bx, by, bx + 28, by + 76], 12, fill=(222, 196, 150), outline=(150, 120, 80))
        for i, c in enumerate(bands):
            d.rectangle([bx, by + 12 + i * 15, bx + 28, by + 19 + i * 15], fill=c)
    else:
        d.line([x, y, x + length, y], fill=(170, 170, 170), width=4)
        bx, by = x + length // 2 - 38, y - 14
        d.rounded_rectangle([bx, by, bx + 76, by + 28], 12, fill=(222, 196, 150), outline=(150, 120, 80))
        for i, c in enumerate(bands):
            d.rectangle([bx + 12 + i * 15, by, bx + 19 + i * 15, by + 28], fill=c)


def led(d, x, y, color):
    d.line([x - 10, y, x - 10, y + 70], fill=(170, 170, 170), width=4)
    d.line([x + 10, y, x + 10, y + 55], fill=(170, 170, 170), width=4)
    d.rectangle([x - 26, y - 8, x + 26, y + 2], fill=tuple(int(c * .8) for c in color))
    d.chord([x - 24, y - 70, x + 24, y - 20], 180, 360, fill=color)
    d.rectangle([x - 24, y - 46, x + 24, y - 6], fill=color)
    d.ellipse([x - 12, y - 60, x - 2, y - 40], fill=(255, 255, 255))


def capacitor(d, x, y):
    d.line([x - 12, y + 110, x - 12, y + 160], fill=(170, 170, 170), width=4)
    d.line([x + 12, y + 110, x + 12, y + 150], fill=(170, 170, 170), width=4)
    d.rounded_rectangle([x - 38, y, x + 38, y + 115], 10, fill=(30, 40, 120), outline=(15, 20, 60), width=2)
    d.rectangle([x + 16, y + 4, x + 30, y + 111], fill=(190, 190, 210))
    d.ellipse([x - 38, y - 10, x + 38, y + 14], fill=(160, 160, 170), outline=(110, 110, 120))
    d.text((x - 28, y + 45), "470uF", fill=(230, 230, 230))


def ic_chip(d, x, y):
    d.rectangle([x, y, x + 210, y + 80], fill=(30, 30, 30), outline=(10, 10, 10))
    for i in range(8):
        px = x + 15 + i * 25
        d.rectangle([px, y - 14, px + 10, y], fill=(190, 190, 190))
        d.rectangle([px, y + 80, px + 10, y + 94], fill=(190, 190, 190))
    d.ellipse([x + 4, y + 30, x + 20, y + 50], fill=(60, 60, 60))
    d.text((x + 70, y + 32), "NE555P", fill=(210, 210, 210))


def transistor(d, x, y):
    for dx in (-14, 0, 14):
        d.line([x + dx, y + 40, x + dx, y + 100], fill=(170, 170, 170), width=4)
    d.chord([x - 28, y - 10, x + 28, y + 46], 180, 360, fill=(25, 25, 25))
    d.rectangle([x - 28, y + 18, x + 28, y + 46], fill=(25, 25, 25))
    d.text((x - 20, y + 22), "2N2222", fill=(200, 200, 200))


def diode(d, x, y):
    d.line([x, y, x + 150, y], fill=(170, 170, 170), width=4)
    d.rectangle([x + 45, y - 11, x + 105, y + 11], fill=(20, 20, 20))
    d.rectangle([x + 92, y - 11, x + 100, y + 11], fill=(210, 210, 210))


def battery(d, x, y, cut_wire=False):
    d.rounded_rectangle([x, y, x + 140, y + 230], 10, fill=(20, 20, 20))
    d.rectangle([x, y + 120, x + 140, y + 230], fill=(205, 130, 20))
    d.rectangle([x + 25, y - 18, x + 55, y], fill=(180, 180, 180))
    d.ellipse([x + 80, y - 20, x + 115, y + 2], fill=(180, 180, 180))
    d.text((x + 45, y + 60), "9V", fill=(240, 240, 240))
    if cut_wire:   # red wire broken: a gap with frayed copper ends
        d.line([x + 40, y - 18, x + 40, y - 60, 190, y - 60], fill=(200, 30, 30), width=5)
        d.line([235, y - 60, 360, y - 60], fill=(200, 30, 30), width=5)
        for dy in (-4, 0, 4):
            d.line([190, y - 60, 199, y - 60 + dy * 2], fill=(205, 127, 50), width=1)
            d.line([235, y - 60, 226, y - 60 + dy * 2], fill=(205, 127, 50), width=1)
    else:
        d.line([x + 40, y - 18, x + 40, y - 60, 360, y - 60], fill=(200, 30, 30), width=5)
    d.line([x + 97, y - 18, x + 97, y - 40, 360, y - 40], fill=(20, 20, 20), width=5)


def draw_circuit(path, cut_wire=False, missing_led=False):
    random.seed(3)   # same desk texture every time, so only the fault differs
    img = Image.new("RGB", (W, H), (120, 98, 72))
    d = ImageDraw.Draw(img)
    for _ in range(4000):  # wood-ish desk texture
        x, y = random.randrange(W), random.randrange(H)
        d.line([x, y, x + random.randint(20, 80), y], fill=(110 + random.randint(-12, 12), 88, 62))
    breadboard(d, 340, 120, 1240, 760)
    battery(d, 60, 420, cut_wire=cut_wire)
    resistor(d, 420, 260)
    resistor(d, 640, 260)
    resistor(d, 1040, 520, vertical=True)
    led(d, 900, 280, (230, 30, 30))
    if not missing_led:
        led(d, 990, 280, (30, 200, 60))
    capacitor(d, 470, 470)
    ic_chip(d, 620, 410)
    transistor(d, 900, 520)
    diode(d, 600, 640)
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img.save(path, quality=92)
    print(f"Saved {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "fault_test"), exist_ok=True)
    draw_circuit(os.path.join(here, "sample_circuit.jpg"))
    draw_circuit(os.path.join(here, "fault_test", "fault_cut_wire.jpg"), cut_wire=True)
    draw_circuit(os.path.join(here, "fault_test", "fault_missing_led.jpg"), missing_led=True)


if __name__ == "__main__":
    main()
