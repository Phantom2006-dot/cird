"""Circuit Component Identifier: desktop app (Tkinter).

    python gui.py                 # open the app
    python gui.py my_photo.jpg    # open the app with a photo already loaded

How to use it:
  1. Wait for "Ready" in the status bar (the AI models load once, 20-60 s).
  2. Click "Open Image..." and pick a photo of a circuit.
  3. Click "Identify". Boxes appear on the photo, and the parts are listed on
     the right. Click a row in the list to highlight that part on the photo.
  4. Drag "Min score" to hide low-confidence guesses.
  5. "Save Result..." saves the labelled image (and a CSV of the detections).

The detection itself is done by identify_circuit.py (YOLO-World + CLIP);
this file only adds the window around it.
"""
import argparse
import csv
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

import identify_circuit as ic

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_TYPES = [("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
HIGHLIGHT = (255, 0, 255)


def _font(size):
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def render(img, dets, max_w, max_h, selected=None):
    """Scale `img` to fit max_w x max_h and draw the boxes at screen size,
    so labels stay readable however big the photo is."""
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    view = img.resize((w, h), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(view)
    font = _font(13)
    order = sorted(range(len(dets)), key=lambda i: -_area(dets[i]["box"]))
    if selected is not None and selected in order:   # selected box drawn last, on top
        order.remove(selected)
        order.append(selected)
    for i in order:
        det = dets[i]
        x1, y1, x2, y2 = (int(v * scale) for v in det["box"])
        sel = i == selected
        color = HIGHLIGHT if sel else ic.COLORS.get(det["label"], (255, 0, 0))
        d.rectangle([x1, y1, x2, y2], outline=color, width=4 if sel else 2)
        text = f"{i + 1}. {det['label']} {det['score']:.2f}"
        tw = d.textlength(text, font=font) + 6
        ty = y1 - 16 if y1 >= 16 else y1
        d.rectangle([x1, ty, x1 + tw, ty + 16], fill=color)
        d.text((x1 + 3, ty + 1), text, fill=(0, 0, 0), font=font)
    return view


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


class App:
    def __init__(self, root, initial_image=None):
        self.root = root
        self.identifier = None
        self.image_path = None
        self.image = None          # PIL image (original size)
        self.all_dets = []         # everything the model returned
        self.dets = []             # after the min-score filter
        self.selected = None
        self.busy = False
        self.events = queue.Queue()  # worker threads -> GUI thread
        self._photo = None
        self._resize_job = None

        root.title("Circuit Component Identifier")
        root.geometry("1200x760")
        root.minsize(820, 520)
        self._build()
        self._set_status("Loading AI models (first start takes 20-60 seconds)...")
        threading.Thread(target=self._load_models, daemon=True).start()
        root.after(100, self._poll)
        if initial_image:
            self.open_image(initial_image)

    # ------------------------------------------------------------ layout
    def _build(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Open Image...", command=self.choose_image).pack(side="left")
        self.btn_identify = ttk.Button(bar, text="Identify", command=self.run_identify,
                                       state="disabled")
        self.btn_identify.pack(side="left", padx=6)
        self.btn_save = ttk.Button(bar, text="Save Result...", command=self.save_result,
                                   state="disabled")
        self.btn_save.pack(side="left")

        ttk.Label(bar, text="   Min score:").pack(side="left")
        self.min_score = tk.DoubleVar(value=0.20)
        ttk.Scale(bar, from_=0.20, to=0.95, variable=self.min_score, length=180,
                  command=lambda _v: self._apply_filter()).pack(side="left", padx=4)
        self.min_score_label = ttk.Label(bar, text="0.20", width=5)
        self.min_score_label.pack(side="left")

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self.canvas = tk.Canvas(body, bg="#2b2b2b", highlightthickness=0)
        self.canvas.bind("<Configure>", self._on_resize)
        body.add(self.canvas, weight=3)

        side = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(side, weight=1)

        ttk.Label(side, text="Summary", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.summary = ttk.Treeview(side, columns=("count", "best"), height=8)
        self.summary.heading("#0", text="Component")
        self.summary.heading("count", text="Count")
        self.summary.heading("best", text="Best score")
        self.summary.column("#0", width=130)
        self.summary.column("count", width=60, anchor="center")
        self.summary.column("best", width=80, anchor="center")
        self.summary.pack(fill="x", pady=(2, 10))

        ttk.Label(side, text="Detections (click to highlight)",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        frame = ttk.Frame(side)
        frame.pack(fill="both", expand=True, pady=(2, 0))
        self.detlist = ttk.Treeview(frame, columns=("label", "score"), show="headings")
        self.detlist.heading("label", text="Component")
        self.detlist.heading("score", text="Score")
        self.detlist.column("label", width=150)
        self.detlist.column("score", width=70, anchor="center")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.detlist.yview)
        self.detlist.configure(yscrollcommand=scroll.set)
        self.detlist.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.detlist.bind("<<TreeviewSelect>>", self._on_select)

        self.status = ttk.Label(self.root, relief="sunken", anchor="w", padding=(6, 2))
        self.status.pack(side="bottom", fill="x")

    # ------------------------------------------------------------ background work
    def _load_models(self):
        try:
            identifier = ic.Identifier()
            identifier._load_clip()        # load CLIP now so the first Identify is quick
            self.events.put(("models_ready", identifier))
        except Exception as e:             # shown in a dialog, not a crash
            self.events.put(("error", f"Could not load the AI models:\n\n{e}"))

    def _detect(self, path):
        try:
            start = time.time()
            dets = self.identifier.detect(path)
            self.events.put(("detected", (path, dets, time.time() - start)))
        except Exception as e:
            self.events.put(("error", f"Identification failed:\n\n{e}"))

    def _poll(self):
        """Tkinter isn't thread-safe: workers post events, the GUI thread handles them."""
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "models_ready":
                    self.identifier = data
                    self._set_status("Ready. Open an image and click Identify.")
                    self._update_buttons()
                    if getattr(self, "on_ready", None):
                        self.on_ready()
                elif kind == "detected":
                    path, dets, secs = data
                    self.busy = False
                    if path == self.image_path:
                        self.all_dets = dets
                        self._apply_filter()
                        self._set_status(f"Found {len(self.all_dets)} part(s) in {secs:.1f} s. "
                                         f"{os.path.basename(path)}")
                    self._update_buttons()
                    if getattr(self, "on_detected", None):
                        self.on_detected()
                elif kind == "error":
                    self.busy = False
                    self._update_buttons()
                    self._set_status("Error, see the message.")
                    messagebox.showerror("Circuit Identifier", data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------ actions
    def choose_image(self):
        path = filedialog.askopenfilename(title="Open circuit image", filetypes=IMAGE_TYPES,
                                          initialdir=HERE)
        if path:
            self.open_image(path)

    def open_image(self, path):
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            messagebox.showerror("Circuit Identifier", f"Could not open the image:\n\n{e}")
            return
        self.image_path, self.image = path, img.convert("RGB")
        self.all_dets, self.dets, self.selected = [], [], None
        self._fill_tables()
        self._redraw()
        self.root.title(f"Circuit Component Identifier - {os.path.basename(path)}")
        if self.identifier:
            self._set_status(f"Loaded {os.path.basename(path)} "
                             f"({img.width}x{img.height}). Click Identify.")
        self._update_buttons()

    def run_identify(self):
        if not (self.identifier and self.image_path) or self.busy:
            return
        self.busy = True
        self._update_buttons()
        self._set_status("Identifying components...")
        threading.Thread(target=self._detect, args=(self.image_path,), daemon=True).start()

    def save_result(self):
        if not self.image_path:
            return
        base = os.path.splitext(os.path.basename(self.image_path))[0] + "_detected.jpg"
        out = filedialog.asksaveasfilename(
            title="Save labelled image", initialfile=base,
            initialdir=os.path.dirname(self.image_path), defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not out:
            return
        self.save_to(out)
        self._set_status(f"Saved {out} (+ .csv list of parts)")

    def save_to(self, out):
        ic.draw(self.image_path, self.dets, out)
        with open(os.path.splitext(out)[0] + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["#", "component", "score", "x1", "y1", "x2", "y2"])
            for i, d in enumerate(self.dets, 1):
                w.writerow([i, d["label"], d["score"], *d["box"]])

    # ------------------------------------------------------------ view updates
    def _apply_filter(self):
        m = round(self.min_score.get(), 2)
        self.min_score_label.config(text=f"{m:.2f}")
        self.dets = [d for d in self.all_dets if d["score"] >= m]
        self.selected = None
        self._fill_tables()
        self._redraw()
        self._update_buttons()

    def _fill_tables(self):
        self.summary.delete(*self.summary.get_children())
        self.detlist.delete(*self.detlist.get_children())
        counts = ic.summarize(self.dets)
        for name, n in counts.most_common():
            best = max(d["score"] for d in self.dets if d["label"] == name)
            self.summary.insert("", "end", text=name, values=(n, f"{best:.2f}"))
        if not counts and self.all_dets:
            self.summary.insert("", "end", text="(all hidden by Min score)", values=("", ""))
        for i, d in enumerate(self.dets):
            self.detlist.insert("", "end", iid=str(i), values=(f"{i + 1}. {d['label']}",
                                                             f"{d['score']:.2f}"))

    def _on_select(self, _event):
        sel = self.detlist.selection()
        self.selected = int(sel[0]) if sel else None
        self._redraw()

    def _on_resize(self, _event):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self._redraw)

    def _redraw(self):
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.image is None:
            self.canvas.create_text(cw // 2, ch // 2, fill="#bbbbbb", font=("TkDefaultFont", 14),
                                    text="Click \"Open Image...\" to choose a circuit photo")
            return
        if cw < 10 or ch < 10:
            return
        view = render(self.image, self.dets, cw - 10, ch - 10, self.selected)
        self._photo = ImageTk.PhotoImage(view)   # keep a reference or Tk drops it
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _update_buttons(self):
        ready = self.identifier is not None and self.image_path and not self.busy
        self.btn_identify.config(state="normal" if ready else "disabled")
        self.btn_save.config(state="normal" if self.all_dets and not self.busy else "disabled")

    def _set_status(self, text):
        self.status.config(text=text)


def main():
    p = argparse.ArgumentParser(description="Circuit Component Identifier (desktop app)")
    p.add_argument("image", nargs="?", help="photo to open at start")
    p.add_argument("--autotest", metavar="SCREENSHOT",
                   help="(for testing) open the image, identify, save a screenshot, quit")
    a = p.parse_args()

    root = tk.Tk()
    app = App(root, a.image)
    if a.autotest:
        def ready():
            app.run_identify()

        def detected():
            app.detlist.selection_set("0")   # highlight the first detection
            root.update()
            root.after(500, shoot)

        def shoot():
            from PIL import ImageGrab
            root.update()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height())).save(a.autotest)
            app.save_to(os.path.splitext(a.autotest)[0] + "_result.jpg")
            print(f"AUTOTEST found={len(app.all_dets)} shown={len(app.dets)} "
                  f"rows={len(app.detlist.get_children())}")
            root.destroy()

        app.on_ready, app.on_detected = ready, detected
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
