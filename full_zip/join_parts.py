"""Join the 4 parts back into circuit_identification.zip (any OS).

    python join_parts.py
"""
import glob
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = "4ef5e88f3d91b18a6010f15fd850a0f899510b6d426d88c0b23affbae8d39d11"

parts = sorted(glob.glob(os.path.join(HERE, "circuit_identification.zip.0*")))
if len(parts) != 4:
    raise SystemExit(f"Expected 4 parts, found {len(parts)}. Download all of them first.")
out = os.path.join(HERE, "circuit_identification.zip")
h = hashlib.sha256()
with open(out, "wb") as dst:
    for p in parts:
        with open(p, "rb") as src:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
                h.update(chunk)
print(f"Created {out}")
print("OK - identical to the original." if h.hexdigest() == EXPECTED
      else f"WARNING - checksum mismatch ({h.hexdigest()}); re-download the parts.")
