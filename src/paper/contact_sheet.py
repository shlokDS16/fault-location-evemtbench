"""Tile the pages of a PDF into contact-sheet PNGs for a quick visual check.
Usage: python src/paper/contact_sheet.py <in.pdf> <out_prefix> [dpi] [pages_per_sheet] [columns]"""
import sys

import fitz
from PIL import Image

pdf, prefix = sys.argv[1], sys.argv[2]
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 60
per = int(sys.argv[4]) if len(sys.argv) > 4 else 6
cols = int(sys.argv[5]) if len(sys.argv) > 5 else 3
doc = fitz.open(pdf)
imgs = [Image.frombytes("RGB", (p.width, p.height), p.samples) for p in (pg.get_pixmap(dpi=dpi) for pg in doc)]
for s in range(0, len(imgs), per):
    chunk = imgs[s:s + per]
    w, h = chunk[0].size
    rows = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), "gray")
    for i, im in enumerate(chunk):
        sheet.paste(im, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    sheet.save(f"{prefix}_{s // per + 1}.png")
    print(f"{prefix}_{s // per + 1}.png pages {s + 1}-{s + len(chunk)}")
