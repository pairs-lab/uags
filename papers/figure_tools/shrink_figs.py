"""Re-encode the large raster images of figure PDFs as JPEG, in place.

Matplotlib stores colormapped renders as Flate palette images, which compress
poorly: the paper reached 6.2 MB and the supplement 17 MB. Each image larger
than --min-kb is decoded to RGB (or grey) and stored as a JPEG (quality 90, no
chroma subsampling) if that is at least 20% smaller. Images with a soft mask
are left as they are. Text and vector content are not touched. A file is only
rewritten if it gets smaller.

Needs PyMuPDF and Pillow, which are not in the project venv:
    uv venv /tmp/pdfenv && uv pip install --python /tmp/pdfenv/bin/python pymupdf pillow
    /tmp/pdfenv/bin/python papers/figure_tools/shrink_figs.py papers/figs/teaser.pdf ...
"""
import argparse
import io
import os

import pymupdf
from PIL import Image


def shrink(path, quality=90, min_kb=20):
    doc = pymupdf.open(path)
    seen, n_done = set(), 0
    for page in doc:
        for img in page.get_images(full=True):
            xref, smask = img[0], img[1]
            if xref in seen or smask:
                continue
            seen.add(xref)
            raw = doc.xref_stream_raw(xref)
            if len(raw) < min_kb * 1024:
                continue
            pix = pymupdf.Pixmap(doc, xref)
            if pix.alpha:
                continue
            if pix.colorspace.n not in (1, 3):
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            mode = 'L' if pix.colorspace.n == 1 else 'RGB'
            im = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=quality, subsampling=0, optimize=True)
            if buf.tell() < 0.8 * len(raw):
                page.replace_image(xref, stream=buf.getvalue())
                n_done += 1
    size0 = os.path.getsize(path)
    tmp = path + '.tmp.pdf'
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    size1 = os.path.getsize(tmp)
    if n_done and size1 < size0:
        os.replace(tmp, path)
        print(f'{path}: {n_done} images re-encoded, {size0 / 1e6:.2f} -> {size1 / 1e6:.2f} MB')
    else:
        os.remove(tmp)
        print(f'{path}: kept ({size0 / 1e6:.2f} MB; re-encoded {n_done}, would be {size1 / 1e6:.2f} MB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pdfs', nargs='+')
    ap.add_argument('--quality', type=int, default=90)
    ap.add_argument('--min-kb', type=float, default=20)
    a = ap.parse_args()
    for p in a.pdfs:
        shrink(p, a.quality, a.min_kb)
