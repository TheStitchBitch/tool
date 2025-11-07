from __future__ import annotations
import io, json, math, zipfile
from typing import Dict, Tuple, List, Optional

from flask import Flask, request, send_file, jsonify, make_response
from PIL import Image, ImageDraw, ImageFont

# Optional embroidery support
try:
    from pyembroidery import EmbPattern, write_dst, write_pes  # type: ignore
    HAS_PYEMB = True
except Exception:
    HAS_PYEMB = False

app = Flask(__name__)

# ---------------------- security / config ----------------------
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB limit
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
BOLD_EVERY = 10
MAX_DIM = 8000  # max allowed image dimension (px)

# ---------------------- utilities ----------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def open_image(fs) -> Image.Image:
    """Open upload and normalize to RGB on white for stable quantization."""
    img = Image.open(fs.stream)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")

def resize_for_stitch_width(img: Image.Image, stitch_w: int) -> Image.Image:
    w, h = img.size
    # pre-shrink huge images for speed
    if max(w, h) > 2000:
        img = img.copy()
        img.thumbnail((2000, 2000))
        w, h = img.size
    ratio = stitch_w / float(w)
    new_h = max(1, int(round(h * ratio)))
    return img.resize((stitch_w, new_h), Image.Resampling.LANCZOS)

def quantize(img: Image.Image, k: int) -> Image.Image:
    """Median-cut palette, no dithering for crisp cells."""
    return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=k,
                       dither=Image.Dither.NONE).convert("RGB")

def palette_counts(img: Image.Image) -> Dict[Tuple[int,int,int], int]:
    counts: Dict[Tuple[int,int,int], int] = {}
    for rgb in img.getdata():
        counts[rgb] = counts.get(rgb, 0) + 1
    return counts

def to_hex(rgb: Tuple[int,int,int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"

def luminance(rgb: Tuple[int,int,int]) -> float:
    r, g, b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b

def draw_grid(base: Image.Image, cell_px: int, bold_every: int = BOLD_EVERY) -> Image.Image:
    """Scale each stitch to a cell and overlay a grid (bold every N)."""
    sx, sy = base.size
    out = base.resize((sx*cell_px, sy*cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    thin = (0, 0, 0, 70)
    bold = (0, 0, 0, 170)
    for x in range(sx + 1):
        draw.line([(x*cell_px, 0), (x*cell_px, sy*cell_px)],
                  fill=(bold if x % bold_every == 0 else thin), width=1)
    for y in range(sy + 1):
        draw.line([(0, y*cell_px), (sx*cell_px, y*cell_px)],
                  fill=(bold if y % bold_every == 0 else thin), width=1)
    return out

def assign_symbols(colors: List[Tuple[int,int,int]]) -> Dict[Tuple[int,int,int], str]:
    """Deterministic symbol per palette color."""
    glyphs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+*#@&%=?/\\^~<>□■●▲◆★✚")
    return {c: glyphs[i % len(glyphs)] for i, c in enumerate(colors)}

def draw_symbols_on_grid(base: Image.Image, cell_px: int, sym_map: Dict[Tuple[int,int,int], str]) -> Image.Image:
    """Overlay symbol per stitch, then grid."""
    sx, sy = base.size
    out = base.resize((sx*cell_px, sy*cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for y in range(sy):
        for x in range(sx):
            rgb = base.getpixel((x, y))
            sym = sym_map[rgb]
            fill = (0,0,0) if luminance(rgb) > 140 else (255,255,255)
            draw.text((x*cell_px + cell_px//2, y*cell_px + cell_px//2),
                      sym, font=font, fill=fill, anchor="mm")
    # grid overlay
    thin = (0, 0, 0, 70)
    bold = (0, 0, 0, 170)
    for x in range(sx + 1):
        draw.line([(x*cell_px, 0), (x*cell_px, sy*cell_px)],
                  fill=(bold if x % 10 == 0 else thin), width=1)
    for y in range(sy + 1):
        draw.line([(0, y*cell_px), (sx*cell_px, y*cell_px)],
                  fill=(bold if y % 10 == 0 else thin), width=1)
    return out

# ---------------------- stitch & knit math ----------------------
def skeins_per_color(stitches: int, cloth_count: int, strands: int, waste: float) -> float:
    """
    One DMC skein = 8 m * 6 strands = 48 m single-strand.
    Per-skein bundle length = 48/strands m.
    Bundle length per stitch ≈ 2*sqrt(2)*(2.54/count) cm; add waste fraction.
    """
    per_stitch_cm = 2.0 * math.sqrt(2.0) * (2.54 / float(cloth_count))
    per_stitch_cm *= (1.0 + waste)
    bundle_cm_per_skein = (800.0 * 6.0) / float(strands)
    return (stitches * per_stitch_cm) / bundle_cm_per_skein

def knit_aspect_resize(img: Image.Image, stitches_w: int, row_aspect: float = 0.8) -> Image.Image:
    """Knitting charts: cells look shorter than wide; compress rows for preview."""
    resized = resize_for_stitch_width(img, stitches_w)
    w, h = resized.size
    preview_h = max(1, int(round(h * row_aspect)))
    return resized.resize((w, preview_h), Image.Resampling.NEAREST)

# ---------------------- embroidery helpers ----------------------
def to_monochrome(img: Image.Image, threshold: int = 180) -> Image.Image:
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p > threshold else 0, mode="1")
    return bw.convert("L")

def serpentine_points(bw: Image.Image, step: int = 3) -> List[Tuple[int,int]]:
    """Naive run-stitch path by row scanning; good for simple outlines."""
    w, h = bw.size
    pts: List[Tuple[int,int]] = []
    data = bw.load()
    for y in range(0, h, step):
        xs = range(0, w, step) if (y // step) % 2 == 0 else range(w-1, -1, -step)
        row_pts = [(x, y) for x in xs if data[x, y] < 128]
        if row_pts:
            if pts and pts[-1] != row_pts[0]:
                pts.append(row_pts[0])  # jump
            pts.extend(row_pts)
    return pts

def write_embroidery_outputs(paths: List[Tuple[int,int]], scale: float = 1.0) -> Dict[str, bytes]:
    """Emit DST/PES if pyembroidery is available; always emit SVG polyline."""
    out: Dict[str, bytes] = {}
    if paths:
        svg_points = " ".join([f"{int(x*scale)},{int(y*scale)}" for x, y in paths])
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(paths[-1][0]*scale+10)} {int(paths[-1][1]*scale+10)}"><polyline fill="none" stroke="black" stroke-width="1" points="{svg_points}"/></svg>'
        out["embroidery.svg"] = svg.encode("utf-8")
    if HAS_PYEMB and paths:
        pat = EmbPattern()
        last: Optional[Tuple[int,int]] = None
        for (x, y) in paths:
            if last is None:
                pat.add_stitch_absolute(0, 0, 2)  # move
            pat.add_stitch_absolute(x, y)
            last = (x, y)
        pat.end()
        buf_dst = io.BytesIO(); write_dst(pat, buf_dst); out["pattern.dst"] = buf_dst.getvalue()
        buf_pes = io.BytesIO(); write_pes(pat, buf_pes); out["pattern.pes"] = buf_pes.getvalue()
    return out

# ---------------------- HTTP ----------------------
@app.get("/health")
def health() -> dict:
    return {"ok": True}

@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413

@app.errorhandler(Exception)
def on_error(_e):
    return make_response(jsonify({"error": "server_error"}), 500)

@app.get("/")
def index() -> str:
    return f"""<!doctype html><meta charset="utf-8">
<title>StitchBitch — Pattern Tools</title>
<body style="font-family:sans-serif;max-width:820px;margin:32px auto;line-height:1.45">
<h2>StitchBitch — Pattern Tools</h2>
<p>Upload an image and choose a pattern type. One free export/day. For machine files, install <code>pyembroidery</code> on the server or use the SVG fallback.</p>
<form method="POST" action="/api/convert" enctype="multipart/form-data">
  <p><input type="file" name="file" accept="image/*" required></p>

  <fieldset style="border:1px solid #ddd;padding:12px"><legend>Pattern Type</legend>
    <label><input type="radio" name="ptype" value="cross" checked> Cross-stitch</label>
    <label style="margin-left:16px"><input type="radio" name="ptype" value="knit"> Knitting (colorwork)</label>
    <label style="margin-left:16px"><input type="radio" name="ptype" value="emb"> Embroidery (run-stitch)</label>
  </fieldset>

  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px">
    <label>Width (stitches) <input type="number" name="width" value="120" min="20" max="400" required></label>
    <label>Max colors <input type="number" name="colors" value="16" min="2" max="60" required></label>
    <label>Cloth count (st/in) <input type="number" name="count" value="14" min="10" max="22" required></label>
    <label>Strands <input type="number" name="strands" value="2" min="1" max="6" required></label>
    <label>Waste % <input type="number" name="waste" value="20" min="0" max="60" required></label>
  </div>

  <p><label><input type="checkbox" name="symbols" checked> Add symbol chart overlay</label>
     <label style="margin-left:16px"><input type="checkbox" name="pdf" checked> Also export single-page PDF</label></p>

  <fieldset style="border:1px solid #ddd;padding:12px"><legend>Embroidery Options</legend>
    <p style="margin:0 0 8px 0;font-size:12px;opacity:.8">Simple run-stitch from image; for full digitizing use your toolchain or Ink/Stitch.</p>
    <label>Threshold <input type="number" name="emb_thresh" value="180" min="0" max="255"></label>
    <label style="margin-left:16px">Step px <input type="number" name="emb_step" value="3" min="1" max="10"></label>
  </fieldset>

  <button type="submit">Generate ZIP</button>
</form>
<p style="font-size:12px;opacity:.7">Cross/Knit outputs: grid.png, legend.csv, meta.json, optional pattern.pdf. Embroidery: pattern.dst/pattern.pes (if available) and embroidery.svg.</p>
</body>"""

@app.post("/api/convert")
def convert():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "missing_file"}), 400
    ctype = (file.mimetype or "").lower()
    if ctype not in ALLOWED_MIME:
        return jsonify({"error": "unsupported_type", "got": ctype}), 400

    try:
        ptype = request.form.get("ptype", "cross")
        stitch_w = clamp(int(request.form.get("width", 120)), 20, 400)
        max_colors = clamp(int(request.form.get("colors", 16)), 2, 60)
        cloth_count = clamp(int(request.form.get("count", 14)), 10, 22)
        strands = clamp(int(request.form.get("strands", 2)), 1, 6)
        waste_pct = clamp(int(request.form.get("waste", 20)), 0, 60)
        want_symbols = request.form.get("symbols") is not None
        want_pdf = request.form.get("pdf") is not None
        emb_thresh = clamp(int(request.form.get("emb_thresh", 180)), 0, 255)
        emb_step = clamp(int(request.form.get("emb_step", 3)), 1, 10)
    except Exception:
        return jsonify({"error": "invalid_parameters"}), 400

    try:
        base = open_image(file)
    except Exception:
        return jsonify({"error": "decode_failed"}), 400
    if max(base.size) > MAX_DIM:
        return jsonify({"error": "image_too_large", "max_dim": MAX_DIM}), 400

    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        if ptype in ("cross", "knit"):
            small = resize_for_stitch_width(base, stitch_w) if ptype == "cross" else knit_aspect_resize(base, stitch_w, 0.8)
            quant = quantize(small, max_colors)
            counts = palette_counts(quant)
            sx, sy = quant.size

            finished_w_in = round(sx / float(cloth_count), 2)
            finished_h_in = round(sy / float(cloth_count), 2)

            grid_img = draw_grid(quant, cell_px=CELL_PX)
            pdf_bytes: Optional[bytes] = None
            if want_symbols or want_pdf:
                pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
                sym_map = assign_symbols(pal)
                sym_img = draw_symbols_on_grid(quant, cell_px=CELL_PX, sym_map=sym_map)
                if want_pdf:
                    buf = io.BytesIO()
                    sym_img.convert("RGB").save(buf, format="PDF", resolution=300.0)
                    pdf_bytes = buf.getvalue()
                grid_img = sym_img

            # legend.csv
            total = sum(counts.values()) or 1
            lines = ["hex,r,g,b,stitches,percent,skeins_est"]
            for (r,g,b), c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                skeins = skeins_per_color(c, cloth_count, strands, waste_pct/100.0)
                lines.append(f"{to_hex((r,g,b))},{r},{g},{b},{c},{(100*c/total):.2f},{skeins:.2f}")
            z.writestr("legend.csv", ("\n".join(lines)).encode("utf-8"))

            meta = {
                "type": ptype,
                "stitches_w": sx, "stitches_h": sy, "colors": len(counts),
                "cloth_count": cloth_count, "strands": strands, "waste_percent": waste_pct,
                "finished_size_in": [finished_w_in, finished_h_in],
                "notes": "Knitting preview compresses row height visually; verify gauge."
            }
            z.writestr("meta.json", json.dumps(meta, indent=2))

            buf_png = io.BytesIO()
            grid_img.save(buf_png, format="PNG")
            z.writestr("grid.png", buf_png.getvalue())
            if pdf_bytes:
                z.writestr("pattern.pdf", pdf_bytes)

        elif ptype == "emb":
            small = resize_for_stitch_width(base, stitch_w)
            bw = to_monochrome(small, threshold=emb_thresh)
            pts = serpentine_points(bw, step=emb_step)
            for name, data in write_embroidery_outputs(pts, scale=1.0).items():
                z.writestr(name, data)
            z.writestr("meta.json", json.dumps({"type": "emb", "points": len(pts), "pyembroidery": HAS_PYEMB}, indent=2))
        else:
            return jsonify({"error": "unknown_ptype"}), 400

    out_zip.seek(0)
    fname = f"pattern_{request.form.get('ptype','cross')}.zip"
    return send_file(out_zip, mimetype="application/zip", as_attachment=True, download_name=fname)

if __name__ == "__main__":
    # For local dev; on Render use: gunicorn app:app --bind 0.0.0.0:$PORT
    app.run(host="127.0.0.1", port=5050)
