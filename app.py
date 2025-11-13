from __future__ import annotations
import io
import json
import math
import os
import zipfile
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from flask import (
    Flask,
    request,
    send_file,
    jsonify,
    make_response,
    render_template_string,
    redirect,
    url_for,
    session,
)
from PIL import Image, ImageDraw, ImageFont

# Optional embroidery support
try:
    from pyembroidery import EmbPattern, write_dst, write_pes  # type: ignore
    HAS_PYEMB = True
except Exception:
    HAS_PYEMB = False

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ---------------------- CONFIG ----------------------
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
BOLD_EVERY = 10
MAX_DIM = 8000  # max width/height in pixels
FREE_EXPORTS_PER_DAY = 1  # free tier limit


# ---------------------- UTILITIES ----------------------
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
    """Resize while preserving aspect ratio to target stitch width."""
    w, h = img.size
    if max(w, h) > 2000:
        img = img.copy()
        img.thumbnail((2000, 2000))
        w, h = img.size
    ratio = stitch_w / float(w)
    new_h = max(1, int(round(h * ratio)))
    return img.resize((stitch_w, new_h), Image.Resampling.LANCZOS)


def quantize(img: Image.Image, k: int) -> Image.Image:
    """Median-cut palette, no dithering for crisp cells."""
    return img.convert(
        "P", palette=Image.Palette.ADAPTIVE, colors=k, dither=Image.Dither.NONE
    ).convert("RGB")


def palette_counts(img: Image.Image) -> Dict[Tuple[int, int, int], int]:
    counts: Dict[Tuple[int, int, int], int] = {}
    for rgb in img.getdata():
        counts[rgb] = counts.get(rgb, 0) + 1
    return counts


def to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def draw_grid(base: Image.Image, cell_px: int, bold_every: int = BOLD_EVERY) -> Image.Image:
    """Scale each stitch to a cell and overlay a grid (bold every N)."""
    sx, sy = base.size
    out = base.resize((sx * cell_px, sy * cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    thin = (0, 0, 0, 70)
    bold = (0, 0, 0, 170)
    for x in range(sx + 1):
        draw.line(
            [(x * cell_px, 0), (x * cell_px, sy * cell_px)],
            fill=(bold if x % bold_every == 0 else thin),
            width=1,
        )
    for y in range(sy + 1):
        draw.line(
            [(0, y * cell_px), (sx * cell_px, y * cell_px)],
            fill=(bold if y % bold_every == 0 else thin),
            width=1,
        )
    return out


def assign_symbols(colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], str]:
    """Deterministic symbol per palette color."""
    glyphs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+*#@&%=?/\\^~<>□■●▲◆★✚")
    return {c: glyphs[i % len(glyphs)] for i, c in enumerate(colors)}


def draw_symbols_on_grid(
    base: Image.Image, cell_px: int, sym_map: Dict[Tuple[int, int, int], str]
) -> Image.Image:
    """Overlay symbol per stitch, then grid."""
    sx, sy = base.size
    out = base.resize((sx * cell_px, sy * cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for y in range(sy):
        for x in range(sx):
            rgb = base.getpixel((x, y))
            sym = sym_map[rgb]
            fill = (0, 0, 0) if luminance(rgb) > 140 else (255, 255, 255)
            draw.text(
                (x * cell_px + cell_px // 2, y * cell_px + cell_px // 2),
                sym,
                font=font,
                fill=fill,
                anchor="mm",
            )
    thin = (0, 0, 0, 70)
    bold = (0, 0, 0, 170)
    for x in range(sx + 1):
        draw.line(
            [(x * cell_px, 0), (x * cell_px, sy * cell_px)],
            fill=(bold if x % 10 == 0 else thin),
            width=1,
        )
    for y in range(sy + 1):
        draw.line(
            [(0, y * cell_px), (sx * cell_px, y * cell_px)],
            fill=(bold if y % 10 == 0 else thin),
            width=1,
        )
    return out


def skeins_per_color(stitches: int, cloth_count: int, strands: int, waste: float) -> float:
    per_stitch_cm = 2 * math.sqrt(2) * (2.54 / cloth_count) * (1 + waste)
    skein_cm = (800 * 6) / strands
    return (stitches * per_stitch_cm) / skein_cm


def knit_aspect_resize(img: Image.Image, stitches_w: int, row_aspect: float = 0.8) -> Image.Image:
    """Knitting charts: visually shorter rows."""
    resized = resize_for_stitch_width(img, stitches_w)
    w, h = resized.size
    preview_h = max(1, int(round(h * row_aspect)))
    return resized.resize((w, preview_h), Image.Resampling.NEAREST)


def to_monochrome(img: Image.Image, threshold: int = 180) -> Image.Image:
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p > threshold else 0, mode="1")
    return bw.convert("L")


def serpentine_points(bw: Image.Image, step: int = 3) -> List[Tuple[int, int]]:
    """Naive run-stitch path by row scanning."""
    w, h = bw.size
    pts: List[Tuple[int, int]] = []
    data = bw.load()
    for y in range(0, h, step):
        xs = range(0, w, step) if (y // step) % 2 == 0 else range(w - 1, -1, -step)
        row_pts = [(x, y) for x in xs if data[x, y] < 128]
        if row_pts:
            if pts and pts[-1] != row_pts[0]:
                pts.append(row_pts[0])
            pts.extend(row_pts)
    return pts


def write_embroidery_outputs(paths: List[Tuple[int, int]], scale: float = 1.0) -> Dict[str, bytes]:
    """Emit DST/PES if pyembroidery is available; always emit SVG polyline."""
    out: Dict[str, bytes] = {}
    if paths:
        svg_points = " ".join([f"{int(x * scale)},{int(y * scale)}" for x, y in paths])
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
            f'{int(paths[-1][0] * scale + 10)} {int(paths[-1][1] * scale + 10)}">'
            f'<polyline fill="none" stroke="black" stroke-width="1" points="{svg_points}"/></svg>'
        )
        out["embroidery.svg"] = svg.encode("utf-8")
    if HAS_PYEMB and paths:
        pat = EmbPattern()
        last: Optional[Tuple[int, int]] = None
        for (x, y) in paths:
            if last is None:
                pat.add_stitch_absolute(0, 0, 2)  # move
            pat.add_stitch_absolute(x, y)
            last = (x, y)
        pat.end()
        buf_dst = io.BytesIO()
        write_dst(pat, buf_dst)
        out["pattern.dst"] = buf_dst.getvalue()
        buf_pes = io.BytesIO()
        write_pes(pat, buf_pes)
        out["pattern.pes"] = buf_pes.getvalue()
    return out


# ---------------------- PAYWALL HELPERS ----------------------
def today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def is_pro() -> bool:
    return bool(session.get("is_pro", False))


def can_use_free_export() -> bool:
    if is_pro():
        return True
    today = today_str()
    last = session.get("free_day")
    used = int(session.get("free_used", 0))
    if last != today:
        session["free_day"] = today
        session["free_used"] = 0
        return True
    return used < FREE_EXPORTS_PER_DAY


def register_export() -> None:
    if is_pro():
        return
    today = today_str()
    last = session.get("free_day")
    used = int(session.get("free_used", 0))
    if last != today:
        session["free_day"] = today
        session["free_used"] = 1
    else:
        session["free_used"] = used + 1


# ---------------------- ROUTES ----------------------
@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/debug/pro/on")
def debug_pro_on():
    session["is_pro"] = True
    return "Pro mode ON."


@app.get("/debug/pro/off")
def debug_pro_off():
    session["is_pro"] = False
    return "Pro mode OFF."


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


@app.errorhandler(Exception)
def on_error(_e):
    return make_response(jsonify({"error": "server_error"}), 500)


@app.get("/")
def index() -> str:
    return render_template_string(HOMEPAGE_HTML)


@app.get("/pricing")
def pricing() -> str:
    return render_template_string(PRICING_HTML)


@app.post("/api/convert")
def convert():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "missing_file"}), 400
    if (file.mimetype or "").lower() not in ALLOWED_MIME:
        return jsonify({"error": "unsupported_type"}), 400

    try:
        ptype = request.form.get("ptype", "cross")
        stitch_style = request.form.get("stitch_style", "full")
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
        return jsonify({"error": "image_too_large"}), 400

    # paywall: 1 free export per day
    if not can_use_free_export():
        # add ?reason=limit so pricing page can show a popup
        return redirect(url_for("pricing", reason="limit"))

    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        if ptype in ("cross", "knit"):
            small = (
                resize_for_stitch_width(base, stitch_w)
                if ptype == "cross"
                else knit_aspect_resize(base, stitch_w)
            )
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
                    pdf_buf = io.BytesIO()
                    sym_img.convert("RGB").save(pdf_buf, format="PDF", resolution=300.0)
                    pdf_bytes = pdf_buf.getvalue()
                grid_img = sym_img

            total_stitches = sum(counts.values()) or 1
            lines = ["hex,r,g,b,stitches,percent,skeins_est"]
            for (r, g, b), c in sorted(
                counts.items(), key=lambda kv: kv[1], reverse=True
            ):
                skeins = skeins_per_color(
                    c, cloth_count, strands, waste_pct / 100.0
                )
                lines.append(
                    f"{to_hex((r,g,b))},{r},{g},{b},{c},{(100*c/total_stitches):.2f},{skeins:.2f}"
                )
            z.writestr("legend.csv", "\n".join(lines))

            note = (
                "Knitting preview compresses row height; verify gauge."
                if ptype == "knit"
                else "Cross-stitch grid with 10×10 guides."
            )
            meta = {
                "type": ptype,
                "stitch_style": stitch_style,
                "stitches_w": sx,
                "stitches_h": sy,
                "colors": len(counts),
                "cloth_count": cloth_count,
                "strands": strands,
                "waste_percent": waste_pct,
                "finished_size_in": [finished_w_in, finished_h_in],
                "notes": note,
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
            for name, data in write_embroidery_outputs(pts).items():
                z.writestr(name, data)
            z.writestr(
                "meta.json",
                json.dumps(
                    {
                        "type": "emb",
                        "stitch_style": "run",
                        "points": len(pts),
                        "pyembroidery": HAS_PYEMB,
                    },
                    indent=2,
                ),
            )
        else:
            return jsonify({"error": "unknown_ptype"}), 400

    out_zip.seek(0)
    register_export()
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{ptype}.zip",
    )


# ---------------------- INLINE HTML ----------------------
HOMEPAGE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PatternCraft — Turn art into patterns</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{
  --bg:#F7F4EF;--fg:#222;--muted:#6b6b6b;
  --line:#e8e4de;--radius:14px;--shadow:0 6px 20px rgba(0,0,0,.08);
  --accent:#4C7CF3;--accent-soft:#e3ebff;
}
body{margin:0;background:linear-gradient(135deg,#f9f6ef,#f1f4ff);color:var(--fg);
     font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter}
.wrap{max-width:980px;margin:0 auto;padding:24px 16px 40px}
h1{font-size:2.3rem;margin:0 0 6px}
h2{margin:0 0 10px}
.card{background:#fff;border-radius:var(--radius);
      border:1px solid var(--line);box-shadow:var(--shadow);padding:18px}
.hero{display:flex;flex-wrap:wrap;gap:20px;margin-bottom:24px;align-items:center}
.hero-text{flex:1 1 280px}
.muted{color:var(--muted);font-size:13px}
.pill{padding:9px 16px;border-radius:999px;background:var(--accent);color:#fff;
      border:none;cursor:pointer;font-size:14px}
.pill-secondary{background:#fff;color:var(--fg);border:1px solid var(--line)}
.features{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));margin-bottom:24px}
.feature-title{font-weight:600;margin-bottom:4px}
.make-layout{display:flex;gap:20px;flex-wrap:wrap}
.make-main{flex:1 1 260px}
.make-sample{flex:1 1 220px;display:flex;flex-direction:column;gap:8px}
.sample-box{border-radius:12px;border:1px dashed var(--line);
            background:repeating-linear-gradient(
               90deg, #f4f2ec 0, #f4f2ec 9px, #e0ddd6 10px),
                       repeating-linear-gradient(
               180deg, #f4f2ec 0, #f4f2ec 9px, #e0ddd6 10px);
            background-blend-mode:multiply;
            aspect-ratio:4/3;position:relative;overflow:hidden}
.sample-overlay{position:absolute;inset:10%;display:grid;place-items:center;
                font-size:11px;color:#333;background:rgba(255,255,255,0.6)}
.file{border:2px dashed var(--accent);border-radius:18px;
      padding:18px;display:flex;align-items:center;gap:12px;cursor:pointer;
      background:var(--accent-soft);transition:background .15s,border-color .15s,transform .1s}
.file:hover{background:#d9e3ff;border-color:#365ed1;transform:translateY(-1px)}
.file input{display:none}
.file-label-main{font-weight:600;font-size:15px}
.file-label-sub{font-size:12px;color:var(--muted)}
.free-note{margin-top:6px;font-size:12px;color:#1c5c2f;background:#e1f4e5;
          border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px}
.free-dot{width:8px;height:8px;border-radius:999px;background:#1c5c2f}
fieldset{border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0}
legend{font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:12px}
.row > label{flex:1 1 150px}
.hidden{display:none}
@media (max-width:720px){
  .hero{flex-direction:column}
}
</style>
</head>
<body>
<div class="wrap">

<div class="hero">
  <div class="hero-text">
    <h1>Turn art into stitchable patterns</h1>
    <p class="muted">
      PatternCraft converts your artwork into cross-stitch grids, knitting charts,
      and embroidery-ready files with one upload.
    </p>
    <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap">
      <button class="pill" onclick="document.getElementById('make').scrollIntoView({behavior:'smooth'})">
        Upload art
      </button>
      <button class="pill pill-secondary" onclick="location.href='#how'">
        See what you get
      </button>
    </div>
  </div>
  <div class="card" style="flex:1 1 280px">
    <h2 style="margin-top:0;font-size:1.1rem">Why PatternCraft</h2>
    <ul class="muted" style="padding-left:18px">
      <li>Clean grids with bold 10×10 guides</li>
      <li>Floss estimates per color for planning</li>
      <li>Knitting charts that respect row proportions</li>
      <li>Embroidery outputs ready for your workflow</li>
    </ul>
  </div>
</div>

<div id="how" class="features">
  <div class="card">
    <div class="feature-title">Cross-stitch</div>
    <p class="muted">Upload art, pick size and colors, get a crisp grid with symbols and rough skein estimates.</p>
  </div>
  <div class="card">
    <div class="feature-title">Knitting charts</div>
    <p class="muted">Generate colorwork charts with realistic row aspect and multiple stitch styles.</p>
  </div>
  <div class="card">
    <div class="feature-title">Embroidery</div>
    <p class="muted">Create run-stitch paths and export embroidery-friendly files or SVG for fine-tuning.</p>
  </div>
</div>

<div id="make" class="card">
  <h2 style="margin-top:0">Make a pattern</h2>
  <div class="make-layout">
    <div class="make-main">
      <form method="POST" action="/api/convert" enctype="multipart/form-data">
        <label class="file">
          <input type="file" name="file" accept="image/*" required onchange="pickFile(this)">
          <div>
            <div class="file-label-main">Upload picture here</div>
            <div class="file-label-sub">Drop in your artwork or click to browse from your device.</div>
          </div>
        </label>
        <div class="free-note">
          <div class="free-dot"></div>
          <span>You have a free pattern conversion available — upload art!</span>
        </div>

        <fieldset>
          <legend>Pattern type</legend>
          <label><input type="radio" name="ptype" value="cross" checked> Cross-stitch</label>
          <label><input type="radio" name="ptype" value="knit"> Knitting</label>
          <label><input type="radio" name="ptype" value="emb"> Embroidery</label>
        </fieldset>

        <fieldset>
          <legend>Stitch + size</legend>
          <div class="row">
            <label>Stitch width
              <input type="number" name="width" value="120" min="20" max="400">
            </label>
            <label>Max colors
              <input type="number" name="colors" value="16" min="2" max="60">
            </label>
            <label>Stitch style
              <select id="stitch_style" name="stitch_style"></select>
            </label>
          </div>
        </fieldset>

        <fieldset id="crossKnitBlock">
          <legend>Fabric + floss</legend>
          <div class="row">
            <label>Cloth count (st/in)
              <input type="number" name="count" value="14" min="10" max="22">
            </label>
            <label>Strands
              <input type="number" name="strands" value="2" min="1" max="6">
            </label>
            <label>Waste %
              <input type="number" name="waste" value="20" min="0" max="60">
            </label>
          </div>
          <label><input type="checkbox" name="symbols" checked> Symbol overlay</label>
          <label><input type="checkbox" name="pdf" checked> Also export PDF</label>
        </fieldset>

        <fieldset id="embBlock" class="hidden">
          <legend>Embroidery options</legend>
          <p class="muted">Simple run-stitch path from your image. For advanced digitizing, continue in your usual embroidery software.</p>
          <div class="row">
            <label>Threshold
              <input type="number" name="emb_thresh" value="180" min="0" max="255">
            </label>
            <label>Step px
              <input type="number" name="emb_step" value="3" min="1" max="10">
            </label>
          </div>
        </fieldset>

        <div style="margin-top:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
          <button class="pill" type="submit">Generate ZIP</button>
          <span class="muted">We’ll bundle the grid, legend, meta, and optional PDF into one download.</span>
        </div>
      </form>
    </div>

    <div class="make-sample">
      <div class="muted" style="font-weight:600;">Sample output preview</div>
      <div class="sample-box">
        <div class="sample-overlay">
          <div>
            <div style="font-weight:600;margin-bottom:4px;">Cross-stitch grid</div>
            <div>10×10 guides, symbol overlay, printable PDF feel.</div>
          </div>
        </div>
      </div>
      <div class="muted" style="font-size:12px;">
        Actual downloads include <strong>grid.png</strong>, <strong>legend.csv</strong>, <strong>meta.json</strong>,
        and optional <strong>pattern.pdf</strong> or embroidery files.
      </div>
    </div>
  </div>
</div>

</div>
<script>
function pickFile(inp){
  document.getElementById('fname')?.textContent;
}

function setStyleOptions(type){
  const sel = document.getElementById('stitch_style');
  sel.innerHTML = '';
  let opts = [];
  if(type === 'cross'){
    opts = [
      ['full','Full stitches'],
      ['half','Half stitches'],
      ['back','Backstitch overlay']
    ];
  } else if(type === 'knit'){
    opts = [
      ['stockinette','Stockinette'],
      ['garter','Garter'],
      ['seed','Seed'],
      ['rib1','Rib 1×1']
    ];
  } else {
    opts = [['run','Run stitch']];
  }
  for(const [val,label] of opts){
    const o = document.createElement('option');
    o.value = val; o.textContent = label;
    sel.appendChild(o);
  }
}
function onTypeChange(){
  const type = document.querySelector('input[name="ptype"]:checked').value;
  const crossKnit = document.getElementById('crossKnitBlock');
  const emb = document.getElementById('embBlock');
  if(type === 'emb'){
    crossKnit.classList.add('hidden');
    emb.classList.remove('hidden');
  } else {
    crossKnit.classList.remove('hidden');
    emb.classList.add('hidden');
  }
  setStyleOptions(type);
}
document.querySelectorAll('input[name="ptype"]').forEach(r => {
  r.addEventListener('change', onTypeChange);
});
onTypeChange();
</script>
</body>
</html>
"""

PRICING_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PatternCraft — Pricing</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:0;background:#F7F4EF;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter}
.wrap{max-width:780px;margin:0 auto;padding:32px 16px}
h1{margin-top:0}
.plans{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.card{background:#fff;border-radius:14px;border:1px solid #e8e4de;padding:18px;box-shadow:0 6px 20px rgba(0,0,0,.06)}
.price{font-size:1.6rem;font-weight:600}
.muted{color:#6b6b6b;font-size:13px}
.btn{display:inline-block;margin-top:10px;padding:8px 14px;border-radius:999px;
     background:#222;color:#fff;text-decoration:none;font-size:14px}
</style>
</head>
<body>
<div class="wrap">
  <h1>PatternCraft pricing</h1>
  <p class="muted">Start free. Upgrade only when you’re actually using it.</p>
  <div class="plans">
    <div class="card">
      <div class="price">Free</div>
      <ul class="muted">
        <li>1 export / day</li>
        <li>Cross-stitch, knitting, embroidery</li>
        <li>No card required</li>
      </ul>
      <a href="/" class="btn">Keep using free</a>
    </div>
    <div class="card">
      <div class="price">$9 / month</div>
      <ul class="muted">
        <li>Unlimited exports</li>
        <li>Priority processing</li>
        <li>Supports new features</li>
      </ul>
      <!-- Replace with your Stripe Payment Link -->
      <a href="https://buy.stripe.com/your-monthly-link" class="btn">Subscribe</a>
    </div>
    <div class="card">
      <div class="price">$3 / 50 exports</div>
      <ul class="muted">
        <li>Pay as you go</li>
        <li>Good for occasional projects</li>
        <li>No subscription</li>
      </ul>
      <!-- Replace with your Stripe Payment Link -->
      <a href="https://buy.stripe.com/your-credits-link" class="btn">Buy credits</a>
    </div>
  </div>
</div>

<script>
  (function () {
    const params = new URLSearchParams(window.location.search);
    if (params.get('reason') === 'limit') {
      alert('You used your free PatternCraft conversion for today. Upgrade to keep generating patterns.');
    }
  }());
</script>

</body>
</html>
"""

if __name__ == "__main__":
    # local dev; on Render use gunicorn app:app --bind 0.0.0.0:$PORT
    app.run(host="127.0.0.1", port=5050, debug=True)
