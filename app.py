from __future__ import annotations
import base64, io, zipfile, colorsys, math, json
from typing import Dict, List, Tuple, Iterable

from flask import Flask, request, render_template_string, jsonify, send_file, abort
from PIL import Image, ImageDraw, ImageOps

app = Flask(__name__)

# ----------------- Marketing + SEO -----------------
SITE_TITLE = "StitchBitch • Free Stitch Pattern Maker"
SITE_DESC = "Turn any image into a clean stitch pattern. Free tool with grid, numbering, PDF, palette, and fabric size calculator."
SITE_URL  = "http://127.0.0.1:5000"  # replace in prod

# ----------------- Image helpers -----------------

def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def resize_image(img: Image.Image, width: int) -> Image.Image:
    w, h = img.size
    width = clamp_int(width, 20, 1200)
    if width <= 0 or width == w:
        return img
    new_h = max(1, int(h * (width / float(w))))
    return img.resize((width, new_h), Image.Resampling.LANCZOS)

def quantize_image(img: Image.Image, colors: int, dither: bool) -> Image.Image:
    colors = clamp_int(colors, 2, 64)
    d = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors, dither=d).convert("RGB")

def draw_grid_numbered(img: Image.Image, grid_px: int, bold_every: int = 10, show_numbers: bool = True) -> Image.Image:
    """Nearest-scale pixel art with grid and 10x10 numbering."""
    w, h = img.size
    cell = clamp_int(grid_px, 4, 40)
    big = img.resize((w * cell, h * cell), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(big)

    # Grid
    for x in range(w + 1):
        thick = 2 if (x % bold_every == 0) else 1
        alpha = 180 if (x % bold_every == 0) else 70
        c = (0, 0, 0, alpha)
        for t in range(thick):
            draw.line([(x * cell + t, 0), (x * cell + t, h * cell)], fill=c)
    for y in range(h + 1):
        thick = 2 if (y % bold_every == 0) else 1
        alpha = 180 if (y % bold_every == 0) else 70
        c = (0, 0, 0, alpha)
        for t in range(thick):
            draw.line([(0, y * cell + t), (w * cell, y * cell + t)], fill=c)

    # 10x10 numbering along top and left
    if show_numbers:
        for x in range(bold_every, w + 1, bold_every):
            label = str(x)
            tw, th = draw.textlength(label), 10  # approx text height
            draw.text((x * cell + 2, 2), label, fill=(0, 0, 0))
        for y in range(bold_every, h + 1, bold_every):
            draw.text((2, y * cell + 2), str(y), fill=(0, 0, 0))
    return big

def palette_counts(img: Image.Image) -> Dict[Tuple[int,int,int], int]:
    counts: Dict[Tuple[int,int,int], int] = {}
    for rgb in _to_rgb(img).getdata():
        counts[rgb] = counts.get(rgb, 0) + 1
    return counts

def rgb_to_hex(rgb: Tuple[int,int,int]) -> str:
    return "#%02x%02x%02x" % rgb

def sorted_palette(pc: Dict[Tuple[int,int,int], int]) -> List[Tuple[Tuple[int,int,int], int]]:
    def key(kv):
        (r,g,b), n = kv
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        return (-n, round(h,3), round(s,3), round(v,3))
    return sorted(pc.items(), key=key)

def legend_strip(pairs: List[Tuple[Tuple[int,int,int], int]], width: int) -> Image.Image:
    if not pairs:
        return Image.new("RGB", (max(1,width), 24), "white")
    w = max(120, width)
    bar_h = 22
    img = Image.new("RGB", (w, bar_h), "white")
    draw = ImageDraw.Draw(img)
    total = sum(n for _, n in pairs)
    x = 0
    for rgb, n in pairs:
        seg = max(1, int(w * (n / total)))
        draw.rectangle([(x, 0), (min(w-1, x+seg), bar_h-1)], fill=rgb)
        x += seg
    return img

def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode("ascii")

# ----------------- Fabric size calculator -----------------

def fabric_sizes(stitches_w: int, stitches_h: int, counts: Iterable[int]=(11,14,16,18), margin_in: float=2.0):
    """
    Return list of dicts: count, finished_w_in, finished_h_in, cut_w_in, cut_h_in (with margins).
    """
    out = []
    for c in counts:
        fw = stitches_w / c
        fh = stitches_h / c
        cw = fw + margin_in*2
        ch = fh + margin_in*2
        out.append({
            "count": c,
            "finished_in": [round(fw,2), round(fh,2)],
            "finished_cm": [round(fw*2.54,1), round(fh*2.54,1)],
            "cut_in": [round(cw,2), round(ch,2)],
        })
    return out

# ----------------- Light DMC approximation (small set) -----------------
# Approximate hex for common DMC shades (subset). Not official. For reference only.
_DMC = [
    ("BLANC","White","#ffffff"),
    ("310","Black","#000000"),
    ("ECRU","Ecru","#f2e2ce"),
    ("321","Red","#c32026"),
    ("666","Bright Red","#d10f1b"),
    ("498","Dark Red","#8e1d2c"),
    ("352","Coral Lt","#f7a195"),
    ("3712","Salmon Md","#ea8d8b"),
    ("3713","Salmon Vy Lt","#ffd3cf"),
    ("720","Orange Spice","#c6521c"),
    ("741","Tangerine Md","#ffa31a"),
    ("742","Tangerine Lt","#ffc03b"),
    ("743","Yellow Md","#ffd35e"),
    ("745","Yellow Pale","#fff2b3"),
    ("744","Yellow Pale Lt","#fff7cc"),
    ("727","Topaz Vy Lt","#fff0a3"),
    ("743","Yellow Md","#ffd35e"),
    ("738","Tan Vy Lt","#e9c9a2"),
    ("739","Tan Ult Vy Lt","#f3dcc2"),
    ("3826","Golden Brown","#a3692b"),
    ("3827","Pale Golden Brown","#f3b774"),
    ("3852","Straw Dk","#c9972b"),
    ("3822","Straw Lt","#f2d277"),
    ("334","Baby Blue Md","#7299c6"),
    ("3325","Baby Blue Lt","#b9d6f2"),
    ("3755","Baby Blue Vy Lt","#cde3f6"),
    ("3761","Sky Blue Lt","#cfe6ef"),
    ("798","Delft Blue Dk","#2f4e8b"),
    ("820","Royal Blue Vy Dk","#1a2a6b"),
    ("939","Navy Blue Vy Dk","#111b3a"),
    ("995","Electric Blue Dk","#0aa3d9"),
    ("996","Electric Blue Lt","#35c6ff"),
    ("909","Emerald Green Vy Dk","#0b6a4a"),
    ("912","Emerald Green Lt","#3fbf9a"),
    ("703","Chartreuse","#7ac70c"),
    ("704","Chartreuse Bright","#9be31b"),
    ("907","Parrot Green Lt","#b4d92a"),
    ("500","Blue Green Vy Dk","#0f3b33"),
    ("522","Fern Green Lt","#afb79a"),
    ("844","Beaver Gray Ult Dk","#5a5a5a"),
    ("3799","Pewter Gray Vy Dk","#363636"),
    ("415","Pearl Gray","#c7c7c7"),
    ("762","Pearl Gray Vy Lt","#eeeef0"),
    ("434","Brown Lt","#b37d43"),
    ("801","Coffee Brown Dk","#6b4229"),
    ("898","Coffee Brown Vy Dk","#4f2d1a"),
    ("3371","Black Brown","#2b1a10"),
    ("950","Desert Sand Lt","#f2c7a5"),
    ("948","Peach Vy Lt","#ffe0cf"),
    ("754","Peach Lt","#f6c9b4"),
    ("3824","Apricot Lt","#f7b4a4"),
    ("550","Violet Vy Dk","#4b2166")
]

def _hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))  # type: ignore

DMC_PALETTE: List[Tuple[str,str,Tuple[int,int,int]]] = [(code,name,_hex_to_rgb(hx)) for code,name,hx in _DMC]

def nearest_dmc(rgb: Tuple[int,int,int]) -> Tuple[str,str,str,int]:
    """Return (code, name, hex, distance)."""
    r,g,b = rgb
    best = None
    best_d = 1e9
    for code,name,drgb in DMC_PALETTE:
        d = (r-drgb[0])**2 + (g-drgb[1])**2 + (b-drgb[2])**2
        if d < best_d:
            best_d = d
            best = (code, name, rgb_to_hex(drgb), int(math.sqrt(d)))
    assert best is not None
    return best  # type: ignore

# ----------------- Inline UI -----------------

INDEX_HTML = """
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<meta name="description" content="{{ desc }}">
<meta property="og:title" content="{{ title }}">
<meta property="og:description" content="{{ desc }}">
<meta property="og:type" content="website">
<meta property="og:url" content="{{ url }}">
<script type="application/ld+json">
{{ jsonld }}
</script>
<style>
:root{ --fg:#111; --muted:#666; --accent:#e4006d; --line:#eee; --card:#fafafa; --radius:14px; --wrap:1100px; }
*{box-sizing:border-box} html{scroll-behavior:smooth} html,body{margin:0;padding:0}
body{font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--fg);background:#fff}
.wrap{max-width:var(--wrap);margin:0 auto;padding:20px}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);z-index:5}
.brand{font-weight:800;letter-spacing:.2px}
.hero h1{font-size:40px;margin:12px 0 4px;letter-spacing:-.3px}
.badge{display:inline-block;background:#ffe6f3;color:#bb0055;border:1px solid #ffd6ec;border-radius:999px;padding:4px 10px;font-weight:700;margin-bottom:8px}
.sub{color:var(--muted);margin:0 0 16px}
.grid{display:grid;gap:16px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:16px;transition:transform .35s ease, font-size .35s ease}
label{display:block;font-weight:600;margin-bottom:6px}
input[type=file],input[type=number],input[type=checkbox],button,select{font:inherit}
input[type=number]{width:120px;padding:8px;border:1px solid var(--line);border-radius:8px}
.row{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.btn{display:inline-block;padding:10px 16px;border-radius:999px;border:1px solid var(--accent);background:var(--accent);color:#fff;text-decoration:none;font-weight:700;cursor:pointer}
.btn.ghost{background:transparent;color:var(--accent)}
.preview{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}
.preview img{width:100%;height:auto;border-radius:12px;border:1px solid var(--line);background:#fff}
.palette{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.swatch{display:flex;align-items:center;gap:8px;padding:8px;border:1px solid var(--line);border-radius:8px;background:#fff}
.swatch .box{width:20px;height:20px;border-radius:4px;border:1px solid #ddd}
.small{font-size:14px;color:var(--muted)}
footer{border-top:1px solid var(--line);margin-top:16px}
@media (max-width:900px){ .preview{grid-template-columns:1fr} .controls{grid-template-columns:1fr}}
/* Emphasis on HOW section while in view */
#how .card h3{transition:font-size .35s ease}
#how .card ol, #how .card p{transition:font-size .35s ease}
body.how-active #how .card{transform:scale(1.04)}
body.how-active #how .card h3{font-size:1.6rem}
body.how-active #how .card ol, body.how-active #how .card p{font-size:1.05rem}
</style>

<header>
  <div class="wrap row" style="justify-content:space-between">
    <div class="brand">StitchBitch.club</div>
    <nav class="row">
      <a class="btn ghost" id="howLink" href="#how">How it works</a>
      <a class="btn" href="#tool">Open the tool</a>
    </nav>
  </div>
</header>

<section class="wrap hero">
  <div class="badge">Free tool — no sign‑up</div>
  <h1>Turn any image into a stitch‑ready pattern.</h1>
  <p class="sub">Grid + numbering, palette, fabric sizes, and a printable PDF. Always free.</p>
  <div class="row">
    <a class="btn" href="#tool">Start free</a>
    <button class="btn ghost" id="shareBtn" title="Share this tool">Share</button>
  </div>
</section>

<section id="tool" class="wrap grid">
  <div class="card controls">
    <div>
      <label>Image</label>
      <input id="file" type="file" accept="image/*" />
      <p class="small">Tip: width 80–160 and 8–16 colors often look best.</p>
    </div>
    <div class="row">
      <div><label>Width (px)</label><input id="width" type="number" min="20" max="1200" value="140"></div>
      <div><label>Colors</label><input id="colors" type="number" min="2" max="64" value="12"></div>
      <div><label>Grid cell (px)</label><input id="grid_px" type="number" min="4" max="40" value="10"></div>
      <div>
        <label>&nbsp;</label>
        <label class="row"><input id="dither" type="checkbox"> Dither</label>
        <label class="row"><input id="grid_on" type="checkbox" checked> Show grid</label>
        <label class="row"><input id="numbers_on" type="checkbox" checked> Numbers</label>
        <label class="row"><input id="use_dmc" type="checkbox"> DMC approx</label>
      </div>
      <div class="row" style="margin-top:8px">
        <button class="btn" id="previewBtn">Preview</button>
        <button class="btn ghost" id="downloadZipBtn" disabled>Download ZIP</button>
        <button class="btn ghost" id="downloadPdfBtn" disabled>Download PDF</button>
      </div>
    </div>
  </div>

  <div class="preview">
    <div class="card">
      <img id="preview" alt="Preview appears here">
      <div class="small" id="meta"></div>
      <div id="fabric" class="small"></div>
    </div>
    <div class="card">
      <img id="legend" alt="Palette legend" style="margin-bottom:12px">
      <div id="palette" class="palette"></div>
    </div>
  </div>
</section>

<section id="how" class="wrap">
  <div class="card">
    <h3>How it works</h3>
    <ol>
      <li>Upload an image. We resize to your width.</li>
      <li>We reduce colors to your palette count.</li>
      <li>We add a stitch grid and 10×10 numbering.</li>
      <li>Download a ZIP or a printable PDF with palette and sizes.</li>
    </ol>
    <p class="small">Private by default. Files are processed in memory and returned to you.</p>
  </div>
</section>

<footer class="wrap small">
  <div class="row" style="justify-content:space-between">
    <div>Banter, not bait. Opt‑in energy only.</div>
    <div>© StitchBitch.club</div>
  </div>
</footer>

<script>
const fileEl = document.getElementById('file');
const widthEl = document.getElementById('width');
const colorsEl = document.getElementById('colors');
const gridPxEl = document.getElementById('grid_px');
const ditherEl = document.getElementById('dither');
const gridOnEl = document.getElementById('grid_on');
const numbersOnEl = document.getElementById('numbers_on');
const useDmcEl = document.getElementById('use_dmc');

const previewImg = document.getElementById('preview');
const legendImg = document.getElementById('legend');
const paletteBox = document.getElementById('palette');
const meta = document.getElementById('meta');
const fabricBox = document.getElementById('fabric');

const previewBtn = document.getElementById('previewBtn');
const downloadZipBtn = document.getElementById('downloadZipBtn');
const downloadPdfBtn = document.getElementById('downloadPdfBtn');

const howSection = document.getElementById('how');
const toolSection = document.getElementById('tool');
const howLink = document.getElementById('howLink');
const shareBtn = document.getElementById('shareBtn');

// Smooth scroll link
howLink.addEventListener('click', (e)=>{ e.preventDefault(); howSection.scrollIntoView({behavior:'smooth', block:'start'}); });

// Emphasis toggle while HOW is in view
function updateHowEmphasis(){
  const vh = window.innerHeight || document.documentElement.clientHeight;
  const how = howSection.getBoundingClientRect();
  const tool = toolSection.getBoundingClientRect();
  const howInView = (how.top < vh*0.66) && (how.bottom > vh*0.34);
  const nearTool = tool.top < vh*0.5;
  document.body.classList.toggle('how-active', howInView && !nearTool);
}
window.addEventListener('scroll', updateHowEmphasis, {passive:true});
window.addEventListener('resize', updateHowEmphasis);
document.addEventListener('DOMContentLoaded', updateHowEmphasis);

// Share button (growth loop)
shareBtn.addEventListener('click', async ()=>{
  const text = "Free stitch pattern maker: grid, numbering, PDF, palette. Try it:";
  const url = location.origin + location.pathname + "#tool";
  if (navigator.share) {
    try { await navigator.share({title: document.title, text, url}); } catch(e){}
  } else {
    await navigator.clipboard.writeText(text + " " + url);
    alert("Link copied. Share anywhere.");
  }
});

// Build form
function buildForm(endpoint){
  const fd = new FormData();
  if(!fileEl.files[0]) { alert("Choose an image first."); return null; }
  fd.append('file', fileEl.files[0]);
  fd.append('width', widthEl.value);
  fd.append('colors', colorsEl.value);
  fd.append('grid_px', gridPxEl.value);
  fd.append('dither', ditherEl.checked ? "1" : "0");
  fd.append('grid_on', gridOnEl.checked ? "1" : "0");
  fd.append('numbers_on', numbersOnEl.checked ? "1" : "0");
  fd.append('use_dmc', useDmcEl.checked ? "1" : "0");
  return {url:endpoint, fd:fd};
}

// Preview
previewBtn.addEventListener('click', async (e)=>{
  e.preventDefault();
  const req = buildForm('/api/preview');
  if(!req) return;
  previewBtn.disabled = true;
  try{
    const r = await fetch(req.url, {method:'POST', body:req.fd});
    if(!r.ok){ const t = await r.text(); throw new Error(t); }
    const data = await r.json();
    previewImg.src = data.preview_png;
    legendImg.src = data.legend_png;
    meta.textContent = `Pattern: ${data.width} × ${data.height} stitches • Colors: ${data.colors_used} • Stitches: ${data.stitches}`;
    // Fabric sizes
    fabricBox.innerHTML = data.fabric.map(f =>
      `Aida ${f.count}: finished ${f.finished_in[0]}″ × ${f.finished_in[1]}″ • cut ${f.cut_in[0]}″ × ${f.cut_in[1]}″`
    ).join("<br>");
    // Palette
    paletteBox.innerHTML = '';
    for(const sw of data.palette){
      const right = data.dmc && data.dmc[sw.hex] ? `<br><span class="small">DMC ${data.dmc[sw.hex].code} — ${data.dmc[sw.hex].name}</span>` : '';
      const div = document.createElement('div');
      div.className = 'swatch';
      div.innerHTML = `<div class="box" style="background:${sw.hex}"></div><div><strong>${sw.hex}</strong><br><span class="small">${sw.count} px</span>${right}</div>`;
      paletteBox.appendChild(div);
    }
    downloadZipBtn.disabled = false;
    downloadPdfBtn.disabled = false;
  }catch(err){
    alert("Preview error: " + err.message);
  }finally{
    previewBtn.disabled = false;
  }
});

// Downloads
async function doDownload(endpoint, filename){
  const req = buildForm(endpoint);
  if(!req) return;
  try{
    const r = await fetch(req.url, {method:'POST', body:req.fd});
    if(!r.ok){ const t = await r.text(); throw new Error(t); }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }catch(err){
    alert("Download error: " + err.message);
  }
}
downloadZipBtn.addEventListener('click', e => { e.preventDefault(); doDownload('/api/download_zip', 'stitchbitch_pattern.zip'); });
downloadPdfBtn.addEventListener('click', e => { e.preventDefault(); doDownload('/api/download_pdf', 'stitchbitch_pattern.pdf'); });
</script>
"""

# ----------------- Core request processing -----------------

def _parse_bool(v: str | None, default: bool=False) -> bool:
    if v is None: return default
    return v in ("1","true","True","YES","yes","on")

def _process_from_request():
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "no file uploaded")
    try:
        width = int(request.form.get("width", "140"))
        colors = int(request.form.get("colors", "12"))
        grid_px = int(request.form.get("grid_px", "10"))
        dither = _parse_bool(request.form.get("dither"), False)
        grid_on = _parse_bool(request.form.get("grid_on"), True)
        numbers_on = _parse_bool(request.form.get("numbers_on"), True)
        use_dmc = _parse_bool(request.form.get("use_dmc"), False)
    except ValueError:
        abort(400, "bad parameters")

    original = _to_rgb(Image.open(f.stream))
    resized = resize_image(original, width)
    quant = quantize_image(resized, colors, dither=dither)
    with_grid = draw_grid_numbered(quant, grid_px, show_numbers=numbers_on) if grid_on else quant.copy()

    pc = palette_counts(quant)
    pairs = sorted_palette(pc)

    # DMC mapping (approx)
    dmc_map: Dict[str, Dict[str,str]] = {}
    if use_dmc:
        for rgb, _ in pairs:
            code, name, hx, _d = nearest_dmc(rgb)
            dmc_map[rgb_to_hex(rgb)] = {"code": code, "name": name, "hex": hx}

    return original, resized, quant, with_grid, pairs, dmc_map

# ----------------- Routes -----------------

@app.get("/")
def index() -> str:
    jsonld = {
      "@context":"https://schema.org",
      "@type":"SoftwareApplication",
      "name":"StitchBitch Pattern Maker",
      "applicationCategory":"GraphicsApplication",
      "operatingSystem":"Web",
      "offers":{"@type":"Offer","price":"0","priceCurrency":"USD"},
      "description": SITE_DESC,
      "url": SITE_URL,
    }
    return render_template_string(INDEX_HTML,
        title=SITE_TITLE, desc=SITE_DESC, url=SITE_URL, jsonld=json.dumps(jsonld))

@app.post("/api/preview")
def api_preview():
    original, resized, quant, with_grid, pairs, dmc_map = _process_from_request()
    legend = legend_strip(pairs, with_grid.width)
    data = {
        "width": quant.width,
        "height": quant.height,
        "stitches": quant.width * quant.height,
        "colors_used": min(64, len(pairs)),
        "preview_png": data_url(with_grid),
        "legend_png": data_url(legend),
        "palette": [{"hex": rgb_to_hex(rgb), "count": n} for rgb, n in pairs[:64]],
        "fabric": fabric_sizes(quant.width, quant.height),
        "dmc": dmc_map or None
    }
    return jsonify(data)

@app.post("/api/download_zip")
def api_download_zip():
    original, resized, quant, with_grid, pairs, dmc_map = _process_from_request()
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("original.png", png_bytes(original))
        z.writestr("resized.png", png_bytes(resized))
        z.writestr("quantized.png", png_bytes(quant))
        z.writestr("with_grid.png", png_bytes(with_grid))
        # palette CSV
        csv_lines = ["hex,count"]
        for rgb, n in pairs:
            csv_lines.append(f"{rgb_to_hex(rgb)},{n}")
        z.writestr("palette.csv", "\n".join(csv_lines).encode("utf-8"))
        # dmc CSV if requested
        if dmc_map:
            dmc_lines = ["orig_hex,dmc_code,dmc_name,dmc_hex"]
            for hx, meta in dmc_map.items():
                dmc_lines.append(f"{hx},{meta['code']},{meta['name']},{meta['hex']}")
            z.writestr("palette_dmc_approx.csv", "\n".join(dmc_lines).encode("utf-8"))
        # README
        readme = (
            "StitchBitch Pattern Pack (Free)\n"
            f"Size: {quant.width} x {quant.height} stitches\n"
            f"Colors used: {min(64, len(pairs))}\n"
            "Files:\n"
            "- original.png (normalized RGB)\n"
            "- resized.png (requested width)\n"
            "- quantized.png (reduced palette)\n"
            "- with_grid.png (grid + 10x10 numbering)\n"
            "- palette.csv (hex + counts)\n"
        )
        if dmc_map:
            readme += "- palette_dmc_approx.csv (nearest DMC approximation; not official)\n"
        z.writestr("README.txt", readme.encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="stitchbitch_pattern.zip")

@app.post("/api/download_pdf")
def api_download_pdf():
    """Generate a 1‑page printable PDF with the chart, legend, and basics."""
    original, resized, quant, with_grid, pairs, dmc_map = _process_from_request()
    # Compose an 8.5x11 inch page at 200 DPI to keep memory low
    DPI = 200
    page_w, page_h = int(8.5*DPI), int(11*DPI)
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)

    # Title
    title = "StitchBitch Pattern"
    draw.text((int(0.5*DPI), int(0.4*DPI)), title, fill=(0,0,0))
    meta = f"{quant.width}×{quant.height} stitches • colors: {min(64,len(pairs))}"
    draw.text((int(0.5*DPI), int(0.7*DPI)), meta, fill=(0,0,0))

    # Fit chart in left column
    max_chart_w = int(5.5*DPI)
    max_chart_h = int(7.5*DPI)
    chart = with_grid
    scale = min(max_chart_w/chart.width, max_chart_h/chart.height, 1.0)
    chart_resized = chart.resize((int(chart.width*scale), int(chart.height*scale)), Image.Resampling.NEAREST)
    page.paste(chart_resized, (int(0.5*DPI), int(1.2*DPI)))

    # Legend and palette on right
    right_x = int(6.3*DPI)
    y = int(1.2*DPI)
    legend = legend_strip(pairs, width=int(1.6*DPI))
    page.paste(legend, (right_x, y)); y += legend.height + int(0.2*DPI)

    # Palette swatches (up to 40)
    for rgb, n in pairs[:40]:
        sw = Image.new("RGB", (int(0.3*DPI), int(0.3*DPI)), rgb)
        page.paste(sw, (right_x, y))
        label = f"{rgb_to_hex(rgb)}  {n} px"
        if dmc_map and rgb_to_hex(rgb) in dmc_map:
            dm = dmc_map[rgb_to_hex(rgb)]
            label += f"  DMC {dm['code']} {dm['name']}"
        draw.text((right_x + int(0.35*DPI), y + 2), label, fill=(0,0,0))
        y += int(0.35*DPI)

    # Fabric sizes
    y += int(0.3*DPI)
    draw.text((right_x, y), "Fabric sizes (Aida):", fill=(0,0,0)); y += int(0.2*DPI)
    for f in fabric_sizes(quant.width, quant.height):
        line = f"{f['count']} ct  finished {f['finished_in'][0]}\"×{f['finished_in'][1]}\"  cut {f['cut_in'][0]}\"×{f['cut_in'][1]}\""
        draw.text((right_x, y), line, fill=(0,0,0)); y += int(0.18*DPI)

    # Export single-page PDF
    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=200.0)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="stitchbitch_pattern.pdf")

# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)