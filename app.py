from __future__ import annotations

import io
import json
import math
import os
import time
import uuid
import zipfile
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
    abort,
    Response,
)
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import generate_password_hash, check_password_hash

# Optional embroidery support
try:
    from pyembroidery import EmbPattern, write_dst, write_pes  # type: ignore
    HAS_PYEMB = True
except Exception:
    HAS_PYEMB = False

BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PATTERNS_DIR = os.path.join(BASE_DIR, "patterns")
os.makedirs(PATTERNS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Config
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels
PATTERN_RETENTION_SECONDS = 7 * 24 * 60 * 60  # 1 week


# ---------------------- USER STORAGE ----------------------
def load_users() -> Dict[str, dict]:
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return {}


def save_users(users: Dict[str, dict]) -> None:
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)


def get_current_user() -> Optional[dict]:
    email = session.get("user_email")
    if not email:
        return None
    users = load_users()
    return users.get(email)


def prune_old_patterns(user: dict) -> bool:
    """Remove patterns older than PATTERN_RETENTION_SECONDS; delete files too."""
    now_ts = time.time()
    changed = False
    kept: List[dict] = []
    for p in user.get("patterns", []):
        created = p.get("created_at") or 0
        if now_ts - created <= PATTERN_RETENTION_SECONDS:
            kept.append(p)
        else:
            for key in ("grid_file", "legend_file", "zip_file"):
                fname = p.get(key)
                if fname:
                    fpath = os.path.join(PATTERNS_DIR, fname)
                    try:
                        os.remove(fpath)
                    except FileNotFoundError:
                        pass
            changed = True
    if changed:
        user["patterns"] = kept
    return changed


# ---------------------- IMAGE / PATTERN HELPERS ----------------------
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


def draw_grid(base: Image.Image, cell_px: int) -> Image.Image:
    """Scale each stitch to a cell and overlay a 10x10 grid."""
    sx, sy = base.size
    out = base.resize((sx * cell_px, sy * cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
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
            [(0, y * cell_px), (sx * cell_px, sy * cell_px)],
            fill=(bold if y % 10 == 0 else thin),
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
            [(0, y * cell_px), (sx * cell_px, sy * cell_px)],
            fill=(bold if y % 10 == 0 else thin),
            width=1,
        )
    return out


def skeins_per_color(stitches: int, cloth_count: int, strands: int, waste: float) -> float:
    per_stitch_cm = 2 * math.sqrt(2) * (2.54 / cloth_count) * (1 + waste)
    skein_cm = (800 * 6) / strands
    return (stitches * per_stitch_cm) / skein_cm


def knit_aspect_resize(img: Image.Image, stitches_w: int, row_aspect: float = 0.8) -> Image.Image:
    """Knitting charts: visually shorter rows for preview."""
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
                pat.add_stitch_absolute(0, 0, 2)
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


# ---------------------- BASIC ROUTES ----------------------
@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


ERROR_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>We hit a snag — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{
      margin:0;
      font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#FFF9E6;
      color:#111827;
      display:flex;
      align-items:center;
      justify-content:center;
      min-height:100vh;
    }
    .card{
      background:#fff;
      border-radius:18px;
      padding:24px 22px;
      max-width:420px;
      width:100%;
      box-shadow:0 14px 40px rgba(15,23,42,.24);
      border:1px solid #fde68a;
    }
    h1{margin:0 0 10px;font-size:1.5rem;}
    p{margin:6px 0;}
    a{
      display:inline-block;
      margin-top:12px;
      padding:10px 18px;
      border-radius:999px;
      border:none;
      background:#f97316;
      color:#fff;
      font-weight:600;
      text-decoration:none;
      box-shadow:0 10px 26px rgba(248,113,22,.4);
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>We hit a snag</h1>
    <p>Something went wrong while processing your request.</p>
    <p>You can go back to PatternCraft.app and try again.</p>
    <a href="/">Back to PatternCraft.app</a>
  </div>
</body>
</html>
"""


@app.errorhandler(Exception)
def on_error(e):
    app.logger.exception("Unhandled error: %s", e)
    return make_response(render_template_string(ERROR_HTML), 500)


@app.get("/favicon.svg")
def favicon_svg() -> Response:
    return Response(FAVICON_SVG, mimetype="image/svg+xml")


# ---------------------- SIGNUP / LOGIN ----------------------
@app.get("/")
def index() -> str:
    user = get_current_user()
    return render_template_string(HOMEPAGE_HTML, user=user)


@app.get("/signup")
def signup() -> str:
    user = get_current_user()
    msg = request.args.get("msg", "")
    return render_template_string(SIGNUP_HTML, user=user, message=msg)


@app.post("/signup")
def signup_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    if not email or "@" not in email:
        return render_template_string(
            SIGNUP_HTML, user=None, message="Please enter a valid email address."
        )
    if len(password) < 8:
        return render_template_string(
            SIGNUP_HTML, user=None, message="Password must be at least 8 characters."
        )
    if password != confirm:
        return render_template_string(
            SIGNUP_HTML, user=None, message="Passwords do not match."
        )

    users = load_users()
    if email in users:
        return render_template_string(
            SIGNUP_HTML,
            user=None,
            message="This email already has an account. Please log in instead.",
        )

    users[email] = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": time.time(),
        "patterns": [],
    }
    save_users(users)
    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/login")
def login() -> str:
    user = get_current_user()
    msg = request.args.get("msg", "")
    return render_template_string(LOGIN_HTML, user=user, message=msg)


@app.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        return render_template_string(
            LOGIN_HTML,
            user=None,
            message="Please enter both email and password.",
        )

    users = load_users()
    stored = users.get(email)
    if not stored or not check_password_hash(stored.get("password_hash", ""), password):
        return render_template_string(
            LOGIN_HTML,
            user=None,
            message="Incorrect email or password.",
        )

    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("index"))


# ---------------------- PATTERN STORAGE HELPERS ----------------------
def _ensure_user_and_patterns() -> Tuple[Dict[str, dict], dict]:
    email = session.get("user_email")
    if not email:
        return {}, None  # type: ignore[return-value]
    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return {}, None  # type: ignore[return-value]
    if "patterns" not in user or not isinstance(user["patterns"], list):
        user["patterns"] = []
    if prune_old_patterns(user):
        users[email] = user
        save_users(users)
    return users, user


# ---------------------- PATTERN GENERATOR (ACCOUNT-GATED) ----------------------
@app.post("/api/convert")
def convert():
    users, user = _ensure_user_and_patterns()
    if not user:
        return redirect(
            url_for("login", msg="Log in or create a free account to generate patterns.")
        )

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
        pattern_name = (request.form.get("pattern_name") or "").strip()
    except Exception:
        return jsonify({"error": "invalid_parameters"}), 400

    try:
        base = open_image(file)
    except Exception:
        return jsonify({"error": "decode_failed"}), 400
    if max(base.size) > MAX_DIM:
        return jsonify({"error": "image_too_large", "max_dim": MAX_DIM}), 400

    out_zip = io.BytesIO()
    legend_rows: List[dict] = []

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
                percent = (100 * c / total_stitches)
                lines.append(
                    f"{to_hex((r,g,b))},{r},{g},{b},{c},{percent:.2f},{skeins:.2f}"
                )
                legend_rows.append(
                    {
                        "hex": to_hex((r, g, b)),
                        "r": r,
                        "g": g,
                        "b": b,
                        "stitches": c,
                        "percent": round(percent, 2),
                        "skeins_est": round(skeins, 2),
                    }
                )
            z.writestr("legend.csv", "\n".join(lines))

            note = (
                "Knitting preview compresses row height; verify gauge."
                if ptype == "knit"
                else "Cross-stitch grid with 10x10 guides."
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

    # Persist pattern assets for "My Patterns"
    pattern_id = uuid.uuid4().hex
    created_ts = time.time()
    default_name = f"Pattern {len(user.get('patterns', [])) + 1}"
    name = pattern_name or default_name

    zip_bytes = out_zip.getvalue()

    # Save ZIP
    zip_filename = f"{pattern_id}.zip"
    with open(os.path.join(PATTERNS_DIR, zip_filename), "wb") as f:
        f.write(zip_bytes)

    # Save grid preview PNG
    # Recreate grid image for disk save (simpler than reusing buffer)
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
        if want_symbols:
            pal = sorted(counts.keys(), key=lambda c: c)
            sym_map = assign_symbols(pal)
            grid_img = draw_symbols_on_grid(quant, cell_px=CELL_PX, sym_map=sym_map)
        grid_filename = f"{pattern_id}_grid.png"
        grid_img.save(os.path.join(PATTERNS_DIR, grid_filename), format="PNG")
    else:
        sx = sy = 0
        finished_w_in = finished_h_in = 0.0
        grid_filename = ""

    # Save legend JSON
    legend_filename = f"{pattern_id}_legend.json"
    with open(os.path.join(PATTERNS_DIR, legend_filename), "w", encoding="utf-8") as f:
        json.dump(legend_rows, f, indent=2)

    entry = {
        "id": pattern_id,
        "name": name,
        "ptype": ptype,
        "created_at": created_ts,
        "stitches_w": sx,
        "stitches_h": sy,
        "colors": len(legend_rows),
        "cloth_count": cloth_count,
        "finished_size_in": [finished_w_in, finished_h_in],
        "grid_file": grid_filename,
        "legend_file": legend_filename,
        "zip_file": zip_filename,
    }
    user["patterns"].insert(0, entry)
    # ensure retention policy on new list
    prune_old_patterns(user)
    users[email := user["email"]] = user
    save_users(users)

    out_zip.seek(0)
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{ptype}.zip",
    )


# ---------------------- MY PATTERNS ----------------------
@app.get("/patterns")
def patterns_index():
    users, user = _ensure_user_and_patterns()
    if not user:
        return redirect(url_for("login", msg="Log in to see your saved patterns."))
    patterns = user.get("patterns", [])
    return render_template_string(
        PATTERNS_HTML,
        user=user,
        patterns=patterns,
    )


@app.get("/patterns/<pattern_id>/grid")
def pattern_grid(pattern_id: str):
    users, user = _ensure_user_and_patterns()
    if not user:
        return redirect(url_for("login", msg="Log in to view that pattern."))
    pattern = next((p for p in user.get("patterns", []) if p.get("id") == pattern_id), None)
    if not pattern:
        abort(404)
    fname = pattern.get("grid_file")
    if not fname:
        abort(404)
    fpath = os.path.join(PATTERNS_DIR, fname)
    if not os.path.exists(fpath):
        abort(404)
    return send_file(fpath, mimetype="image/png")


@app.get("/patterns/<pattern_id>/download")
def pattern_download(pattern_id: str):
    users, user = _ensure_user_and_patterns()
    if not user:
        return redirect(url_for("login", msg="Log in to download that pattern."))
    pattern = next((p for p in user.get("patterns", []) if p.get("id") == pattern_id), None)
    if not pattern:
        abort(404)
    fname = pattern.get("zip_file")
    if not fname:
        abort(404)
    fpath = os.path.join(PATTERNS_DIR, fname)
    if not os.path.exists(fpath):
        abort(404)
    return send_file(
        fpath,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{pattern.get('name') or 'pattern'}.zip",
    )


@app.get("/patterns/<pattern_id>")
def pattern_detail(pattern_id: str):
    users, user = _ensure_user_and_patterns()
    if not user:
        return redirect(url_for("login", msg="Log in to view that pattern."))
    pattern = next((p for p in user.get("patterns", []) if p.get("id") == pattern_id), None)
    if not pattern:
        abort(404)

    legend_rows: List[dict] = []
    legend_file = pattern.get("legend_file")
    if legend_file:
        path = os.path.join(PATTERNS_DIR, legend_file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    legend_rows = data
        except Exception:
            legend_rows = []

    return render_template_string(
        PATTERN_DETAIL_HTML,
        user=user,
        pattern=pattern,
        legend_rows=legend_rows,
    )


# ---------------------- INLINE HTML: HOMEPAGE ----------------------
FAVICON_SVG = r"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#facc15'/>
      <stop offset='100%' stop-color='#f97316'/>
    </linearGradient>
  </defs>
  <rect x='4' y='4' width='56' height='56' rx='14' fill='url(#g)'/>
  <path d='M18 22h20a8 8 0 0 1 0 16H24' fill='none' stroke='#1f2937' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/>
  <path d='M24 18v28' fill='none' stroke='#1f2937' stroke-width='4' stroke-linecap='round'/>
  <circle cx='44' cy='24' r='3' fill='#1f2937'/>
</svg>"""


HOMEPAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="robots" content="noindex,noarchive">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    :root{
      --bg:#FFF9E6;--bg-soft:#FFFAF0;
      --fg:#252422;--muted:#6b6b6b;
      --line:#f3e8c8;--radius:18px;
      --accent:#f97316;--accent-soft:#FFEDD5;--accent-strong:#c05621;
      --pill:#f97316;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#fef3c7 0,#fff7ed 36%,transparent 60%),
        radial-gradient(circle at bottom right,#fde68a 0,#fffbeb 40%,transparent 60%),
        linear-gradient(to bottom,#fffbeb,#fff7ed);
    }
    a{color:#ea580c;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1120px;margin:0 auto;padding:24px 16px 48px}
    h1{font-size:2.5rem;margin:0 0 8px;}
    h2{margin:0 0 10px;font-size:1.1rem;}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      display:flex;align-items:center;gap:8px;
      font-weight:800;font-size:20px;letter-spacing:.04em;text-transform:uppercase;
    }
    .brand-mark{
      width:26px;height:26px;border-radius:9px;
      background:linear-gradient(135deg,#facc15,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:14px;font-weight:900;color:#1f2937;
    }
    .top-links{font-size:13px;color:#4b5563}
    .top-links a{margin-left:10px;font-weight:600;}
    .top-links span{opacity:.7}
    .card{
      background:#fff;
      border-radius:var(--radius);
      border:1px solid var(--line);
      padding:22px 20px;
      box-shadow:0 18px 50px rgba(15,23,42,.16);
    }
    .hero{
      display:grid;
      grid-template-columns:minmax(0,1.2fr) minmax(260px,1fr);
      gap:20px;margin-bottom:24px;align-items:stretch;
    }
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:4px 10px;border-radius:999px;
      background:rgba(255,255,255,.9);border:1px solid #fed7aa;
      font-size:11px;color:#92400e;text-transform:uppercase;letter-spacing:.08em;
    }
    .chip-dot{width:8px;height:8px;border-radius:999px;background:#22c55e}
    .hero-tagline{color:var(--muted);max-width:480px;}
    .muted{color:var(--muted);font-size:13px}
    .pill{
      padding:12px 24px;border-radius:999px;
      background:linear-gradient(135deg,var(--pill),#fb923c);
      color:#fff;border:none;cursor:pointer;
      font-size:15px;font-weight:700;letter-spacing:.02em;
      box-shadow:0 10px 26px rgba(248,113,22,.46);
      transition:transform .08s,box-shadow .08s;
      display:inline-flex;align-items:center;justify-content:center;
      text-decoration:none;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 32px rgba(248,113,22,.6);}
    .pill-secondary{
      background:#fff;color:var(--fg);
      border:1px solid rgba(248,153,64,.6);
      box-shadow:0 4px 14px rgba(248,171,88,.35);
    }
    .pill-secondary:hover{
      box-shadow:0 7px 18px rgba(248,171,88,.45);
    }
    .hero-cta-row{
      display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;align-items:center;
    }
    .hero-note{font-size:12px;color:#7c2d12;margin-top:10px;background:#ffedd5;border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px;}
    .badge-row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
    .badge{
      font-size:11px;padding:4px 8px;border-radius:999px;
      background:#fef3c7;color:#92400e;border:1px solid #facc15;font-weight:600;
    }
    .why-card h2{margin-bottom:6px;}
    .why-list{margin:8px 0 0;padding-left:18px;font-size:13px;color:#4b5563}
    .why-list li{margin:3px 0;}
    .section-title{font-size:1.15rem;margin-bottom:6px}
    .make-layout{display:grid;gap:18px;grid-template-columns:minmax(0,1.3fr)}
    .file{
      border:2px dashed var(--accent);
      border-radius:22px;
      padding:18px 18px;
      display:flex;align-items:center;gap:12px;
      cursor:pointer;
      background:var(--accent-soft);
      transition:background .15s,border-color .15s,transform .1s,box-shadow .1s;
    }
    .file:hover{
      background:#fed7aa;border-color:#ea580c;
      transform:translateY(-1px);
      box-shadow:0 7px 18px rgba(234,88,12,.4);
    }
    .file-ready{
      background:#dcfce7;
      border-color:#16a34a;
      box-shadow:0 7px 18px rgba(22,163,74,.4);
    }
    .file input{display:none}
    .file-label-main{font-weight:800;font-size:14px;text-transform:uppercase;letter-spacing:.10em}
    .file-label-sub{font-size:12px;color:#7c2d12}
    fieldset{border:1px solid var(--line);border-radius:14px;padding:10px 10px 8px;margin:10px 0}
    legend{font-size:13px;padding:0 4px}
    .row{display:flex;flex-wrap:wrap;gap:12px}
    .row > label{flex:1 1 160px;font-size:13px}
    .row input,.row select{
      width:100%;margin-top:3px;padding:7px 9px;border-radius:10px;
      border:1px solid #fde68a;font-size:13px;background:#FFFBEB;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#f97316;box-shadow:0 0 0 1px rgba(249,115,22,.45);
    }
    label{font-size:13px}
    .controls-note{font-size:11px;color:#a16207;margin-top:4px}
    .hidden{display:none}
    .banner-free{
      margin-top:8px;
      font-size:12px;
      color:#166534;
      background:#dcfce7;
      border-radius:999px;
      padding:6px 10px;
      display:inline-flex;
      align-items:center;
      gap:6px;
    }
    .dot-free{width:8px;height:8px;border-radius:999px;background:#16a34a}
    .two-col{
      display:grid;
      grid-template-columns:1.1fr 1fr;
      gap:18px;
      margin-top:18px;
    }
    .mini-panel{
      background:var(--bg-soft);
      border-radius:14px;
      padding:14px 14px 10px;
      border:1px dashed #facc15;
      font-size:13px;
      color:#4b5563;
    }
    .mini-panel h3{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#92400e;}
    .timeline{list-style:none;padding:0;margin:0;}
    .timeline li{display:flex;gap:8px;margin-bottom:8px;}
    .timeline-step{
      width:18px;height:18px;border-radius:999px;
      background:#f97316;color:#fff;
      display:flex;align-items:center;justify-content:center;
      font-size:11px;font-weight:700;flex-shrink:0;
    }
    .timeline p{margin:0;font-size:12px;}
    .mini-list{margin:0;padding-left:16px;font-size:12px;}
    .mini-list li{margin:2px 0;}
    .patterns-link{
      margin-top:10px;
      font-size:13px;
    }
    .patterns-link a{font-weight:600;}
    @media (max-width:900px){
      .hero{grid-template-columns:1fr}
      .two-col{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">PC</div>
      <span>PatternCraft.app</span>
    </div>
    <div class="top-links">
      {% if user %}
        <span>Signed in as {{ user.email }}</span>
        · <a href="/patterns">My patterns</a>
        · <a href="/logout">Sign out</a>
      {% else %}
        <a href="/signup">Create Free Account</a>
        · <a href="/login">Log in</a>
      {% endif %}
    </div>
  </div>

  <div class="hero">
    <div>
      <div class="chip">
        <span class="chip-dot"></span>
        <span>For cross-stitch, knitting, and quilting</span>
      </div>
      <h1>Turn art into stitchable patterns</h1>
      <p class="hero-tagline">
        PatternCraft.app converts your artwork into cross-stitch grids, knitting charts,
        and embroidery-ready line art. Upload a picture, choose your settings, and download
        a full pattern and legend you can print or save.
      </p>
      <div class="hero-cta-row">
        {% if user %}
          <a class="pill" href="#make">Open the pattern tool</a>
          <a class="pill pill-secondary" href="/patterns">View my patterns</a>
        {% else %}
          <a class="pill" href="/signup?msg=Create+your+free+PatternCraft.app+account+to+start+making+patterns.">
            Create Free Account
          </a>
          <a class="pill pill-secondary" href="/login">Log in</a>
        {% endif %}
      </div>
      {% if not user %}
      <div class="hero-note">
        <span>Free forever:</span>
        <span>Create a single account and enjoy ongoing access to full-size pattern exports and legends.</span>
      </div>
      {% endif %}
      <div class="badge-row">
        <span class="badge">Unlimited patterns after signup</span>
        <span class="badge">Patterns saved under “My patterns” for 7 days</span>
      </div>
    </div>

    <div class="card why-card">
      <h2>Why makers use PatternCraft.app</h2>
      <p class="muted">A purpose-built pattern tool with stitchers in mind:</p>
      <ul class="why-list">
        <li>Clean grids with bold 10x10 guides and symbol overlays</li>
        <li>Color legends with hex and RGB values for accurate palettes</li>
        <li>Fabric size estimates based on stitch count and cloth count</li>
        <li>Knitting charts that respect row proportions</li>
        <li>Embroidery line outputs ready for your machine software</li>
      </ul>
    </div>
  </div>

  <div id="make" class="card">
    <h2 class="section-title">Make a pattern</h2>
    <p class="muted">
      {% if user %}
        Upload a picture, choose stitch type and size, and download a ZIP with your grid, legend, and metadata.
        Your last week of patterns is always available under <a href="/patterns">My patterns</a>.
      {% else %}
        First create a free account to start making patterns. Once signed in, you can generate full-size patterns
        and legends with no watermark.
      {% endif %}
    </p>

    <div class="two-col">
      <div class="make-main">
        {% if user %}
        <form method="POST" action="/api/convert" enctype="multipart/form-data">
          <label class="file">
            <input id="fileInput" type="file" name="file" accept="image/*" required onchange="pickFile(this)">
            <div>
              <div class="file-label-main">UPLOAD PICTURE HERE</div>
              <div class="file-label-sub">
                Drop in your artwork or tap to browse from your device.
              </div>
            </div>
          </label>

          <fieldset>
            <legend>Pattern basics</legend>
            <div class="row">
              <label>Pattern name (optional)
                <input type="text" name="pattern_name" placeholder="Rainy window, floral grid, logo, etc.">
              </label>
            </div>
          </fieldset>

          <fieldset>
            <legend>Pattern type</legend>
            <label><input type="radio" name="ptype" value="cross" checked> Cross-stitch</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="knit"> Knitting</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="emb"> Embroidery line art</label>
          </fieldset>

          <fieldset>
            <legend>Stitch &amp; size</legend>
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
            <p class="controls-note">Defaults work well for most art. Adjust once you know your style.</p>
          </fieldset>

          <fieldset id="crossKnitBlock">
            <legend>Fabric &amp; floss</legend>
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
            <label style="margin-left:10px"><input type="checkbox" name="pdf" checked> Also export PDF</label>
          </fieldset>

          <fieldset id="embBlock" class="hidden">
            <legend>Embroidery options</legend>
            <p class="muted">
              Generates a single run-stitch line drawing from your image. For advanced digitizing, continue in your embroidery software.
            </p>
            <div class="row">
              <label>Threshold
                <input type="number" name="emb_thresh" value="180" min="0" max="255">
              </label>
              <label>Step px
                <input type="number" name="emb_step" value="3" min="1" max="10">
              </label>
            </div>
          </fieldset>

          <div style="margin-top:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
            <button class="pill" id="generateBtn" type="submit">Generate pattern ZIP</button>
            <span class="muted">
              Download includes grid.png, legend.csv, meta.json, and optional pattern.pdf or embroidery files.
            </span>
          </div>
        </form>
        {% else %}
          <div class="mini-panel">
            <h3>Step 1 — Create your free account</h3>
            <p>Sign up once and unlock unlimited full-size pattern exports and legends, saved under “My patterns” for 7 days.</p>
            <ul class="mini-list">
              <li>Use any email you check regularly.</li>
              <li>You’ll receive an occasional monthly pattern ideas email (you can unsubscribe anytime).</li>
            </ul>
            <div style="margin-top:10px;display:flex;gap:10px;flex-wrap:wrap;">
              <a class="pill" href="/signup?msg=Create+your+free+PatternCraft.app+account+to+start+making+patterns.">
                Create Free Account
              </a>
              <a class="pill pill-secondary" href="/login">Log in</a>
            </div>
          </div>
        {% endif %}
      </div>

      <div class="mini-panel">
        <h3>How the tool works</h3>
        <ul class="timeline">
          <li>
            <div class="timeline-step">1</div>
            <div>
              <p><strong>Pick your stitch type.</strong></p>
              <p>Choose cross-stitch, knitting, or simple embroidery line art.</p>
            </div>
          </li>
          <li>
            <div class="timeline-step">2</div>
            <div>
              <p><strong>Set size &amp; palette.</strong></p>
              <p>Adjust stitch width, cloth count, and max colors to fit your fabric.</p>
            </div>
          </li>
          <li>
            <div class="timeline-step">3</div>
            <div>
              <p><strong>Download &amp; stitch.</strong></p>
              <p>Print the grid and legend, or keep them digital while you stitch.</p>
            </div>
          </li>
        </ul>
        {% if user %}
        <div class="patterns-link">
          View everything you’ve generated in the last week on your <a href="/patterns">My patterns</a> page.
        </div>
        {% else %}
        <div class="patterns-link">
          Once you’re signed in, every pattern you generate is saved to a personal “My patterns” page for 7 days.
        </div>
        {% endif %}
      </div>
    </div>
  </div>

</div>
<script>
  function pickFile(inp){
    const wrapper = inp.closest('label');
    const label = wrapper ? wrapper.querySelector('.file-label-main') : null;
    const generateBtn = document.getElementById('generateBtn');

    if (!inp.files || !inp.files[0]){
      if (wrapper) wrapper.classList.remove('file-ready');
      if (label) label.textContent = 'UPLOAD PICTURE HERE';
      if (generateBtn) generateBtn.classList.remove('pill-ready');
      return;
    }

    if (wrapper) wrapper.classList.add('file-ready');
    if (label) label.textContent = 'IMAGE ATTACHED';
    if (generateBtn) generateBtn.classList.add('pill-ready');
  }

  function setStyleOptions(type){
    const sel = document.getElementById('stitch_style');
    if (!sel) return;
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
        ['rib1','Rib 1x1']
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
    const typeInput = document.querySelector('input[name="ptype"]:checked');
    const type = typeInput ? typeInput.value : 'cross';
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


# ---------------------- INLINE HTML: SIGNUP / LOGIN / PATTERNS ----------------------
SIGNUP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your free account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    body{margin:0;background:#FFF9E6;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#fff;border-radius:18px;border:1px solid #facc15;
      padding:24px 22px;box-shadow:0 18px 46px rgba(15,23,42,.22);
    }
    h1{margin:0 0 10px;font-size:1.7rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:11px;
      border:1px solid #fde68a;font-size:14px;background:#FFFBEB;
    }
    input:focus{
      outline:none;border-color:#f97316;box-shadow:0 0 0 1px rgba(249,115,22,.45);
    }
    .pill{
      margin-top:16px;padding:11px 22px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#f97316,#fb923c);color:#fff;
      font-size:15px;font-weight:700;cursor:pointer;
      box-shadow:0 10px 26px rgba(248,113,22,.46);
      width:100%;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 32px rgba(248,113,22,.6);}
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#ea580c;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4b5563;padding-left:18px;margin-top:10px}
    .note{
      margin-top:10px;
      font-size:12px;
      color:#92400e;
      background:#fffbeb;
      border-radius:12px;
      padding:8px 10px;
      border:1px dashed #facc15;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Create your free PatternCraft.app account</h1>
    <p class="muted">
      A single free account unlocks ongoing access to full-size pattern grids and legends. Patterns you make are
      stored under <strong>My patterns</strong> for 7 days so you can revisit, reprint, or download them again.
    </p>
    <ul>
      <li>No credit card required — just an email and password.</li>
      <li>Unlimited pattern generation once you’re signed in.</li>
      <li>Works for cross-stitch, knit charts, and embroidery line art.</li>
    </ul>
    <form method="POST" action="/signup">
      <label>Email
        <input type="email" name="email" placeholder="you@example.com" required>
      </label>
      <label>Password (min 8 characters)
        <input type="password" name="password" required>
      </label>
      <label>Confirm password
        <input type="password" name="confirm" required>
      </label>
      <button class="pill" type="submit">Create Free Account</button>
    </form>
    {% if message %}
      <div class="msg">{{ message }}</div>
    {% endif %}
    <p class="muted" style="margin-top:10px;">
      Already have an account? <a href="/login">Log in instead</a>.
    </p>
    <div class="note">
      By creating an account you agree to receive a short monthly PatternCraft.app email with pattern ideas and updates.
      You can unsubscribe with a single click at any time.
    </div>
  </div>
</div>
</body>
</html>
"""


LOGIN_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Log in — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    body{margin:0;background:#FFF9E6;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#fff;border-radius:18px;border:1px solid #facc15;
      padding:24px 22px;box-shadow:0 18px 46px rgba(15,23,42,.22);
    }
    h1{margin:0 0 10px;font-size:1.7rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:11px;
      border:1px solid #fde68a;font-size:14px;background:#FFFBEB;
    }
    input:focus{
      outline:none;border-color:#f97316;box-shadow:0 0 0 1px rgba(249,115,22,.45);
    }
    .pill{
      margin-top:16px;padding:11px 22px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#4c51bf,#6366f1);color:#fff;
      font-size:15px;font-weight:700;cursor:pointer;
      box-shadow:0 10px 26px rgba(79,70,229,.46);
      width:100%;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 32px rgba(79,70,229,.6);}
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#ea580c;text-decoration:none;}
    a:hover{text-decoration:underline;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Log in to PatternCraft.app</h1>
    <p class="muted">
      Enter the email and password you used when you created your free account. Once you’re signed in, you can
      generate new patterns and revisit anything created in the last 7 days.
    </p>
    <form method="POST" action="/login">
      <label>Email
        <input type="email" name="email" placeholder="you@example.com" required>
      </label>
      <label>Password
        <input type="password" name="password" required>
      </label>
      <button class="pill" type="submit">Log in</button>
    </form>
    {% if message %}
      <div class="msg">{{ message }}</div>
    {% endif %}
    <p class="muted" style="margin-top:10px;">
      New here? <a href="/signup">Create a free account</a>.
    </p>
  </div>
</div>
</body>
</html>
"""


PATTERNS_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>My patterns — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#FFF9E6;
      color:#111827;
    }
    a{color:#ea580c;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1100px;margin:0 auto;padding:24px 16px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{display:flex;align-items:center;gap:8px;font-weight:800;}
    .brand-mark{
      width:24px;height:24px;border-radius:9px;
      background:linear-gradient(135deg,#facc15,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:13px;font-weight:900;color:#1f2937;
    }
    .card{
      background:#fff;border-radius:18px;border:1px solid #facc15;
      padding:20px 18px;box-shadow:0 16px 40px rgba(15,23,42,.18);
    }
    h1{margin:0 0 6px;font-size:1.6rem;}
    .muted{font-size:13px;color:#6b7280}
    .grid{
      margin-top:16px;
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
      gap:14px;
    }
    .pattern-card{
      border-radius:16px;
      border:1px solid #f3e8c8;
      padding:10px 10px 12px;
      background:#FFFBEB;
      display:flex;
      flex-direction:column;
      gap:8px;
    }
    .pattern-thumb{
      border-radius:12px;
      overflow:hidden;
      background:#0f172a;
      padding:4px;
      display:flex;
      align-items:center;
      justify-content:center;
    }
    .pattern-thumb img{
      max-width:100%;
      max-height:180px;
      display:block;
      border-radius:8px;
    }
    .pattern-meta{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:8px;
    }
    .pattern-name{font-size:14px;font-weight:600;}
    .pattern-info{font-size:12px;color:#6b7280;}
    .pattern-actions{
      display:flex;
      gap:8px;
      margin-top:4px;
    }
    .btn-small{
      flex:1;
      padding:8px 10px;
      border-radius:999px;
      border:none;
      font-size:12px;
      font-weight:600;
      cursor:pointer;
      text-align:center;
      text-decoration:none;
    }
    .btn-view{
      background:#f97316;color:#fff;
      box-shadow:0 6px 16px rgba(248,113,22,.45);
    }
    .btn-download{
      background:#fff;color:#7c2d12;
      border:1px solid #fed7aa;
    }
    .empty{
      margin-top:16px;
      font-size:14px;
      color:#6b7280;
      background:#fffbeb;
      border-radius:14px;
      padding:10px 12px;
      border:1px dashed #facc15;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">PC</div>
      <span>PatternCraft.app</span>
    </div>
    <div class="muted">
      <a href="/">Back to tool</a> · <a href="/logout">Sign out</a>
    </div>
  </div>

  <div class="card">
    <h1>My patterns</h1>
    <p class="muted">
      Patterns you generate while signed in are stored here for 7 days. You can reopen them, reprint, or download the ZIP again.
    </p>

    {% if not patterns %}
      <div class="empty">
        You don’t have any saved patterns yet. Once you create your first pattern, it will appear here automatically.
      </div>
    {% else %}
      <div class="grid">
        {% for p in patterns %}
          <div class="pattern-card">
            <div class="pattern-thumb">
              {% if p.grid_file %}
                <img src="/patterns/{{ p.id }}/grid" alt="Pattern grid preview">
              {% else %}
                <span class="muted">No preview available</span>
              {% endif %}
            </div>
            <div class="pattern-meta">
              <div>
                <div class="pattern-name">{{ p.name or ('Pattern ' ~ loop.index) }}</div>
                <div class="pattern-info">
                  {{ p.ptype|upper }} · {{ p.stitches_w }}x{{ p.stitches_h }} stitches · {{ p.colors }} colors
                </div>
              </div>
            </div>
            <div class="pattern-actions">
              <a class="btn-small btn-view" href="/patterns/{{ p.id }}">View details</a>
              <a class="btn-small btn-download" href="/patterns/{{ p.id }}/download">Download ZIP</a>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
</div>
</body>
</html>
"""


PATTERN_DETAIL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ pattern.name }} — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#FFF9E6;
      color:#111827;
    }
    a{color:#ea580c;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1100px;margin:0 auto;padding:24px 16px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{display:flex;align-items:center;gap:8px;font-weight:800;}
    .brand-mark{
      width:24px;height:24px;border-radius:9px;
      background:linear-gradient(135deg,#facc15,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:13px;font-weight:900;color:#1f2937;
    }
    .card{
      background:#fff;border-radius:18px;border:1px solid #facc15;
      padding:20px 18px;box-shadow:0 16px 40px rgba(15,23,42,.18);
    }
    h1{margin:0 0 6px;font-size:1.5rem;}
    .muted{font-size:13px;color:#6b7280}
    .layout{
      margin-top:16px;
      display:grid;
      grid-template-columns:minmax(0,1.3fr) minmax(280px,1fr);
      gap:16px;
    }
    .grid-pane{
      border-radius:16px;
      background:#FFFBEB;
      border:1px solid #f3e8c8;
      padding:8px;
    }
    .grid-pane img{
      max-width:100%;
      display:block;
      border-radius:12px;
      background:#0f172a;
    }
    .legend-pane{
      border-radius:16px;
      background:#FFFBEB;
      border:1px solid #f3e8c8;
      padding:10px;
      font-size:12px;
    }
    table{
      width:100%;
      border-collapse:collapse;
      font-size:12px;
    }
    th,td{
      padding:4px 6px;
      border-bottom:1px solid #f3e8c8;
      text-align:left;
    }
    th{font-weight:600;background:#fef3c7;}
    .swatch{
      width:18px;height:18px;border-radius:4px;border:1px solid #e5e7eb;
      display:inline-block;
    }
    .actions{
      margin-top:10px;
      display:flex;
      gap:10px;
      flex-wrap:wrap;
    }
    .btn{
      padding:9px 16px;
      border-radius:999px;
      font-size:13px;
      font-weight:600;
      border:none;
      cursor:pointer;
      text-decoration:none;
      text-align:center;
    }
    .btn-primary{
      background:#f97316;color:#fff;
      box-shadow:0 7px 20px rgba(248,113,22,.48);
    }
    .btn-secondary{
      background:#fff;color:#7c2d12;
      border:1px solid #fed7aa;
    }
    @media (max-width:900px){
      .layout{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">PC</div>
      <span>PatternCraft.app</span>
    </div>
    <div class="muted">
      <a href="/patterns">My patterns</a> · <a href="/">Back to tool</a>
    </div>
  </div>

  <div class="card">
    <h1>{{ pattern.name }}</h1>
    <p class="muted">
      {{ pattern.ptype|upper }} · {{ pattern.stitches_w }}x{{ pattern.stitches_h }} stitches · {{ pattern.colors }} colors
      {% if pattern.finished_size_in %}
        · approx {{ pattern.finished_size_in[0] }}″ × {{ pattern.finished_size_in[1] }}″ on {{ pattern.cloth_count }}ct
      {% endif %}
    </p>

    <div class="layout">
      <div class="grid-pane">
        <img src="/patterns/{{ pattern.id }}/grid" alt="Pattern grid preview">
      </div>
      <div class="legend-pane">
        <table>
          <thead>
            <tr>
              <th>Color</th>
              <th>HEX</th>
              <th>RGB</th>
              <th>Stitches</th>
              <th>%</th>
              <th>Skeins est.</th>
            </tr>
          </thead>
          <tbody>
            {% for row in legend_rows %}
            <tr>
              <td><span class="swatch" style="background: {{ row.hex }};"></span></td>
              <td>{{ row.hex }}</td>
              <td>{{ row.r }},{{ row.g }},{{ row.b }}</td>
              <td>{{ row.stitches }}</td>
              <td>{{ row.percent }}</td>
              <td>{{ row.skeins_est }}</td>
            </tr>
            {% endfor %}
            {% if not legend_rows %}
            <tr><td colspan="6" class="muted">Legend data not available for this pattern.</td></tr>
            {% endif %}
          </tbody>
        </table>
        <div class="actions">
          <a class="btn btn-primary" href="/patterns/{{ pattern.id }}/download">Download full ZIP</a>
          <a class="btn btn-secondary" href="/patterns">Back to My patterns</a>
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
