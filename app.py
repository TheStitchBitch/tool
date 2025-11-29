from __future__ import annotations

import io
import json
import math
import os
import time
import uuid
import zipfile
import shutil
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
from werkzeug.exceptions import HTTPException

# Optional embroidery support
try:
    from pyembroidery import EmbPattern, write_dst, write_pes  # type: ignore
    HAS_PYEMB = True
except Exception:
    HAS_PYEMB = False

BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PATTERNS_ROOT = os.path.join(BASE_DIR, "user_patterns")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Config
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels
PATTERN_TTL_SECONDS = 7 * 24 * 60 * 60  # keep patterns ~7 days


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
    """
    Scale each stitch to a cell and overlay a 10×10 grid.

    This is the standard “professional chart” treatment: one square = one stitch,
    with bold 10×10 guides.
    """
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
    """Overlay symbol per stitch, then grid (no watermarking, no fade)."""
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


def knit_aspect_resize(
    img: Image.Image, stitches_w: int, row_aspect: float = 1.0
) -> Image.Image:
    """
    Knitting charts: keep squares to scale (one square = one stitch).

    row_aspect is left in case you ever want to squeeze rows, but by default we
    keep charts square so what you see looks like a standard colorwork chart.
    """
    resized = resize_for_stitch_width(img, stitches_w)
    if abs(row_aspect - 1.0) < 1e-6:
        return resized
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


def write_embroidery_outputs(
    paths: List[Tuple[int, int]], scale: float = 1.0
) -> Dict[str, bytes]:
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


# ---------------------- PATTERN PERSISTENCE ----------------------


def _email_to_folder(email: str) -> str:
    safe = email.replace("@", "_at_").replace(".", "_dot_")
    return os.path.join(PATTERNS_ROOT, safe)


def _pattern_dir(email: str, pattern_id: str) -> str:
    return os.path.join(_email_to_folder(email), pattern_id)


def save_pattern_files(
    email: str,
    meta: dict,
    grid_img: Optional[Image.Image],
    legend_csv: Optional[str],
    pdf_bytes: Optional[bytes],
    extra_files: Optional[Dict[str, bytes]] = None,
) -> str:
    """Persist pattern assets to disk and return pattern_id."""
    os.makedirs(PATTERNS_ROOT, exist_ok=True)
    folder = _email_to_folder(email)
    os.makedirs(folder, exist_ok=True)

    pattern_id = uuid.uuid4().hex
    pdir = _pattern_dir(email, pattern_id)
    os.makedirs(pdir, exist_ok=True)

    meta = dict(meta)
    meta.setdefault("created_at", time.time())

    # Save grid image
    if grid_img is not None:
        grid_path = os.path.join(pdir, "grid.png")
        grid_img.save(grid_path, format="PNG")

    # Save legend
    if legend_csv is not None:
        legend_path = os.path.join(pdir, "legend.csv")
        with open(legend_path, "w", encoding="utf-8") as f:
            f.write(legend_csv)

    # Save PDF
    if pdf_bytes:
        pdf_path = os.path.join(pdir, "pattern.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

    # Save any extra files (embroidery, etc.)
    if extra_files:
        for name, data in extra_files.items():
            safe_name = name.replace("..", "").lstrip("/\\")
            with open(os.path.join(pdir, safe_name), "wb") as f:
                f.write(data)

    # Save meta last
    meta_path = os.path.join(pdir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return pattern_id


def _cleanup_old_patterns(email: str) -> None:
    folder = _email_to_folder(email)
    if not os.path.isdir(folder):
        return
    now = time.time()
    cutoff = now - PATTERN_TTL_SECONDS
    for pid in os.listdir(folder):
        pdir = os.path.join(folder, pid)
        if not os.path.isdir(pdir):
            continue
        meta_path = os.path.join(pdir, "meta.json")
        created = None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                created = meta.get("created_at")
        except Exception:
            pass
        if not isinstance(created, (int, float)):
            try:
                created = os.path.getmtime(pdir)
            except Exception:
                created = now
        if created < cutoff:
            try:
                shutil.rmtree(pdir)
            except Exception:
                pass


def list_user_patterns(email: str) -> List[dict]:
    folder = _email_to_folder(email)
    if not os.path.isdir(folder):
        return []

    _cleanup_old_patterns(email)
    patterns: List[dict] = []
    now = time.time()

    for pid in os.listdir(folder):
        pdir = os.path.join(folder, pid)
        if not os.path.isdir(pdir):
            continue
        meta_path = os.path.join(pdir, "meta.json")
        grid_path = os.path.join(pdir, "grid.png")
        legend_path = os.path.join(pdir, "legend.csv")
        meta: dict = {}
        created = None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                created = meta.get("created_at")
        except Exception:
            meta = {}
        if not isinstance(created, (int, float)):
            try:
                created = os.path.getmtime(pdir)
            except Exception:
                created = now
        patterns.append(
            {
                "id": pid,
                "meta": meta,
                "created_at": created,
                "has_grid": os.path.isfile(grid_path),
                "has_legend": os.path.isfile(legend_path),
            }
        )

    patterns.sort(key=lambda p: p["created_at"], reverse=True)
    return patterns


def get_pattern_paths(email: str, pattern_id: str) -> Optional[dict]:
    pdir = _pattern_dir(email, pattern_id)
    if not os.path.isdir(pdir):
        return None
    meta_path = os.path.join(pdir, "meta.json")
    grid_path = os.path.join(pdir, "grid.png")
    legend_path = os.path.join(pdir, "legend.csv")
    pdf_path = os.path.join(pdir, "pattern.pdf")
    info = {
        "dir": pdir,
        "meta_path": meta_path,
        "grid_path": grid_path if os.path.isfile(grid_path) else None,
        "legend_path": legend_path if os.path.isfile(legend_path) else None,
        "pdf_path": pdf_path if os.path.isfile(pdf_path) else None,
    }
    return info


# ---------------------- BASIC ROUTES ----------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


ERROR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>We hit a snag — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#FFF9E6;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:40px 16px}
    .card{background:#fff;border-radius:18px;border:1px solid #fde68a;padding:24px;box-shadow:0 14px 40px rgba(15,23,42,.15)}
    h1{margin:0 0 10px;font-size:1.7rem}
    p{margin:6px 0;font-size:14px;color:#4b5563}
    a{color:#2563eb;text-decoration:none;font-weight:600}
    a:hover{text-decoration:underline}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>We hit a snag</h1>
      <p>Something went wrong while processing your request.</p>
      <p>You can go back to PatternCraft.app and try again.</p>
      <p style="margin-top:14px"><a href="/">← Back to PatternCraft.app</a></p>
    </div>
  </div>
</body>
</html>"""


@app.errorhandler(Exception)
def on_error(e):
    # Let normal HTTP errors (404, 403, etc.) pass through
    if isinstance(e, HTTPException):
        return e
    return make_response(ERROR_HTML, 500)


FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#fbbf24'/>
      <stop offset='100%' stop-color='#f97316'/>
    </linearGradient>
  </defs>
  <rect x='4' y='4' width='56' height='56' rx='12' fill='url(#g)'/>
  <path d='M16 20h32v4H16zM16 30h32v4H16zM16 40h18v4H16z' fill='#111827'/>
  <text x='50%' y='52%' text-anchor='middle' dominant-baseline='middle'
        font-family='system-ui, -apple-system, Arial' font-size='18'
        fill='#111827'>PC</text>
</svg>"""


@app.get("/favicon.ico")
def favicon():
    return Response(FAVICON_SVG, mimetype="image/svg+xml")


# ---------------------- SIGNUP / LOGIN ----------------------


@app.get("/")
def index() -> str:
    user = get_current_user()
    patterns = list_user_patterns(user["email"]) if user else []
    return render_template_string(HOMEPAGE_HTML, user=user, patterns=patterns)


@app.get("/signup")
def signup() -> str:
    user = get_current_user()
    if user:
        return redirect(url_for("index"))
    msg = request.args.get("msg", "")
    return render_template_string(SIGNUP_HTML, message=msg)


@app.post("/signup")
def signup_post():
    user = get_current_user()
    if user:
        return redirect(url_for("index"))

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    if not email or "@" not in email:
        return render_template_string(
            SIGNUP_HTML, message="Please enter a valid email address."
        )
    if len(password) < 8:
        return render_template_string(
            SIGNUP_HTML, message="Password must be at least 8 characters."
        )
    if password != confirm:
        return render_template_string(
            SIGNUP_HTML, message="Passwords do not match."
        )

    users = load_users()
    if email in users:
        return render_template_string(
            SIGNUP_HTML,
            message="This email already has an account. Please log in instead.",
        )

    users[email] = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": time.time(),
    }
    save_users(users)
    session["user_email"] = email
    session["login_failures"] = 0
    return redirect(url_for("index"))


@app.get("/login")
def login() -> str:
    user = get_current_user()
    if user:
        return redirect(url_for("index"))

    msg = request.args.get("msg", "")
    failures = int(session.get("login_failures", 0) or 0)
    attempts_left = max(0, 3 - failures)
    return render_template_string(
        LOGIN_HTML,
        message=msg,
        attempts_left=attempts_left,
    )


@app.post("/login")
def login_post():
    user = get_current_user()
    if user:
        return redirect(url_for("index"))

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    failures = int(session.get("login_failures", 0) or 0)

    if not email or not password:
        attempts_left = max(0, 3 - failures)
        return render_template_string(
            LOGIN_HTML,
            message="Please enter both email and password.",
            attempts_left=attempts_left,
        )

    users = load_users()
    stored = users.get(email)
    if not stored or not check_password_hash(stored.get("password_hash", ""), password):
        failures += 1
        session["login_failures"] = failures
        if failures >= 3:
            session["login_failures"] = 0
            return redirect(
                url_for(
                    "signup",
                    msg="We couldn’t match that email and password after several attempts. Create a PatternCraft.app account to get started.",
                )
            )
        attempts_left = max(0, 3 - failures)
        return render_template_string(
            LOGIN_HTML,
            message="Incorrect email or password.",
            attempts_left=attempts_left,
        )

    # success
    session["user_email"] = email
    session["login_failures"] = 0
    return redirect(url_for("index"))


@app.get("/logout")
def logout():
    session.pop("user_email", None)
    session["login_failures"] = 0
    return redirect(url_for("index"))


# ---------------------- PATTERN GENERATOR (ACCOUNT-GATED) ----------------------


@app.post("/api/convert")
def convert():
    email = session.get("user_email")
    if not email:
        return redirect(
            url_for(
                "login",
                msg="Log in or create a free account to generate patterns.",
            )
        )

    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(
            url_for(
                "signup",
                msg="Create your PatternCraft.app account to continue.",
            )
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
    except Exception:
        return jsonify({"error": "invalid_parameters"}), 400

    try:
        base = open_image(file)
    except Exception:
        return jsonify({"error": "decode_failed"}), 400
    if max(base.size) > MAX_DIM:
        return jsonify({"error": "image_too_large", "max_dim": MAX_DIM}), 400

    out_zip = io.BytesIO()
    pdf_bytes: Optional[bytes] = None
    legend_csv: Optional[str] = None
    grid_img: Optional[Image.Image] = None
    extra_files: Dict[str, bytes] = {}
    meta: dict = {}

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        if ptype in ("cross", "knit"):
            # True grids: one square = one stitch
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

            # Base color grid with 10×10 lines
            base_grid = draw_grid(quant, cell_px=CELL_PX)

            # Optional symbol overlay (only if explicitly requested)
            symbol_grid: Optional[Image.Image] = None
            if want_symbols:
                pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
                sym_map = assign_symbols(pal)
                symbol_grid = draw_symbols_on_grid(
                    quant, cell_px=CELL_PX, sym_map=sym_map
                )

            # Decide which image to show as the core pattern
            if want_symbols and symbol_grid is not None:
                grid_img = symbol_grid
            else:
                grid_img = base_grid

            # PDF respects overlay choice as well
            if want_pdf:
                pdf_source = symbol_grid if (want_symbols and symbol_grid is not None) else base_grid
                pdf_buf = io.BytesIO()
                pdf_source.convert("RGB").save(pdf_buf, format="PDF", resolution=300.0)
                pdf_bytes = pdf_buf.getvalue()

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
            legend_csv = "\n".join(lines)
            z.writestr("legend.csv", legend_csv)

            if ptype == "knit":
                note = (
                    "Colorwork knitting chart. One square = one stitch; "
                    "your swatch and gauge control the finished size."
                )
            else:
                note = (
                    "Cross-stitch chart with 10×10 guides. One square = one full stitch."
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
                "created_at": time.time(),
            }
            z.writestr("meta.json", json.dumps(meta, indent=2))

            if grid_img is not None:
                buf_png = io.BytesIO()
                grid_img.save(buf_png, format="PNG")
                z.writestr("grid.png", buf_png.getvalue())

            if pdf_bytes:
                z.writestr("pattern.pdf", pdf_bytes)

        elif ptype == "emb":
            # Simple embroidery line art: run-stitch path only, no extra overlays
            small = resize_for_stitch_width(base, stitch_w)
            bw = to_monochrome(small, threshold=emb_thresh)
            pts = serpentine_points(bw, step=emb_step)
            extra_files.update(write_embroidery_outputs(pts))
            meta = {
                "type": "emb",
                "stitch_style": "run",
                "points": len(pts),
                "pyembroidery": HAS_PYEMB,
                "created_at": time.time(),
                "notes": "Simple run-stitch embroidery line art. Refine in your machine software if needed.",
            }
            z.writestr("meta.json", json.dumps(meta, indent=2))
        else:
            return jsonify({"error": "unknown_ptype"}), 400

    out_zip.seek(0)

    # Persist pattern so the user can revisit it
    if email and grid_img is not None:
        save_pattern_files(
            email=email,
            meta=meta,
            grid_img=grid_img,
            legend_csv=legend_csv,
            pdf_bytes=pdf_bytes,
            extra_files=extra_files,
        )

    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{ptype}.zip",
    )


# ---------------------- SAVED PATTERNS UI ----------------------


@app.get("/patterns")
def my_patterns():
    user = get_current_user()
    if not user:
        return redirect(url_for("login", msg="Log in to see your saved patterns."))

    patterns = list_user_patterns(user["email"])
    return render_template_string(PATTERNS_HTML, user=user, patterns=patterns)


@app.get("/patterns/<pattern_id>")
def pattern_detail(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login", msg="Log in to view that pattern."))

    paths = get_pattern_paths(user["email"], pattern_id)
    if not paths:
        abort(404)

    meta = {}
    try:
        with open(paths["meta_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    return render_template_string(
        PATTERN_DETAIL_HTML,
        user=user,
        pattern_id=pattern_id,
        meta=meta,
        has_grid=paths["grid_path"] is not None,
        has_legend=paths["legend_path"] is not None,
        has_pdf=paths["pdf_path"] is not None,
    )


@app.get("/patterns/<pattern_id>/image")
def pattern_image(pattern_id: str):
    user = get_current_user()
    if not user:
        abort(403)
    paths = get_pattern_paths(user["email"], pattern_id)
    if not paths or not paths["grid_path"]:
        abort(404)
    return send_file(paths["grid_path"], mimetype="image/png")


@app.get("/patterns/<pattern_id>/legend.csv")
def pattern_legend(pattern_id: str):
    user = get_current_user()
    if not user:
        abort(403)
    paths = get_pattern_paths(user["email"], pattern_id)
    if not paths or not paths["legend_path"]:
        abort(404)
    return send_file(
        paths["legend_path"],
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"pattern_{pattern_id}_legend.csv",
    )


@app.get("/patterns/<pattern_id>/download")
def pattern_download(pattern_id: str):
    user = get_current_user()
    if not user:
        abort(403)
    paths = get_pattern_paths(user["email"], pattern_id)
    if not paths:
        abort(404)

    # Rebuild a ZIP from saved assets
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if paths["grid_path"]:
            with open(paths["grid_path"], "rb") as f:
                z.writestr("grid.png", f.read())
        if paths["legend_path"]:
            with open(paths["legend_path"], "rb") as f:
                z.writestr("legend.csv", f.read())
        if paths["pdf_path"]:
            with open(paths["pdf_path"], "rb") as f:
                z.writestr("pattern.pdf", f.read())
        if os.path.isfile(paths["meta_path"]):
            with open(paths["meta_path"], "r", encoding="utf-8") as f:
                z.writestr("meta.json", f.read())

    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{pattern_id}.zip",
    )


# ---------------------- INLINE HTML: PAGES ----------------------


HOMEPAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="robots" content="noarchive,noimageindex">
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <style>
    :root{
      --bg:#FFF9E6;
      --fg:#1f2933;
      --muted:#6b7280;
      --line:#facc15;
      --radius:18px;
      --shadow:0 20px 55px rgba(15,23,42,.20);
      --accent:#f97316;
      --accent-soft:#fffbeb;
      --accent-strong:#ea580c;
      --pill:#f59e0b;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#fef3c7 0,#fffbeb 40%,transparent 65%),
        radial-gradient(circle at bottom right,#ffedd5 0,#fffbeb 55%,transparent 75%),
        #fffbeb;
    }
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1120px;margin:0 auto;padding:24px 16px 48px}
    h1{font-size:2.7rem;margin:0 0 8px;letter-spacing:-.03em}
    h2{margin:0 0 10px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:22px;
    }
    .brand-mark{
      display:flex;align-items:center;gap:8px;
    }
    .brand-icon{
      width:30px;height:30px;border-radius:10px;
      background:linear-gradient(135deg,#fbbf24,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:15px;font-weight:800;color:#111827;
      box-shadow:0 10px 24px rgba(248,181,0,.55);
    }
    .brand-text{
      font-weight:800;font-size:19px;letter-spacing:.06em;text-transform:uppercase;
    }
    .top-links{
      font-size:13px;color:#4b5563;display:flex;align-items:center;gap:8px;
      flex-wrap:wrap;
    }
    .nav-link{
      font-size:13px;color:#4b5563;
      padding:6px 12px;border-radius:999px;
      background:rgba(255,255,255,.85);
      border:1px solid rgba(248,181,0,.35);
      text-decoration:none;
    }
    .nav-link:hover{
      background:#fff7cc;
      box-shadow:0 6px 16px rgba(248,181,0,.35);
    }
    .pill{
      padding:13px 24px;border-radius:999px;
      background:linear-gradient(135deg,#f97316,#ea580c);
      color:#fff;border:none;cursor:pointer;
      font-size:15px;font-weight:650;letter-spacing:.03em;
      box-shadow:0 18px 40px rgba(248,113,22,.52);
      transition:transform .08s,box-shadow .08s;
      display:inline-flex;align-items:center;justify-content:center;
      text-decoration:none;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 24px 55px rgba(248,113,22,.6);
    }
    .pill.secondary{
      background:#fff;color:#1f2933;
      border:1px solid rgba(148,163,184,.7);
      box-shadow:0 12px 32px rgba(148,163,184,.55);
    }
    .pill.secondary:hover{
      box-shadow:0 18px 40px rgba(148,163,184,.7);
    }
    .pill-ready{
      background:linear-gradient(135deg,#22c55e,#15803d);
      box-shadow:0 18px 40px rgba(22,163,74,.6);
    }
    .pill-ready:hover{
      box-shadow:0 24px 55px rgba(22,163,74,.7);
    }
    .hero{
      display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,2.2fr);
      gap:26px;margin-bottom:26px;align-items:center;
    }
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:6px 12px;border-radius:999px;
      background:#fff7cc;border:1px solid #facc15;
      font-size:11px;color:#854d0e;text-transform:uppercase;letter-spacing:.08em;
    }
    .chip-dot{width:8px;height:8px;border-radius:999px;background:#22c55e}
    .hero-tagline{color:var(--muted);max-width:450px;}
    .hero-cta-row{
      display:flex;gap:10px;margin-top:18px;flex-wrap:wrap;align-items:center;
    }
    .hero-note{font-size:12px;color:#475569;margin-top:10px;max-width:540px;}
    .badge-row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
    .badge{
      font-size:11px;padding:6px 10px;border-radius:999px;
      background:#fef3c7;color:#92400e;border:1px dashed #fbbf24;
    }
    .card{
      background:#fff;border-radius:var(--radius);
      border:1px solid rgba(250,204,21,.6);
      box-shadow:var(--shadow);
      padding:20px;
    }
    .section-title{font-size:1.1rem;margin-bottom:6px}
    .why-list{
      list-style:none;margin:0;padding:0;font-size:14px;color:#4b5563;
    }
    .why-list li{
      padding-left:22px;position:relative;margin-bottom:4px;
    }
    .why-list li::before{
      content:"◆";
      position:absolute;left:4px;top:1px;font-size:11px;color:#f97316;
    }
    .make-card{margin-top:6px;}
    .make-layout{display:grid;gap:18px;grid-template-columns:minmax(0,1.2fr)}
    .file{
      border:2px dashed #f97316;
      border-radius:18px;
      padding:18px;
      display:flex;align-items:center;gap:12px;
      cursor:pointer;
      background:var(--accent-soft);
      transition:background .15s,border-color .15s,transform .1s,box-shadow .1s;
      width:100%;
      text-align:left;
    }
    .file:hover{
      background:#fef3c7;border-color:#ea580c;
      transform:translateY(-1px);
      box-shadow:0 12px 28px rgba(234,88,12,.5);
    }
    .file.file-ready{
      background:#dcfce7;border-color:#16a34a;
      box-shadow:0 12px 30px rgba(22,163,74,.6);
    }
    .file input{display:none}
    .file-label-main{font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.06em}
    .file-label-sub{font-size:12px;color:#6b7280}
    fieldset{border:1px solid rgba(250,204,21,.8);border-radius:12px;padding:10px;margin:10px 0}
    legend{font-size:13px;padding:0 4px;color:#92400e}
    .row{display:flex;flex-wrap:wrap;gap:12px}
    .row > label{flex:1 1 150px;font-size:13px}
    .row input,.row select{
      width:100%;margin-top:3px;padding:7px 9px;border-radius:10px;
      border:1px solid #e5e7eb;font-size:13px;
      background:#fffbeb;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#f59e0b;box-shadow:0 0 0 1px rgba(245,158,11,.45);
      background:#fff;
    }
    label{font-size:13px}
    .controls-note{font-size:11px;color:#94a3b8;margin-top:4px}
    .hidden{display:none}
    .pill-row{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .pill-row span{font-size:13px;color:#4b5563}
    @media (max-width:880px){
      .hero{grid-template-columns:1fr}
      .make-layout{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<div class="wrap">

  <header class="topbar">
    <div class="brand-mark">
      <div class="brand-icon">PC</div>
      <div class="brand-text">PatternCraft.app</div>
    </div>
    <div class="top-links">
      {% if user %}
        <a class="nav-link" href="/patterns">My patterns</a>
        <span>Signed in as {{ user.email }}</span>
        <a class="nav-link" href="/logout">Sign out</a>
      {% else %}
        <a class="nav-link" href="/login">Log in</a>
        <a class="nav-link" href="/signup">Create account</a>
      {% endif %}
    </div>
  </header>

  <main>
    <section class="hero">
      <div>
        <div class="chip">
          <span class="chip-dot"></span>
          <span>Picture in → stitchable pattern out</span>
        </div>
        <h1>Turn photos into stitchable patterns</h1>
        <p class="hero-tagline">
          PatternCraft.app turns your artwork into three kinds of charts: classic cross‑stitch,
          knitting colorwork, and simple embroidery line art. Upload once – get a clean grid,
          color legend, and ready‑to‑print files.
        </p>

        <div class="hero-cta-row">
          {% if user %}
            <a class="pill" href="#make">Start a new pattern</a>
            <a class="pill secondary" href="/patterns">View saved patterns</a>
          {% else %}
            <a class="pill" href="/signup">Start free pattern account</a>
            <a class="pill secondary" href="/login">Already have an account? Log in</a>
          {% endif %}
        </div>
        <p class="hero-note">
          Free account includes unlimited full‑size patterns and legends. Patterns are kept for about a week
          under <strong>My patterns</strong>. We also send an optional monthly pattern ideas email you can
          unsubscribe from at any time.
        </p>
        <div class="badge-row">
          <span class="badge">Built by and for stitchers</span>
          <span class="badge">Cross‑stitch · knitting · simple embroidery</span>
        </div>
      </div>

      <aside class="card">
        <h2 class="section-title">Why makers use PatternCraft.app</h2>
        <ul class="why-list">
          <li>Clean grids with bold 10×10 guides and optional symbol overlays</li>
          <li>Color legends with hex and RGB values for accurate palettes</li>
          <li>Fabric size estimates based on stitch count and cloth count</li>
          <li>Knitting charts where each square is one stitch for colorwork</li>
          <li>Embroidery line outputs ready for your machine software</li>
        </ul>
      </aside>
    </section>

    <section id="make" class="card make-card">
      <h2 class="section-title">Make a pattern</h2>
      <p class="controls-note">
        Pick your pattern type (cross‑stitch, knitting, or simple embroidery line art), adjust a few settings,
        and download a ZIP with your grid, legend, and metadata.
      </p>
      <div class="make-layout">
        <div class="make-main">
          <form method="POST" action="/api/convert" enctype="multipart/form-data">
            {% if user %}
              <label class="file" id="fileLabel">
                <input id="fileInput" type="file" name="file" accept="image/*" required onchange="pickFile(this)">
                <div>
                  <div class="file-label-main" id="fileLabelText">UPLOAD PICTURE HERE</div>
                  <div class="file-label-sub">
                    Drop in your artwork or tap to browse from your device.
                  </div>
                </div>
              </label>
            {% else %}
              <button type="button" class="file" onclick="window.location.href='/login?msg=Log+in+or+create+a+free+account+to+upload+an+image.'">
                <div>
                  <div class="file-label-main">UPLOAD PICTURE HERE</div>
                  <div class="file-label-sub">
                    You’ll be asked to create a free account or log in before generating a pattern.
                  </div>
                </div>
              </button>
            {% endif %}

            {% if not user %}
              <p class="controls-note" style="margin-top:8px;">
                New here? <a href="/signup">Create a free account</a>. Already joined? <a href="/login">Log in</a>.
              </p>
            {% endif %}

            {% if user %}
            <fieldset>
              <legend>Pattern type</legend>
              <label><input type="radio" name="ptype" value="cross" checked> Cross‑stitch chart</label>
              <label style="margin-left:12px"><input type="radio" name="ptype" value="knit"> Knitting colorwork chart</label>
              <label style="margin-left:12px"><input type="radio" name="ptype" value="emb"> Simple embroidery line art</label>
              <p class="controls-note" style="margin-top:6px;">
                Cross‑stitch and knitting options give you a true grid (one square = one stitch).
                Embroidery creates a clean run‑stitch line drawing.
              </p>
            </fieldset>

            <fieldset>
              <legend>Stitch & size</legend>
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
              <p class="controls-note">Defaults work well for most art. Adjust once you know your preferences.</p>
            </fieldset>

            <fieldset id="crossKnitBlock">
              <legend>Fabric & floss</legend>
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
              <label><input type="checkbox" name="symbols" checked> Symbol overlay on grid</label>
              <label style="margin-left:10px"><input type="checkbox" name="pdf" checked> Also export PDF</label>
              <p class="controls-note">
                Symbols are optional – uncheck if you prefer color‑only charts. PDF always matches what you see on the grid.
              </p>
            </fieldset>

            <fieldset id="embBlock" class="hidden">
              <legend>Embroidery options</legend>
              <p class="controls-note">
                Creates a simple run‑stitch path. For advanced digitizing, continue in your embroidery software.
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

            <div class="pill-row">
              <button class="pill" id="generateBtn" type="submit">Generate pattern ZIP</button>
              <span>
                Download includes <strong>grid.png</strong>, <strong>legend.csv</strong>, and <strong>meta.json</strong>,
                plus an optional PDF or embroidery files.
              </span>
            </div>
            {% endif %}
          </form>
        </div>
      </div>
    </section>

    {% if user and patterns %}
    <section class="card" style="margin-top:22px">
      <h2 class="section-title">Recent patterns</h2>
      <p class="controls-note">
        Patterns you’ve created in the last week. Older patterns roll off automatically.
      </p>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:10px">
        {% for p in patterns %}
          <article style="border-radius:14px;border:1px solid rgba(250,204,21,.8);padding:10px;background:#fffbeb;">
            <div style="font-size:13px;font-weight:600;margin-bottom:4px;">
              {{ p.meta.type|default('pattern')|capitalize }} · {{ p.meta.stitches_w|default('?') }}×{{ p.meta.stitches_h|default('?') }} stitches
            </div>
            <div style="font-size:12px;color:#6b7280;margin-bottom:6px;">
              {{ p.meta.colors|default('?') }} colors · saved recently
            </div>
            {% if p.has_grid %}
              <a href="/patterns/{{ p.id }}" style="display:block;border-radius:10px;overflow:hidden;border:1px solid #facc15;background:#fff;">
                <img src="/patterns/{{ p.id }}/image" alt="Pattern preview" style="width:100%;display:block;">
              </a>
            {% endif %}
            <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
              <a class="pill secondary" style="padding:8px 14px;font-size:13px" href="/patterns/{{ p.id }}">Open</a>
              <a class="pill" style="padding:8px 14px;font-size:13px" href="/patterns/{{ p.id }}/download">Download ZIP</a>
            </div>
          </article>
        {% endfor %}
      </div>
      <p class="controls-note" style="margin-top:10px;">
        Need a pattern long‑term? Keep the ZIP you download – it includes everything you need to re‑print later.
      </p>
    </section>
    {% endif %}
  </main>
</div>

<script>
  function pickFile(inp){
    const wrapper = document.getElementById('fileLabel');
    const label = document.getElementById('fileLabelText');
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
    const typeInput = document.querySelector('input[name="ptype"]:checked');
    const type = typeInput ? typeInput.value : 'cross';
    const crossKnit = document.getElementById('crossKnitBlock');
    const emb = document.getElementById('embBlock');
    if(type === 'emb'){
      if (crossKnit) crossKnit.classList.add('hidden');
      if (emb) emb.classList.remove('hidden');
    } else {
      if (crossKnit) crossKnit.classList.remove('hidden');
      if (emb) emb.classList.add('hidden');
    }
    setStyleOptions(type);
  }

  document.querySelectorAll('input[name="ptype"]').forEach(r => {
    r.addEventListener('change', onTypeChange);
  });
  onTypeChange();
</script>
</body>
</html>"""


SIGNUP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your free account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <style>
    body{margin:0;background:#FFF9E6;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#fff;border-radius:18px;
      border:1px solid #fde68a;padding:22px;
      box-shadow:0 20px 50px rgba(15,23,42,.2)
    }
    h1{margin:0 0 10px;font-size:1.7rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #e5e7eb;font-size:14px;background:#fffbeb;
    }
    input:focus{
      outline:none;border-color:#f59e0b;box-shadow:0 0 0 1px rgba(245,158,11,.45);
      background:#fff;
    }
    .pill{
      margin-top:16px;padding:10px 20px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#f97316,#ea580c);color:#fff;
      font-size:15px;font-weight:600;cursor:pointer;
      box-shadow:0 18px 42px rgba(248,113,22,.55);
      width:100%;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 24px 55px rgba(248,113,22,.65);}
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4b5563;padding-left:18px;margin-top:10px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Create your free PatternCraft.app account</h1>
    <p class="muted">
      Your free account gives you unlimited full‑size patterns and legends. Patterns you generate are saved
      under <strong>My patterns</strong> for about a week so you can re‑download or print again.
    </p>
    <ul>
      <li>Use a real email address – it’s how you log back in.</li>
      <li>We send at most one monthly publication with pattern ideas and tips. You can unsubscribe any time.</li>
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
      Already have an account? <a href="/login">Log in</a>.
    </p>
  </div>
</div>
</body>
</html>"""


LOGIN_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Log in — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <style>
    body{margin:0;background:#FFF9E6;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#fff;border-radius:18px;
      border:1px solid #fde68a;padding:22px;
      box-shadow:0 20px 50px rgba(15,23,42,.2)
    }
    h1{margin:0 0 10px;font-size:1.7rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #e5e7eb;font-size:14px;background:#fffbeb;
    }
    input:focus{
      outline:none;border-color:#f59e0b;box-shadow:0 0 0 1px rgba(245,158,11,.45);
      background:#fff;
    }
    .pill{
      margin-top:16px;padding:10px 20px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#4c51bf,#4338ca);color:#fff;
      font-size:15px;font-weight:600;cursor:pointer;
      box-shadow:0 18px 42px rgba(79,70,229,.55);
      width:100%;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 24px 55px rgba(79,70,229,.7);}
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Log in to PatternCraft.app</h1>
    <p class="muted">
      Use the email and password you created for your free PatternCraft.app account.
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
    {% if attempts_left is not none %}
      <p class="muted" style="margin-top:6px;">
        {% if attempts_left > 0 %}
          You have {{ attempts_left }} more attempt{{ 's' if attempts_left != 1 else '' }} before we suggest creating a new account.
        {% else %}
          If you’re having trouble logging in, you can create a new account with your email.
        {% endif %}
      </p>
    {% endif %}
    <p class="muted" style="margin-top:10px;">
      New here? <a href="/signup">Create a free account</a> for unlimited patterns.
    </p>
  </div>
</div>
</body>
</html>"""


PATTERNS_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>My patterns — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <style>
    body{
      margin:0;background:#FFF9E6;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827
    }
    .wrap{max-width:1120px;margin:0 auto;padding:24px 16px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand-mark{display:flex;align-items:center;gap:8px;}
    .brand-icon{
      width:26px;height:26px;border-radius:9px;
      background:linear-gradient(135deg,#fbbf24,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:13px;font-weight:800;color:#111827;
      box-shadow:0 8px 18px rgba(248,181,0,.45);
    }
    .brand-text{font-weight:800;font-size:18px;letter-spacing:.06em;text-transform:uppercase;}
    .nav-link{
      font-size:13px;color:#4b5563;
      padding:6px 10px;border-radius:999px;
      background:rgba(255,255,255,.85);border:1px solid rgba(248,181,0,.35);
      text-decoration:none;
      margin-left:6px;
    }
    .nav-link:hover{
      background:#fff7cc;
      box-shadow:0 6px 16px rgba(248,181,0,.35);
    }
    h1{margin:0 0 8px;font-size:1.9rem}
    .muted{font-size:13px;color:#6b7280}
    .card{
      background:#fff;border-radius:18px;
      border:1px solid #fde68a;padding:20px;
      box-shadow:0 18px 48px rgba(15,23,42,.18)
    }
  </style>
</head>
<body>
<div class="wrap">
  <header class="topbar">
    <div class="brand-mark">
      <div class="brand-icon">PC</div>
      <div class="brand-text">PatternCraft.app</div>
    </div>
    <div>
      <a class="nav-link" href="/">Tool</a>
      <a class="nav-link" href="/logout">Sign out</a>
    </div>
  </header>

  <section class="card">
    <h1>My patterns</h1>
    <p class="muted">
      Patterns are kept here for about a week after you generate them. Download the ZIP for long‑term storage.
    </p>

    {% if not patterns %}
      <p class="muted" style="margin-top:10px;">
        You haven’t generated any patterns yet. <a href="/">Start a new pattern</a>.
      </p>
    {% else %}
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-top:12px">
        {% for p in patterns %}
          <article style="border-radius:14px;border:1px solid rgba(250,204,21,.8);padding:10px;background:#fffbeb;">
            <div style="font-size:13px;font-weight:600;margin-bottom:4px;">
              {{ p.meta.type|default('pattern')|capitalize }} · {{ p.meta.stitches_w|default('?') }}×{{ p.meta.stitches_h|default('?') }} stitches
            </div>
            <div style="font-size:12px;color:#6b7280;margin-bottom:6px;">
              {{ p.meta.colors|default('?') }} colors
            </div>
            {% if p.has_grid %}
              <a href="/patterns/{{ p.id }}" style="display:block;border-radius:10px;overflow:hidden;border:1px solid #facc15;background:#fff;">
                <img src="/patterns/{{ p.id }}/image" alt="Pattern preview" style="width:100%;display:block;">
              </a>
            {% endif %}
            <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
              <a class="nav-link" style="background:#fff;padding:7px 12px;font-size:13px" href="/patterns/{{ p.id }}">Open</a>
              <a class="nav-link" style="background:#f97316;color:#fff;border-color:#ea580c;padding:7px 12px;font-size:13px" href="/patterns/{{ p.id }}/download">Download ZIP</a>
            </div>
          </article>
        {% endfor %}
      </div>
    {% endif %}
  </section>
</div>
</body>
</html>"""


PATTERN_DETAIL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pattern detail — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <style>
    body{
      margin:0;background:#FFF9E6;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827
    }
    .wrap{max-width:1120px;margin:0 auto;padding:24px 16px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand-mark{display:flex;align-items:center;gap:8px;}
    .brand-icon{
      width:26px;height:26px;border-radius:9px;
      background:linear-gradient(135deg,#fbbf24,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:13px;font-weight:800;color:#111827;
      box-shadow:0 8px 18px rgba(248,181,0,.45);
    }
    .brand-text{font-weight:800;font-size:18px;letter-spacing:.06em;text-transform:uppercase;}
    .nav-link{
      font-size:13px;color:#4b5563;
      padding:6px 10px;border-radius:999px;
      background:rgba(255,255,255,.85);border:1px solid rgba(248,181,0,.35);
      text-decoration:none;
      margin-left:6px;
    }
    .nav-link:hover{background:#fff7cc;box-shadow:0 6px 16px rgba(248,181,0,.35);}
    .layout{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2fr);gap:20px}
    .card{
      background:#fff;border-radius:18px;
      border:1px solid #fde68a;padding:18px;
      box-shadow:0 18px 48px rgba(15,23,42,.18)
    }
    h1{margin:0 0 8px;font-size:1.7rem}
    .muted{font-size:13px;color:#6b7280}
    .meta-list{font-size:13px;color:#374151;list-style:none;margin:0;padding:0}
    .meta-list li{margin-bottom:3px;}
  </style>
</head>
<body>
<div class="wrap">
  <header class="topbar">
    <div class="brand-mark">
      <div class="brand-icon">PC</div>
      <div class="brand-text">PatternCraft.app</div>
    </div>
    <div>
      <a class="nav-link" href="/">Tool</a>
      <a class="nav-link" href="/patterns">My patterns</a>
      <a class="nav-link" href="/logout">Sign out</a>
    </div>
  </header>

  <main class="layout">
    <section class="card">
      <h1>Pattern preview</h1>
      <p class="muted">
        This is the stitched grid saved with this pattern. Use the legend and details on the right
        when stitching or re‑printing.
      </p>
      {% if has_grid %}
        <div style="border-radius:14px;overflow:hidden;border:1px solid #facc15;background:#fff;margin-top:10px;max-height:600px;">
          <img src="/patterns/{{ pattern_id }}/image" alt="Pattern grid" style="width:100%;display:block;">
        </div>
      {% else %}
        <p class="muted">No grid image was stored for this pattern.</p>
      {% endif %}
      <div style="margin-top:10px;">
        <a class="nav-link" style="background:#f97316;color:#fff;border-color:#ea580c;padding:7px 12px;font-size:13px" href="/patterns/{{ pattern_id }}/download">
          Download pattern ZIP
        </a>
        {% if has_pdf %}
          <span class="muted" style="margin-left:6px;">ZIP includes a printable PDF version.</span>
        {% endif %}
      </div>
    </section>

    <aside class="card">
      <h2 style="margin:0 0 6px;font-size:1.2rem">Details & legend</h2>
      <ul class="meta-list">
        <li><strong>Type:</strong> {{ meta.type|default('pattern')|capitalize }}</li>
        <li><strong>Stitch style:</strong> {{ meta.stitch_style|default('standard') }}</li>
        <li><strong>Stitch size:</strong> {{ meta.stitches_w|default('?') }}×{{ meta.stitches_h|default('?') }} stitches</li>
        {% if meta.cloth_count %}
          <li><strong>Cloth count:</strong> {{ meta.cloth_count }} stitches / inch</li>
        {% endif %}
        {% if meta.finished_size_in %}
          <li><strong>Approx. finished size:</strong> {{ meta.finished_size_in[0] }}″ × {{ meta.finished_size_in[1] }}″</li>
        {% endif %}
        {% if meta.colors %}
          <li><strong>Colors:</strong> {{ meta.colors }}</li>
        {% endif %}
      </ul>
      {% if meta.notes %}
        <p class="muted" style="margin-top:8px;">{{ meta.notes }}</p>
      {% endif %}

      {% if has_legend %}
        <h3 style="margin-top:14px;font-size:1rem">Legend</h3>
        <p class="muted" style="margin-top:2px;">
          Download the full legend as CSV:
          <a href="/patterns/{{ pattern_id }}/legend.csv">legend.csv</a>
        </p>
      {% else %}
        <p class="muted" style="margin-top:10px;">No legend was stored for this pattern.</p>
      {% endif %}
    </aside>
  </main>
</div>
</body>
</html>"""


if __name__ == "__main__":
    app.run(debug=True)
