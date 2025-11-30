from __future__ import annotations
import base64
import io
import json
import math
import os
import uuid
import zipfile
from datetime import datetime, timedelta
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
from werkzeug.security import generate_password_hash, check_password_hash

# Optional embroidery support
try:
    from pyembroidery import EmbPattern, write_dst, write_pes  # type: ignore

    HAS_PYEMB = True
except Exception:
    HAS_PYEMB = False

# -------------------------------------------------------------------
# App + storage paths
# -------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PATTERN_DIR = os.path.join(DATA_DIR, "patterns")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PATTERN_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # px
PATTERN_TTL_DAYS = 7

# Simple PNG favicon (pastel tile with "PC" monogram)
FAVICON_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAABYUlEQVR4nO2bQW7CMBREQ8TtegQ2bLkB"
    "lsJ2EU8Ab4V7sjG+1SKBB2xSlI5OYj4Q3AjjaN+6XuPDvzM+E0BMMEHH3wG8QV4TcdlOLFtuQ2wHqghB"
    "6dPlZ2AxBj8xNQJIEh4Jc0Fb6MFJYWxCbBexfYk8BZ4T8U6vLZrJPGnHtVbLoVbKQ4ttc2eyh7bnm5L1"
    "rSgj1tqj9Q1HNHuYtYc9TMekb9pQzFtXaVDWKi+J8mCzP6q/S7GF7YkLOI09rXug++oqNyEt9/t4GkEz"
    "kI2jxnMfFMGXWTpX7GuCYNqD7HS4r0DNZ0D/R6Gmdm7JR8msx4O/o7qGL29bpYWd0mq9r03hmkQX4rJX"
    "qKt6tVaS7mR9Tqv+1nC3+OsaQJ4bLf2pcYnBsduqNHTwfzjoLVq9JrS1Ug2eYgzAfH2yD1juOOcPsZji"
    "qE5Zwv0ETxZotjnc+oOsRk0Uf+H/H3zGfp6g19TRvAe+08GWShhQGQAAAABJRU5ErkJggg=="
)

# -------------------------------------------------------------------
# User + pattern storage helpers
# -------------------------------------------------------------------


def _ensure_users_structure(data: dict) -> Dict[str, dict]:
    """Ensure all users have expected fields."""
    if not isinstance(data, dict):
        return {}
    for email, u in data.items():
        if not isinstance(u, dict):
            data[email] = {}
            u = data[email]
        u.setdefault("email", email)
        u.setdefault("password_hash", "")
        u.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
        u.setdefault("patterns", [])  # list of pattern dicts
    return data  # type: ignore[return-value]


def prune_old_patterns(users: Dict[str, dict], days: int = PATTERN_TTL_DAYS) -> bool:
    """Drop patterns older than TTL and delete their files."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    changed = False

    for u in users.values():
        patterns = u.get("patterns") or []
        new_patterns = []
        for p in patterns:
            created_str = p.get("created_at") or ""
            try:
                created = datetime.fromisoformat(created_str.replace("Z", ""))
            except Exception:
                created = datetime.utcnow()
            if created >= cutoff:
                new_patterns.append(p)
                continue

            # delete files for expired pattern
            for key in ("zip_path", "preview_path"):
                path = p.get(key)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            changed = True

        if new_patterns != patterns:
            u["patterns"] = new_patterns

    return changed


def load_users() -> Dict[str, dict]:
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    except Exception:
        data = {}

    data = _ensure_users_structure(data)
    if prune_old_patterns(data):
        save_users(data)
    return data


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


def save_pattern_for_user(
    email: str,
    meta: dict,
    zip_bytes: bytes,
    grid_png_bytes: bytes,
) -> None:
    users = load_users()
    user = users.setdefault(email, {"email": email, "password_hash": "", "patterns": []})

    pattern_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat() + "Z"

    zip_path = os.path.join(PATTERN_DIR, f"{pattern_id}.zip")
    preview_path = os.path.join(PATTERN_DIR, f"{pattern_id}_grid.png")

    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    with open(preview_path, "wb") as f:
        f.write(grid_png_bytes)

    record = {
        "id": pattern_id,
        "created_at": created_at,
        "ptype": meta.get("type", ""),
        "stitches_w": meta.get("stitches_w"),
        "stitches_h": meta.get("stitches_h"),
        "colors": meta.get("colors"),
        "zip_path": zip_path,
        "preview_path": preview_path,
        "meta": meta,
    }
    patterns = user.get("patterns") or []
    patterns.insert(0, record)
    user["patterns"] = patterns
    users[email] = user
    save_users(users)


def find_user_pattern(email: str, pattern_id: str) -> Optional[dict]:
    users = load_users()
    u = users.get(email)
    if not u:
        return None
    for p in u.get("patterns", []):
        if p.get("id") == pattern_id:
            return p
    return None


# -------------------------------------------------------------------
# Image / pattern helpers
# -------------------------------------------------------------------


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def open_image(fs) -> Image.Image:
    img = Image.open(fs.stream)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def resize_for_stitch_width(img: Image.Image, stitch_w: int) -> Image.Image:
    w, h = img.size
    if max(w, h) > 2000:
        img = img.copy()
        img.thumbnail((2000, 2000))
        w, h = img.size
    ratio = stitch_w / float(w)
    new_h = max(1, int(round(h * ratio)))
    return img.resize((stitch_w, new_h), Image.Resampling.LANCZOS)


def knit_aspect_resize(img: Image.Image, stitches_w: int, row_aspect: float = 0.8) -> Image.Image:
    resized = resize_for_stitch_width(img, stitches_w)
    w, h = resized.size
    preview_h = max(1, int(round(h * row_aspect)))
    return resized.resize((w, preview_h), Image.Resampling.NEAREST)


def quantize(img: Image.Image, k: int) -> Image.Image:
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
    glyphs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+*#@&%=?/\\^~<>□■●▲◆★✚")
    return {c: glyphs[i % len(glyphs)] for i, c in enumerate(colors)}


def draw_symbols_on_grid(
    base: Image.Image, cell_px: int, sym_map: Dict[Tuple[int, int, int], str]
) -> Image.Image:
    sx, sy = base.size
    out = base.resize((sx * cell_px, sy * cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    # symbols
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

    # grid lines (no diagonals, no overlays)
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


def to_monochrome(img: Image.Image, threshold: int = 180) -> Image.Image:
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p > threshold else 0, mode="1")
    return bw.convert("L")


def serpentine_points(bw: Image.Image, step: int = 3) -> List[Tuple[int, int]]:
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


# -------------------------------------------------------------------
# Basic routes / errors
# -------------------------------------------------------------------


@app.get("/favicon.ico")
def favicon():
    data = base64.b64decode(FAVICON_PNG_BASE64)
    return send_file(io.BytesIO(data), mimetype="image/png")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


ERROR_500_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>We hit a snag — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#FFF8D8;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{background:#fff;border-radius:18px;border:1px solid #fde68a;padding:24px;box-shadow:0 12px 35px rgba(15,23,42,.18)}
    h1{margin:0 0 8px;font-size:1.7rem}
    p{margin:6px 0;font-size:14px;color:#4b5563}
    a{color:#b45309;text-decoration:none;font-weight:600}
    a:hover{text-decoration:underline}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>We hit a snag</h1>
      <p>Something went wrong while processing your request.</p>
      <p>You can go back to <a href="/">PatternCraft.app</a> and try again.</p>
    </div>
  </div>
</body>
</html>
"""


@app.errorhandler(500)
def on_error(_e):
    return make_response(render_template_string(ERROR_500_HTML), 500)


@app.errorhandler(404)
def not_found(_e):
    return make_response(render_template_string(ERROR_500_HTML), 404)


# -------------------------------------------------------------------
# Auth
# -------------------------------------------------------------------


@app.get("/signup")
def signup() -> str:
    user = get_current_user()
    if user:
        return redirect(url_for("index"))
    msg = request.args.get("msg", "")
    return render_template_string(SIGNUP_HTML, message=msg)


@app.post("/signup")
def signup_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    if not email or "@" not in email:
        return render_template_string(SIGNUP_HTML, message="Please enter a valid email address.")
    if len(password) < 8:
        return render_template_string(SIGNUP_HTML, message="Password must be at least 8 characters.")
    if password != confirm:
        return render_template_string(SIGNUP_HTML, message="Passwords do not match.")

    users = load_users()
    if email in users and users[email].get("password_hash"):
        return render_template_string(
            SIGNUP_HTML,
            message="This email already has an account. Try logging in instead.",
        )

    users[email] = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "patterns": [],
    }
    save_users(users)
    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/login")
def login() -> str:
    user = get_current_user()
    if user:
        return redirect(url_for("index"))
    msg = request.args.get("msg", "")
    return render_template_string(LOGIN_HTML, message=msg)


@app.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        return render_template_string(
            LOGIN_HTML,
            message="Please enter both email and password.",
        )

    users = load_users()
    stored = users.get(email)
    if not stored or not check_password_hash(stored.get("password_hash", ""), password):
        return render_template_string(
            LOGIN_HTML,
            message="We couldn’t find that email and password. Double‑check and try again.",
        )

    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("index"))


# -------------------------------------------------------------------
# Patterns: generate + list + preview + download
# -------------------------------------------------------------------


@app.get("/")
def index() -> str:
    user = get_current_user()
    return render_template_string(INDEX_HTML, user=user)


@app.post("/api/convert")
def convert():
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in or create a free account to generate patterns."))

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
    grid_png_bytes: Optional[bytes] = None
    meta: dict = {}

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

            pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
            if want_symbols:
                sym_map = assign_symbols(pal)
                grid_img = draw_symbols_on_grid(quant, cell_px=CELL_PX, sym_map=sym_map)
            else:
                grid_img = draw_grid(quant, cell_px=CELL_PX)

            # legend.csv
            total_stitches = sum(counts.values()) or 1
            lines = ["hex,r,g,b,stitches,percent,skeins_est"]
            for (r, g, b), c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                skeins = skeins_per_color(c, cloth_count, strands, waste_pct / 100.0)
                lines.append(
                    f"{to_hex((r,g,b))},{r},{g},{b},{c},{(100*c/total_stitches):.2f},{skeins:.2f}"
                )
            z.writestr("legend.csv", "\n".join(lines))

            note = (
                "Knitting chart. Row height visually compressed; always check your gauge."
                if ptype == "knit"
                else "Cross‑stitch grid with 10×10 guides."
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
            grid_png_bytes = buf_png.getvalue()
            z.writestr("grid.png", grid_png_bytes)

            if want_pdf:
                pdf_buf = io.BytesIO()
                grid_img.convert("RGB").save(pdf_buf, format="PDF", resolution=300.0)
                z.writestr("pattern.pdf", pdf_buf.getvalue())

        elif ptype == "emb":
            small = resize_for_stitch_width(base, stitch_w)
            bw = to_monochrome(small, threshold=emb_thresh)
            pts = serpentine_points(bw, step=emb_step)
            for name, data in write_embroidery_outputs(pts).items():
                z.writestr(name, data)
            meta = {
                "type": "emb",
                "stitch_style": "run",
                "points": len(pts),
                "pyembroidery": HAS_PYEMB,
                "notes": "Simple line‑art embroidery run path.",
            }
            z.writestr("meta.json", json.dumps(meta, indent=2))

            # preview: just the monochrome image scaled to cells, no extra overlays
            grid_img = draw_grid(bw.convert("RGB"), cell_px=CELL_PX)
            buf_png = io.BytesIO()
            grid_img.save(buf_png, format="PNG")
            grid_png_bytes = buf_png.getvalue()
            z.writestr("grid.png", grid_png_bytes)
        else:
            return jsonify({"error": "unknown_ptype"}), 400

    out_zip.seek(0)
    zip_bytes = out_zip.getvalue()

    if grid_png_bytes is not None:
        save_pattern_for_user(email, meta, zip_bytes, grid_png_bytes)

    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{ptype}.zip",
    )


@app.get("/patterns")
def list_patterns():
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to see your saved patterns."))

    users = load_users()
    user = users.get(email) or {"patterns": []}
    patterns = user.get("patterns") or []
    return render_template_string(PATTERNS_HTML, user=user, patterns=patterns)


@app.get("/patterns/<pattern_id>")
def pattern_detail(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login"))

    pat = find_user_pattern(email, pattern_id)
    if not pat:
        return make_response(render_template_string(ERROR_500_HTML), 404)

    return render_template_string(PATTERN_DETAIL_HTML, pattern=pat)


@app.get("/patterns/<pattern_id>/image")
def pattern_image(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login"))

    pat = find_user_pattern(email, pattern_id)
    if not pat:
        return make_response(render_template_string(ERROR_500_HTML), 404)

    path = pat.get("preview_path")
    if not path or not os.path.exists(path):
        return make_response(render_template_string(ERROR_500_HTML), 404)

    return send_file(path, mimetype="image/png")


@app.get("/patterns/<pattern_id>/download")
def pattern_download(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login"))

    pat = find_user_pattern(email, pattern_id)
    if not pat:
        return make_response(render_template_string(ERROR_500_HTML), 404)

    path = pat.get("zip_path")
    if not path or not os.path.exists(path):
        return make_response(render_template_string(ERROR_500_HTML), 404)

    return send_file(path, mimetype="application/zip", as_attachment=True)


# -------------------------------------------------------------------
# Inline HTML templates
# -------------------------------------------------------------------

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <style>
    :root{
      --bg:#FFF8D8;--fg:#222;--muted:#6b7280;
      --line:#fde68a;--radius:18px;--shadow:0 14px 40px rgba(15,23,42,.20);
      --accent:#f59e0b;--accent-dark:#b45309;--accent-soft:#fffbeb;
      --pill:#f97316;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:var(--fg);
      background:radial-gradient(circle at top,#fffbe6 0,#fff1c1 40%,#fff8d8 70%);
    }
    a{color:var(--accent-dark);text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1040px;margin:0 auto;padding:24px 16px 48px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:22px;
    }
    .brand{
      font-weight:900;font-size:20px;letter-spacing:.12em;
      text-transform:uppercase;color:#92400e;
    }
    .nav{font-size:14px;color:#78350f;display:flex;gap:12px;align-items:center;}
    .nav a{font-weight:500;}
    .hero{
      display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2fr);
      gap:22px;margin-bottom:28px;align-items:center;
    }
    .card{
      background:#fff;border-radius:var(--radius);
      border:1px solid var(--line);
      box-shadow:var(--shadow);
      padding:22px;
    }
    h1{font-size:2.4rem;margin:0 0 10px;color:#92400e;}
    h2{margin:0 0 10px;font-size:1.2rem;color:#92400e;}
    .hero-tagline{color:var(--muted);max-width:460px;font-size:15px;}
    .pill{
      padding:12px 24px;border-radius:999px;
      background:linear-gradient(135deg,#facc15,#f97316);
      color:#78350f;border:none;cursor:pointer;
      font-size:15px;font-weight:700;letter-spacing:.02em;
      box-shadow:0 10px 26px rgba(245,158,11,.65);
      transition:transform .08s,box-shadow .08s;
      display:inline-flex;align-items:center;justify-content:center;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 34px rgba(245,158,11,.8);}
    .pill.secondary{
      background:#fff;color:#92400e;border:2px solid #fbbf24;
      box-shadow:0 6px 18px rgba(245,158,11,.35);
    }
    .pill.secondary:hover{box-shadow:0 8px 22px rgba(245,158,11,.5);}
    .hero-cta{display:flex;gap:12px;margin-top:14px;flex-wrap:wrap;}
    .hero-note{font-size:13px;color:#6b7280;margin-top:8px;}
    .why{
      font-size:14px;color:#4b5563;padding-left:18px;margin:8px 0 0;
    }
    .why li{margin:4px 0;}
    .badge-row{margin-top:10px;font-size:12px;color:#92400e;}
    .badge{
      display:inline-block;margin-right:8px;margin-bottom:4px;
      padding:5px 10px;border-radius:999px;
      background:#fffbeb;border:1px solid #fed7aa;
      font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.08em;
    }
    .tool-card{margin-top:8px;}
    .file{
      border:2px dashed #fbbf24;
      border-radius:18px;
      padding:18px;
      display:flex;align-items:center;gap:12px;
      cursor:pointer;
      background:#fffbeb;
      transition:background .15s,border-color .15s,transform .1s,box-shadow .1s;
    }
    .file:hover{
      background:#fef9c3;border-color:#f59e0b;
      transform:translateY(-1px);
      box-shadow:0 8px 22px rgba(245,158,11,.45);
    }
    .file input{display:none;}
    .file-label-main{
      font-weight:800;font-size:14px;text-transform:uppercase;letter-spacing:.12em;color:#92400e;
    }
    .file-label-sub{font-size:12px;color:#6b7280;}
    fieldset{border:1px solid #fed7aa;border-radius:14px;padding:12px;margin:10px 0;}
    legend{font-size:13px;padding:0 6px;color:#92400e;font-weight:600;}
    .row{display:flex;flex-wrap:wrap;gap:12px;}
    .row > label{flex:1 1 150px;font-size:13px;}
    .row input,.row select{
      width:100%;margin-top:4px;padding:7px 9px;border-radius:10px;
      border:1px solid #facc15;font-size:13px;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#f97316;box-shadow:0 0 0 1px rgba(249,115,22,.45);
    }
    label{font-size:13px;color:#374151;}
    .controls-note{font-size:11px;color:#9ca3af;margin-top:4px;}
    .hidden{display:none;}
    .muted{font-size:13px;color:#6b7280;}
    @media (max-width:860px){
      .hero{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">PATTERNCRAFT.APP</div>
    <div class="nav">
      {% if user %}
        <span>Signed in as {{ user.email }}</span>
        <a href="/patterns">My patterns</a>
        <a href="/logout">Sign out</a>
      {% else %}
        <a href="/login">Log in</a>
        <a href="/signup"><strong>Create Free Account</strong></a>
      {% endif %}
    </div>
  </div>

  <div class="hero">
    <div>
      <h1>Turn pictures into stitch‑ready patterns.</h1>
      <p class="hero-tagline">
        Upload a photo or artwork and PatternCraft.app translates it into a clear grid:
        cross‑stitch, knitting chart, or simple embroidery line art — with color legend and sizing info.
      </p>
      {% if not user %}
      <div class="hero-cta">
        <a class="pill" href="/signup">Get Started!</a>
        <a class="pill secondary" href="/login">Already have an account? Log in</a>
      </div>
      <p class="hero-note">
        Free account includes unlimited full‑size patterns and a monthly pattern ideas email.
      </p>
      {% else %}
      <div class="hero-cta">
        <a class="pill" href="#tool">Make a new pattern</a>
        <a class="pill secondary" href="/patterns">View saved patterns</a>
      </div>
      <p class="hero-note">
        You’re signed in. Upload art below to generate a new pattern, or revisit anything in <a href="/patterns">My patterns</a>.
      </p>
      {% endif %}

      <div class="badge-row">
        <span class="badge">Cross‑stitch grids</span>
        <span class="badge">Knitting charts</span>
        <span class="badge">Embroidery line art</span>
      </div>
    </div>

    <div class="card">
      <h2>Why makers use PatternCraft.app</h2>
      <ul class="why">
        <li>Clean grids with bold 10×10 guides and optional symbols.</li>
        <li>Color legends with hex/RGB values for accurate palettes.</li>
        <li>Fabric size estimates based on stitch count and cloth count.</li>
        <li>Knitting charts that respect row proportions.</li>
        <li>Simple line‑art outputs for basic embroidery.</li>
      </ul>
    </div>
  </div>

  {% if user %}
  <div id="tool" class="card tool-card">
    <h2>Make a pattern</h2>
    <p class="muted">
      Choose your pattern type, adjust a few options if needed, and download a full ZIP.
      Each pattern you generate is also saved under <a href="/patterns">My patterns</a> for one week.
    </p>

    <form method="POST" action="/api/convert" enctype="multipart/form-data">
      <label class="file">
        <input id="fileInput" type="file" name="file" accept="image/*" required>
        <div>
          <div class="file-label-main">Upload picture here</div>
          <div class="file-label-sub">
            A clear photo, illustration, or logo works best. Higher resolution gives more detail.
          </div>
        </div>
      </label>

      <fieldset>
        <legend>Pattern type</legend>
        <label><input type="radio" name="ptype" value="cross" checked> Cross‑stitch</label>
        <label style="margin-left:12px"><input type="radio" name="ptype" value="knit"> Knitting chart</label>
        <label style="margin-left:12px"><input type="radio" name="ptype" value="emb"> Simple embroidery line art</label>
      </fieldset>

      <fieldset>
        <legend>Stitch &amp; color controls</legend>
        <div class="row">
          <label>Stitch width (stitches across)
            <input type="number" name="width" value="120" min="20" max="400">
          </label>
          <label>Max colors
            <input type="number" name="colors" value="16" min="2" max="60">
          </label>
          <label>Stitch style
            <select id="stitch_style" name="stitch_style"></select>
          </label>
        </div>
        <p class="controls-note">
          For cross‑stitch, start around 80–150 stitches across. Knitting charts compress row height slightly to mimic real fabric.
        </p>
      </fieldset>

      <fieldset id="crossKnitBlock">
        <legend>Fabric &amp; floss / yarn</legend>
        <div class="row">
          <label>Cloth count (stitches / inch)
            <input type="number" name="count" value="14" min="10" max="22">
          </label>
          <label>Strands
            <input type="number" name="strands" value="2" min="1" max="6">
          </label>
          <label>Waste %
            <input type="number" name="waste" value="20" min="0" max="60">
          </label>
        </div>
        <label><input type="checkbox" name="symbols" checked> Symbol overlay on the grid</label>
        <label style="margin-left:10px"><input type="checkbox" name="pdf" checked> Also export a printable PDF</label>
      </fieldset>

      <fieldset id="embBlock" class="hidden">
        <legend>Embroidery options</legend>
        <p class="muted">
          We trace a simple run‑stitch path from your artwork for basic line embroidery. For full digitizing,
          continue in your usual embroidery software.
        </p>
        <div class="row">
          <label>Threshold (0–255)
            <input type="number" name="emb_thresh" value="180" min="0" max="255">
          </label>
          <label>Step size (px)
            <input type="number" name="emb_step" value="3" min="1" max="10">
          </label>
        </div>
      </fieldset>

      <div style="margin-top:14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <button class="pill" type="submit">Generate pattern ZIP</button>
        <span class="muted">
          Download includes <code>grid.png</code>, <code>legend.csv</code>, <code>meta.json</code>, and optional <code>pattern.pdf</code> or embroidery files.
        </span>
      </div>
    </form>
  </div>
  {% else %}
  <div class="card tool-card">
    <h2>Create a free account to start</h2>
    <p class="muted">
      Once you’ve signed up, you can generate unlimited full‑size patterns, and every pattern is saved under
      “My patterns” for a week so you can re‑download or print later.
    </p>
    <div class="hero-cta">
      <a class="pill" href="/signup">Create Free Account</a>
      <a class="pill secondary" href="/login">Log in</a>
    </div>
  </div>
  {% endif %}

</div>
<script>
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

SIGNUP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your free account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <style>
    body{margin:0;background:#FFF8D8;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:36px 16px 44px}
    .card{background:#fff;border-radius:18px;border:1px solid #fde68a;padding:24px;box-shadow:0 14px 40px rgba(15,23,42,.2)}
    h1{margin:0 0 10px;font-size:1.8rem;color:#92400e;}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px;color:#374151;}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #facc15;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#f97316;box-shadow:0 0 0 1px rgba(249,115,22,.45);
    }
    .pill{
      margin-top:16px;padding:11px 22px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#facc15,#f97316);color:#78350f;
      font-size:15px;font-weight:700;cursor:pointer;
      box-shadow:0 10px 26px rgba(245,158,11,.65);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 34px rgba(245,158,11,.8);}
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#b45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4b5563;padding-left:18px;margin-top:10px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Create your free PatternCraft.app account</h1>
    <p class="muted">
      Free membership includes unlimited full‑size pattern conversions and access to “My patterns” for one week
      of recent projects. We’ll also send a low‑key monthly email with pattern ideas — you can unsubscribe any time.
    </p>
    <ul>
      <li>Use an email you check regularly; that’s how you’ll log in.</li>
      <li>Your patterns are tied to this account and visible only to you.</li>
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
  </div>
</div>
</body>
</html>
"""

LOGIN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Log in — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <style>
    body{margin:0;background:#FFF8D8;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:36px 16px 44px}
    .card{background:#fff;border-radius:18px;border:1px solid #fde68a;padding:24px;box-shadow:0 14px 40px rgba(15,23,42,.2)}
    h1{margin:0 0 10px;font-size:1.8rem;color:#92400e;}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px;color:#374151;}
    input{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #facc15;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#f97316;box-shadow:0 0 0 1px rgba(249,115,22,.45);
    }
    .pill{
      margin-top:16px;padding:11px 22px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#facc15,#f97316);color:#78350f;
      font-size:15px;font-weight:700;cursor:pointer;
      box-shadow:0 10px 26px rgba(245,158,11,.65);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 34px rgba(245,158,11,.8);}
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#b45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Log in</h1>
    <p class="muted">
      Use the email and password from when you first created your PatternCraft.app account.
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
      New here? <a href="/signup">Create a free account</a> for unlimited patterns.
    </p>
  </div>
</div>
</body>
</html>
"""

PATTERNS_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>My patterns — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <style>
    body{margin:0;background:#FFF8D8;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:1040px;margin:0 auto;padding:28px 16px 48px}
    .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;}
    .brand{font-weight:900;font-size:18px;letter-spacing:.12em;text-transform:uppercase;color:#92400e;}
    a{color:#b45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .pill{
      padding:9px 18px;border-radius:999px;border:none;
      background:linear-gradient(135deg,#facc15,#f97316);color:#78350f;
      font-size:14px;font-weight:700;cursor:pointer;
      box-shadow:0 8px 22px rgba(245,158,11,.6);
    }
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:10px;}
    .card{background:#fff;border-radius:18px;border:1px solid #fde68a;padding:14px 14px 16px;box-shadow:0 10px 32px rgba(15,23,42,.18);}
    .card h3{margin:0 0 4px;font-size:15px;color:#92400e;}
    .meta{font-size:12px;color:#6b7280;margin-bottom:8px;}
    img{max-width:100%;border-radius:12px;border:1px solid #e5e7eb;display:block;}
    .actions{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;}
    .small{font-size:12px;color:#6b7280;margin-top:2px;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">MY PATTERNS</div>
    <div>
      <a href="/">← Back to tool</a>
      &nbsp;·&nbsp;
      <a href="/logout">Sign out</a>
    </div>
  </div>

  {% if patterns %}
    <p class="small">
      Patterns from the last 7 days are kept here. Older projects drop off automatically to keep things tidy.
    </p>
    <div class="cards">
      {% for p in patterns %}
      <div class="card">
        <h3>{{ p.ptype|upper }} pattern</h3>
        <div class="meta">
          Created {{ p.created_at[:10] }} · {{ p.stitches_w }}×{{ p.stitches_h }} stitches · {{ p.colors }} colors
        </div>
        <a href="/patterns/{{ p.id }}"><img src="/patterns/{{ p.id }}/image" alt="Pattern preview {{ loop.index }}"></a>
        <div class="actions">
          <a class="pill" href="/patterns/{{ p.id }}">View details</a>
          <a class="pill" href="/patterns/{{ p.id }}/download">Download ZIP</a>
        </div>
      </div>
      {% endfor %}
    </div>
  {% else %}
    <p class="small">
      No saved patterns yet. Generate your first pattern from the <a href="/">tool page</a> —
      each new pattern appears here automatically.
    </p>
  {% endif %}
</div>
</body>
</html>
"""

PATTERN_DETAIL_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pattern details — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <style>
    body{margin:0;background:#FFF8D8;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:960px;margin:0 auto;padding:28px 16px 48px}
    a{color:#b45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;}
    .brand{font-weight:900;font-size:18px;letter-spacing:.12em;text-transform:uppercase;color:#92400e;}
    .pill{
      padding:9px 18px;border-radius:999px;border:none;
      background:linear-gradient(135deg,#facc15,#f97316);color:#78350f;
      font-size:14px;font-weight:700;cursor:pointer;
      box-shadow:0 8px 22px rgba(245,158,11,.6);
    }
    .layout{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(260px,1fr);gap:16px;}
    .panel{background:#fff;border-radius:18px;border:1px solid #fde68a;padding:16px;box-shadow:0 10px 32px rgba(15,23,42,.18);}
    img{max-width:100%;border-radius:12px;border:1px solid #e5e7eb;display:block;}
    h2{margin:0 0 8px;font-size:1.1rem;color:#92400e;}
    .meta{font-size:13px;color:#6b7280;margin-bottom:6px;}
    .small{font-size:12px;color:#6b7280;}
    pre{font-size:11px;background:#fffbeb;border-radius:10px;padding:8px;overflow:auto;border:1px solid #fef3c7;}
    @media (max-width:820px){
      .layout{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">PATTERN DETAILS</div>
    <div>
      <a href="/patterns">← Back to My patterns</a>
    </div>
  </div>

  <div class="layout">
    <div class="panel">
      <h2>Pattern grid</h2>
      <div class="meta">
        {{ pattern.ptype|upper }} · {{ pattern.stitches_w }}×{{ pattern.stitches_h }} stitches · {{ pattern.colors }} colors
        <br>Created {{ pattern.created_at[:10] }}
      </div>
      <img src="/patterns/{{ pattern.id }}/image" alt="Pattern preview">
      <p class="small" style="margin-top:8px;">
        This preview matches the <code>grid.png</code> inside your ZIP. Use it to quickly check color balance and chart clarity.
      </p>
    </div>

    <div class="panel">
      <h2>Legend &amp; details</h2>
      <p class="small">
        The ZIP includes <code>legend.csv</code> with hex/RGB values, stitch counts, and skein estimates for each color.
        Use that file to match your own floss, yarn, or fabric palette.
      </p>
      <p class="small" style="margin-top:6px;">
        Download your full pattern package here:
      </p>
      <p>
        <a class="pill" href="/patterns/{{ pattern.id }}/download">Download pattern ZIP</a>
      </p>
      <p class="small" style="margin-top:10px;">
        Quick meta snapshot (from <code>meta.json</code> in your ZIP):
      </p>
      <pre>{{ pattern.meta | tojson(indent=2) }}</pre>
    </div>
  </div>
</div>
</body>
</html>
"""

# -------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

