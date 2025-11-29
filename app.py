from __future__ import annotations

import io
import json
import math
import os
import uuid
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
from werkzeug.security import generate_password_hash, check_password_hash

# Optional embroidery support
try:
    from pyembroidery import EmbPattern, write_dst, write_pes  # type: ignore

    HAS_PYEMB = True
except Exception:
    HAS_PYEMB = False

# ---------------------- APP & STORAGE PATHS ----------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PATTERNS_DIR = os.path.join(BASE_DIR, "user_patterns")
os.makedirs(PATTERNS_DIR, exist_ok=True)

# Config
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels


# ---------------------- UTILS: USERS & PATTERN STORAGE ----------------------


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


def safe_email_key(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_dot_")


def get_user_pattern_dir(email: str) -> str:
    d = os.path.join(PATTERNS_DIR, safe_email_key(email))
    os.makedirs(d, exist_ok=True)
    return d


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
    """Scale each stitch to a cell and overlay a 10×10 grid (no watermark)."""
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
    """Overlay symbol per stitch, then grid (no fade / blur)."""
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
    img: Image.Image, stitches_w: int, row_aspect: float = 0.8
) -> Image.Image:
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


# ---------------------- BASIC ROUTES & ERRORS ----------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


ERROR_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Something went wrong — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;background:#FFFDF2;color:#1F2933;}
    .wrap{max-width:520px;margin:0 auto;padding:40px 16px;}
    .card{background:#fff;border-radius:16px;border:1px solid #F4E5A9;box-shadow:0 12px 30px rgba(15,23,42,.12);padding:22px;}
    h1{margin:0 0 8px;font-size:1.7rem;}
    p{margin:6px 0;font-size:14px;color:#4B5563;}
    a{color:#C97700;text-decoration:none;font-weight:600;}
    a:hover{text-decoration:underline;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>We hit a snag</h1>
    <p>Something went wrong while processing your request.</p>
    <p>You can <a href="/">go back to PatternCraft.app</a> and try again.</p>
  </div>
</div>
</body>
</html>
"""


@app.errorhandler(Exception)
def on_error(e):
    app.logger.exception("Unhandled error: %s", e)
    return make_response(render_template_string(ERROR_HTML), 500)


# ---------------------- MAIN PAGES ----------------------


@app.get("/")
def index() -> str:
    user = get_current_user()
    err = request.args.get("err", "")
    return render_template_string(HOMEPAGE_HTML, user=user, error=err)


@app.get("/patterns")
def my_patterns() -> str:
    user = get_current_user()
    if not user:
        return redirect(
            url_for("login", msg="Log in to see patterns saved to your account.")
        )
    patterns = user.get("patterns", [])
    # newest first
    patterns = sorted(patterns, key=lambda p: p.get("created_at", ""), reverse=True)
    return render_template_string(PATTERNS_HTML, user=user, patterns=patterns)


@app.get("/patterns/<pattern_id>/grid")
def pattern_grid(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    email = user["email"]
    for p in user.get("patterns", []):
        if p.get("id") == pattern_id:
            user_dir = get_user_pattern_dir(email)
            grid_path = os.path.join(user_dir, f"{pattern_id}_grid.png")
            if os.path.exists(grid_path):
                return send_file(grid_path, mimetype="image/png")
    return make_response("Not found", 404)


@app.get("/patterns/<pattern_id>/legend")
def pattern_legend(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    email = user["email"]
    for p in user.get("patterns", []):
        if p.get("id") == pattern_id:
            user_dir = get_user_pattern_dir(email)
            legend_path = os.path.join(user_dir, f"{pattern_id}_legend.csv")
            if os.path.exists(legend_path):
                with open(legend_path, "r", encoding="utf-8") as f:
                    text = f.read()
                return make_response(
                    text, 200, {"Content-Type": "text/plain; charset=utf-8"}
                )
    return make_response("Not found", 404)


@app.get("/patterns/<pattern_id>/download")
def pattern_download(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    email = user["email"]
    for p in user.get("patterns", []):
        if p.get("id") == pattern_id:
            user_dir = get_user_pattern_dir(email)
            zip_path = os.path.join(user_dir, f"{pattern_id}.zip")
            if os.path.exists(zip_path):
                return send_file(
                    zip_path,
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name=f"patterncraft_{pattern_id}.zip",
                )
    return make_response("Not found", 404)


# ---------------------- SIGNUP / LOGIN ----------------------


@app.get("/signup")
def signup() -> str:
    user = get_current_user()
    msg = request.args.get("msg", "")
    return render_template_string(SIGNUP_HTML, user=user, message=msg)


@app.post("/signup")
def signup_post():
    user = get_current_user()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    if not email or "@" not in email:
        return render_template_string(
            SIGNUP_HTML, user=user, message="Please enter a valid email address."
        )
    if len(password) < 8:
        return render_template_string(
            SIGNUP_HTML, user=user, message="Password must be at least 8 characters."
        )
    if password != confirm:
        return render_template_string(
            SIGNUP_HTML, user=user, message="Passwords do not match."
        )

    users = load_users()
    if email in users:
        return render_template_string(
            SIGNUP_HTML,
            user=user,
            message="This email already has an account. Please log in instead.",
        )

    users[email] = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "free_used": False,  # 1 full pattern per account while in beta
        "patterns": [],
    }
    save_users(users)
    session["user_email"] = email
    session["login_failures"] = 0
    return redirect(url_for("index"))


@app.get("/login")
def login() -> str:
    user = get_current_user()
    msg = request.args.get("msg", "")
    failures = int(session.get("login_failures", 0) or 0)
    attempts_left = max(0, 3 - failures)
    return render_template_string(
        LOGIN_HTML,
        user=user,
        message=msg,
        attempts_left=attempts_left,
    )


@app.post("/login")
def login_post():
    user = get_current_user()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    failures = int(session.get("login_failures", 0) or 0)

    if not email or not password:
        attempts_left = max(0, 3 - failures)
        return render_template_string(
            LOGIN_HTML,
            user=user,
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
                    msg="We couldn’t match that email and password after several attempts. Create a free PatternCraft.app account to get started.",
                )
            )
        attempts_left = max(0, 3 - failures)
        return render_template_string(
            LOGIN_HTML,
            user=user,
            message="Incorrect email or password.",
            attempts_left=attempts_left,
        )

    session["user_email"] = email
    session["login_failures"] = 0
    return redirect(url_for("index"))


@app.get("/logout")
def logout():
    session.pop("user_email", None)
    session["login_failures"] = 0
    return redirect(url_for("index"))


# ---------------------- PATTERN GENERATOR ----------------------


@app.post("/api/convert")
def convert():
    # Require an account
    email = session.get("user_email")
    if not email:
        return redirect(
            url_for(
                "login",
                msg="Create a free account to upload your first PatternCraft.app pattern.",
            )
        )

    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(
            url_for("signup", msg="Create your free PatternCraft.app account to continue.")
        )

    patterns = user.get("patterns", [])
    if len(patterns) >= 1:
        # One full pattern per account while in beta
        return redirect(
            url_for(
                "index",
                err="Beta limit: one full pattern per account for now. Your existing patterns are saved under My Patterns.",
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

    original_name = getattr(file, "filename", "") or "pattern"

    pattern_id = uuid.uuid4().hex[:12]
    user_dir = get_user_pattern_dir(email)
    zip_path = os.path.join(user_dir, f"{pattern_id}.zip")
    grid_path = os.path.join(user_dir, f"{pattern_id}_grid.png")
    legend_path = os.path.join(user_dir, f"{pattern_id}_legend.csv")
    meta_path = os.path.join(user_dir, f"{pattern_id}_meta.json")

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
                sym_img = draw_symbols_on_grid(
                    quant, cell_px=CELL_PX, sym_map=sym_map
                )
                if want_pdf:
                    pdf_buf = io.BytesIO()
                    sym_img.convert("RGB").save(
                        pdf_buf, format="PDF", resolution=300.0
                    )
                    pdf_bytes = pdf_buf.getvalue()
                grid_img = sym_img

            # Legend CSV with color codes and floss estimate
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

            note = (
                "Knitting preview compresses row height; verify gauge."
                if ptype == "knit"
                else "Cross-stitch grid with 10×10 guides and symbol overlay."
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
                "source_filename": original_name,
            }
            meta_json = json.dumps(meta, indent=2)
            z.writestr("meta.json", meta_json)

            buf_png = io.BytesIO()
            grid_img.save(buf_png, format="PNG")
            grid_bytes = buf_png.getvalue()
            z.writestr("grid.png", grid_bytes)
            if pdf_bytes:
                z.writestr("pattern.pdf", pdf_bytes)

            # Persist for "My Patterns"
            with open(grid_path, "wb") as f:
                f.write(grid_bytes)
            with open(legend_path, "w", encoding="utf-8") as f:
                f.write(legend_csv)
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(meta_json)

        elif ptype == "emb":
            small = resize_for_stitch_width(base, stitch_w)
            bw = to_monochrome(small, threshold=emb_thresh)
            pts = serpentine_points(bw, step=emb_step)
            emb_files = write_embroidery_outputs(pts)
            for name, data in emb_files.items():
                z.writestr(name, data)
            meta = {
                "type": "emb",
                "stitch_style": "run",
                "points": len(pts),
                "pyembroidery": HAS_PYEMB,
                "source_filename": original_name,
            }
            meta_json = json.dumps(meta, indent=2)
            z.writestr("meta.json", meta_json)
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(meta_json)
        else:
            return jsonify({"error": "unknown_ptype"}), 400

    # Save ZIP for later downloads
    zip_bytes = out_zip.getvalue()
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    # Update user record: mark free_used and attach pattern metadata
    pattern_entry = {
        "id": pattern_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "ptype": ptype,
        "title": original_name or "Pattern",
    }
    patterns = user.get("patterns") or []
    patterns.append(pattern_entry)
    user["patterns"] = patterns
    user["free_used"] = True
    users[email] = user
    save_users(users)

    # Return ZIP immediately
    out_zip.seek(0)
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{ptype}.zip",
    )


# ---------------------- INLINE HTML: HOMEPAGE ----------------------

HOMEPAGE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" type="image/svg+xml"
        href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' ry='7' fill='%23FACC15'/%3E%3Cpath d='M9 22V9h7.2a4.3 4.3 0 1 1 0 8.6H12v4.4H9zm3-7h4.1a2.3 2.3 0 0 0 0-4.6H12V15z' fill='%231F2933'/%3E%3C/svg%3E">
  <style>
    :root{
      --bg:#FFFDF2;
      --fg:#1F2933;
      --muted:#6B7280;
      --line:#F4E5A9;
      --card:#FFFFFF;
      --radius:18px;
      --shadow:0 16px 40px rgba(15,23,42,.16);
      --accent:#F59E0B;
      --accent-strong:#C97700;
      --accent-soft:#FFF3C4;
      --danger:#DC2626;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#FEF3C7 0,#FFFDF2 45%,transparent 60%),
        radial-gradient(circle at bottom right,#FDE68A 0,#FFF7ED 45%,transparent 60%),
        linear-gradient(to bottom,#FFFDF2,#FFF7ED);
    }
    a{color:var(--accent-strong);text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1100px;margin:0 auto;padding:26px 18px 52px;}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:20px;
    }
    .brand{
      display:flex;align-items:center;gap:8px;
      font-weight:900;font-size:20px;letter-spacing:.05em;text-transform:uppercase;
    }
    .brand-mark{
      width:26px;height:26px;border-radius:9px;
      background:linear-gradient(135deg,#FACC15,#F97316);
      display:flex;align-items:center;justify-content:center;
      font-size:14px;font-weight:800;color:#1F2933;
    }
    .top-links{font-size:13px;color:#4B5563;display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
    .top-links strong{font-weight:600;}
    .top-links a{font-weight:500;}
    .pill{
      display:inline-flex;align-items:center;justify-content:center;
      min-height:42px;padding:0 24px;border-radius:999px;
      border:none;cursor:pointer;
      font-size:14px;font-weight:700;letter-spacing:.02em;
      background:linear-gradient(135deg,#FACC15,#F97316);
      color:#1F2933;
      box-shadow:0 10px 26px rgba(217,119,6,.45);
      transition:transform .08s,box-shadow .08s,background .08s;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 32px rgba(217,119,6,.55);}
    .pill.secondary{
      background:#FFFFFF;
      border:1px solid #E5E7EB;
      box-shadow:0 6px 18px rgba(15,23,42,.12);
      color:#111827;
    }
    .pill.secondary:hover{box-shadow:0 10px 26px rgba(15,23,42,.16);}
    .pill.outline{
      background:transparent;
      color:#C97700;
      border:1px solid #FACC15;
      box-shadow:none;
    }
    .pill.outline:hover{
      background:#FFF7C2;
      box-shadow:0 6px 18px rgba(250,204,21,.45);
    }
    .hero{
      display:grid;
      grid-template-columns:minmax(0,3fr) minmax(280px,2.2fr);
      gap:22px;margin-bottom:26px;align-items:center;
    }
    h1{font-size:2.4rem;margin:6px 0 6px;}
    .kicker{
      font-size:12px;text-transform:uppercase;letter-spacing:.15em;
      color:#9CA3AF;font-weight:700;
    }
    .hero-tagline{color:var(--muted);max-width:460px;font-size:14px;}
    .hero-cta-row{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;}
    .hero-note{font-size:12px;color:#4B5563;margin-top:10px;}
    .badge-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;}
    .badge{
      font-size:11px;padding:5px 10px;border-radius:999px;
      background:#FEF3C7;color:#92400E;border:1px solid #FDE68A;
      text-transform:uppercase;letter-spacing:.08em;font-weight:700;
    }
    .badge-ghost{
      background:#EEF2FF;color:#4338CA;border-color:#E0E7FF;
    }
    .card{
      background:var(--card);
      border-radius:var(--radius);
      border:1px solid var(--line);
      box-shadow:var(--shadow);
      padding:20px 18px;
    }
    .hero-right-title{
      font-size:14px;font-weight:700;margin-bottom:6px;
      text-transform:uppercase;letter-spacing:.12em;color:#9CA3AF;
    }
    .hero-right-main{font-size:14px;color:#111827;margin:0 0 6px;}
    .hero-right-list{font-size:13px;color:#4B5563;padding-left:18px;margin:8px 0;}
    .hero-right-divider{
      margin:12px 0;border-top:1px dashed #E5E7EB;
    }
    .hero-right-foot{font-size:12px;color:#6B7280;}

    .section-title{font-size:1.1rem;margin:0 0 4px;}
    .section-sub{font-size:13px;color:#6B7280;margin:0 0 12px;}

    .two-col{
      display:grid;
      grid-template-columns:minmax(0,1.3fr) minmax(260px,1fr);
      gap:18px;
    }

    .file{
      border:2px dashed #FBBF24;
      border-radius:18px;
      padding:18px;
      display:flex;align-items:center;gap:12px;
      cursor:pointer;
      background:#FFFBEB;
      transition:background .15s,border-color .15s,transform .08s,box-shadow .08s;
    }
    .file:hover{
      background:#FEF3C7;border-color:#F59E0B;
      transform:translateY(-1px);
      box-shadow:0 10px 22px rgba(217,119,6,.3);
    }
    .file.disabled{
      opacity:.7;cursor:pointer;
    }
    .file input{display:none;}
    .file-label-main{
      font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.08em;
    }
    .file-label-sub{font-size:12px;color:#6B7280;}
    .file-ready{
      background:#ECFDF3;border-color:#16A34A;
      box-shadow:0 10px 22px rgba(22,163,74,.35);
    }

    fieldset{
      border:1px solid var(--line);
      border-radius:12px;
      padding:10px;
      margin:10px 0;
    }
    legend{font-size:13px;padding:0 4px;color:#6B7280;}
    .row{display:flex;flex-wrap:wrap;gap:12px;}
    .row > label{flex:1 1 150px;font-size:13px;}
    .row input,.row select{
      width:100%;margin-top:3px;padding:7px 10px;border-radius:9px;
      border:1px solid #E5E7EB;font-size:13px;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#F59E0B;box-shadow:0 0 0 1px rgba(245,158,11,.4);
    }
    label{font-size:13px;}
    .controls-note{font-size:11px;color:#9CA3AF;margin-top:4px;}
    .hidden{display:none;}

    .free-note{
      margin-top:8px;font-size:12px;color:#166534;background:#DCFCE7;
      border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px;
    }
    .free-dot{width:8px;height:8px;border-radius:999px;background:#16A34A;}

    .error-banner{
      margin-top:8px;font-size:12px;color:#B91C1C;background:#FEE2E2;
      border-radius:9px;padding:8px 10px;
    }

    .why-card{
      background:#FFFBEB;border-radius:var(--radius);
      border:1px dashed #FACC15;padding:18px;margin-top:22px;
    }
    .why-card h2{margin:0 0 6px;font-size:1.05rem;}
    .why-list{font-size:13px;color:#4B5563;padding-left:18px;margin:6px 0;}
    .why-list li{margin:3px 0;}

    .my-pattern-link{
      font-size:13px;margin-top:10px;color:#4B5563;
    }

    @media (max-width:880px){
      .hero{grid-template-columns:1fr;}
      .two-col{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">PC</div>
      <div>PatternCraft.app</div>
    </div>
    <div class="top-links">
      {% if user %}
        <span><strong>Signed in:</strong> {{ user.email }}</span>
        · <a href="/patterns">My patterns</a>
        · <a href="/logout">Sign out</a>
      {% else %}
        <a href="/login">Log in</a>
        · <a href="/signup" class="pill outline">Create Free Account</a>
      {% endif %}
    </div>
  </div>

  <div class="hero">
    <div>
      <div class="kicker">Picture in · pattern out</div>
      <h1>Turn art into stitchable patterns</h1>
      <p class="hero-tagline">
        Upload your artwork once and PatternCraft.app turns it into a stitchable grid with
        color legend and stitch counts. Designed for cross‑stitch, knitting charts, and line embroidery.
      </p>
      <div class="hero-cta-row">
        {% if user %}
          <button class="pill" onclick="document.getElementById('make').scrollIntoView({behavior:'smooth'})">
            Open the pattern tool
          </button>
          <a class="pill secondary" href="/patterns">View my patterns</a>
        {% else %}
          <a class="pill" href="/signup">
            Create Free Account
          </a>
          <a class="pill secondary" href="/login">
            Log in
          </a>
        {% endif %}
      </div>
      <div class="hero-note">
        Every account includes <strong>one full pattern on us</strong>, plus an optional
        monthly email with ideas and tips for new designs.
      </div>
      <div class="badge-row">
        <span class="badge">One free pattern per account (beta)</span>
        <span class="badge badge-ghost">Patterns saved to your account</span>
      </div>
      {% if error %}
        <div class="error-banner" style="max-width:460px;">{{ error }}</div>
      {% endif %}
    </div>

    <div class="card">
      <div class="hero-right-title">Why makers use PatternCraft.app</div>
      <p class="hero-right-main">
        A purpose‑built pattern tool with stitchers in mind.
      </p>
      <ul class="hero-right-list">
        <li>Clean grids with bold 10×10 guides and symbol overlays</li>
        <li>Color legends with hex and RGB values for accurate palettes</li>
        <li>Fabric size estimates based on stitch count and cloth count</li>
        <li>Knitting charts that respect row proportions</li>
        <li>Embroidery line outputs ready for your machine software</li>
      </ul>
      <div class="hero-right-divider"></div>
      <div class="hero-right-foot">
        Your patterns are saved to your account so you can return later to view, print, or download again.
      </div>
    </div>
  </div>

  <div id="make" class="card">
    <div class="two-col">
      <div>
        <h2 class="section-title">Make a pattern</h2>
        <p class="section-sub">
          Create a free account or log in to generate a pattern. While we’re in beta,
          every account can create one full pattern that stays saved under <strong>My patterns</strong>.
        </p>

        <form method="POST" action="/api/convert" enctype="multipart/form-data">
          <label class="file {% if not user %}disabled{% endif %}" onclick="redirectIfAnon(event)">
            <input id="fileInput" type="file" name="file" accept="image/*"
                   {% if not user %} disabled{% endif %} onchange="pickFile(this)">
            <div>
              <div class="file-label-main" id="fileLabelMain">UPLOAD PICTURE HERE</div>
              <div class="file-label-sub">
                {% if user %}
                  Drop in your artwork or tap to browse from your device.
                {% else %}
                  Create a free account or log in to upload an image.
                {% endif %}
              </div>
            </div>
          </label>

          {% if not user %}
            <div class="free-note">
              <div class="free-dot"></div>
              <span>
                New here? <a href="/signup">Create your free account</a>.
                Already joined? <a href="/login">Log in</a>.
              </span>
            </div>
          {% endif %}

          {% if user %}
          <fieldset>
            <legend>Pattern type</legend>
            <label><input type="radio" name="ptype" value="cross" checked> Cross‑stitch</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="knit"> Knitting</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="emb"> Embroidery</label>
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
            <p class="controls-note">
              Simple run‑stitch path from your image. For advanced digitizing, continue in your embroidery software.
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
            <span class="controls-note">
              Download includes grid.png, legend.csv, meta.json, and optional pattern.pdf or embroidery files.
            </span>
          </div>
          {% endif %}
        </form>

        {% if user %}
          <div class="my-pattern-link">
            Finished patterns are saved to your account. Visit <a href="/patterns">My patterns</a> to re‑view and re‑download.
          </div>
        {% endif %}
      </div>

      <div>
        <div class="why-card">
          <h2>How it works</h2>
          <ul class="why-list">
            <li>Select your stitch type and size.</li>
            <li>Upload a photo, illustration, or logo.</li>
            <li>PatternCraft analyzes it and builds a stitch grid.</li>
            <li>You get a pattern image, legend with codes, and a ZIP you can print or archive.</li>
          </ul>
          <p class="controls-note">
            While we’re in beta, accounts are limited to one full pattern each so we can keep things fast and reliable.
          </p>
        </div>
      </div>
    </div>
  </div>

</div>
<script>
  const IS_LOGGED_IN = {{ "true" if user else "false" }};

  function redirectIfAnon(evt){
    if(!IS_LOGGED_IN){
      evt.preventDefault();
      window.location.href = "/login?msg=" + encodeURIComponent(
        "Create a free account to upload your first PatternCraft.app pattern."
      );
    }
  }

  function pickFile(inp){
    const wrapper = inp.closest('label');
    const labelMain = document.getElementById('fileLabelMain');
    const generateBtn = document.getElementById('generateBtn');
    if(!inp.files || !inp.files[0]){
      if(wrapper) wrapper.classList.remove('file-ready');
      if(labelMain) labelMain.textContent = 'UPLOAD PICTURE HERE';
      if(generateBtn) generateBtn.classList.remove('file-ready');
      return;
    }
    if(wrapper) wrapper.classList.add('file-ready');
    if(labelMain) labelMain.textContent = 'IMAGE ATTACHED';
    if(generateBtn) generateBtn.classList.add('file-ready');
  }

  function setStyleOptions(type){
    const sel = document.getElementById('stitch_style');
    if(!sel) return;
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
    if(crossKnit && emb){
      if(type === 'emb'){
        crossKnit.classList.add('hidden');
        emb.classList.remove('hidden');
      } else {
        crossKnit.classList.remove('hidden');
        emb.classList.add('hidden');
      }
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

SIGNUP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your free account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#FFFDF2;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#1F2933;}
    .wrap{max-width:520px;margin:0 auto;padding:40px 16px 46px;}
    .card{
      background:#FFFFFF;border-radius:18px;border:1px solid #F4E5A9;
      padding:22px;box-shadow:0 16px 40px rgba(15,23,42,.18);
    }
    h1{margin:0 0 8px;font-size:1.8rem;}
    .muted{font-size:13px;color:#6B7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:11px;
      border:1px solid #E5E7EB;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#F59E0B;box-shadow:0 0 0 1px rgba(245,158,11,.4);
    }
    .pill{
      margin-top:16px;padding:10px 22px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#FACC15,#F97316);color:#1F2933;
      font-size:14px;font-weight:700;cursor:pointer;
      box-shadow:0 12px 30px rgba(217,119,6,.45);
      display:inline-flex;align-items:center;justify-content:center;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 15px 36px rgba(217,119,6,.52);}
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;}
    a{color:#C97700;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4B5563;padding-left:18px;margin-top:10px;}
    .free-chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:5px 10px;border-radius:999px;
      background:#FEF3C7;color:#92400E;font-size:11px;
      text-transform:uppercase;letter-spacing:.09em;font-weight:700;
      margin-bottom:8px;
    }
    .dot{width:8px;height:8px;border-radius:999px;background:#16A34A;}
    .disclaimer{
      margin-top:10px;font-size:11px;color:#6B7280;
      background:#FFFBEB;border-radius:12px;padding:8px 10px;border:1px dashed #FACC15;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="free-chip"><span class="dot"></span> Free account · one pattern included</div>
    <h1>Create your free PatternCraft.app account</h1>
    <p class="muted">
      Sign up once and your patterns stay attached to your email. Every new account includes
      <strong>one full pattern</strong> you can generate, save, and return to under <em>My patterns</em>.
    </p>
    <ul>
      <li>Use an email you actually check — we’ll send your account notices there.</li>
      <li>Passwords are private and hashed; we never store them in plain text.</li>
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
      Already have an account? <a href="/login">Log in here</a>.
    </p>
    <div class="disclaimer">
      By creating an account, you agree that PatternCraft.app may send a short monthly email with
      new pattern ideas and updates. You can unsubscribe at any time.
    </div>
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
  <style>
    body{margin:0;background:#FFFDF2;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#1F2933;}
    .wrap{max-width:520px;margin:0 auto;padding:40px 16px 46px;}
    .card{
      background:#FFFFFF;border-radius:18px;border:1px solid #F4E5A9;
      padding:22px;box-shadow:0 16px 40px rgba(15,23,42,.18);
    }
    h1{margin:0 0 8px;font-size:1.7rem;}
    .muted{font-size:13px;color:#6B7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:11px;
      border:1px solid #E5E7EB;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#F59E0B;box-shadow:0 0 0 1px rgba(245,158,11,.4);
    }
    .pill{
      margin-top:16px;padding:10px 22px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#FACC15,#F97316);color:#1F2933;
      font-size:14px;font-weight:700;cursor:pointer;
      box-shadow:0 12px 30px rgba(217,119,6,.45);
      display:inline-flex;align-items:center;justify-content:center;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 15px 36px rgba(217,119,6,.52);}
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;}
    a{color:#C97700;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .alt{
      margin-top:12px;padding-top:10px;border-top:1px solid #F4E5A9;
      font-size:13px;color:#6B7280;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Log in to PatternCraft.app</h1>
    <p class="muted">
      Use the email and password you created when you first generated a pattern.
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
    <div class="alt">
      New to PatternCraft.app? <a href="/signup">Create a free account</a> —
      includes one free pattern and access to monthly pattern ideas.
    </div>
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
  <title>My Patterns — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#FFFDF2;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#1F2933;}
    .wrap{max-width:1100px;margin:0 auto;padding:26px 18px 46px;}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      display:flex;align-items:center;gap:8px;
      font-weight:900;font-size:20px;letter-spacing:.05em;text-transform:uppercase;
    }
    .brand-mark{
      width:26px;height:26px;border-radius:9px;
      background:linear-gradient(135deg,#FACC15,#F97316);
      display:flex;align-items:center;justify-content:center;
      font-size:14px;font-weight:800;color:#1F2933;
    }
    .top-links{font-size:13px;color:#4B5563;display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
    a{color:#C97700;text-decoration:none;}
    a:hover{text-decoration:underline;}
    h1{margin:0 0 6px;font-size:1.6rem;}
    .muted{font-size:13px;color:#6B7280;margin:0 0 16px;}

    .grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
      gap:16px;
    }
    .card{
      background:#FFFFFF;border-radius:16px;border:1px solid #F4E5A9;
      padding:14px 14px 16px;
      box-shadow:0 10px 26px rgba(15,23,42,.12);
      display:flex;flex-direction:column;gap:8px;
    }
    .card h2{
      margin:0;font-size:14px;font-weight:700;
      display:flex;justify-content:space-between;align-items:center;
    }
    .badge{
      font-size:11px;padding:3px 8px;border-radius:999px;
      background:#FEF3C7;color:#92400E;border:1px solid #FDE68A;
      text-transform:uppercase;letter-spacing:.08em;font-weight:700;
    }
    .meta{font-size:11px;color:#9CA3AF;}
    .pattern-preview{
      margin-top:6px;
      border-radius:12px;
      border:1px solid #E5E7EB;
      background:#F9FAFB;
      overflow:hidden;
      max-height:220px;
      display:flex;align-items:center;justify-content:center;
    }
    .pattern-preview img{
      display:block;width:100%;height:auto;
      object-fit:contain;
    }
    .actions{
      margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;
    }
    .pill-small{
      display:inline-flex;align-items:center;justify-content:center;
      padding:7px 14px;border-radius:999px;border:none;
      font-size:12px;font-weight:600;cursor:pointer;
      background:#FACC15;color:#1F2933;
    }
    .pill-small.secondary{
      background:#FFFFFF;border:1px solid #E5E7EB;color:#111827;
    }
    .pill-small:hover{filter:brightness(0.97);}
    .empty{
      margin-top:12px;padding:12px;border-radius:12px;
      background:#FFFBEB;border:1px dashed #FACC15;font-size:13px;color:#92400E;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">PC</div>
      <div>PatternCraft.app</div>
    </div>
    <div class="top-links">
      <span>Signed in as {{ user.email }}</span>
      · <a href="/">Back to tool</a>
      · <a href="/logout">Sign out</a>
    </div>
  </div>

  <h1>My patterns</h1>
  <p class="muted">
    These are the patterns you’ve generated with PatternCraft.app. Open a grid to print, view its legend,
    or download the original ZIP again.
  </p>

  {% if not patterns %}
    <div class="empty">
      No patterns yet. Once you generate a pattern, it will appear here so you can revisit and print it any time.
    </div>
  {% else %}
    <div class="grid">
      {% for p in patterns %}
        <div class="card">
          <h2>
            <span>{{ p.title or ("Pattern " ~ loop.index) }}</span>
            <span class="badge">{{ p.ptype|upper }}</span>
          </h2>
          <div class="meta">
            Created {{ p.created_at }}
          </div>
          <div class="pattern-preview">
            <img src="/patterns/{{ p.id }}/grid" alt="Pattern preview">
          </div>
          <div class="actions">
            <a class="pill-small" href="/patterns/{{ p.id }}/download">Download ZIP</a>
            <a class="pill-small secondary" href="/patterns/{{ p.id }}/legend" target="_blank">View legend</a>
          </div>
        </div>
      {% endfor %}
    </div>
  {% endif %}
</div>
</body>
</html>
"""

if __name__ == "__main__":
    # For local testing; on Render you use gunicorn "app:app"
    app.run(debug=True)
