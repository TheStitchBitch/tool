from __future__ import annotations
import io
import json
import math
import os
import zipfile
import base64
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

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Simple JSON “database” for users
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# Config
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels


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
    """Scale each stitch to a cell and overlay a 10×10 grid."""
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


def add_strong_watermark(img: Image.Image) -> Image.Image:
    """
    Overlay a dense diagonal watermark so the free preview
    is not practically usable as a working pattern.
    Applied only to the on-page preview sample.
    """
    wm = img.convert("RGBA")
    w, h = wm.size

    text_canvas = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(text_canvas)

    try:
        font_size = max(20, min(w, h) // 6)
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        font_size = 24

    text = "PREVIEW ONLY • PATTERNCRAFT.APP"
    step_x = font_size * 6
    step_y = font_size * 3

    for y in range(0, h * 2, step_y):
        for x in range(0, w * 2, step_x):
            tdraw.text((x, y), text, font=font, fill=(255, 255, 255, 230))

    rotated = text_canvas.rotate(30, expand=True)
    rw, rh = rotated.size
    left = (rw - w) // 2
    top = (rh - h) // 2
    crop = rotated.crop((left, top, left + w, top + h))

    result = Image.alpha_composite(wm, crop)
    return result.convert("RGB")


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


@app.errorhandler(Exception)
def on_error(_e):
    return make_response(jsonify({"error": "server_error"}), 500)


@app.get("/")
def index() -> str:
    user = get_current_user()
    return render_template_string(HOMEPAGE_HTML, user=user, sample_preview=None, sample_error=None)


@app.get("/pricing")
def pricing() -> str:
    user = get_current_user()
    reason = request.args.get("reason", "")
    message = ""
    if reason == "used_free":
        message = "You’ve already used your PatternCraft.app pattern for this account. Choose a plan to keep generating patterns."
    return render_template_string(PRICING_HTML, user=user, message=message)


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
        "subscription": "free",
        "free_used": False,
    }
    save_users(users)
    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/login")
def login() -> str:
    user = get_current_user()
    return render_template_string(LOGIN_HTML, user=user, message="")


@app.post("/login")
def login_post():
    user = get_current_user()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        return render_template_string(
            LOGIN_HTML,
            user=user,
            message="Please enter both email and password.",
        )

    users = load_users()
    stored = users.get(email)
    if not stored or not check_password_hash(stored.get("password_hash", ""), password):
        return render_template_string(
            LOGIN_HTML,
            user=user,
            message="Incorrect email or password.",
        )

    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("index"))


# ---------------------- SAMPLE PREVIEW (FREE, ON-PAGE, WATERMARKED) ----------------------
@app.post("/sample-preview")
def sample_preview():
    """Allow anyone to upload a sample image and see a watermarked preview grid inline on the homepage."""
    user = get_current_user()
    file = request.files.get("sample_file")
    if not file:
        return render_template_string(
            HOMEPAGE_HTML,
            user=user,
            sample_preview=None,
            sample_error="Please upload an image to preview.",
        )
    try:
        img = open_image(file)
    except Exception:
        return render_template_string(
            HOMEPAGE_HTML,
            user=user,
            sample_preview=None,
            sample_error="Could not read that image. Try a PNG or JPG.",
        )
    if max(img.size) > MAX_DIM:
        return render_template_string(
            HOMEPAGE_HTML,
            user=user,
            sample_preview=None,
            sample_error=f"Image is too large. Try something under {MAX_DIM}px on a side.",
        )

    # Lightweight preview defaults
    stitch_w = 80
    max_colors = 12
    small = resize_for_stitch_width(img, stitch_w)
    quant = quantize(small, max_colors)
    grid_img = draw_grid(quant, cell_px=10)

    # Apply strong watermark ONLY to this sample preview
    grid_img = add_strong_watermark(grid_img)

    buf = io.BytesIO()
    grid_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    return render_template_string(
        HOMEPAGE_HTML,
        user=user,
        sample_preview=data_url,
        sample_error=None,
    )


# ---------------------- FREE SAMPLE QUILT PATTERN ZIP ----------------------
@app.get("/sample-pattern.zip")
def sample_pattern_zip():
    """
    Serve a sample quilt-style pattern ZIP with a grid and legend.
    Uses a synthetic patchwork-style chart generated in code.
    """
    # Create a small quilt-style patchwork image
    w, h = 40, 40
    base = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(base)
    colors = [
        (239, 68, 68),    # red
        (249, 115, 22),   # orange
        (234, 179, 8),    # gold
        (34, 197, 94),    # green
        (59, 130, 246),   # blue
    ]

    patch = 8  # size of each quilt block in pixels
    for py in range(0, h, patch):
        for px in range(0, w, patch):
            block_idx = (px // patch + py // patch) % len(colors)
            base_color = colors[block_idx]
            alt_color = colors[(block_idx + 2) % len(colors)]

            # Different block styles for a “quilt” feel
            style = (px // patch + 2 * (py // patch)) % 3

            if style == 0:
                # Solid block
                draw.rectangle((px, py, px + patch - 1, py + patch - 1), fill=base_color)
            elif style == 1:
                # Diagonal half-square triangle
                for y in range(patch):
                    for x in range(patch):
                        if x >= y:
                            draw.point((px + x, py + y), fill=base_color)
                        else:
                            draw.point((px + x, py + y), fill=alt_color)
            else:
                # Stripes inside block
                for y in range(patch):
                    color = base_color if y % 2 == 0 else alt_color
                    draw.line((px, py + y, px + patch - 1, py + y), fill=color)

    # Treat as a quilt-style cross-stitch chart
    max_colors = len(colors)
    quant = quantize(base, max_colors)
    counts = palette_counts(quant)
    sx, sy = quant.size

    cloth_count = 10
    strands = 2
    waste_pct = 20
    finished_w_in = round(sx / float(cloth_count), 2)
    finished_h_in = round(sy / float(cloth_count), 2)

    # Symbol grid
    grid_img = draw_grid(quant, cell_px=CELL_PX)
    pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
    sym_map = assign_symbols(pal)
    sym_img = draw_symbols_on_grid(quant, cell_px=CELL_PX, sym_map=sym_map)
    grid_img = sym_img

    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # legend.csv with color counts and skein estimates
        total_stitches = sum(counts.values()) or 1
        lines = ["hex,r,g,b,stitches,percent,skeins_est"]
        for (r, g, b), c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            skeins = skeins_per_color(c, cloth_count, strands, waste_pct / 100.0)
            lines.append(
                f"{to_hex((r,g,b))},{r},{g},{b},{c},{(100*c/total_stitches):.2f},{skeins:.2f}"
            )
        z.writestr("legend.csv", "\n".join(lines))

        meta = {
            "type": "quilt_sample",
            "stitch_style": "patchwork",
            "stitches_w": sx,
            "stitches_h": sy,
            "colors": len(counts),
            "cloth_count": cloth_count,
            "strands": strands,
            "waste_percent": waste_pct,
            "finished_size_in": [finished_w_in, finished_h_in],
            "notes": "Sample quilt-style color grid generated by PatternCraft.app with grid.png and legend.csv.",
        }
        z.writestr("meta.json", json.dumps(meta, indent=2))

        buf_png = io.BytesIO()
        grid_img.save(buf_png, format="PNG")
        z.writestr("grid.png", buf_png.getvalue())

    out_zip.seek(0)
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name="patterncraft_sample_quilt.zip",
    )


# ---------------------- PATTERN GENERATOR (ACCOUNT-GATED) ----------------------
@app.post("/api/convert")
def convert():
    # Require an account
    email = session.get("user_email")
    if not email:
        return redirect(url_for("signup", msg="Create your PatternCraft.app account to generate patterns."))

    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(url_for("signup", msg="Please create your PatternCraft.app account to continue."))

    subscription = user.get("subscription", "free")
    mark_free_used = False

    if subscription in ("pro_monthly", "pro_yearly", "pro_unlimited"):
        pass
    else:
        if user.get("free_used"):
            return redirect(url_for("pricing", reason="used_free"))
        else:
            mark_free_used = True

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

    if mark_free_used:
        user["free_used"] = True
        users[email] = user
        save_users(users)

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
  <style>
    :root{
      --bg:#F7F4EF;--fg:#222;--muted:#6b6b6b;
      --line:#e8e4de;--radius:14px;--shadow:0 10px 30px rgba(15,23,42,.15);
      --accent:#4C7CF3;--accent-soft:#e3ebff;--accent-strong:#173d99;
      --pill:#f97316;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#fde68a 0,#f1f5f9 35%,transparent 55%),
        radial-gradient(circle at top right,#bfdbfe 0,#f9fafb 40%,transparent 60%),
        linear-gradient(to bottom,#f3f4f6,#fefce8);
    }
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1040px;margin:0 auto;padding:24px 16px 48px}
    h1{font-size:2.6rem;margin:0 0 8px}
    h2{margin:0 0 10px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{font-weight:800;font-size:20px;letter-spacing:.04em;text-transform:uppercase;}
    .top-links{font-size:13px;color:#4b5563}
    .top-links a{margin-left:8px;}
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:4px 10px;border-radius:999px;
      background:rgba(255,255,255,.8);border:1px solid rgba(148,163,184,.4);
      font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.08em;
    }
    .chip-dot{width:8px;height:8px;border-radius:999px;background:#22c55e}
    .card{
      background:#fff;border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.3);
      box-shadow:var(--shadow);
      padding:20px;
    }
    .hero{
      display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2fr);
      gap:20px;margin-bottom:28px;align-items:center;
    }
    .hero-tagline{color:var(--muted);max-width:420px;}
    .muted{color:var(--muted);font-size:13px}
    .pill{
      padding:11px 20px;border-radius:999px;
      background:linear-gradient(135deg,var(--pill),#f97316);
      color:#fff;border:none;cursor:pointer;
      font-size:14px;font-weight:600;letter-spacing:.02em;
      box-shadow:0 7px 18px rgba(248,113,22,.35);
      transition:transform .08s,box-shadow .08s;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(248,113,22,.45);}
    .pill-secondary{
      background:#fff;color:var(--fg);
      border:1px solid rgba(148,163,184,.5);
      box-shadow:none;
    }
    .pill-secondary:hover{
      box-shadow:0 4px 14px rgba(148,163,184,.4);
    }
    .hero-cta-row{
      display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;align-items:center;
    }
    .hero-note{font-size:12px;color:#475569;margin-top:8px;}
    .badge-row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
    .badge{
      font-size:11px;padding:4px 8px;border-radius:999px;
      background:#e5edff;color:#1d4ed8;border:1px solid rgba(129,140,248,.5);
    }
    .sample-grid{display:grid;gap:14px;grid-template-columns:1.2fr 1.1fr}
    .sample-card{border-radius:14px;border:1px solid var(--line);padding:10px;background:#fff;box-shadow:0 6px 18px rgba(15,23,42,.08)}
    .sample-label{font-weight:600;font-size:13px;margin-bottom:6px}
    .sample-art{border-radius:12px;overflow:hidden;background:#0f172a}
    .sample-art img{width:100%;display:block;object-fit:cover;max-height:260px}
    .sample-pattern{
      border-radius:12px;border:1px solid #e5e7eb;
      background-image:
        linear-gradient(to right, rgba(15,23,42,.15) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(15,23,42,.15) 1px, transparent 1px);
      background-size:12px 12px;
      height:220px;position:relative;overflow:hidden;
    }
    .sample-pattern::before{
      content:"";
      position:absolute;inset:18px;
      border-radius:8px;
      background:repeating-linear-gradient(
        45deg,
        rgba(249,115,22,.15) 0,
        rgba(249,115,22,.15) 6px,
        transparent 6px,
        transparent 12px
      );
      mix-blend-mode:multiply;
    }
    .sample-pattern::after{
      content:"✦ ○ ◆";
      position:absolute;inset:0;
      display:flex;align-items:center;justify-content:center;
      font-size:24px;font-weight:700;color:rgba(15,23,42,.45);
    }
    .sample-note{font-size:12px;color:var(--muted);margin-top:6px}
    .section-title{font-size:1.1rem;margin-bottom:6px}
    .make-layout{display:grid;gap:18px;grid-template-columns:minmax(0,1.2fr)}
    .file{
      border:2px dashed var(--accent);
      border-radius:18px;
      padding:18px;
      display:flex;align-items:center;gap:12px;
      cursor:pointer;
      background:var(--accent-soft);
      transition:background .15s,border-color .15s,transform .1s,box-shadow .1s;
    }
    .file:hover{
      background:#dbe4ff;border-color:#365ed1;
      transform:translateY(-1px);
      box-shadow:0 6px 15px rgba(37,99,235,.3);
    }
    .file input{display:none}
    .file-label-main{font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.06em}
    .file-label-sub{font-size:12px;color:var(--muted)}
    .free-note{
      margin-top:6px;font-size:12px;color:#065f46;background:#d1fae5;
      border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px;
    }
    .free-dot{width:8px;height:8px;border-radius:999px;background:#10b981}
    fieldset{border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0}
    legend{font-size:13px;padding:0 4px}
    .row{display:flex;flex-wrap:wrap;gap:12px}
    .row > label{flex:1 1 150px;font-size:13px}
    .row input,.row select{
      width:100%;margin-top:3px;padding:6px 8px;border-radius:8px;
      border:1px solid #cbd5f5;font-size:13px;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#4f46e5;box-shadow:0 0 0 1px rgba(79,70,229,.35);
    }
    label{font-size:13px}
    .controls-note{font-size:11px;color:#94a3b8;margin-top:4px}
    .hidden{display:none}
    .error-banner{
      margin-top:8px;font-size:12px;color:#b91c1c;background:#fee2e2;
      border-radius:8px;padding:6px 8px;
    }
    @media (max-width:860px){
      .hero{grid-template-columns:1fr}
      .sample-grid{grid-template-columns:1fr}
      .make-layout{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">PatternCraft.app</div>
    <div class="top-links">
      <a href="/pricing">Pricing</a>
      {% if user %}
        · Signed in as {{ user.email }} ({{ user.subscription }} plan)
        · <a href="/logout">Sign out</a>
      {% else %}
        · <a href="/login">Log in</a> · <a href="/signup">Create account</a>
      {% endif %}
    </div>
  </div>

  <div class="hero">
    <div class="hero-text">
      <div class="chip">
        <span class="chip-dot"></span>
        <span>For cross-stitch, knitting, and quilting</span>
      </div>
      <h1>Turn art into stitchable patterns</h1>
      <p class="hero-tagline">
        PatternCraft.app converts your artwork into cross-stitch grids, knitting charts,
        and embroidery-ready files with one upload. Export a full ZIP you can print or take to your machine.
      </p>
      <div class="hero-cta-row">
        <button class="pill" onclick="document.getElementById('make').scrollIntoView({behavior:'smooth'})">
          See the tool
        </button>
        <button class="pill pill-secondary" onclick="location.href='#how'">
          See how it works
        </button>
      </div>
      <div class="hero-note">
        Step 1: <a href="#how">Preview a finished pattern</a> ·
        Step 2: <a href="/signup">Create your PatternCraft.app account</a> ·
        Step 3: Upload your art and generate your own pattern.
      </div>
      <div class="badge-row">
        <span class="badge">One pattern included with every account</span>
        <span class="badge">Designed for hobbyists & pattern sellers</span>
      </div>
    </div>

    <div class="card">
      <h2 class="section-title">Why PatternCraft.app</h2>
      <ul class="muted" style="padding-left:18px">
        <li>Clean grids with bold 10×10 guides</li>
        <li>Floss estimates per color for planning</li>
        <li>Knitting charts that respect row proportions</li>
        <li>Embroidery outputs ready for your workflow</li>
      </ul>
    </div>
  </div>

  <div id="how" class="card" style="margin-bottom:24px">
    <h2 class="section-title">See a PatternCraft.app conversion</h2>
    <div class="sample-grid">

      <div class="sample-card">
        <div class="sample-label">Artwork in</div>
        <form method="POST" action="/sample-preview" enctype="multipart/form-data">
          <label class="file">
            <input type="file" name="sample_file" accept="image/*" required>
            <div>
              <div class="file-label-main">TRY A FREE PREVIEW</div>
              <div class="file-label-sub">
                Upload a sample image — we’ll convert it and show the pattern on the right.
              </div>
            </div>
          </label>
          <button class="pill pill-secondary" type="submit" style="margin-top:8px;">
            See pattern preview
          </button>
        </form>
        {% if sample_error %}
          <div class="error-banner" style="margin-top:8px;">{{ sample_error }}</div>
        {% endif %}
        <div class="sample-note" style="margin-top:8px;">
          This on-page preview is for demonstration only. Your full, clean pattern comes from the ZIP you generate below.
        </div>
      </div>

      <div class="sample-card">
        <div class="sample-label">Pattern out (preview)</div>
        {% if sample_preview %}
          <div class="sample-art">
            <img src="{{ sample_preview }}" alt="PatternCraft.app sample pattern preview">
          </div>
          <div class="sample-note">
            This watermarked grid is a live rendering of your upload. Exports from your account are clean and ready to stitch.
          </div>
        {% else %}
          <div class="sample-pattern"></div>
          <div class="sample-note">
            Once you upload art on the left, we’ll show the stitched grid here.
          </div>
        {% endif %}
      </div>

    </div>
  </div>

  <div id="make" class="card">
    <h2 class="section-title">Make a pattern</h2>
    <p class="muted">
      Create a PatternCraft.app account or log in to generate patterns. Every account includes one pattern on us.
      After that, plans on the pricing page keep you creating.
    </p>
    <div class="make-layout">
      <div class="make-main">
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

          <div style="display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap">
            <div class="free-note">
              <div class="free-dot"></div>
              <span>
                New here? <a href="/signup">Create your account</a>.
                Already joined? <a href="/login">Log in</a>.
              </span>
            </div>
          </div>

          <fieldset>
            <legend>Pattern type</legend>
            <label><input type="radio" name="ptype" value="cross" checked> Cross-stitch</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="knit"> Knitting</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="emb"> Embroidery</label>
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
            <p class="controls-note">Defaults work well for most art. Adjust once you know your style.</p>
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
            <label><input type="checkbox" name="symbols" checked> Symbol overlay</label>
            <label style="margin-left:10px"><input type="checkbox" name="pdf" checked> Also export PDF</label>
          </fieldset>

          <fieldset id="embBlock" class="hidden">
            <legend>Embroidery options</legend>
            <p class="muted">
              Simple run-stitch path from your image. For advanced digitizing, continue in your embroidery software.
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

          <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
            <button class="pill" type="submit">Generate pattern ZIP</button>
            <span class="muted">
              Your download includes grid.png, legend.csv, meta.json, and optional pattern.pdf or embroidery files.
            </span>
          </div>
        </form>
      </div>
    </div>
  </div>

</div>
<script>
  function pickFile(inp){
    const wrapper = inp.closest('label');
    const label = wrapper ? wrapper.querySelector('.file-label-main') : null;
    if (!inp.files || !inp.files[0] || !label) return;
    label.textContent = 'Selected: ' + inp.files[0].name;
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

# ---------------------- INLINE HTML: SIGNUP / LOGIN / PRICING ----------------------
SIGNUP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#F7F4EF;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{background:#fff;border-radius:14px;border:1px solid #e5e7eb;padding:20px;box-shadow:0 10px 30px rgba(15,23,42,.15)}
    h1{margin:0 0 10px;font-size:1.6rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #cbd5e1;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#4f46e5;box-shadow:0 0 0 1px rgba(79,70,229,.35);
    }
    .pill{
      margin-top:14px;padding:9px 18px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#f97316,#ea580c);color:#fff;
      font-size:14px;font-weight:600;cursor:pointer;
      box-shadow:0 7px 18px rgba(248,113,22,.35);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(248,113,22,.45);}
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4b5563;padding-left:18px;margin-top:10px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Create your PatternCraft.app account</h1>
    <p class="muted">
      Every account includes one pattern on us. Your account lets you come back later,
      use different devices, and upgrade to a plan when you’re ready.
    </p>
    <ul>
      <li>Use your best email — we’ll send occasional pattern ideas and updates.</li>
      <li>Choose a password you’ll remember; you’ll use it to log back in.</li>
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
      <button class="pill" type="submit">Create account</button>
    </form>
    {% if message %}
      <div class="msg">{{ message }}</div>
    {% endif %}
    <p class="muted" style="margin-top:10px;">
      Already set this up? <a href="/login">Log in instead</a>.
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
  <style>
    body{margin:0;background:#F7F4EF;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{background:#fff;border-radius:14px;border:1px solid #e5e7eb;padding:20px;box-shadow:0 10px 30px rgba(15,23,42,.15)}
    h1{margin:0 0 10px;font-size:1.6rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #cbd5e1;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#4f46e5;box-shadow:0 0 0 1px rgba(79,70,229,.35);
    }
    .pill{
      margin-top:14px;padding:9px 18px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#4c51bf,#4338ca);color:#fff;
      font-size:14px;font-weight:600;cursor:pointer;
      box-shadow:0 7px 18px rgba(79,70,229,.35);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(79,70,229,.45);}
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
      Use the email and password you created when you first tried PatternCraft.app.
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
      New to PatternCraft.app? <a href="/signup">Create an account</a>.
    </p>
  </div>
</div>
</body>
</html>
"""

PRICING_HTML = """
<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PatternCraft • Pricing</title>
<style>
:root{
  --fg:#111; --muted:#666; --accent:#e4006d; --line:#eee; --card:#fafafa;
  --radius:14px; --wrap:1100px;
}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{
  font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  color:var(--fg);background:#fff;
}
.wrap{max-width:var(--wrap);margin:0 auto;padding:20px}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);z-index:5}
.brand{font-weight:800;letter-spacing:.2px}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.btn{
  display:inline-block;padding:10px 16px;border-radius:999px;
  border:1px solid var(--accent);background:var(--accent);color:#fff;
  text-decoration:none;font-weight:700;cursor:pointer
}
.btn.ghost{background:transparent;color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:16px}
.card h3{margin:0 0 6px}
.price{font-size:28px;font-weight:800;margin:4px 0}
.small{font-size:14px;color:var(--muted)}
.list{margin:8px 0 12px;padding-left:20px}
.badge{
  display:inline-block;background:#ffe6f3;color:#bb0055;border-radius:999px;
  padding:4px 12px;font-size:13px;font-weight:600;margin-bottom:6px
}
.notice{
  margin-top:10px;padding:10px 12px;border-radius:10px;
  background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;
  font-size:14px;
}
footer{border-top:1px solid var(--line);margin-top:24px}
@media (max-width:700px){ .cards{grid-template-columns:1fr} }

/* how-it-works visual */
.steps{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:16px;
  margin-top:16px;
}
.step-card{
  background:var(--card);
  border:1px solid var(--line);
  border-radius:var(--radius);
  padding:16px;
}
.step-num{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:26px;height:26px;
  border-radius:999px;
  background:var(--accent);
  color:#fff;
  font-size:14px;
  font-weight:700;
  margin-bottom:8px;
}
</style>

<header>
  <div class="wrap row" style="justify-content:space-between">
    <div class="brand">PatternCraft.app</div>
    <nav class="row">
      <a class="btn ghost" href="/">Tool</a>
      <a class="btn" href="/pricing">Pricing</a>
    </nav>
  </div>
</header>

<section class="wrap">
  <div class="badge">Simple, transparent pricing</div>
  <h1>Choose the plan that fits your stitching</h1>
  <p class="small">Start with a single pattern, save with a pack, or go unlimited.</p>

  {% if message %}
  <div class="notice">{{ message }}</div>
  {% endif %}

  <div class="cards">
    <!-- Single Pattern -->
    <div class="card">
      <h3>Single Pattern</h3>
      <div class="price">$25</div>
      <p class="small">Best for one-off projects.</p>
      <ul class="list">
        <li>1 professional pattern conversion</li>
        <li>High-resolution output</li>
        <li>Advanced color reduction</li>
        <li>Customizable grid (10×10 + numbering)</li>
        <li>Multi-format export (PNG, PDF, CSV)</li>
        <li>Fabric size calculator</li>
      </ul>
      <button class="btn" type="button">Buy single (coming soon)</button>
      <p class="small">One payment, one finished pattern.</p>
    </div>

    <!-- 10-Pattern Pack -->
    <div class="card">
      <h3>10-Pattern Pack</h3>
      <div class="price">$60</div>
      <p class="small">Great for consistent hobby use.</p>
      <ul class="list">
        <li>10 pattern conversions</li>
        <li>Credits never expire</li>
        <li>Priority processing over free tier</li>
        <li>All export formats included</li>
        <li>Premium palette options</li>
      </ul>
      <button class="btn ghost" type="button">Buy 10-pack (coming soon)</button>
      <p class="small">Save big vs buying singles.</p>
    </div>

    <!-- 3-Month Unlimited -->
    <div class="card">
      <h3>3-Month Unlimited</h3>
      <div class="price">$75 / 3 months</div>
      <p class="small">Short-term unlimited access.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>Higher-resolution output</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <button class="btn ghost" type="button">Start 3-month (coming soon)</button>
      <p class="small">Perfect for focused projects or seasons.</p>
    </div>

    <!-- Annual Pro Unlimited -->
    <div class="card">
      <h3>Pro Annual Unlimited</h3>
      <div class="price">$99 / year</div>
      <p class="small">Unlimited patterns all year.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>4× resolution for large projects</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <button class="btn ghost" type="button">Go Pro annual (coming soon)</button>
      <p class="small">Best value if you stitch more than 4 patterns a year.</p>
    </div>
  </div>
</section>

<section class="wrap">
  <h2>How PatternCraft.app works</h2>
  <p class="small">From photo to stitch-ready pattern in three simple steps.</p>
  <div class="steps">
    <div class="step-card">
      <div class="step-num">1</div>
      <h3>Upload your image</h3>
      <p class="small">
        Start with a photo, artwork, or logo. PatternCraft analyzes it for stitchable detail.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3>Choose size & colors</h3>
      <p class="small">
        Set stitch width, cloth count, and palette size. Preview your grid with bold 10×10 guides.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3>Download your pattern ZIP</h3>
      <p class="small">
        Download a ZIP with grid.png, legend.csv, meta.json, and optional PDF or embroidery files,
        ready to print or import into your workflow.
      </p>
    </div>
  </div>
</section>

<footer class="wrap small">
  <div class="row" style="justify-content:space-between">
    <div>© PatternCraft.app</div>
    <div><a href="/" class="btn ghost">Back to tool</a></div>
  </div>
</footer>
"""

# (no explicit __main__ block needed on Render; you can add one for local dev if you like)
# if __name__ == "__main__":
#     app.run(debug=True)

