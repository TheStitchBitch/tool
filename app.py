from __future__ import annotations

import io
import json
import math
import os
import time
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple

from flask import (
    Flask,
    jsonify,
    make_response,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import check_password_hash, generate_password_hash

# ---------------------- CONFIG & PATHS ----------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PATTERN_DIR = os.path.join(BASE_DIR, "patterns")

os.makedirs(PATTERN_DIR, exist_ok=True)

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels
PATTERN_TTL_SECONDS = 7 * 24 * 60 * 60  # 1 week


# ---------------------- UTIL: USERS ----------------------


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
    return load_users().get(email)


# ---------------------- UTIL: PATTERNS ----------------------


def cleanup_old_patterns() -> None:
    """Delete patterns older than PATTERN_TTL_SECONDS."""
    now = time.time()
    for name in os.listdir(PATTERN_DIR):
        if not name.endswith(".meta.json"):
            continue
        meta_path = os.path.join(PATTERN_DIR, name)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            # If meta is unreadable, drop the whole bundle
            base = name[: -len(".meta.json")]
            for suffix in ("_grid.png", "_legend.csv", ".zip", ".meta.json"):
                try:
                    os.remove(os.path.join(PATTERN_DIR, base + suffix))
                except FileNotFoundError:
                    pass
            continue

        ts = float(meta.get("created_at", 0.0))
        if now - ts > PATTERN_TTL_SECONDS:
            base = name[: -len(".meta.json")]
            for suffix in ("_grid.png", "_legend.csv", ".zip", ".meta.json"):
                try:
                    os.remove(os.path.join(PATTERN_DIR, base + suffix))
                except FileNotFoundError:
                    pass


def save_pattern_for_user(
    email: str,
    original_filename: str,
    ptype: str,
    grid_bytes: bytes,
    legend_text: str,
    meta_extra: dict,
    zip_bytes: bytes,
) -> str:
    pattern_id = uuid.uuid4().hex[:12]
    created_at = time.time()

    base = os.path.join(PATTERN_DIR, pattern_id)
    grid_path = base + "_grid.png"
    legend_path = base + "_legend.csv"
    meta_path = base + ".meta.json"
    zip_path = base + ".zip"

    with open(grid_path, "wb") as f:
        f.write(grid_bytes)

    with open(legend_path, "w", encoding="utf-8") as f:
        f.write(legend_text)

    meta = {
        "id": pattern_id,
        "user": email,
        "created_at": created_at,
        "ptype": ptype,
        "original_filename": original_filename,
    }
    meta.update(meta_extra or {})
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    return pattern_id


def load_user_patterns(email: str) -> List[dict]:
    cleanup_old_patterns()
    patterns: List[dict] = []
    now = time.time()
    for name in os.listdir(PATTERN_DIR):
        if not name.endswith(".meta.json"):
            continue
        meta_path = os.path.join(PATTERN_DIR, name)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("user") != email:
            continue
        ts = float(meta.get("created_at", 0.0))
        if now - ts > PATTERN_TTL_SECONDS:
            continue
        pid = meta.get("id")
        if not pid:
            continue
        patterns.append(
            {
                "id": pid,
                "original_filename": meta.get("original_filename") or "Pattern",
                "ptype": meta.get("ptype", "cross"),
                "created_at": time.strftime("%Y-%m-%d", time.localtime(ts)),
            }
        )
    patterns.sort(key=lambda p: p["created_at"], reverse=True)
    return patterns


def get_pattern_paths(pattern_id: str) -> Optional[dict]:
    base = os.path.join(PATTERN_DIR, pattern_id)
    meta_path = base + ".meta.json"
    grid_path = base + "_grid.png"
    legend_path = base + "_legend.csv"
    zip_path = base + ".zip"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return {
        "meta": meta,
        "meta_path": meta_path,
        "grid_path": grid_path,
        "legend_path": legend_path,
        "zip_path": zip_path,
    }


# ---------------------- IMAGE HELPERS ----------------------


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


def draw_grid(base: Image.Image, cell_px: int, show_grid: bool = True) -> Image.Image:
    """Scale each stitch to a cell and optionally overlay a 10×10 grid."""
    sx, sy = base.size
    out = base.resize((sx * cell_px, sy * cell_px), Image.Resampling.NEAREST)
    if not show_grid:
        return out
    draw = ImageDraw.Draw(out)
    thin = (0, 0, 0, 80)
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
    base: Image.Image,
    cell_px: int,
    sym_map: Dict[Tuple[int, int, int], str],
    show_grid: bool = True,
) -> Image.Image:
    """Overlay symbol per stitch, then optional grid. No diagonal overlays."""
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

    if show_grid:
        thin = (0, 0, 0, 80)
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
    resized = resize_for_stitch_width(img, stitches_w)
    w, h = resized.size
    preview_h = max(1, int(round(h * row_aspect)))
    return resized.resize((w, preview_h), Image.Resampling.NEAREST)


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
    if not paths:
        return out
    svg_points = " ".join([f"{int(x * scale)},{int(y * scale)}" for x, y in paths])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
        f'{int(paths[-1][0] * scale + 10)} {int(paths[-1][1] * scale + 10)}">'
        f'<polyline fill="none" stroke="black" stroke-width="1" points="{svg_points}"/></svg>'
    )
    out["embroidery.svg"] = svg.encode("utf-8")
    return out


# ---------------------- ROUTES: BASIC & AUTH ----------------------


@app.errorhandler(500)
def on_error(_e):
    html = """
    <!doctype html>
    <title>We hit a snag — PatternCraft.app</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <body style="font-family:system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;background:#FFFDF3;margin:0;padding:40px;display:flex;justify-content:center;">
      <div style="max-width:520px;background:white;padding:24px;border-radius:16px;box-shadow:0 18px 45px rgba(15,23,42,.18);border:1px solid #F1E4B8;">
        <h1 style="margin-top:0;font-size:1.6rem;">We hit a snag</h1>
        <p>Something went wrong while processing your request.</p>
        <p>You can go back to <a href="/" style="color:#b45309;font-weight:600;text-decoration:none;">PatternCraft.app</a> and try again.</p>
      </div>
    </body>
    """
    return make_response(html, 500)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def index() -> str:
    user = get_current_user()
    patterns = load_user_patterns(user["email"]) if user else []
    return render_template_string(HOMEPAGE_HTML, user=user, patterns=patterns)


@app.get("/signup")
def signup_get() -> str:
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
        return render_template_string(
            SIGNUP_HTML, message="Password must be at least 8 characters long."
        )
    if password != confirm:
        return render_template_string(SIGNUP_HTML, message="Passwords do not match.")

    users = load_users()
    if email in users:
        return render_template_string(
            SIGNUP_HTML,
            message="This email already has an account. Try logging in instead.",
        )

    users[email] = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": time.time(),
    }
    save_users(users)
    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/login")
def login_get() -> str:
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
        return render_template_string(LOGIN_HTML, message="Please enter both email and password.")

    users = load_users()
    stored = users.get(email)
    if not stored or not check_password_hash(stored.get("password_hash", ""), password):
        return render_template_string(LOGIN_HTML, message="Incorrect email or password.")

    session["user_email"] = email
    return redirect(url_for("index"))


@app.get("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("index"))


# ---------------------- ROUTES: PATTERNS ----------------------


@app.get("/patterns")
def patterns_page():
    user = get_current_user()
    if not user:
        return redirect(url_for("login_get", msg="Log in to see your saved patterns."))
    patterns = load_user_patterns(user["email"])
    return render_template_string(PATTERNS_HTML, user=user, patterns=patterns)


@app.get("/patterns/<pattern_id>/grid")
def pattern_grid(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login_get"))
    info = get_pattern_paths(pattern_id)
    if not info or info["meta"].get("user") != user["email"]:
        return make_response("Not found", 404)
    return send_file(info["grid_path"], mimetype="image/png")


@app.get("/patterns/<pattern_id>/legend")
def pattern_legend(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login_get"))
    info = get_pattern_paths(pattern_id)
    if not info or info["meta"].get("user") != user["email"]:
        return make_response("Not found", 404)
    return send_file(
        info["legend_path"],
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{pattern_id}_legend.csv",
    )


@app.get("/patterns/<pattern_id>/download")
def pattern_download(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login_get"))
    info = get_pattern_paths(pattern_id)
    if not info or info["meta"].get("user") != user["email"]:
        return make_response("Not found", 404)
    if not os.path.exists(info["zip_path"]):
        return make_response("Not found", 404)
    return send_file(
        info["zip_path"],
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{pattern_id}.zip",
    )


# ---------------------- ROUTES: CONVERT ----------------------


ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}


@app.post("/api/convert")
def convert():
    user = get_current_user()
    if not user:
        return redirect(url_for("login_get", msg="Create a free account to generate patterns."))

    file = request.files.get("file")
    if not file:
        return make_response(jsonify({"error": "missing_file"}), 400)
    if (file.mimetype or "").lower() not in ALLOWED_MIME:
        return make_response(jsonify({"error": "unsupported_type"}), 400)

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
        show_grid = request.form.get("show_grid") is not None
    except Exception:
        return make_response(jsonify({"error": "invalid_parameters"}), 400)

    try:
        base = open_image(file)
    except Exception:
        return make_response(jsonify({"error": "decode_failed"}), 400)
    if max(base.size) > MAX_DIM:
        return make_response(
            jsonify({"error": "image_too_large", "max_dim": MAX_DIM}),
            400,
        )

    original_filename = getattr(file, "filename", "") or "upload"

    out_zip = io.BytesIO()
    grid_png_bytes: bytes
    legend_text: str
    meta_extra: dict = {}

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

            grid_img = draw_grid(quant, cell_px=CELL_PX, show_grid=show_grid)
            pdf_bytes: Optional[bytes] = None
            if want_symbols or want_pdf:
                pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
                sym_map = assign_symbols(pal)
                sym_img = draw_symbols_on_grid(
                    quant, cell_px=CELL_PX, sym_map=sym_map, show_grid=show_grid
                )
                if want_pdf:
                    pdf_buf = io.BytesIO()
                    sym_img.convert("RGB").save(pdf_buf, format="PDF", resolution=300.0)
                    pdf_bytes = pdf_buf.getvalue()
                grid_img = sym_img

            total_stitches = sum(counts.values()) or 1
            lines = ["hex,r,g,b,stitches,percent,skeins_est"]
            for (r, g, b), c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                skeins = skeins_per_color(c, cloth_count, strands, waste_pct / 100.0)
                lines.append(
                    f"{to_hex((r,g,b))},{r},{g},{b},{c},{(100*c/total_stitches):.2f},{skeins:.2f}"
                )
            legend_text = "\n".join(lines)
            z.writestr("legend.csv", legend_text)

            note = (
                "Knitting chart with row height adjusted for typical gauge."
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
            meta_extra = meta
            z.writestr("meta.json", json.dumps(meta, indent=2))

            buf_png = io.BytesIO()
            grid_img.save(buf_png, format="PNG")
            grid_png_bytes = buf_png.getvalue()
            z.writestr("grid.png", grid_png_bytes)
            if want_pdf and pdf_bytes:
                z.writestr("pattern.pdf", pdf_bytes)

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
                "notes": "Simple line-art embroidery path extracted from the image.",
            }
            meta_extra = meta
            legend_text = "Simple embroidery export; see SVG path."
            z.writestr("meta.json", json.dumps(meta, indent=2))
            blank = Image.new("RGB", small.size, "white")
            buf_png = io.BytesIO()
            blank.save(buf_png, format="PNG")
            grid_png_bytes = buf_png.getvalue()
            z.writestr("grid.png", grid_png_bytes)
        else:
            return make_response(jsonify({"error": "unknown_ptype"}), 400)

    out_zip.seek(0)
    zip_bytes = out_zip.getvalue()

    # Save pattern bundle for "My patterns"
    save_pattern_for_user(
        email=user["email"],
        original_filename=original_filename,
        ptype=ptype,
        grid_bytes=grid_png_bytes,
        legend_text=legend_text,
        meta_extra=meta_extra,
        zip_bytes=zip_bytes,
    )

    out_zip.seek(0)
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
  <title>PatternCraft.app — Picture in, pattern out</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root{
      --bg:#FFFDF3;
      --card:#FFFFFF;
      --card-soft:#FFFBEB;
      --accent:#F59E0B;
      --accent-strong:#B45309;
      --accent-soft:#FEF3C7;
      --fg:#1F2933;
      --muted:#6B7280;
      --border:#F1E4B8;
      --pill-radius:999px;
      --radius:18px;
      --shadow:0 18px 45px rgba(15,23,42,.18);
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#FEF9C3 0,#FFFDF3 50%,transparent 70%),
        radial-gradient(circle at bottom right,#FFEDD5 0,#FFFDF3 40%,transparent 65%),
        linear-gradient(to bottom,#FFFDF3,#FFF7ED);
    }
    a{color:var(--accent-strong);text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1080px;margin:0 auto;padding:24px 16px 40px;}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      display:flex;align-items:center;gap:8px;
      font-weight:800;font-size:20px;letter-spacing:.05em;text-transform:uppercase;
    }
    .brand-logo{
      width:28px;height:28px;border-radius:9px;
      background:conic-gradient(from 140deg,#FBBF24,#FDBA74,#F9A8D4,#93C5FD,#FBBF24);
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 0 0 2px rgba(255,255,255,.8),0 10px 25px rgba(248,181,0,.55);
      font-size:14px;font-weight:900;color:#1F2933;
    }
    .top-links{
      font-size:13px;color:var(--muted);
      display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    }
    .pill{
      display:inline-flex;align-items:center;justify-content:center;
      min-width:150px;
      padding:11px 20px;border-radius:var(--pill-radius);
      border:none;
      background:linear-gradient(135deg,#FBBF24,#F97316);
      color:#1F2933;font-weight:700;font-size:14px;
      cursor:pointer;
      box-shadow:0 14px 30px rgba(248,181,0,.60);
      text-decoration:none;
      transition:transform .08s,box-shadow .08s,filter .08s;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 18px 40px rgba(180,83,9,.75);
      filter:brightness(1.02);
    }
    .pill.secondary{
      background:#FFF;border:1px solid var(--border);
      box-shadow:0 8px 20px rgba(148,163,184,.25);
      color:var(--accent-strong);
      min-width:130px;
    }
    .pill.secondary:hover{
      box-shadow:0 12px 28px rgba(148,163,184,.35);
    }
    .pill.small{min-width:0;padding:8px 14px;font-size:13px;}
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:4px 10px;border-radius:999px;
      background:#FEF3C7;border:1px solid #FDE68A;
      font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#92400E;
    }
    .chip-dot{width:7px;height:7px;border-radius:999px;background:#22C55E;}
    .layout{
      display:grid;grid-template-columns:minmax(0,2.1fr) minmax(0,2fr);
      gap:22px;align-items:start;
    }
    .card{
      background:var(--card);border-radius:var(--radius);
      border:1px solid var(--border);box-shadow:var(--shadow);
      padding:20px;
    }
    h1{font-size:2.4rem;margin:8px 0 10px;}
    .hero-text p{margin:0 0 10px;color:var(--muted);max-width:480px;}
    .hero-cta{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;}
    .hero-note{font-size:12px;color:var(--muted);margin-top:8px;}
    .hero-note strong{color:#92400E;}
    ul.feature-list{margin:10px 0 0;padding-left:18px;font-size:13px;color:var(--muted);}
    ul.feature-list li{margin-bottom:3px;}
    .section-title{font-size:1.05rem;margin:0 0 6px;}
    .helper{font-size:12px;color:var(--muted);}
    .tool-grid{display:grid;gap:16px;}
    .field-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;}
    .field-row label{flex:1 1 150px;font-size:12px;}
    .field-row input,.field-row select{
      width:100%;margin-top:4px;padding:6px 8px;border-radius:8px;
      border:1px solid #E5E7EB;font-size:13px;
    }
    .field-row input:focus,.field-row select:focus{
      outline:none;border-color:#F59E0B;box-shadow:0 0 0 1px rgba(245,158,11,.45);
    }
    fieldset{
      border:1px solid #F3E3B0;border-radius:12px;
      padding:8px 10px;margin:10px 0;
    }
    legend{font-size:12px;color:#92400E;padding:0 4px;}
    label{font-size:12px;}
    .file{
      border:2px dashed #FBBF24;background:var(--card-soft);
      border-radius:14px;padding:14px;
      display:flex;align-items:center;gap:12px;cursor:pointer;
      transition:background .1s,border-color .1s,box-shadow .1s,transform .1s;
    }
    .file.ready{
      background:#ECFCCB;border-color:#65A30D;
      box-shadow:0 10px 26px rgba(101,163,13,.45);
    }
    .file:hover{
      background:#FFFBEB;border-color:#F97316;
      transform:translateY(-1px);
      box-shadow:0 8px 20px rgba(248,181,0,.45);
    }
    .file input{display:none;}
    .file-main{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;}
    .file-sub{font-size:12px;color:var(--muted);}
    .note-pill{
      display:inline-flex;align-items:center;gap:6px;
      padding:5px 10px;border-radius:999px;
      background:#ECFCCB;color:#365314;font-size:11px;margin-top:6px;
    }
    .note-dot{width:7px;height:7px;border-radius:999px;background:#22C55E;}
    .patterns-section{margin-top:22px;}
    .pattern-grid{
      display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
      gap:14px;margin-top:12px;
    }
    .pattern-card{
      background:var(--card-soft);border-radius:14px;
      border:1px solid #F3E3B0;padding:10px;
    }
    .pattern-card img{width:100%;border-radius:10px;display:block;}
    .pattern-meta{font-size:12px;color:var(--muted);margin-top:6px;}
    .pattern-actions{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;}
    .muted{color:var(--muted);}
    .hidden{display:none;}
    @media (max-width:860px){
      .layout{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="brand">
        <div class="brand-logo">PC</div>
        <span>PatternCraft.app</span>
      </div>
      <div class="top-links">
        {% if user %}
          Signed in as <strong>{{ user.email }}</strong>
          · <a href="{{ url_for('patterns_page') }}">My patterns</a>
          · <a href="{{ url_for('logout') }}">Sign out</a>
        {% else %}
          <a href="{{ url_for('login_get') }}">Log in</a>
          ·
          <a href="{{ url_for('signup_get') }}">Sign up</a>
        {% endif %}
      </div>
    </header>

    <main class="layout">
      <section class="hero-text">
        <div class="chip">
          <span class="chip-dot"></span>
          <span>For cross-stitch, knitting, and simple embroidery</span>
        </div>
        <h1>Picture in, pattern out.</h1>
        <p>
          Turn artwork, quilt mockups, or favorite photos into stitchable charts.
          PatternCraft.app makes tidy grids, color legends, and fabric sizing
          that feel like they came from a seasoned pattern designer.
        </p>

        <div class="hero-cta">
          {% if not user %}
            <a class="pill" href="{{ url_for('signup_get') }}">Create Free Account</a>
            <a class="pill secondary" href="{{ url_for('login_get') }}">Log in</a>
          {% else %}
            <button class="pill" type="button" onclick="document.getElementById('tool').scrollIntoView({behavior:'smooth'});">
              Open the pattern tool
            </button>
            <a class="pill secondary" href="{{ url_for('patterns_page') }}">View my patterns</a>
          {% endif %}
        </div>

        <div class="hero-note">
          <strong>Always free for stitchers.</strong>
          Create an account once, then generate full‑size patterns and legends.
          Recent patterns live in your <a href="{{ url_for('patterns_page') }}">My patterns</a> shelf for a week.
        </div>

        <div style="margin-top:14px;">
          <h2 class="section-title">Why makers use PatternCraft.app</h2>
          <ul class="feature-list">
            <li>Clean grids with bold 10×10 guides and optional symbol overlays.</li>
            <li>Color legends with hex and RGB values for accurate palettes.</li>
            <li>Fabric size estimates driven by stitch count and cloth count.</li>
            <li>Knitting charts with gently compressed row height for truer gauge.</li>
            <li>Simple line‑art exports when you just want embroidery outlines.</li>
          </ul>
        </div>
      </section>

      <section class="card" id="tool">
        {% if not user %}
          <h2 class="section-title">Create a free account to start</h2>
          <p class="helper">
            Once you’re signed in, you can upload artwork, choose cross‑stitch, knitting,
            or simple embroidery, and download full pattern ZIPs with grid and legend.
          </p>
          <a class="pill" href="{{ url_for('signup_get') }}">Create Free Account</a>
          <p class="helper" style="margin-top:10px;">
            You’ll also receive an occasional pattern ideas email from PatternCraft.app.
          </p>
        {% else %}
          <h2 class="section-title">Make a new pattern</h2>
          <p class="helper">
            Upload a picture, pick your stitch style, and PatternCraft.app builds a
            ready‑to‑print chart and color legend. Cross‑stitch charts use full squares;
            knitting charts compress rows slightly to match how knit fabric behaves.
          </p>

          <form method="POST" action="{{ url_for('convert') }}" enctype="multipart/form-data" class="tool-grid">
            <label class="file" id="fileLabel">
              <input id="fileInput" type="file" name="file" accept="image/*" required onchange="onFileChosen(this)">
              <div>
                <div class="file-main">Upload picture here</div>
                <div class="file-sub">
                  Choose a quilt mockup, illustration, or simple photo. Higher contrast works best.
                </div>
              </div>
            </label>

            <div class="note-pill">
              <span class="note-dot"></span>
              <span>Full‑size patterns and legends are included with your free account.</span>
            </div>

            <fieldset>
              <legend>Pattern type</legend>
              <label><input type="radio" name="ptype" value="cross" checked onchange="onTypeChange()"> Cross‑stitch chart</label><br>
              <label><input type="radio" name="ptype" value="knit" onchange="onTypeChange()"> Knitting chart</label><br>
              <label><input type="radio" name="ptype" value="emb" onchange="onTypeChange()"> Simple embroidery line art</label>
            </fieldset>

            <fieldset id="sizeBlock">
              <legend>Size & color</legend>
              <div class="field-row">
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
              <p class="helper">Defaults suit most artwork. Increase stitch width for more detail, or reduce colors for simpler palettes.</p>
            </fieldset>

            <fieldset id="fabricBlock">
              <legend>Fabric & legend</legend>
              <div class="field-row">
                <label>Cloth count (st/in)
                  <input type="number" name="count" value="14" min="10" max="22">
                </label>
                <label>Strands
                  <input type="number" name="strands" value="2" min="1" max="6">
                </label>
                <label>Allow for extra (%) 
                  <input type="number" name="waste" value="20" min="0" max="60">
                </label>
              </div>
              <label><input type="checkbox" name="symbols" checked> Symbol overlay on top of color blocks</label><br>
              <label><input type="checkbox" name="show_grid" checked> 10×10 gridlines on chart</label><br>
              <label><input type="checkbox" name="pdf" checked> Also export a printable PDF</label>
              <p class="helper">Your legend includes hex and RGB values plus estimated skeins by color.</p>
            </fieldset>

            <fieldset id="embBlock" class="hidden">
              <legend>Embroidery tuning</legend>
              <p class="helper">
                For line‑art exports, we trace darker areas of your image into a single running path.
                Use this as a starting point in your machine software.
              </p>
              <div class="field-row">
                <label>Line threshold
                  <input type="number" name="emb_thresh" value="180" min="0" max="255">
                </label>
                <label>Step size (px)
                  <input type="number" name="emb_step" value="3" min="1" max="10">
                </label>
              </div>
            </fieldset>

            <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
              <button class="pill small" type="submit" id="generateBtn">Generate pattern ZIP</button>
              <span class="helper">ZIP includes your grid image, legend CSV, meta notes, and optional PDF.</span>
            </div>
          </form>
        {% endif %}
      </section>
    </main>

    {% if user %}
      <section class="patterns-section">
        <h2 class="section-title">My patterns (last 7 days)</h2>
        {% if patterns %}
          <div class="pattern-grid">
            {% for p in patterns %}
              <div class="pattern-card">
                <img src="{{ url_for('pattern_grid', pattern_id=p.id) }}" alt="Pattern preview">
                <div class="pattern-meta">
                  <strong>{{ p.original_filename }}</strong><br>
                  {{ p.ptype|capitalize }} · made {{ p.created_at }}
                </div>
                <div class="pattern-actions">
                  <a class="pill small secondary" href="{{ url_for('pattern_download', pattern_id=p.id) }}">Download ZIP</a>
                  <a class="pill small" href="{{ url_for('pattern_legend', pattern_id=p.id) }}">Legend CSV</a>
                </div>
              </div>
            {% endfor %}
          </div>
        {% else %}
          <p class="helper">New here? Your recent charts will appear here after you generate patterns.</p>
        {% endif %}
      </section>
    {% endif %}
  </div>

<script>
  function onFileChosen(inp){
    const label = document.getElementById('fileLabel');
    if (!label) return;
    if (!inp.files || !inp.files[0]){
      label.classList.remove('ready');
      return;
    }
    label.classList.add('ready');
  }

  function setStyleOptions(type){
    const sel = document.getElementById('stitch_style');
    if (!sel) return;
    sel.innerHTML = '';
    let opts = [];
    if(type === 'cross'){
      opts = [
        ['full','Full cross‑stitches'],
        ['half','Half stitches'],
        ['back','Cross‑stitch with backstitch accents']
      ];
    } else if(type === 'knit'){
      opts = [
        ['stockinette','Stockinette'],
        ['garter','Garter'],
        ['seed','Seed'],
        ['rib1','Rib 1×1']
      ];
    } else {
      opts = [['run','Single running line']];
    }
    for(const [val,label] of opts){
      const o = document.createElement('option');
      o.value = val;
      o.textContent = label;
      sel.appendChild(o);
    }
  }

  function onTypeChange(){
    const checked = document.querySelector('input[name="ptype"]:checked');
    const type = checked ? checked.value : 'cross';
    const emb = document.getElementById('embBlock');
    const fabric = document.getElementById('fabricBlock');
    const size = document.getElementById('sizeBlock');
    if(type === 'emb'){
      emb.classList.remove('hidden');
      fabric.classList.add('hidden');
    } else {
      emb.classList.add('hidden');
      fabric.classList.remove('hidden');
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
  <style>
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
      background:#FFFDF3;
      color:#111827;
      display:flex;
      justify-content:center;
    }
    .wrap{max-width:500px;width:100%;padding:32px 16px 40px;}
    .card{
      background:#FFFFFF;
      border-radius:18px;
      border:1px solid #F1E4B8;
      box-shadow:0 18px 45px rgba(15,23,42,.18);
      padding:22px;
    }
    h1{margin:0 0 8px;font-size:1.7rem;}
    .muted{font-size:13px;color:#6B7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #E5E7EB;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#F59E0B;box-shadow:0 0 0 1px rgba(245,158,11,.45);
    }
    .pill{
      margin-top:16px;
      padding:10px 18px;
      border-radius:999px;
      border:none;
      background:linear-gradient(135deg,#FBBF24,#F97316);
      color:#1F2933;
      font-size:14px;
      font-weight:700;
      cursor:pointer;
      box-shadow:0 14px 30px rgba(248,181,0,.60);
      width:100%;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 18px 40px rgba(180,83,9,.75);
    }
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;}
    a{color:#B45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4B5563;padding-left:18px;margin-top:8px;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Create your free PatternCraft.app account</h1>
      <p class="muted">
        One signup, unlimited full‑size patterns. We keep your recent charts in
        <strong>My patterns</strong> for a week so you can re‑download or print them again.
      </p>
      <ul>
        <li>No billing details needed – this tool is free for stitchers.</li>
        <li>We may send a monthly ideas email with color stories and pattern tips.</li>
      </ul>
      <form method="POST" action="{{ url_for('signup_post') }}">
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
        Already have an account? <a href="{{ url_for('login_get') }}">Log in instead</a>.
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
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
      background:#FFFDF3;
      color:#111827;
      display:flex;
      justify-content:center;
    }
    .wrap{max-width:500px;width:100%;padding:32px 16px 40px;}
    .card{
      background:#FFFFFF;
      border-radius:18px;
      border:1px solid #F1E4B8;
      box-shadow:0 18px 45px rgba(15,23,42,.18);
      padding:22px;
    }
    h1{margin:0 0 8px;font-size:1.7rem;}
    .muted{font-size:13px;color:#6B7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #E5E7EB;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#F59E0B;box-shadow:0 0 0 1px rgba(245,158,11,.45);
    }
    .pill{
      margin-top:16px;
      padding:10px 18px;
      border-radius:999px;
      border:none;
      background:linear-gradient(135deg,#FBBF24,#F97316);
      color:#1F2933;
      font-size:14px;
      font-weight:700;
      cursor:pointer;
      box-shadow:0 14px 30px rgba(248,181,0,.60);
      width:100%;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 18px 40px rgba(180,83,9,.75);
    }
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;}
    a{color:#B45309;text-decoration:none;}
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
      <form method="POST" action="{{ url_for('login_post') }}">
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
        New to PatternCraft.app? <a href="{{ url_for('signup_get') }}">Create a free account</a>.
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
  <style>
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
      background:#FFFDF3;
      color:#111827;
    }
    .wrap{max-width:1000px;margin:0 auto;padding:24px 16px 40px;}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;
    }
    .brand{font-weight:800;letter-spacing:.05em;text-transform:uppercase;}
    .pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:8px 16px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#FBBF24,#F97316);
      color:#1F2933;font-size:13px;font-weight:700;text-decoration:none;
      box-shadow:0 12px 26px rgba(248,181,0,.60);
    }
    .pill.secondary{
      background:#FFF;border:1px solid #F1E4B8;box-shadow:0 6px 18px rgba(148,163,184,.3);
    }
    .card{
      background:#FFFFFF;border-radius:18px;border:1px solid #F1E4B8;
      box-shadow:0 18px 45px rgba(15,23,42,.18);padding:20px;
    }
    h1{margin:0 0 8px;font-size:1.6rem;}
    .helper{font-size:13px;color:#6B7280;}
    .pattern-grid{
      display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
      gap:16px;margin-top:14px;
    }
    .pattern-card{
      background:#FFFBEB;border-radius:14px;border:1px solid #F3E3B0;
      padding:10px;
    }
    .pattern-card img{width:100%;border-radius:10px;display:block;}
    .pattern-meta{font-size:12px;color:#6B7280;margin-top:6px;}
    .pattern-actions{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="brand">PatternCraft.app</div>
      <div>
        <a class="pill secondary" href="{{ url_for('index') }}">Back to tool</a>
        <a class="pill" href="{{ url_for('logout') }}">Sign out</a>
      </div>
    </header>

    <div class="card">
      <h1>My patterns</h1>
      <p class="helper">
        These are the patterns you’ve generated in the last week. Download the ZIP again or
        grab just the legend if you’re re‑planning colors.
      </p>

      {% if patterns %}
        <div class="pattern-grid">
          {% for p in patterns %}
            <div class="pattern-card">
              <img src="{{ url_for('pattern_grid', pattern_id=p.id) }}" alt="Pattern preview">
              <div class="pattern-meta">
                <strong>{{ p.original_filename }}</strong><br>
                {{ p.ptype|capitalize }} · made {{ p.created_at }}
              </div>
              <div class="pattern-actions">
                <a class="pill secondary" href="{{ url_for('pattern_download', pattern_id=p.id) }}">Download ZIP</a>
                <a class="pill secondary" href="{{ url_for('pattern_legend', pattern_id=p.id) }}">Legend CSV</a>
              </div>
            </div>
          {% endfor %}
        </div>
      {% else %}
        <p class="helper" style="margin-top:12px;">
          You haven’t saved any patterns yet. Head back to the tool, upload a picture,
          and your charts will appear here automatically.
        </p>
        <a class="pill" href="{{ url_for('index') }}" style="margin-top:12px;display:inline-flex;">Go to pattern tool</a>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
