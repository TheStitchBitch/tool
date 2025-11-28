from __future__ import annotations
import io
import json
import math
import os
import zipfile
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import uuid

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

import stripe

# Optional embroidery support
try:
    from pyembroidery import EmbPattern, write_dst, write_pes  # type: ignore
    HAS_PYEMB = True
except Exception:
    HAS_PYEMB = False

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ------------- STRIPE CONFIG (LIVE) -------------

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# From latest live prices you shared
# Single Pattern – $25
STRIPE_PRICE_SINGLE = "price_1SXNyWCINTImVye2jayzoKKj"
# 10 Pattern Pack – $60
STRIPE_PRICE_PACK10 = "price_1SXNyRCINTImVye2m433u7pL"
# Pro Annual – $99/year (recurring)
STRIPE_PRICE_ANNUAL = "price_1SXNyNCINTImVye2rcxl5LsO"
# 3‑Month Unlimited – $75 every 3 months (recurring)
STRIPE_PRICE_3MO = "price_1SXTFUCINTImVye2JwOxUN55"

# ------------- SIMPLE JSON “DB” FOR USERS -------------

BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# Per‑user patterns are stored under pattern_store/<user_key>/...
PATTERN_ROOT = os.path.join(BASE_DIR, "pattern_store")
os.makedirs(PATTERN_ROOT, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels


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


def _user_key(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_")


def get_user_patterns(email: str) -> List[dict]:
    users = load_users()
    user = users.get(email)
    if not user:
        return []
    patterns = user.get("patterns")
    if not isinstance(patterns, list):
        return []
    return patterns


def find_pattern_for_user(email: str, pattern_id: str) -> Optional[dict]:
    for p in get_user_patterns(email):
        if p.get("id") == pattern_id:
            return p
    return None


def _pattern_paths(rec: dict) -> Tuple[str, Optional[str]]:
    rel_zip = rec.get("rel_zip") or ""
    zip_path = os.path.join(PATTERN_ROOT, rel_zip)
    rel_prev = rec.get("rel_preview")
    preview_path = os.path.join(PATTERN_ROOT, rel_prev) if rel_prev else None
    return zip_path, preview_path


def store_pattern_for_user(
    email: str,
    original_name: str,
    ptype: str,
    meta: dict,
    preview_bytes: Optional[bytes],
    zip_bytes: bytes,
) -> None:
    """
    Persist a generated pattern for later viewing / download / print.
    This is best‑effort; failures here should not break the main download.
    """
    if not zip_bytes:
        return
    users = load_users()
    user = users.get(email)
    if not user:
        return

    user_key = _user_key(email)
    user_dir = os.path.join(PATTERN_ROOT, user_key)
    os.makedirs(user_dir, exist_ok=True)

    pattern_id = uuid.uuid4().hex
    zip_rel = f"{user_key}/{pattern_id}.zip"
    zip_path = os.path.join(PATTERN_ROOT, zip_rel)

    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    preview_rel = None
    if preview_bytes:
        preview_rel = f"{user_key}/{pattern_id}_preview.png"
        preview_path = os.path.join(PATTERN_ROOT, preview_rel)
        with open(preview_path, "wb") as f:
            f.write(preview_bytes)

    record = {
        "id": pattern_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "name": original_name or "Pattern",
        "ptype": ptype,
        "meta": meta or {},
        "rel_zip": zip_rel,
        "rel_preview": preview_rel,
    }

    patterns = user.get("patterns") or []
    patterns.append(record)
    user["patterns"] = patterns
    users[email] = user
    save_users(users)


# ------------- IMAGE / PATTERN HELPERS -------------

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


# ------------- BASIC ROUTES / HEALTH / ROBOTS -------------

@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/robots.txt")
def robots_txt():
    body = (
        "User-agent: *\n"
        "Disallow: /api/\n"
        "Disallow: /patterns\n"
        "Disallow: /sample-pattern.zip\n"
        "Allow: /\n"
    )
    return make_response(body, 200, {"Content-Type": "text/plain"})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


@app.errorhandler(Exception)
def on_error(e):
    return make_response(jsonify({"error": "server_error"}), 500)


# ------------- HOMEPAGE / PRICING / CHECKOUT / SUCCESS -------------

@app.get("/")
def index() -> str:
    user = get_current_user()
    return render_template_string(HOMEPAGE_HTML, user=user)


@app.get("/pricing")
def pricing() -> str:
    user = get_current_user()
    reason = request.args.get("reason", "")
    message = ""
    if reason == "used_free":
        message = "You’ve already used your included PatternCraft.app pattern for this account. Choose a plan to keep generating patterns."
    elif reason == "no_credits":
        message = "You’ve used all credits on this account. Pick a pack or unlimited plan to continue."
    elif reason == "checkout_error":
        message = "We couldn’t start checkout. Please try again or contact support."
    return render_template_string(PRICING_HTML, user=user, message=message)


@app.post("/checkout")
def create_checkout():
    """
    Stripe Checkout entry-point.
    Requires login so we can tie purchases to an email.
    """
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log+in+or+create+an+account+before+purchasing+a+plan."))

    plan = (request.form.get("plan") or "").strip()

    if plan == "single":
        price_id = STRIPE_PRICE_SINGLE
        mode = "payment"
    elif plan == "pack10":
        price_id = STRIPE_PRICE_PACK10
        mode = "payment"
    elif plan == "unlimited_3m":
        price_id = STRIPE_PRICE_3MO
        mode = "subscription"
    elif plan == "unlimited_year":
        price_id = STRIPE_PRICE_ANNUAL
        mode = "subscription"
    else:
        return redirect(url_for("pricing"))

    try:
        checkout_session = stripe.checkout.Session.create(
            mode=mode,
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=email,
            client_reference_id=email,
            success_url=url_for("success", _external=True) + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=url_for("pricing", _external=True),
        )
    except Exception:
        return redirect(url_for("pricing", reason="checkout_error"))

    return redirect(checkout_session.url)


@app.get("/success")
def success():
    user = get_current_user()
    return render_template_string(SUCCESS_HTML, user=user)


# ------------- SIGNUP / LOGIN / LOGOUT -------------

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
        # subscription: free, single, pack10, unlimited_3m, unlimited_year
        "subscription": "free",
        "free_used": False,
        "credits": 0,  # for credit-based packs
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
                    msg="We couldn’t match that email and password after several attempts. Create a PatternCraft.app account to get started.",
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


# ------------- PATTERN HISTORY ROUTES -------------

@app.get("/patterns")
def patterns_list():
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log+in+to+see+your+saved+patterns."))
    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(url_for("signup", msg="Create+a+PatternCraft.app+account+to+get+started."))

    patterns = get_user_patterns(email)
    patterns_sorted = sorted(
        patterns,
        key=lambda p: p.get("created_at", ""),
        reverse=True,
    )
    return render_template_string(PATTERN_LIST_HTML, user=user, patterns=patterns_sorted)


@app.get("/patterns/<pattern_id>")
def pattern_detail(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log+in+to+view+your+patterns."))
    rec = find_pattern_for_user(email, pattern_id)
    if not rec:
        return "Pattern not found.", 404
    meta = rec.get("meta") or {}
    legend = meta.get("legend") or []
    preview_url = (
        url_for("pattern_preview", pattern_id=pattern_id)
        if rec.get("rel_preview")
        else None
    )
    return render_template_string(
        PATTERN_DETAIL_HTML,
        user=get_current_user(),
        pattern=rec,
        legend=legend,
        preview_url=preview_url,
    )


@app.get("/patterns/<pattern_id>/download")
def pattern_download(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log+in+to+download+your+patterns."))
    rec = find_pattern_for_user(email, pattern_id)
    if not rec or not rec.get("rel_zip"):
        return "Pattern not found.", 404
    zip_path, _preview = _pattern_paths(rec)
    if not os.path.exists(zip_path):
        return "Pattern file missing.", 404
    download_name = f"{(rec.get('name') or 'pattern').replace(' ', '_')}_{pattern_id}.zip"
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@app.get("/patterns/<pattern_id>/preview")
def pattern_preview(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log+in+to+preview+your+patterns."))
    rec = find_pattern_for_user(email, pattern_id)
    if not rec or not rec.get("rel_preview"):
        return "No preview available for this pattern.", 404
    _zip_path, preview_path = _pattern_paths(rec)
    if not preview_path or not os.path.exists(preview_path):
        return "Preview file missing.", 404
    return send_file(preview_path, mimetype="image/png")


@app.get("/patterns/<pattern_id>/print")
def pattern_print(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log+in+to+print+your+patterns."))
    rec = find_pattern_for_user(email, pattern_id)
    if not rec:
        return "Pattern not found.", 404
    preview_url = (
        url_for("pattern_preview", pattern_id=pattern_id)
        if rec.get("rel_preview")
        else None
    )
    return render_template_string(
        PATTERN_PRINT_HTML,
        user=get_current_user(),
        pattern=rec,
        preview_url=preview_url,
    )


# ------------- SAMPLE PATTERN ZIP -------------

@app.get("/sample-pattern.zip")
def sample_pattern_zip():
    """
    Serve a sample quilt-style pattern ZIP with a grid and legend.
    Uses a synthetic patchwork-style chart generated in code.
    """
    w, h = 40, 40
    base = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(base)
    colors = [
        (239, 68, 68),
        (249, 115, 22),
        (234, 179, 8),
        (34, 197, 94),
        (59, 130, 246),
    ]

    patch = 8
    for py in range(0, h, patch):
        for px in range(0, w, patch):
            block_idx = (px // patch + py // patch) % len(colors)
            base_color = colors[block_idx]
            alt_color = colors[(block_idx + 2) % len(colors)]

            style = (px // patch + 2 * (py // patch)) % 3

            if style == 0:
                draw.rectangle((px, py, px + patch - 1, py + patch - 1), fill=base_color)
            elif style == 1:
                for y in range(patch):
                    for x in range(patch):
                        if x >= y:
                            draw.point((px + x, py + y), fill=base_color)
                        else:
                            draw.point((px + x, py + y), fill=alt_color)
            else:
                for y in range(patch):
                    color = base_color if y % 2 == 0 else alt_color
                    draw.line((px, py + y, px + patch - 1, py + y), fill=color)

    max_colors = len(colors)
    quant = quantize(base, max_colors)
    counts = palette_counts(quant)
    sx, sy = quant.size

    cloth_count = 10
    strands = 2
    waste_pct = 20
    finished_w_in = round(sx / float(cloth_count), 2)
    finished_h_in = round(sy / float(cloth_count), 2)

    grid_img = draw_grid(quant, cell_px=CELL_PX)
    pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
    sym_map = assign_symbols(pal)
    sym_img = draw_symbols_on_grid(quant, cell_px=CELL_PX, sym_map=sym_map)
    grid_img = sym_img

    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
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


# ------------- PATTERN GENERATOR (ACCOUNT + MEMBERSHIP GATED) -------------

@app.post("/api/convert")
def convert():
    # Light anti-scraping: block obvious non-browser user-agents
    ua = (request.headers.get("User-Agent") or "").lower()
    if any(bot in ua for bot in ("curl", "wget", "python-requests", "httpclient", "scrapy", "bot", "spider")):
        return jsonify({"error": "automation_blocked"}), 403

    # Require login
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log+in+to+generate+patterns."))

    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(url_for("signup", msg="Create+your+PatternCraft.app+account+to+continue."))

    subscription = user.get("subscription", "free")
    credits = int(user.get("credits", 0) or 0)
    mark_free_used = False
    consume_credit = False

    # Membership logic:
    # - unlimited_3m / unlimited_year: unlimited usage
    # - if credits > 0: consume 1 credit
    # - else: free tier, 1 pattern per account
    if subscription in ("unlimited_3m", "unlimited_year"):
        pass
    elif credits > 0:
        consume_credit = True
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

    original_name = file.filename or "Pattern"
    try:
        ptype = request.form.get("ptype", "cross")
        stitch_style = request.form.get("stitch_style", "full")

        # Higher limits for subscription users (match “higher resolution / advanced color tools” copy)
        width_limit = 400
        color_limit = 60
        if subscription in ("unlimited_3m", "unlimited_year"):
            width_limit = 600
            color_limit = 80

        stitch_w = clamp(int(request.form.get("width", 120)), 20, width_limit)
        max_colors = clamp(int(request.form.get("colors", 16)), 2, color_limit)

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

    pattern_meta: dict = {}
    preview_bytes: Optional[bytes] = None

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

            # Chart with grid + optional symbols (for stitching)
            chart_img = draw_grid(quant, cell_px=CELL_PX)
            pdf_bytes: Optional[bytes] = None
            if want_symbols or want_pdf:
                pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
                sym_map = assign_symbols(pal)
                sym_img = draw_symbols_on_grid(quant, cell_px=CELL_PX, sym_map=sym_map)
                if want_pdf:
                    pdf_buf = io.BytesIO()
                    sym_img.convert("RGB").save(pdf_buf, format="PDF", resolution=300.0)
                    pdf_bytes = pdf_buf.getvalue()
                chart_img = sym_img

            total_stitches = sum(counts.values()) or 1
            lines = ["hex,r,g,b,stitches,percent,skeins_est"]
            legend_rows: List[dict] = []
            for (r, g, b), c in sorted(
                counts.items(), key=lambda kv: kv[1], reverse=True
            ):
                percent = (100 * c / total_stitches)
                skeins = skeins_per_color(
                    c, cloth_count, strands, waste_pct / 100.0
                )
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
                else "Cross-stitch grid with 10×10 guides."
            )
            pattern_meta = {
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
                "legend": legend_rows,
            }
            z.writestr("meta.json", json.dumps(pattern_meta, indent=2))

            # Preview image for on-site viewing: clean art-style, no grid / watermark
            preview_img = quant.resize((sx * CELL_PX, sy * CELL_PX), Image.Resampling.NEAREST)
            buf_preview = io.BytesIO()
            preview_img.save(buf_preview, format="PNG")
            preview_bytes = buf_preview.getvalue()

            # Chart image (with grid / symbols) for stitch reference in ZIP
            buf_chart = io.BytesIO()
            chart_img.save(buf_chart, format="PNG")
            z.writestr("grid.png", buf_chart.getvalue())

            # Also include the clean preview art as art.png in the ZIP
            buf_art = io.BytesIO()
            preview_img.save(buf_art, format="PNG")
            z.writestr("art.png", buf_art.getvalue())

            if pdf_bytes:
                z.writestr("pattern.pdf", pdf_bytes)

        elif ptype == "emb":
            small = resize_for_stitch_width(base, stitch_w)
            bw = to_monochrome(small, threshold=emb_thresh)
            pts = serpentine_points(bw, step=emb_step)
            for name, data in write_embroidery_outputs(pts).items():
                z.writestr(name, data)
            pattern_meta = {
                "type": "emb",
                "stitch_style": "run",
                "points": len(pts),
                "pyembroidery": HAS_PYEMB,
            }
            z.writestr("meta.json", json.dumps(pattern_meta, indent=2))
        else:
            return jsonify({"error": "unknown_ptype"}), 400

    # update membership usage
    if consume_credit and credits > 0:
        user["credits"] = max(0, credits - 1)
    if mark_free_used:
        user["free_used"] = True
    if consume_credit or mark_free_used:
        users[email] = user
        save_users(users)

    # persist pattern for history (pattern + legend together)
    zip_bytes = out_zip.getvalue()
    try:
        store_pattern_for_user(
            email=email,
            original_name=original_name,
            ptype=ptype,
            meta=pattern_meta,
            preview_bytes=preview_bytes,
            zip_bytes=zip_bytes,
        )
    except Exception:
        # best-effort; ignore storage failures
        pass

    out_zip.seek(0)
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{ptype}.zip",
    )


# ------------- INLINE HTML TEMPLATES -------------

HOMEPAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <!-- Favicon: create static/patterncraft-mark.svg with a simple PC monogram -->
  <link rel="icon" href="/static/patterncraft-mark.svg">
  <style>
    :root{
      --bg:#0f172a;
      --fg:#111827;
      --muted:#6b7280;
      --line:#e5e7eb;
      --radius:18px;
      --accent:#ec4899;
      --accent-soft:#fdf2ff;
      --accent-strong:#be185d;
      --pill:#f97316;
      --shell:#020617;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#f9a8d4 0,#eef2ff 38%,transparent 60%),
        radial-gradient(circle at bottom right,#a5b4fc 0,#fdf2ff 40%,transparent 65%),
        linear-gradient(135deg,#020617,#0f172a);
    }
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1040px;margin:0 auto;padding:22px 14px 40px}
    h1{font-size:2.4rem;margin:0 0 6px;letter-spacing:-.03em;}
    h2{margin:0 0 10px;font-size:1.2rem;}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      display:flex;align-items:center;gap:8px;
      font-weight:800;font-size:20px;letter-spacing:.06em;text-transform:uppercase;color:#f9fafb;
    }
    .brand-mark{
      width:24px;height:24px;border-radius:9px;
      background:conic-gradient(from 200deg,var(--accent),#4f46e5,#22c55e,var(--accent));
      display:flex;align-items:center;justify-content:center;
      color:#0f172a;font-size:12px;font-weight:800;
      box-shadow:0 0 0 1px rgba(15,23,42,.4);
    }
    .top-links{
      font-size:13px;color:#e5e7eb;
    }
    .top-links a{margin-left:8px;color:#e5e7eb;}
    .top-links a:hover{color:#fff;}
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:3px 9px;border-radius:999px;
      background:rgba(15,23,42,.85);border:1px solid rgba(148,163,184,.8);
      font-size:11px;color:#e5e7eb;text-transform:uppercase;letter-spacing:.09em;
    }
    .chip-dot{width:7px;height:7px;border-radius:999px;background:#22c55e}
    .card{
      background:rgba(248,250,252,.96);
      border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.4);
      box-shadow:0 18px 45px rgba(15,23,42,.55);
      padding:18px 18px 18px;
      backdrop-filter:blur(18px);
    }
    .hero{
      display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);
      gap:18px;margin-bottom:26px;align-items:center;
    }
    .hero-tagline{color:var(--muted);max-width:430px;font-size:14px;}
    .muted{color:var(--muted);font-size:13px}
    .pill{
      padding:10px 18px;border-radius:999px;
      background:linear-gradient(135deg,var(--pill),#ea580c);
      color:#fff;border:none;cursor:pointer;
      font-size:14px;font-weight:600;letter-spacing:.02em;
      box-shadow:0 8px 24px rgba(248,113,22,.5);
      transition:transform .08s,box-shadow .08s,background .08s;
      display:inline-flex;align-items:center;gap:6px;
      text-decoration:none;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 10px 30px rgba(248,113,22,.6);
    }
    .pill-secondary{
      background:#020617;
      color:#e5e7eb;
      border:1px solid rgba(148,163,184,.7);
      box-shadow:0 6px 18px rgba(15,23,42,.7);
    }
    .pill-secondary:hover{
      background:#0b1120;
      box-shadow:0 10px 26px rgba(15,23,42,.9);
    }
    .pill-ready{
      background:#16a34a;
      box-shadow:0 8px 24px rgba(22,163,74,.6);
    }
    .hero-cta-row{
      display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;align-items:center;
    }
    .hero-note{font-size:12px;color:#e5e7eb;margin-top:8px;}
    .hero-note a{color:#bfdbfe;}
    .hero-note a:hover{color:#fff;}
    .badge-row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
    .badge{
      font-size:11px;padding:3px 8px;border-radius:999px;
      background:rgba(15,23,42,.85);color:#e5e7eb;border:1px solid rgba(148,163,184,.7);
    }

    .demo-shell{
      background:var(--shell);
      border-radius:26px;
      border:1px solid rgba(15,23,42,.9);
      padding:10px 8px 10px;
      box-shadow:0 20px 40px rgba(15,23,42,.9);
      max-width:360px;
      margin:0 auto;
    }
    .demo-header{
      display:flex;justify-content:space-between;align-items:center;
      color:#9ca3af;font-size:11px;padding:0 8px 8px;
    }
    .demo-dots{display:flex;gap:4px;}
    .demo-dot{
      width:6px;height:6px;border-radius:999px;
      background:rgba(148,163,184,.9);
    }
    .demo-screen{
      border-radius:18px;
      overflow:hidden;
      background:#020617;
      border:1px solid rgba(30,64,175,.8);
      position:relative;
    }
    .demo-screen video{
      width:100%;display:block;
    }
    .demo-caption{
      font-size:11px;color:#e5e7eb;
      margin-top:8px;text-align:center;
    }

    .section-card{
      background:rgba(248,250,252,.98);
      border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.4);
      padding:18px;
      box-shadow:0 14px 32px rgba(15,23,42,.45);
      margin-bottom:18px;
    }
    .why-title{
      font-size:1.05rem;
      margin:0 0 4px;
    }
    .why-sub{font-size:13px;color:var(--muted);margin-bottom:10px;}
    .why-list{margin:0;padding-left:20px;font-size:13px;color:#111827;}
    .why-list li{margin-bottom:4px;}

    .make-layout{display:grid;gap:18px;grid-template-columns:minmax(0,1.35fr);}
    .file{
      border:2px dashed rgba(129,140,248,1);
      border-radius:18px;
      padding:16px 14px;
      display:flex;align-items:center;gap:11px;
      cursor:pointer;
      background:rgba(239,246,255,.96);
      transition:background .15s,border-color .15s,transform .1s,box-shadow .1s;
    }
    .file:hover{
      background:#e0ecff;border-color:#4f46e5;
      transform:translateY(-1px);
      box-shadow:0 8px 22px rgba(59,130,246,.45);
    }
    .file-ready{
      background:#dcfce7;
      border-color:#16a34a;
      box-shadow:0 8px 22px rgba(22,163,74,.55);
    }
    .file input{display:none}
    .file-label-main{font-weight:800;font-size:14px;text-transform:uppercase;letter-spacing:.08em}
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
    .legal{
      font-size:11px;color:#9ca3af;margin-top:18px;text-align:center;
    }

    @media (max-width:880px){
      .hero{grid-template-columns:1fr;gap:14px;}
      .demo-shell{max-width:320px;}
    }
  </style>
</head>
<body data-logged-in="{{ '1' if user else '0' }}">
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">PC</div>
      <span>PatternCraft.app</span>
    </div>
    <div class="top-links">
      <a href="/pricing">Pricing</a>
      {% if user %}
        · <a href="/patterns">My patterns</a>
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
        <span>For cross‑stitch, knitting, and quilting</span>
      </div>
      <h1>Turn art into stitchable patterns</h1>
      <p class="hero-tagline">
        PatternCraft.app converts your artwork into cross‑stitch grids, knitting charts,
        and embroidery‑ready files in a few clicks. Upload a picture, set your stitch details,
        and download a pattern ZIP you can print or hand off to your machine software.
      </p>
      <div class="hero-cta-row">
        {% if user %}
          <button class="pill" onclick="document.getElementById('make').scrollIntoView({behavior:'smooth'})">
            <span>Open the tool</span>
          </button>
        {% else %}
          <a class="pill" href="/login?msg=Log+in+to+open+the+PatternCraft+tool.">
            <span>Open the tool</span>
          </a>
        {% endif %}
        <a class="pill pill-secondary" href="/pricing#how">
          See how it works
        </a>
      </div>
      <div class="hero-note">
        Step 1: <a href="/pricing#how">See the process end‑to‑end</a> ·
        Step 2: <a href="/signup">Create your account</a> ·
        Step 3: Upload a photo and generate your pattern ZIP.
      </div>
      <div class="badge-row">
        <span class="badge">One pattern included with every account</span>
        <span class="badge">Built for hobbyists and pattern sellers</span>
      </div>
    </div>

    <div class="demo-shell">
      <div class="demo-header">
        <div class="demo-dots">
          <div class="demo-dot"></div>
          <div class="demo-dot"></div>
          <div class="demo-dot"></div>
        </div>
        <div>Pattern preview</div>
        <div style="font-size:10px;">PC</div>
      </div>
      <div class="demo-screen">
        <!-- Put PC.APP.mp4 into static/PC.APP.mp4 on Render -->
        <video autoplay muted playsinline loop poster="/static/demo-poster.png">
          <source src="/static/PC.APP.mp4" type="video/mp4">
        </video>
      </div>
      <div class="demo-caption">
        Picture in → pattern out. Short demo of selecting a plan, uploading a photo,
        adjusting options, and downloading your finished grid.
      </div>
    </div>
  </div>

  <div class="section-card">
    <h2 class="why-title">Why makers use PatternCraft.app</h2>
    <p class="why-sub">A purpose‑built pattern tool with stitchers in mind:</p>
    <ul class="why-list">
      <li>Clean grids with bold 10×10 guides and optional symbol overlays</li>
      <li>Color legends with hex and RGB values for accurate palettes</li>
      <li>Fabric size estimates based on stitch count and cloth count</li>
      <li>Knitting charts that respect row proportions</li>
      <li>Embroidery line outputs ready for your machine software</li>
    </ul>
  </div>

  <div id="make" class="section-card">
    <h2 class="why-title">Make a pattern</h2>
    <p class="muted">
      Create a PatternCraft.app account or log in to generate patterns. Every account includes one pattern on us.
      After that, the plans on the pricing page keep you creating as often as you’d like.
    </p>
    <div class="make-layout">
      <div class="make-main">
        <form method="POST" action="/api/convert" enctype="multipart/form-data">
          <label class="file" onclick="guardUpload(event)">
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
            <label><input type="radio" name="ptype" value="cross" checked> Cross‑stitch</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="knit"> Knitting</label>
            <label style="margin-left:12px"><input type="radio" name="ptype" value="emb"> Embroidery</label>
          </fieldset>

          <fieldset>
            <legend>Stitch & size</legend>
            <div class="row">
              <label>Stitch width
                <input type="number" name="width" value="120" min="20" max="600">
              </label>
              <label>Max colors
                <input type="number" name="colors" value="16" min="2" max="80">
              </label>
              <label>Stitch style
                <select id="stitch_style" name="stitch_style"></select>
              </label>
            </div>
            <p class="controls-note">Defaults work well for most art. Increase size and colors on Pro plans for more detail.</p>
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

          <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
            <button class="pill" id="generateBtn" type="submit">Generate pattern ZIP</button>
            <span class="muted">
              Your download includes grid.png, art.png, legend.csv, meta.json, and optional pattern.pdf or embroidery files.
            </span>
          </div>
        </form>
      </div>
    </div>
    <div class="legal">
      PatternCraft.app helps you build patterns; always check fabric, floss, and gauge before starting a full project.
    </div>
  </div>

</div>
<script>
  function guardUpload(evt){
    var loggedIn = document.body.getAttribute('data-logged-in') === '1';
    if (!loggedIn){
      evt.preventDefault();
      evt.stopPropagation();
      window.location.href = "/login?msg=Log+in+or+create+a+free+account+to+upload+pictures.+Every+signup+includes+1+free+pattern.";
    }
  }

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
</html>"""

SIGNUP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#0f172a;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#e5e7eb}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:rgba(15,23,42,.98);
      border-radius:16px;border:1px solid #1f2937;
      padding:22px;box-shadow:0 18px 40px rgba(0,0,0,.8);
    }
    h1{margin:0 0 10px;font-size:1.6rem;color:#f9fafb;}
    .muted{font-size:13px;color:#9ca3af}
    label{display:block;font-size:13px;margin-top:12px}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #4b5563;font-size:14px;background:#020617;color:#e5e7eb;
    }
    input:focus{
      outline:none;border-color:#6366f1;box-shadow:0 0 0 1px rgba(129,140,248,.55);
    }
    .pill{
      margin-top:14px;padding:9px 18px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#f97316,#ea580c);color:#fff;
      font-size:14px;font-weight:600;cursor:pointer;
      box-shadow:0 7px 18px rgba(248,113,22,.45);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 10px 26px rgba(248,113,22,.65);}
    .msg{margin-top:10px;font-size:13px;color:#fecaca}
    a{color:#93c5fd;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#9ca3af;padding-left:18px;margin-top:10px}
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
      <li>Use your best email — we occasionally share pattern ideas and product updates.</li>
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
</html>"""

LOGIN_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Log in — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#020617;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#e5e7eb}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:rgba(15,23,42,.98);
      border-radius:16px;border:1px solid #1f2937;
      padding:22px;box-shadow:0 18px 40px rgba(0,0,0,.9);
    }
    h1{margin:0 0 10px;font-size:1.6rem;color:#f9fafb;}
    .muted{font-size:13px;color:#9ca3af}
    label{display:block;font-size:13px;margin-top:12px}
    input{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #4b5563;font-size:14px;background:#020617;color:#e5e7eb;
    }
    input:focus{
      outline:none;border-color:#6366f1;box-shadow:0 0 0 1px rgba(129,140,248,.6);
    }
    .pill{
      margin-top:14px;padding:9px 18px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#4c51bf,#4338ca);color:#fff;
      font-size:14px;font-weight:600;cursor:pointer;
      box-shadow:0 7px 18px rgba(79,70,229,.55);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 10px 26px rgba(79,70,229,.75);}
    .msg{margin-top:10px;font-size:13px;color:#fecaca}
    a{color:#93c5fd;text-decoration:none;}
    a:hover{text-decoration:underline;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Log in to PatternCraft.app</h1>
    <p class="muted">
      Use the email and password you created when you first tried PatternCraft.app.
      New here? Create a free account and get one pattern included.
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
    <a href="/signup"
       class="pill"
       style="display:inline-block;margin-top:10px;background:linear-gradient(135deg,#f97316,#ea580c);box-shadow:0 7px 18px rgba(248,113,22,.55);text-align:center;text-decoration:none;">
      Create free account · 1 pattern included
    </a>
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
  </div>
</div>
</body>
</html>"""

PRICING_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PatternCraft • Pricing</title>
<style>
:root{
  --fg:#0f172a; --muted:#6b7280; --accent:#ec4899; --line:#e5e7eb; --card:#f9fafb;
  --radius:16px; --wrap:1040px;
}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{
  font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  color:var(--fg);
  background:
    radial-gradient(circle at top left,#f9a8d4 0,#eef2ff 40%,transparent 60%),
    radial-gradient(circle at bottom right,#a5b4fc 0,#fdf2ff 40%,transparent 65%),
    linear-gradient(135deg,#020617,#0f172a);
}
.wrap{max-width:var(--wrap);margin:0 auto;padding:20px 14px 32px}
header{position:sticky;top:0;background:#020617d9;border-bottom:1px solid rgba(148,163,184,.5);z-index:5;backdrop-filter:blur(10px);}
.brand{font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:#e5e7eb;font-size:13px;}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.btn{
  display:inline-block;padding:8px 15px;border-radius:999px;
  border:1px solid var(--accent);background:var(--accent);color:#fff;
  text-decoration:none;font-weight:600;cursor:pointer;font-size:13px;
}
.btn.ghost{background:transparent;color:#f9a8d4;border-color:rgba(248,250,252,.5);}
.btn.ghost:hover{background:rgba(15,23,42,.9);}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-top:16px}
.card{
  background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:16px 14px 16px;
  box-shadow:0 16px 34px rgba(15,23,42,.45);
}
.card h3{margin:0 0 6px}
.price{font-size:24px;font-weight:800;margin:4px 0}
.small{font-size:13px;color:var(--muted)}
.list{margin:8px 0 12px;padding-left:20px;font-size:13px}
.badge{
  display:inline-block;background:#fce7f3;color:#9d174d;border-radius:999px;
  padding:4px 12px;font-size:12px;font-weight:600;margin-bottom:6px
}
.notice{
  margin-top:10px;padding:9px 11px;border-radius:10px;
  background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;
  font-size:13px;
}
footer{border-top:1px solid rgba(148,163,184,.6);margin-top:24px;padding-top:10px;color:#cbd5f5;font-size:12px;}
footer a{color:#e5e7eb;text-decoration:none;}
footer a:hover{text-decoration:underline;}
.heading{color:#e5e7eb;margin:8px 0 4px;font-size:1.15rem;}
.subhead{color:#cbd5f5;font-size:13px;margin:0 0 10px;}
@media (max-width:700px){ .cards{grid-template-columns:1fr} }

.steps{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:16px;
  margin-top:16px;
}
.step-card{
  background:rgba(15,23,42,.96);
  border:1px solid rgba(148,163,184,.5);
  border-radius:var(--radius);
  padding:14px;
  color:#e5e7eb;
  box-shadow:0 16px 32px rgba(15,23,42,.9);
}
.step-num{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:24px;height:24px;
  border-radius:999px;
  background:#f97316;
  color:#0f172a;
  font-size:13px;
  font-weight:700;
  margin-bottom:6px;
}
</style>
</head>
<body>
<header>
  <div class="wrap row" style="justify-content:space-between;padding-top:9px;padding-bottom:9px;">
    <div class="brand">PATTERNCRAFT.APP</div>
    <nav class="row">
      <a class="btn ghost" href="/">Tool</a>
      {% if user %}
        <a class="btn ghost" href="/patterns">My patterns</a>
      {% endif %}
      <a class="btn" href="/pricing">Pricing</a>
    </nav>
  </div>
</header>

<section class="wrap">
  <h1 class="heading">Simple, transparent pricing</h1>
  <p class="subhead">Start with a single pattern, save with a pack, or go unlimited on a recurring plan.</p>

  {% if message %}
  <div class="notice">{{ message }}</div>
  {% endif %}

  <div class="cards">
    <!-- Single Pattern -->
    <div class="card">
      <h3>Single Pattern</h3>
      <div class="price">$25</div>
      <p class="small">Single pattern and legend. Use whenever.</p>
      <ul class="list">
        <li>1 professional pattern conversion</li>
        <li>Detailed legend included</li>
        <li>High-resolution grid output</li>
        <li>Use your pattern whenever you like</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="single">
        <button class="btn" type="submit">Buy single</button>
      </form>
      <p class="small">Best for one-off projects or trying PatternCraft.app.</p>
    </div>

    <!-- 10-Pattern Pack -->
    <div class="card">
      <h3>10-Pattern Pack</h3>
      <div class="price">$60</div>
      <p class="small">Great for consistent hobby use.</p>
      <ul class="list">
        <li>10 pattern conversions</li>
        <li>Credits never expire</li>
        <li>Includes all export formats</li>
        <li>Premium palette options</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="pack10">
        <button class="btn ghost" type="submit">Buy 10-pack</button>
      </form>
      <p class="small">Save big vs buying singles.</p>
    </div>

    <!-- 3-Month Unlimited -->
    <div class="card">
      <h3>3-Month Unlimited</h3>
      <div class="price">$75 / 3 months</div>
      <p class="small">Recurring every 3 months. Unlimited patterns while active.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>Higher-resolution output on the tool</li>
        <li>Advanced color tools (more colors allowed)</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_3m">
        <button class="btn ghost" type="submit">Start 3‑month plan</button>
      </form>
      <p class="small">Perfect for focused seasons of heavy stitching.</p>
    </div>

    <!-- Annual Pro Unlimited -->
    <div class="card">
      <h3>Pro Annual Unlimited</h3>
      <div class="price">$99 / year</div>
      <p class="small">Unlimited patterns all year.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>Higher-resolution output for large projects</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_year">
        <button class="btn ghost" type="submit">Go Pro annual</button>
      </form>
      <p class="small">Best value if you stitch more than a few patterns a year.</p>
    </div>
  </div>
</section>

<section class="wrap" id="how">
  <h2 class="heading">How PatternCraft.app works</h2>
  <p class="subhead">From photo to stitch‑ready pattern in a few simple steps.</p>
  <div class="steps">
    <div class="step-card">
      <div class="step-num">1</div>
      <h3>Pick your plan</h3>
      <p class="small">
        Start with a single, grab a credit pack, or choose an unlimited plan if you know you’ll be creating patterns regularly.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3>Upload your image</h3>
      <p class="small">
        Upload a photo, illustration, or logo. PatternCraft analyzes it for stitchable detail and prepares it for grid conversion.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3>Dial in your settings</h3>
      <p class="small">
        Choose stitch type, stitch width, cloth count, and color count. For knitting and embroidery, switch to those chart types with a click.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">4</div>
      <h3>Checkout and generate</h3>
      <p class="small">
        Complete checkout via Stripe, then generate a ZIP that includes your clean chart image, separate art preview, color legend, and metadata for easy reference.
      </p>
    </div>
  </div>
</section>

<footer class="wrap">
  <div class="row" style="justify-content:space-between">
    <div>© PatternCraft.app</div>
    <div><a href="/">Back to tool</a></div>
  </div>
</footer>
</body>
</html>"""

PATTERN_LIST_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>My patterns — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      background:#020617;
      color:#e5e7eb;
    }
    .wrap{max-width:960px;margin:0 auto;padding:24px 14px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      font-weight:800;font-size:18px;letter-spacing:.12em;text-transform:uppercase;color:#f9fafb;
    }
    .nav a{
      color:#cbd5f5;
      font-size:13px;
      margin-left:10px;
      text-decoration:none;
    }
    .nav a:hover{text-decoration:underline;}
    h1{margin:0 0 10px;font-size:1.5rem;color:#f9fafb;}
    .muted{font-size:13px;color:#9ca3af}
    .grid{
      display:flex;
      flex-direction:column;
      gap:12px;
      margin-top:12px;
    }
    .card{
      background:rgba(15,23,42,.98);
      border-radius:14px;
      border:1px solid #1f2937;
      padding:12px 12px;
      display:flex;
      gap:12px;
      box-shadow:0 14px 30px rgba(0,0,0,.85);
    }
    .thumb{
      width:120px;
      border-radius:10px;
      overflow:hidden;
      background:#020617;
      display:flex;
      align-items:center;
      justify-content:center;
      border:1px solid #111827;
      flex-shrink:0;
    }
    .thumb img{
      max-width:100%;
      display:block;
    }
    .thumb-placeholder{
      font-size:11px;
      color:#6b7280;
      padding:8px;
      text-align:center;
    }
    .info{
      flex:1 1 auto;
    }
    .title{
      font-size:14px;
      font-weight:600;
      margin-bottom:2px;
      color:#f9fafb;
    }
    .meta{
      font-size:12px;
      color:#9ca3af;
      margin-bottom:4px;
    }
    .meta span{margin-right:8px;}
    .actions{
      display:flex;
      flex-direction:column;
      justify-content:center;
      gap:6px;
      flex-shrink:0;
    }
    .btn{
      display:inline-block;
      padding:6px 10px;
      border-radius:999px;
      border:none;
      cursor:pointer;
      font-size:12px;
      font-weight:600;
      text-decoration:none;
      text-align:center;
    }
    .btn-primary{
      background:#f97316;
      color:#111827;
      box-shadow:0 6px 18px rgba(248,113,22,.6);
    }
    .btn-primary:hover{
      background:#ea580c;
    }
    .btn-ghost{
      background:transparent;
      color:#e5e7eb;
      border:1px solid #4b5563;
    }
    .btn-ghost:hover{
      background:#111827;
    }
    .empty{
      margin-top:16px;
      font-size:13px;
      color:#9ca3af;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">PATTERNCRAFT.APP</div>
    <div class="nav">
      <a href="/">Tool</a>
      <a href="/pricing">Pricing</a>
      <a href="/logout">Sign out</a>
    </div>
  </div>

  <h1>My patterns</h1>
  <p class="muted">
    Every time you generate a pattern, PatternCraft.app saves a copy here so you can revisit, preview, and print it later.
  </p>

  {% if patterns and patterns|length > 0 %}
    <div class="grid">
      {% for p in patterns %}
        {% set m = p.meta or {} %}
        <div class="card">
          <div class="thumb">
            {% if p.rel_preview %}
              <img src="/patterns/{{ p.id }}/preview" alt="Pattern preview">
            {% else %}
              <div class="thumb-placeholder">
                No preview available for this pattern. Download the ZIP to view details.
              </div>
            {% endif %}
          </div>
          <div class="info">
            <div class="title">{{ p.name or "Pattern" }}</div>
            <div class="meta">
              <span>{{ p.ptype|default("cross")|capitalize }} pattern</span>
              {% if m.stitches_w and m.stitches_h %}
                <span>{{ m.stitches_w }}×{{ m.stitches_h }} stitches</span>
              {% endif %}
              {% if m.colors %}
                <span>{{ m.colors }} colors</span>
              {% endif %}
              {% if p.created_at %}
                <span>Created {{ p.created_at }}</span>
              {% endif %}
            </div>
            {% if m.finished_size_in %}
              <div class="meta">
                Approx. finished size: {{ m.finished_size_in[0] }}" × {{ m.finished_size_in[1] }}" on {{ m.cloth_count }} ct
              </div>
            {% endif %}
          </div>
          <div class="actions">
            <a class="btn btn-primary" href="/patterns/{{ p.id }}">View details</a>
            <a class="btn btn-ghost" href="/patterns/{{ p.id }}/download">Download ZIP</a>
            {% if p.rel_preview %}
              <a class="btn btn-ghost" href="/patterns/{{ p.id }}/print" target="_blank">Open print view</a>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </div>
  {% else %}
    <p class="empty">
      You haven’t saved any patterns yet. Generate one from the <a href="/">tool</a> and it will appear here automatically.
    </p>
  {% endif %}
</div>
</body>
</html>"""

PATTERN_DETAIL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pattern details — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{
      margin:0;
      font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      background:#020617;
      color:#e5e7eb;
    }
    .wrap{max-width:980px;margin:0 auto;padding:24px 14px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      font-weight:800;font-size:18px;letter-spacing:.12em;text-transform:uppercase;color:#f9fafb;
    }
    .nav a{
      color:#cbd5f5;
      font-size:13px;
      margin-left:10px;
      text-decoration:none;
    }
    .nav a:hover{text-decoration:underline;}
    h1{margin:0 0 8px;font-size:1.4rem;color:#f9fafb;}
    .meta{font-size:12px;color:#9ca3af;margin-bottom:10px;}
    .layout{
      display:grid;
      grid-template-columns:minmax(0,1.2fr) minmax(0,1.1fr);
      gap:16px;
    }
    .card{
      background:rgba(15,23,42,.98);
      border-radius:16px;
      border:1px solid #1f2937;
      padding:14px;
      box-shadow:0 16px 32px rgba(0,0,0,.85);
    }
    .preview{
      background:#020617;
      border-radius:12px;
      border:1px solid #111827;
      padding:6px;
      max-height:520px;
      overflow:auto;
    }
    .preview img{
      max-width:100%;
      display:block;
    }
    .actions{
      margin-top:10px;
      display:flex;
      gap:8px;
      flex-wrap:wrap;
    }
    .btn{
      display:inline-block;
      padding:7px 12px;
      border-radius:999px;
      border:none;
      cursor:pointer;
      font-size:12px;
      font-weight:600;
      text-decoration:none;
      text-align:center;
    }
    .btn-primary{
      background:#f97316;
      color:#111827;
      box-shadow:0 6px 18px rgba(248,113,22,.6);
    }
    .btn-primary:hover{background:#ea580c;}
    .btn-ghost{
      background:transparent;
      color:#e5e7eb;
      border:1px solid #4b5563;
    }
    .btn-ghost:hover{background:#111827;}
    table{
      width:100%;
      border-collapse:collapse;
      font-size:12px;
      color:#e5e7eb;
    }
    th,td{
      border-bottom:1px solid #1f2937;
      padding:4px 6px;
      text-align:left;
    }
    th{
      font-weight:600;
      color:#cbd5f5;
      background:#020617;
      position:sticky;
      top:0;
    }
    tbody tr:nth-child(even){
      background:#020617;
    }
    .swatch{
      width:16px;
      height:16px;
      border-radius:4px;
      border:1px solid #0f172a;
      display:inline-block;
      margin-right:4px;
    }
    @media (max-width:900px){
      .layout{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">PATTERNCRAFT.APP</div>
    <div class="nav">
      <a href="/">Tool</a>
      <a href="/patterns">My patterns</a>
      <a href="/pricing">Pricing</a>
      <a href="/logout">Sign out</a>
    </div>
  </div>

  <h1>{{ pattern.name or "Pattern" }}</h1>
  <div class="meta">
    {{ pattern.ptype|default("cross")|capitalize }} pattern
    {% if pattern.created_at %} · Created {{ pattern.created_at }}{% endif %}
    {% set m = pattern.meta or {} %}
    {% if m.stitches_w and m.stitches_h %} · {{ m.stitches_w }}×{{ m.stitches_h }} stitches{% endif %}
    {% if m.colors %} · {{ m.colors }} colors{% endif %}
    {% if m.finished_size_in %}
      · Approx. finished size: {{ m.finished_size_in[0] }}" × {{ m.finished_size_in[1] }}" on {{ m.cloth_count }} ct
    {% endif %}
  </div>

  <div class="layout">
    <div class="card">
      <h2 style="margin:0 0 6px;font-size:1rem;">Pattern preview</h2>
      <p style="margin:0 0 8px;font-size:12px;color:#9ca3af;">
        Clean, high‑resolution preview with no watermark or blur. Download the ZIP for the full chart and PDF.
      </p>
      <div class="preview">
        {% if preview_url %}
          <img src="{{ preview_url }}" alt="Pattern preview">
        {% else %}
          <p style="font-size:12px;color:#9ca3af;">No inline preview available. Download the ZIP to see the pattern image.</p>
        {% endif %}
      </div>
      <div class="actions">
        <a href="/patterns/{{ pattern.id }}/download" class="btn btn-primary">Download ZIP</a>
        {% if preview_url %}
          <a href="/patterns/{{ pattern.id }}/print" target="_blank" class="btn btn-ghost">Open print view</a>
        {% endif %}
      </div>
    </div>

    <div class="card">
      <h2 style="margin:0 0 6px;font-size:1rem;">Color legend</h2>
      <p style="margin:0 0 8px;font-size:12px;color:#9ca3af;">
        Each row shows a palette entry for this pattern. Hex and RGB values help you match floss or yarn accurately.
      </p>
      {% if legend and legend|length > 0 %}
        <div style="max-height:420px;overflow:auto;">
          <table>
            <thead>
              <tr>
                <th>Color</th>
                <th>Hex / RGB</th>
                <th>Stitches</th>
                <th>%</th>
                <th>Skeins est.</th>
              </tr>
            </thead>
            <tbody>
              {% for row in legend %}
                <tr>
                  <td>
                    <span class="swatch" style="background: {{ row.hex }};"></span>
                  </td>
                  <td>{{ row.hex }} · {{ row.r }},{{ row.g }},{{ row.b }}</td>
                  <td>{{ row.stitches }}</td>
                  <td>{{ row.percent }}</td>
                  <td>{{ row.skeins_est }}</td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      {% else %}
        <p style="font-size:12px;color:#9ca3af;">Legend details are not available for this pattern.</p>
      {% endif %}
    </div>
  </div>
</div>
</body>
</html>"""

PATTERN_PRINT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Print pattern — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{
      margin:0;
      font:14px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:#111827;
      background:#f9fafb;
    }
    .wrap{
      max-width:900px;
      margin:0 auto;
      padding:16px;
    }
    h1{
      margin:0 0 6px;
      font-size:1.2rem;
    }
    .meta{
      font-size:12px;
      color:#6b7280;
      margin-bottom:8px;
    }
    .preview{
      margin-top:10px;
      border:1px solid #e5e7eb;
      padding:6px;
      background:#fff;
    }
    .preview img{
      max-width:100%;
      display:block;
    }
    @media print{
      body{background:#fff;}
      .wrap{padding:0;}
    }
  </style>
</head>
<body onload="window.print()">
  <div class="wrap">
    <h1>{{ pattern.name or "Pattern" }}</h1>
    <div class="meta">
      {{ pattern.ptype|default("cross")|capitalize }} pattern
      {% if pattern.created_at %} · Created {{ pattern.created_at }}{% endif %}
    </div>
    {% if preview_url %}
      <div class="preview">
        <img src="{{ preview_url }}" alt="Pattern preview for printing">
      </div>
    {% else %}
      <p>No inline preview available. Download the ZIP from the My Patterns page and print the PDF or grid image from there.</p>
    {% endif %}
  </div>
</body>
</html>"""

SUCCESS_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Payment successful — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#020617;
      color:#e5e7eb;
    }
    .wrap{
      max-width:520px;
      margin:0 auto;
      padding:32px 16px 40px;
    }
    .card{
      background:rgba(15,23,42,.98);
      border-radius:16px;
      border:1px solid #1f2937;
      padding:24px;
      box-shadow:0 18px 40px rgba(0,0,0,.85);
    }
    h1{margin:0 0 10px;font-size:1.7rem;color:#f9fafb;}
    p{margin:6px 0;font-size:14px;color:#e5e7eb}
    a{
      color:#93c5fd;
      text-decoration:none;
      font-weight:600;
    }
    a:hover{text-decoration:underline;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Payment received</h1>
      <p>Thank you{% if user %}, {{ user.email }}{% endif %}. Your PatternCraft.app plan will be updated based on your purchase.</p>
      <p>You can go back to the tool and start generating patterns right away. If your account hasn’t updated yet, it will as soon as your membership settings are refreshed.</p>
      <p style="margin-top:14px;">
        <a href="/">← Back to PatternCraft.app</a>
      </p>
    </div>
  </div>
</body>
</html>"""

if __name__ == "__main__":
    app.run(debug=True)

