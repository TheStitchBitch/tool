from __future__ import annotations

import base64
import datetime as dt
import io
import json
import math
import os
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

# ---------------------- APP & STORAGE SETUP ----------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
PATTERN_DIR = os.path.join(DATA_DIR, "patterns")
PREVIEW_DIR = os.path.join(DATA_DIR, "previews")
os.makedirs(PATTERN_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000

# ---------------------- STRIPE CONFIG ----------------------------

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# Live price IDs from your latest Stripe export (prices (5).csv)
STRIPE_PRICE_SINGLE = "price_1SXNyWCINTImVye2jayzoKKj"   # Single Pattern – $25
STRIPE_PRICE_PACK10 = "price_1SXNyRCINTImVye2m433u7pL"   # 10 Pattern Pack – $60
STRIPE_PRICE_3MO = "price_1SXTFUCINTImVye2JwOxUN55"      # 3-Month Unlimited – billed every 3 months
STRIPE_PRICE_ANNUAL = "price_1SXNyNCINTImVye2rcxl5LsO"   # Annual Unlimited – $99/year

# ---------------------- USER STORAGE -----------------------------


def load_users() -> Dict[str, dict]:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        # If file is corrupted, fail safe and start empty (prevents crashes)
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
    user = users.get(email)
    if not user:
        return None
    # Ensure patterns list exists
    user.setdefault("patterns", [])
    return user


def has_unlimited_access(user: dict) -> bool:
    """Check if user currently has an active unlimited plan."""
    now = dt.datetime.utcnow()
    sub = user.get("subscription", "free")
    if sub == "unlimited_3m":
        exp = user.get("unlimited_3m_expires_at")
        if not exp:
            return False
        try:
            exp_dt = dt.datetime.fromisoformat(exp.replace("Z", "+00:00"))
        except Exception:
            return False
        return exp_dt > now
    if sub == "annual":
        exp = user.get("annual_expires_at")
        if not exp:
            return False
        try:
            exp_dt = dt.datetime.fromisoformat(exp.replace("Z", "+00:00"))
        except Exception:
            return False
        return exp_dt > now
    return False


# ---------------------- IMAGE / PATTERN HELPERS ------------------


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
    """Scale each stitch to a cell and overlay a clean 10×10 grid (no watermark/fade)."""
    sx, sy = base.size
    out = base.resize((sx * cell_px, sy * cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    thin = (0, 0, 0, 60)
    bold = (0, 0, 0, 160)
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
    """Overlay symbol per stitch, then clean grid."""
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

    thin = (0, 0, 0, 60)
    bold = (0, 0, 0, 160)
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
    """Rough DMC skein estimate."""
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


# ---------------------- BASIC ROUTES -----------------------------

@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


@app.get("/favicon.ico")
def favicon():
    """Generate a simple 'PC' emblem favicon on first request."""
    path = os.path.join(DATA_DIR, "favicon.ico")
    if not os.path.exists(path):
        img = Image.new("RGBA", (64, 64), (255, 249, 230, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse((8, 8, 56, 56), fill=(245, 158, 11, 255))
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except Exception:
            font = ImageFont.load_default()
        w, h = draw.textsize("PC", font=font)
        draw.text(((64 - w) / 2, (64 - h) / 2 - 2), "PC", font=font, fill=(255, 255, 255, 255))
        img.save(path, format="ICO")
    return send_file(path, mimetype="image/x-icon")


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
        message = "You’ve already used your free PatternCraft.app pattern. Choose a plan to keep creating."
    elif reason == "no_credits":
        message = "You’ve used your available credits on this account. Pick a pack or unlimited plan to continue."
    elif reason == "checkout_error":
        message = "We couldn’t start checkout. Please try again or contact support."
    return render_template_string(PRICING_HTML, user=user, message=message)


# ---------------------- CHECKOUT + SUCCESS ----------------------

@app.post("/checkout")
def create_checkout():
    """Create a Stripe Checkout Session for the selected plan."""
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in or create an account before purchasing a plan."))

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
    elif plan == "annual":
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
    """Stripe success page; update membership based on the Checkout Session."""
    user = get_current_user()
    session_id = request.args.get("session_id")

    if session_id and user:
        try:
            cs = stripe.checkout.Session.retrieve(session_id, expand=["line_items", "customer"])
            email = (
                cs.get("customer_details", {}).get("email")
                or user.get("email")
            )
            price_id = None
            if cs.get("line_items") and cs["line_items"]["data"]:
                price_id = cs["line_items"]["data"][0]["price"]["id"]

            users = load_users()
            u = users.get(email, user)
            now = dt.datetime.utcnow()

            if price_id == STRIPE_PRICE_SINGLE:
                u["subscription"] = "single"
                u["credits"] = int(u.get("credits", 0) or 0) + 1
            elif price_id == STRIPE_PRICE_PACK10:
                u["subscription"] = "pack10"
                u["credits"] = int(u.get("credits", 0) or 0) + 10
            elif price_id == STRIPE_PRICE_3MO:
                u["subscription"] = "unlimited_3m"
                expires = now + dt.timedelta(days=90)
                u["unlimited_3m_expires_at"] = expires.isoformat() + "Z"
            elif price_id == STRIPE_PRICE_ANNUAL:
                u["subscription"] = "annual"
                expires = now + dt.timedelta(days=365)
                u["annual_expires_at"] = expires.isoformat() + "Z"

            u.setdefault("patterns", [])
            users[email] = u
            save_users(users)
        except Exception:
            pass

    return render_template_string(SUCCESS_HTML, user=user)


# ---------------------- SIGNUP / LOGIN --------------------------

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
        "subscription": "free",  # free / single / pack10 / unlimited_3m / annual
        "free_used": False,
        "credits": 0,
        "patterns": [],
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
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
                    msg="We couldn’t match that email and password after several attempts. Create a PatternCraft.app account to get started (includes one free pattern).",
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


# ---------------------- SAMPLE PATTERN ZIP -----------------------

@app.get("/sample-pattern.zip")
def sample_pattern_zip():
    """Serve a small sample quilt-style pattern ZIP with a grid and legend."""
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
            "notes": "Sample quilt-style color grid generated by PatternCraft.app.",
        }
        z.writestr("meta.json", json.dumps(meta, indent=2))

        buf_png = io.BytesIO()
        sym_img.save(buf_png, format="PNG")
        z.writestr("grid.png", buf_png.getvalue())

    out_zip.seek(0)
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name="patterncraft_sample_quilt.zip",
    )


# ---------------------- PATTERN GENERATOR ------------------------

@app.post("/api/convert")
def convert():
    # Require account
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to generate patterns (every new account includes one free pattern)."))

    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(url_for("signup", msg="Create your PatternCraft.app account to continue."))

    user.setdefault("patterns", [])
    subscription = user.get("subscription", "free")
    credits = int(user.get("credits", 0) or 0)
    mark_free_used = False
    consume_credit = False

    # Membership logic
    if has_unlimited_access(user):
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

    pattern_id = uuid.uuid4().hex
    pattern_name = file.filename or f"Pattern {pattern_id[:8]}"

    out_zip = io.BytesIO()
    grid_img_for_preview: Optional[Image.Image] = None
    sx = sy = 0
    counts: Dict[Tuple[int, int, int], int] = {}

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

            grid_img_for_preview = grid_img.copy()

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

    # Save usage and membership changes
    if consume_credit and credits > 0:
        user["credits"] = max(0, credits - 1)
    if mark_free_used:
        user["free_used"] = True

    # Save pattern preview + ZIP path to user's history
    try:
        zip_bytes = out_zip.getvalue()
        zip_filename = f"{pattern_id}.zip"
        zip_path = os.path.join(PATTERN_DIR, zip_filename)
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        preview_filename = None
        if grid_img_for_preview is not None:
            preview_filename = f"{pattern_id}.png"
            preview_path = os.path.join(PREVIEW_DIR, preview_filename)
            grid_img_for_preview.save(preview_path, format="PNG")

        pattern_record = {
            "id": pattern_id,
            "name": pattern_name,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
            "type": ptype,
            "stitch_style": stitch_style,
            "stitches_w": sx,
            "stitches_h": sy,
            "colors": len(counts) if counts else None,
            "zip_file": zip_filename,
            "preview_file": preview_filename,
        }
        patterns = user.get("patterns") or []
        patterns.append(pattern_record)
        user["patterns"] = patterns
    except Exception:
        # Do not break the download if saving history fails
        pass

    users[email] = user
    save_users(users)

    out_zip.seek(0)
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{pattern_name or 'pattern'}.zip",
    )


# ---------------------- PATTERNS DASHBOARD ----------------------

@app.get("/patterns")
def patterns():
    user = get_current_user()
    if not user:
        return redirect(url_for("login", msg="Log in to view your saved patterns."))
    users = load_users()
    u = users.get(user["email"], {})
    u.setdefault("patterns", [])
    patterns_list = list(reversed(u["patterns"]))
    return render_template_string(PATTERNS_HTML, user=u, patterns=patterns_list)


@app.get("/pattern/preview/<pattern_id>")
def pattern_preview(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login", msg="Log in to view your patterns."))
    users = load_users()
    u = users.get(user["email"])
    if not u:
        return "Not found", 404
    for p in u.get("patterns", []):
        if p.get("id") == pattern_id and p.get("preview_file"):
            path = os.path.join(PREVIEW_DIR, p["preview_file"])
            if os.path.exists(path):
                return send_file(path, mimetype="image/png")
    return "Not found", 404


@app.get("/pattern/download/<pattern_id>")
def pattern_download(pattern_id: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("login", msg="Log in to download your patterns."))
    users = load_users()
    u = users.get(user["email"])
    if not u:
        return "Not found", 404
    for p in u.get("patterns", []):
        if p.get("id") == pattern_id and p.get("zip_file"):
            path = os.path.join(PATTERN_DIR, p["zip_file"])
            if os.path.exists(path):
                fname = (p.get("name") or "pattern").replace("/", "_")
                return send_file(
                    path,
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name=f"{fname}.zip",
                )
    return "Not found", 404


# ---------------------- INLINE HTML: HOME -----------------------

HOMEPAGE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root{
      --bg:#FFF9E6;--fg:#222222;--muted:#6b6b6b;
      --line:#f1e4b8;--radius:16px;--shadow:0 12px 30px rgba(148,131,68,.25);
      --accent:#f59e0b;--accent-dark:#c26e00;--accent-soft:#fef3c7;
      --pill:#f59e0b;--pill-text:#1f2933;
      --chip-bg:#fff7cc;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#fef3c7 0,#fff 35%,transparent 60%),
        radial-gradient(circle at top right,#fde68a 0,#fff 40%,transparent 65%),
        linear-gradient(to bottom,#fff7e6,#fff);
    }
    a{color:#1d4ed8;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1080px;margin:0 auto;padding:20px 16px 40px}
    h1{font-size:2.2rem;margin:0 0 8px}
    h2{margin:0 0 10px;font-size:1.3rem}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      display:flex;align-items:center;gap:8px;
      font-weight:800;font-size:19px;letter-spacing:.08em;text-transform:uppercase;
    }
    .brand-mark{
      width:26px;height:26px;border-radius:10px;
      background:linear-gradient(135deg,#fbbf24,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:13px;font-weight:800;color:white;
      box-shadow:0 4px 10px rgba(245,158,11,.55);
    }
    .top-links{font-size:13px;color:#4b5563}
    .top-links a{margin-left:10px;}
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:4px 10px;border-radius:999px;
      background:var(--chip-bg);border:1px solid rgba(234,179,8,.6);
      font-size:11px;color:#854d0e;text-transform:uppercase;letter-spacing:.08em;
    }
    .chip-dot{width:8px;height:8px;border-radius:999px;background:#22c55e}
    .hero{
      display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2fr);
      gap:18px;margin-bottom:24px;align-items:center;
    }
    .hero-tagline{color:var(--muted);max-width:420px;font-size:14px;}
    .muted{color:var(--muted);font-size:13px}
    .card{
      background:#fffbe9;
      border-radius:var(--radius);
      border:1px solid var(--line);
      box-shadow:var(--shadow);
      padding:20px;
    }
    .pill{
      padding:11px 22px;border-radius:999px;
      background:var(--accent);
      color:var(--pill-text);
      border:none;cursor:pointer;
      font-size:14px;font-weight:700;letter-spacing:.02em;
      box-shadow:0 8px 20px rgba(245,158,11,.5);
      transition:transform .08s,box-shadow .08s,background .08s;
      display:inline-block;
      text-decoration:none;
      text-align:center;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 12px 26px rgba(245,158,11,.6);
      background:var(--accent-dark);
      color:#fff;
    }
    .pill-secondary{
      background:#fff;
      color:#92400e;
      border:1px solid rgba(234,179,8,.7);
      box-shadow:none;
    }
    .pill-secondary:hover{
      box-shadow:0 6px 18px rgba(148,131,68,.35);
    }
    .pill-large{
      padding:13px 26px;
      font-size:15px;
    }
    .hero-cta-row{
      display:flex;gap:12px;margin-top:14px;flex-wrap:wrap;align-items:center;
    }
    .hero-note{font-size:12px;color:#6b7280;margin-top:8px;}
    .badge-row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
    .badge{
      font-size:11px;padding:4px 9px;border-radius:999px;
      background:#fef3c7;color:#92400e;border:1px solid #facc15;
    }
    .why-list{
      padding-left:18px;
      margin:8px 0 0;
      font-size:13px;
      color:#4b5563;
    }
    .why-list li{margin-bottom:4px;}

    .make-layout{display:grid;gap:18px;grid-template-columns:minmax(0,1.25fr);}
    .file{
      border:2px dashed var(--accent);
      border-radius:18px;
      padding:18px;
      display:flex;align-items:center;gap:12px;
      cursor:pointer;
      background:var(--accent-soft);
      transition:background .15s,border-color .15s,transform .1s,box-shadow .1s;
      width:100%;
    }
    .file:hover{
      background:#fef3c7;border-color:#f59e0b;
      transform:translateY(-1px);
      box-shadow:0 6px 18px rgba(148,131,68,.35);
    }
    .file-ready{
      background:#ecfccb;
      border-color:#16a34a;
      box-shadow:0 6px 18px rgba(22,163,74,.45);
    }
    .file input{display:none}
    .file-label-main{font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.06em}
    .file-label-sub{font-size:12px;color:#6b7280}
    .file-btn{
      border-style:solid;
    }
    .free-note{
      margin-top:6px;font-size:12px;color:#166534;background:#dcfce7;
      border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px;
    }
    .free-dot{width:8px;height:8px;border-radius:999px;background:#22c55e}
    fieldset{border:1px solid var(--line);border-radius:12px;padding:10px;margin:10px 0}
    legend{font-size:13px;padding:0 4px;color:#92400e;}
    .row{display:flex;flex-wrap:wrap;gap:12px}
    .row > label{flex:1 1 150px;font-size:13px}
    .row input,.row select{
      width:100%;margin-top:3px;padding:7px 9px;border-radius:10px;
      border:1px solid #e5e7eb;font-size:13px;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#f59e0b;box-shadow:0 0 0 1px rgba(245,158,11,.5);
    }
    label{font-size:13px}
    .controls-note{font-size:11px;color:#9ca3af;margin-top:4px}
    .hidden{display:none}
    .section-title{font-size:1.2rem;margin-bottom:6px}
    .foot-row{font-size:11px;color:#9ca3af;margin-top:8px}

    .pill-ready{
      background:#16a34a !important;
      color:#fff !important;
      box-shadow:0 8px 20px rgba(22,163,74,.6) !important;
    }

    .dashboard-link{font-size:13px;color:#6b7280;margin-top:6px;}

    @media (max-width:860px){
      .hero{grid-template-columns:1fr}
      .make-layout{grid-template-columns:1fr}
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
      <a href="/pricing">Pricing</a>
      <a href="/patterns">My patterns</a>
      {% if user %}
        · Signed in as {{ user.email }}
        · <a href="/logout">Sign out</a>
      {% else %}
        · <a href="/login">Log in</a>
        · <a href="/signup">Create Free Account</a>
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
        PatternCraft.app transforms your photos and artwork into clean, printable grids with legends.
        Choose a plan, upload a picture, and download a stitch-ready pattern in minutes.
      </p>
      <div class="hero-cta-row">
        {% if user %}
          <button class="pill pill-large" onclick="document.getElementById('make').scrollIntoView({behavior:'smooth'})">
            Open the tool
          </button>
        {% else %}
          <a class="pill pill-large" href="/login?msg=Log+in+to+open+the+PatternCraft+tool.">
            Open the tool
          </a>
        {% endif %}
        <a class="pill pill-secondary" href="/pricing#plans">
          View pricing
        </a>
      </div>
      <div class="hero-note">
        Every new account includes <strong>one free pattern</strong>. Upgrade later with a single, pack, or unlimited plan.
      </div>
      <div class="badge-row">
        <span class="badge">Clean printable grids</span>
        <span class="badge">Color legends & size estimates</span>
        <span class="badge">Save & revisit past patterns</span>
      </div>
    </div>

    <div class="card">
      <h2>Why makers use PatternCraft.app</h2>
      <ul class="why-list">
        <li>Clean grids with bold 10×10 guides and symbol overlays</li>
        <li>Color legends with hex and RGB values for accurate palettes</li>
        <li>Fabric size estimates based on stitch count and cloth count</li>
        <li>Knitting charts that respect row proportions</li>
        <li>Embroidery line exports ready for your machine software</li>
      </ul>
    </div>
  </div>

  <div id="make" class="card">
    <h2 class="section-title">Make a pattern</h2>
    <p class="muted">
      Upload artwork, pick your stitch settings, and download a full ZIP with grid, legend, and metadata.
      {% if not user %}
        Create a free account to use the tool (includes one pattern on us).
      {% endif %}
    </p>
    <div class="make-layout">
      <div class="make-main">
        <form method="POST" action="/api/convert" enctype="multipart/form-data">
          {% if user %}
          <label class="file">
            <input id="fileInput" type="file" name="file" accept="image/*" required onchange="pickFile(this)">
            <div>
              <div class="file-label-main">UPLOAD PICTURE HERE</div>
              <div class="file-label-sub">
                Drop in your artwork or tap to browse from your device.
              </div>
            </div>
          </label>
          {% else %}
          <button class="file file-btn" type="button"
            onclick="location.href='/login?msg=Log+in+or+create+a+free+account+to+upload+a+picture+and+generate+your+first+pattern.'">
            <div>
              <div class="file-label-main">UPLOAD PICTURE HERE</div>
              <div class="file-label-sub">
                Log in or create a free account (includes 1 pattern) to use the tool.
              </div>
            </div>
          </button>
          {% endif %}

          {% if not user %}
          <div style="display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap">
            <div class="free-note">
              <div class="free-dot"></div>
              <span>
                New here? <a href="/signup">Create Free Account</a>.
                Already joined? <a href="/login">Log in</a>.
              </span>
            </div>
          </div>
          {% endif %}

          <fieldset>
            <legend>Pattern type</legend>
            <label><input type="radio" name="ptype" value="cross" checked> Cross-stitch</label>
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
            <p class="controls-note">Defaults work well for most art. Adjust once you know your preferences.</p>
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
              Generates a simple run-stitch path you can refine in your embroidery software.
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
            <button class="pill" id="generateBtn" type="submit" {% if not user %}disabled{% endif %}>
              Generate pattern ZIP
            </button>
            <span class="muted">
              Your download includes <code>grid.png</code>, <code>legend.csv</code>, <code>meta.json</code>, and optional <code>pattern.pdf</code> or embroidery files.
            </span>
          </div>

          {% if user %}
          <div class="dashboard-link">
            After generating, your pattern is saved on your account under <a href="/patterns">My patterns</a>.
          </div>
          {% endif %}
        </form>
      </div>
    </div>
    <div class="foot-row">
      Tip: Start with simpler, high-contrast images for the cleanest stitch results. You can adjust colors and size before printing.
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

# ---------------------- INLINE HTML: AUTH & PRICING --------------

SIGNUP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#FFF9E6;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#fff;
      border-radius:16px;
      border:1px solid #f4e4ae;
      padding:22px;
      box-shadow:0 12px 30px rgba(148,131,68,.3);
    }
    h1{margin:0 0 10px;font-size:1.6rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #e5e7eb;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#f59e0b;box-shadow:0 0 0 1px rgba(245,158,11,.6);
    }
    .pill{
      margin-top:16px;padding:11px 22px;border-radius:999px;
      border:none;background:#f59e0b;color:#1f2933;
      font-size:14px;font-weight:700;cursor:pointer;
      box-shadow:0 8px 20px rgba(245,158,11,.5);
      width:100%;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 12px 26px rgba(245,158,11,.6);
      background:#c26e00;color:#fff;
    }
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#1d4ed8;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4b5563;padding-left:18px;margin-top:10px}
    .top-link{font-size:13px;margin-bottom:10px;}
    .top-link a{font-weight:600;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="top-link">
      Already have an account? <a href="/login">Log in</a>.
    </div>
    <h1>Create your PatternCraft.app account</h1>
    <p class="muted">
      Every new account includes <strong>one free pattern</strong>. Use it to test your first project, then upgrade when you’re ready.
    </p>
    <ul>
      <li>Use an email you check regularly.</li>
      <li>Choose a password you’ll remember.</li>
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
      You can manage your patterns anytime under <strong>My patterns</strong>.
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
    body{margin:0;background:#FFF9E6;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#fff;
      border-radius:16px;
      border:1px solid #f4e4ae;
      padding:22px;
      box-shadow:0 12px 30px rgba(148,131,68,.3);
    }
    h1{margin:0 0 10px;font-size:1.6rem}
    .muted{font-size:13px;color:#6b7280}
    label{display:block;font-size:13px;margin-top:12px}
    input{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #e5e7eb;font-size:14px;
    }
    input:focus{
      outline:none;border-color:#f59e0b;box-shadow:0 0 0 1px rgba(245,158,11,.6);
    }
    .pill{
      margin-top:14px;padding:11px 22px;border-radius:999px;
      border:none;background:#f59e0b;color:#1f2933;
      font-size:14px;font-weight:700;cursor:pointer;
      box-shadow:0 8px 20px rgba(245,158,11,.5);
      width:100%;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 12px 26px rgba(245,158,11,.6);
      background:#c26e00;color:#fff;
    }
    .pill-ghost{
      margin-top:10px;
      background:#fff;
      color:#92400e;
      border:1px solid #facc15;
      box-shadow:none;
    }
    .pill-ghost:hover{
      box-shadow:0 8px 18px rgba(148,131,68,.35);
    }
    .msg{margin-top:10px;font-size:13px;color:#b91c1c}
    a{color:#1d4ed8;text-decoration:none;}
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
    <button class="pill pill-ghost" type="button" onclick="location.href='/signup'">
      Create Free Account (1 pattern included)
    </button>
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
</html>
"""

PRICING_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PatternCraft • Pricing</title>
<style>
:root{
  --fg:#111; --muted:#666; --accent:#f59e0b; --line:#eee; --card:#fffaf0;
  --radius:16px; --wrap:1020px;
}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{
  font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  color:var(--fg);background:#FFF9E6;
}
.wrap{max-width:var(--wrap);margin:0 auto;padding:22px 16px 32px}
header{position:sticky;top:0;background:#fff9e6;border-bottom:1px solid #f4e4ae;z-index:5}
.brand{
  display:flex;align-items:center;gap:8px;
  font-weight:800;letter-spacing:.08em;text-transform:uppercase;font-size:14px;
}
.brand-mark{
  width:22px;height:22px;border-radius:9px;
  background:linear-gradient(135deg,#fbbf24,#f97316);
  display:flex;align-items:center;justify-content:center;
  font-size:11px;color:#fff;
}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.btn{
  display:inline-block;padding:9px 16px;border-radius:999px;
  border:1px solid var(--accent);background:var(--accent);color:#1f2933;
  text-decoration:none;font-weight:700;cursor:pointer;font-size:13px;
}
.btn.ghost{background:#fff;color:#92400e;border-color:#facc15}
.btn:hover{
  background:#c26e00;color:#fff;border-color:#c26e00;
}
.btn.ghost:hover{
  box-shadow:0 6px 18px rgba(148,131,68,.35);
}
.cards{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:16px;margin-top:16px;
}
.card{
  background:var(--card);border:1px solid #f4e4ae;border-radius:var(--radius);padding:16px;
  display:flex;flex-direction:column;justify-content:space-between;
}
.card h3{margin:0 0 6px}
.price{font-size:24px;font-weight:800;margin:4px 0}
.small{font-size:13px;color:var(--muted)}
.list{margin:8px 0 12px;padding-left:18px;font-size:13px;color:#4b5563}
.badge{
  display:inline-block;background:#fef3c7;color:#92400e;border-radius:999px;
  padding:4px 12px;font-size:12px;font-weight:600;margin-bottom:6px;
  border:1px solid #facc15;
}
.notice{
  margin-top:10px;padding:10px 12px;border-radius:10px;
  background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;
  font-size:13px;
}
footer{border-top:1px solid #f4e4ae;margin-top:24px;font-size:12px;color:#6b7280}
.steps{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:16px;
  margin-top:16px;
}
.step-card{
  background:#fff;
  border:1px solid #f4e4ae;
  border-radius:var(--radius);
  padding:16px;
  font-size:13px;
}
.step-num{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:26px;height:26px;
  border-radius:999px;
  background:#f59e0b;
  color:#1f2933;
  font-size:14px;
  font-weight:700;
  margin-bottom:8px;
}
@media (max-width:700px){ .cards{grid-template-columns:1fr} }
</style>
</head>
<body>
<header>
  <div class="wrap row" style="justify-content:space-between">
    <div class="brand">
      <div class="brand-mark">PC</div>
      <span>PatternCraft.app</span>
    </div>
    <nav class="row">
      <a class="btn ghost" href="/">Tool</a>
      <a class="btn" href="/pricing">Pricing</a>
      <a class="btn ghost" href="/patterns">My patterns</a>
    </nav>
  </div>
</header>

<section class="wrap" id="plans">
  <div class="badge">Simple, transparent pricing</div>
  <h1>Choose the plan that fits your stitching</h1>
  <p class="small">Start with a single pattern, save with a pack, or go unlimited.</p>

  {% if message %}
  <div class="notice">{{ message }}</div>
  {% endif %}

  <div class="cards">
    <!-- Single Pattern -->
    <div class="card">
      <div>
        <h3>Single Pattern</h3>
        <div class="price">$25</div>
        <p class="small">Single pattern and legend. Use whenever.</p>
        <ul class="list">
          <li>1 professional pattern conversion</li>
          <li>Detailed legend included</li>
          <li>High-resolution grid output</li>
          <li>Use your pattern whenever you like</li>
        </ul>
      </div>
      <div>
        <form method="POST" action="/checkout">
          <input type="hidden" name="plan" value="single">
          <button class="btn" type="submit">Buy Single</button>
        </form>
        <p class="small" style="margin-top:6px;">Best for one-off projects or trying PatternCraft.app.</p>
      </div>
    </div>

    <!-- 10-Pattern Pack -->
    <div class="card">
      <div>
        <h3>10-Pattern Pack</h3>
        <div class="price">$60</div>
        <p class="small">Great for consistent hobby use.</p>
        <ul class="list">
          <li>10 pattern conversions</li>
          <li>Credits never expire</li>
          <li>Includes all export formats</li>
          <li>Premium palette options</li>
        </ul>
      </div>
      <div>
        <form method="POST" action="/checkout">
          <input type="hidden" name="plan" value="pack10">
          <button class="btn ghost" type="submit">Buy 10-Pack</button>
        </form>
        <p class="small" style="margin-top:6px;">Save big vs buying singles.</p>
      </div>
    </div>

    <!-- 3-Month Unlimited -->
    <div class="card">
      <div>
        <h3>3-Month Unlimited</h3>
        <div class="price">$75 / 3 months</div>
        <p class="small">Recurring every 3 months until you cancel.</p>
        <ul class="list">
          <li>Unlimited pattern conversions</li>
          <li>Higher-resolution output</li>
          <li>Advanced color tools</li>
          <li>Priority processing</li>
          <li>All export formats + templates</li>
        </ul>
      </div>
      <div>
        <form method="POST" action="/checkout">
          <input type="hidden" name="plan" value="unlimited_3m">
          <button class="btn ghost" type="submit">Start 3-Month Unlimited</button>
        </form>
        <p class="small" style="margin-top:6px;">Perfect for focused projects or seasons.</p>
      </div>
    </div>

    <!-- Annual Unlimited -->
    <div class="card">
      <div>
        <h3>Annual Unlimited</h3>
        <div class="price">$99 / year</div>
        <p class="small">Unlimited patterns all year.</p>
        <ul class="list">
          <li>Unlimited pattern conversions</li>
          <li>4× resolution for large projects</li>
          <li>Advanced color tools</li>
          <li>Priority processing</li>
          <li>All export formats + templates</li>
        </ul>
      </div>
      <div>
        <form method="POST" action="/checkout">
          <input type="hidden" name="plan" value="annual">
          <button class="btn ghost" type="submit">Go Annual Unlimited</button>
        </form>
        <p class="small" style="margin-top:6px;">Best value if you stitch more than 4 patterns a year.</p>
      </div>
    </div>
  </div>
</section>

<section class="wrap">
  <h2>How PatternCraft.app works</h2>
  <p class="small">From photo to stitch-ready pattern in three simple steps.</p>
  <div class="steps">
    <div class="step-card">
      <div class="step-num">1</div>
      <h3>Select a plan</h3>
      <p>Start with a single pattern, a pack, or unlimited access. Every account begins with one free pattern.</p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3>Upload & choose settings</h3>
      <p>Upload a photo or artwork, set stitch width, fabric count, and colors. PatternCraft builds a clean grid and legend.</p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3>Download & stitch</h3>
      <p>Download your pattern ZIP, print the grid and legend, or import files into your own tools. Your patterns are saved under <strong>My patterns</strong>.</p>
    </div>
  </div>
</section>

<footer class="wrap">
  <div class="row" style="justify-content:space-between">
    <div>© PatternCraft.app</div>
    <div><a href="/" class="btn ghost">Back to tool</a></div>
  </div>
</footer>
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
      background:#FFF9E6;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:#111827;
    }
    .wrap{max-width:1020px;margin:0 auto;padding:22px 16px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:16px;
    }
    .brand{
      display:flex;align-items:center;gap:8px;
      font-weight:800;font-size:14px;letter-spacing:.08em;text-transform:uppercase;
    }
    .brand-mark{
      width:22px;height:22px;border-radius:9px;
      background:linear-gradient(135deg,#fbbf24,#f97316);
      display:flex;align-items:center;justify-content:center;
      font-size:11px;color:#fff;
    }
    a{color:#1d4ed8;text-decoration:none;}
    a:hover{text-decoration:underline;}
    h1{margin:0 0 8px;font-size:1.6rem}
    .muted{font-size:13px;color:#6b7280}
    .grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
      gap:16px;
      margin-top:16px;
    }
    .card{
      background:#fff;
      border-radius:16px;
      border:1px solid #f4e4ae;
      padding:14px;
      box-shadow:0 8px 22px rgba(148,131,68,.25);
      display:flex;
      flex-direction:column;
      gap:8px;
    }
    .preview{
      background:#f9fafb;
      border-radius:12px;
      overflow:hidden;
      border:1px solid #e5e7eb;
      max-height:220px;
    }
    .preview img{
      width:100%;
      height:auto;
      display:block;
    }
    .meta{
      font-size:12px;
      color:#4b5563;
    }
    .meta span{display:inline-block;margin-right:6px;margin-bottom:2px;}
    .pill{
      padding:8px 16px;border-radius:999px;
      border:none;background:#f59e0b;color:#1f2933;
      font-size:13px;font-weight:700;cursor:pointer;
      box-shadow:0 6px 18px rgba(245,158,11,.5);
      text-align:center;
      display:inline-block;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 10px 24px rgba(245,158,11,.6);
      background:#c26e00;color:#fff;
    }
    .top-links{font-size:13px;color:#4b5563}
    .top-links a{margin-left:10px;}
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
      <a href="/">Tool</a>
      <a href="/pricing">Pricing</a>
      <a href="/patterns">My patterns</a>
      {% if user %}
        · {{ user.email }}
        · <a href="/logout">Sign out</a>
      {% endif %}
    </div>
  </div>

  <h1>My patterns</h1>
  <p class="muted">
    Each time you generate a pattern, it’s saved here with a preview, details, and a fresh download link.
  </p>

  {% if patterns %}
  <div class="grid">
    {% for p in patterns %}
    <div class="card">
      <div class="preview">
        {% if p.preview_file %}
          <img src="/pattern/preview/{{ p.id }}" alt="Pattern preview {{ p.name }}">
        {% else %}
          <div style="padding:40px 12px;text-align:center;font-size:12px;color:#9ca3af;">
            No preview available for this pattern.
          </div>
        {% endif %}
      </div>
      <div style="font-size:13px;font-weight:600;margin-top:4px;">
        {{ p.name or ('Pattern ' ~ p.id[:8]) }}
      </div>
      <div class="meta">
        <span>{{ p.type|default('cross')|capitalize }}</span>
        {% if p.stitches_w and p.stitches_h %}
          <span>{{ p.stitches_w }} × {{ p.stitches_h }} stitches</span>
        {% endif %}
        {% if p.colors %}
          <span>{{ p.colors }} colors</span>
        {% endif %}
        {% if p.created_at %}
          <span>Created {{ p.created_at[:10] }}</span>
        {% endif %}
      </div>
      <div>
        <a class="pill" href="/pattern/download/{{ p.id }}">Download pattern ZIP</a>
      </div>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <p class="muted" style="margin-top:12px;">
    You haven’t generated a pattern yet. Go to the <a href="/">tool</a> to create your first pattern (your account includes one free pattern).
  </p>
  {% endif %}
</div>
</body>
</html>
"""

SUCCESS_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Payment successful — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{
      margin:0;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#FFF9E6;
      color:#111827;
    }
    .wrap{
      max-width:520px;
      margin:0 auto;
      padding:32px 16px 40px;
    }
    .card{
      background:#fff;
      border-radius:16px;
      border:1px solid #f4e4ae;
      padding:24px;
      box-shadow:0 12px 30px rgba(148,131,68,.3);
    }
    h1{margin:0 0 10px;font-size:1.7rem}
    p{margin:6px 0;font-size:14px;color:#4b5563}
    a{
      color:#1d4ed8;
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
      <p>Thank you{% if user %}, {{ user.email }}{% endif %}. Your PatternCraft.app plan has been updated.</p>
      <p>You can go back to the tool and start generating patterns right away. If your unlimited or credits plan looks unchanged, refresh the page and try generating a pattern.</p>
      <p style="margin-top:14px;">
        <a href="/">← Back to PatternCraft.app</a> · <a href="/patterns">View my patterns</a>
      </p>
    </div>
  </div>
</body>
</html>
"""

# ---------------------- MAIN -------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

