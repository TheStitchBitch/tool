from __future__ import annotations
import io
import json
import math
import os
import zipfile
import uuid
from datetime import datetime, timezone
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
    Response,
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

# Stripe configuration (set STRIPE_SECRET_KEY in hosting env)
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# Stripe price IDs (from your Stripe dashboard)
# Adjust these if you create new prices in Live mode.
STRIPE_PRICE_SINGLE = "price_1SXC552EltyWEGkhnT1exvQV"      # Single Pattern – $25
STRIPE_PRICE_PACK10 = "price_1SXCBW2EltyWEGkhcBE1KwPW"      # 10 Pattern Pack – $60
STRIPE_PRICE_3MO    = "price_1SXCJK2EltyWEGkhWnoupSSf"      # 3-Month Unlimited – $75 (recurring every 3 months)
STRIPE_PRICE_ANNUAL = "price_1SXCEq2EltyWEGkhoOIFpb1w"      # Annual Unlimited – $99/year

# Simple JSON “database” for users
BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# Folder for storing generated patterns (per user)
PATTERNS_DIR = os.path.join(BASE_DIR, "patterns")
os.makedirs(PATTERNS_DIR, exist_ok=True)

# Config
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels


# ---------------------- BASIC SECURITY / ANTI-SCRAPING ----------------------


@app.after_request
def add_basic_headers(resp: Response) -> Response:
    # Basic security headers
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # Hint to large crawlers/models not to train on this content.
    resp.headers.setdefault("X-Robots-Tag", "noai, noimageai")
    return resp


@app.get("/robots.txt")
def robots() -> Response:
    rules = [
        "User-agent: *",
        "Disallow: /api/",
        "Disallow: /sample-pattern.zip",
        "Disallow: /success",
        "",
    ]
    return Response("\n".join(rules), mimetype="text/plain")


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
    """
    Resize while preserving aspect ratio to target stitch width.
    Use NEAREST to keep cells crisp (avoid blur).
    """
    w, h = img.size
    if max(w, h) > 2000:
        img = img.copy()
        img.thumbnail((2000, 2000))
        w, h = img.size
    ratio = stitch_w / float(w)
    new_h = max(1, int(round(h * ratio)))
    return img.resize((stitch_w, new_h), Image.Resampling.NEAREST)


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
    Scale each stitch to a cell and overlay only bold 10×10 grid lines.
    No fine per-cell lines, no watermark – clean printable pattern grid.
    """
    sx, sy = base.size
    out = base.resize((sx * cell_px, sy * cell_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    bold = (0, 0, 0, 170)
    for x in range(sx + 1):
        if x % 10 == 0:
            draw.line(
                [(x * cell_px, 0), (x * cell_px, sy * cell_px)],
                fill=bold,
                width=1,
            )
    for y in range(sy + 1):
        if y % 10 == 0:
            draw.line(
                [(0, y * cell_px), (sx * cell_px, sy * cell_px)],
                fill=bold,
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
    """
    Overlay symbol per stitch, then bold 10×10 grid (no per-cell lines).
    Keeps pattern sharp, readable, and free of extra distortion.
    """
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
    bold = (0, 0, 0, 170)
    for x in range(sx + 1):
        if x % 10 == 0:
            draw.line(
                [(x * cell_px, 0), (x * cell_px, sy * cell_px)],
                fill=bold,
                width=1,
            )
    for y in range(sy + 1):
        if y % 10 == 0:
            draw.line(
                [(0, y * cell_px), (sx * cell_px, sy * cell_px)],
                fill=bold,
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


# ---------------------- BASIC ROUTES / ERRORS / FAVICON ----------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True}


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f97316"/>
      <stop offset="1" stop-color="#facc15"/>
    </linearGradient>
  </defs>
  <rect x="4" y="4" width="56" height="56" rx="14" fill="url(#g)"/>
  <g stroke="#fef9c3" stroke-width="1" opacity="0.65">
    <line x1="16" y1="12" x2="16" y2="52"/>
    <line x1="28" y1="12" x2="28" y2="52"/>
    <line x1="40" y1="12" x2="40" y2="52"/>
    <line x1="52" y1="12" x2="52" y2="52"/>
    <line x1="12" y1="16" x2="52" y2="16"/>
    <line x1="12" y1="28" x2="52" y2="28"/>
    <line x1="12" y1="40" x2="52" y2="40"/>
    <line x1="12" y1="52" x2="52" y2="52"/>
  </g>
  <text x="18" y="37" fill="#78350f" font-size="20" font-family="system-ui, sans-serif" font-weight="700">P</text>
  <text x="34" y="43" fill="#7c2d12" font-size="16" font-family="system-ui, sans-serif" font-weight="600">C</text>
</svg>
"""


@app.get("/favicon.svg")
def favicon_svg() -> Response:
    return Response(FAVICON_SVG, mimetype="image/svg+xml")


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


@app.errorhandler(Exception)
def on_error(_e):
    return make_response(jsonify({"error": "server_error"}), 500)


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


# ---------------------- CHECKOUT + SUCCESS ----------------------


@app.post("/checkout")
def create_checkout():
    """
    Create a Stripe Checkout Session for the selected plan.
    Requires the user to be logged in so we can tie the purchase to their account.
    """
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
        "subscription": "free",  # free, single, pack10, unlimited_3m, unlimited_year
        "free_used": False,
        "credits": 0,            # used for credit-based plans
        "patterns": [],          # history of generated patterns
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


# ---------------------- SAMPLE PATTERN ZIP (QUILT DEMO) ----------------------


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


# ---------------------- PATTERN GENERATOR (ACCOUNT + MEMBERSHIP GATED) ----------------------


@app.post("/api/convert")
def convert():
    # Require an account
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to generate patterns."))

    # Basic anti-scraping guard
    ua = (request.headers.get("User-Agent") or "").lower()
    suspicious_tokens = ("python-requests", "curl", "wget", "scrapy", "httpclient", "aiohttp")
    if any(token in ua for token in suspicious_tokens):
        return jsonify({"error": "automated_access_blocked"}), 403

    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(url_for("signup", msg="Create your PatternCraft.app account to continue."))

    subscription = user.get("subscription", "free")
    credits = int(user.get("credits", 0) or 0)
    mark_free_used = False
    consume_credit = False

    # Membership logic:
    # - unlimited_3m / unlimited_year: unlimited usage
    # - if credits > 0: consume 1 credit per convert
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

    pattern_id = uuid.uuid4().hex[:12]
    preview_img: Optional[Image.Image] = None
    pattern_record: dict = {}
    original_name = getattr(file, "filename", "") or "Uploaded artwork"

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

            preview_img = grid_img

            total_stitches = sum(counts.values()) or 1
            lines = ["hex,r,g,b,stitches,percent,skeins_est"]
            legend_entries: List[dict] = []
            for (r, g, b), c in sorted(
                counts.items(), key=lambda kv: kv[1], reverse=True
            ):
                skeins = skeins_per_color(
                    c, cloth_count, strands, waste_pct / 100.0
                )
                percent_val = 100 * c / total_stitches
                lines.append(
                    f"{to_hex((r,g,b))},{r},{g},{b},{c},{percent_val:.2f},{skeins:.2f}"
                )
                legend_entries.append(
                    {
                        "hex": to_hex((r, g, b)),
                        "r": r,
                        "g": g,
                        "b": b,
                        "stitches": c,
                        "percent": round(percent_val, 2),
                        "skeins_est": round(skeins, 2),
                    }
                )
            z.writestr("legend.csv", "\n".join(lines))

            note = (
                "Knitting preview compresses row height; verify gauge."
                if ptype == "knit"
                else "Cross-stitch grid with bold 10×10 guides and symbol overlay."
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

            pattern_record = {
                "id": pattern_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ptype": ptype,
                "stitch_style": stitch_style,
                "stitches_w": sx,
                "stitches_h": sy,
                "colors": len(counts),
                "file": f"{pattern_id}.zip",
                "preview": f"{pattern_id}.png",
                "legend": legend_entries,
                "original_name": original_name,
            }

        elif ptype == "emb":
            small = resize_for_stitch_width(base, stitch_w)
            bw = to_monochrome(small, threshold=emb_thresh)
            pts = serpentine_points(bw, step=emb_step)
            for name, data in write_embroidery_outputs(pts).items():
                z.writestr(name, data)
            emb_meta = {
                "type": "emb",
                "stitch_style": "run",
                "points": len(pts),
                "pyembroidery": HAS_PYEMB,
            }
            z.writestr("meta.json", json.dumps(emb_meta, indent=2))

            preview_img = bw.convert("RGB")

            pattern_record = {
                "id": pattern_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ptype": "emb",
                "stitch_style": "run",
                "stitches_w": small.size[0],
                "stitches_h": small.size[1],
                "colors": 1,
                "file": f"{pattern_id}.zip",
                "preview": f"{pattern_id}.png",
                "legend": [],
                "original_name": original_name,
            }

        else:
            return jsonify({"error": "unknown_ptype"}), 400

    out_zip.seek(0)
    zip_bytes = out_zip.getvalue()

    zip_path = os.path.join(PATTERNS_DIR, f"{pattern_id}.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    if preview_img is not None:
        preview_path = os.path.join(PATTERNS_DIR, f"{pattern_id}.png")
        try:
            preview_img.save(preview_path, format="PNG")
        except Exception:
            pattern_record["preview"] = None
    else:
        pattern_record["preview"] = None

    if consume_credit and credits > 0:
        user["credits"] = max(0, credits - 1)
    if mark_free_used:
        user["free_used"] = True

    patterns = user.get("patterns") or []
    patterns.append(pattern_record)
    user["patterns"] = patterns
    users[email] = user
    save_users(users)

    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pattern_{ptype}.zip",
    )


# ---------------------- PATTERN HISTORY ROUTES ----------------------


@app.get("/patterns")
def patterns_page() -> str:
    user = get_current_user()
    if not user:
        return redirect(url_for("login", msg="Log in to see your patterns."))
    patterns = user.get("patterns") or []
    patterns = sorted(patterns, key=lambda p: p.get("created_at", ""), reverse=True)
    return render_template_string(PATTERNS_HTML, user=user, patterns=patterns)


def _get_user_pattern(email: str, pattern_id: str) -> Optional[dict]:
    users = load_users()
    user = users.get(email)
    if not user:
        return None
    for p in user.get("patterns") or []:
        if p.get("id") == pattern_id:
            return p
    return None


@app.get("/patterns/<pattern_id>/download")
def pattern_download(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to download your patterns."))
    pat = _get_user_pattern(email, pattern_id)
    if not pat:
        return "Pattern not found", 404
    filename = pat.get("file")
    if not filename:
        return "Pattern file missing", 404
    path = os.path.join(PATTERNS_DIR, filename)
    if not os.path.exists(path):
        return "Pattern file missing", 404
    return send_file(path, mimetype="application/zip", as_attachment=True, download_name=filename)


@app.get("/patterns/<pattern_id>/preview.png")
def pattern_preview_png(pattern_id: str):
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to see your pattern preview."))
    pat = _get_user_pattern(email, pattern_id)
    if not pat:
        return "Pattern not found", 404
    preview_name = pat.get("preview")
    if not preview_name:
        return "No preview available for this pattern", 404
    path = os.path.join(PATTERNS_DIR, preview_name)
    if not os.path.exists(path):
        return "Preview image missing", 404
    return send_file(path, mimetype="image/png")


@app.get("/patterns/<pattern_id>")
def pattern_view(pattern_id: str) -> str:
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to view your patterns."))
    pat = _get_user_pattern(email, pattern_id)
    if not pat:
        return "Pattern not found", 404
    user = get_current_user()
    return render_template_string(PATTERN_VIEW_HTML, user=user, pattern=pat)


# ---------------------- INLINE HTML: HOMEPAGE ----------------------


HOMEPAGE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    :root{
      --bg:#FFF7D6;--fg:#1F2933;--muted:#6B7280;
      --line:#F3E8C6;--radius:14px;--shadow:0 14px 30px rgba(180,137,52,.35);
      --accent:#F97316;--accent-soft:#FFF1C9;--accent-strong:#B45309;
      --pill:#F97316;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#FEF9C3 0,#FFFBEB 40%,#FFF7D6 100%);
    }
    a{color:#B45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1040px;margin:0 auto;padding:24px 16px 48px}
    h1{font-size:2.6rem;margin:0 0 8px}
    h2{margin:0 0 10px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      font-weight:800;font-size:20px;letter-spacing:.08em;text-transform:uppercase;
      display:flex;align-items:center;gap:8px;
    }
    .brand-mark{
      width:22px;height:22px;border-radius:6px;
      background:linear-gradient(135deg,#F97316,#FACC15);
      position:relative;overflow:hidden;
      box-shadow:0 4px 10px rgba(248,181,55,.55);
    }
    .brand-mark::before{
      content:"";position:absolute;inset:4px;
      border-radius:4px;
      background-image:
        linear-gradient(to right,rgba(255,255,255,.6) 1px,transparent 1px),
        linear-gradient(to bottom,rgba(255,255,255,.6) 1px,transparent 1px);
      background-size:4px 4px;opacity:.8;
    }
    .top-links{font-size:13px;color:var(--muted)}
    .top-links a{margin-left:8px;}
    .card{
      background:#FFFEFA;
      border-radius:var(--radius);
      border:1px solid var(--line);
      box-shadow:var(--shadow);
      padding:20px;
    }
    .hero{
      display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2fr);
      gap:20px;margin-bottom:24px;align-items:center;
    }
    .hero-tagline{color:var(--muted);max-width:440px;}
    .muted{color:var(--muted);font-size:13px}
    .pill{
      padding:11px 24px;border-radius:999px;
      background:linear-gradient(135deg,var(--pill),#EA580C);
      color:#fff;border:none;cursor:pointer;
      font-size:15px;font-weight:700;letter-spacing:.03em;
      box-shadow:0 14px 32px rgba(248,113,22,.55);
      transition:transform .08s,box-shadow .08s,background-color .08s;
      display:inline-block;text-decoration:none;text-align:center;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 18px 40px rgba(248,113,22,.7);}
    .pill-secondary{
      background:linear-gradient(135deg,#FFF7D6,#FFFBEB);
      color:var(--fg);
      border:1px solid rgba(234,179,8,.7);
      box-shadow:0 8px 20px rgba(250,204,21,.4);
    }
    .pill-secondary:hover{
      box-shadow:0 12px 28px rgba(250,204,21,.55);
    }
    .pill-ready{
      background:linear-gradient(135deg,#16A34A,#22C55E);
      box-shadow:0 16px 36px rgba(22,163,74,.65);
    }
    .pill-ready:hover{
      box-shadow:0 20px 42px rgba(22,163,74,.8);
    }
    .hero-cta-row{
      display:flex;gap:12px;margin-top:14px;flex-wrap:wrap;align-items:center;
    }
    .hero-note{font-size:12px;color:#854D0E;margin-top:8px;}
    .badge-row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:4px 10px;border-radius:999px;
      background:rgba(255,255,255,.9);border:1px solid rgba(250,204,21,.6);
      font-size:11px;color:#92400E;text-transform:uppercase;letter-spacing:.08em;
    }
    .chip-dot{width:8px;height:8px;border-radius:999px;background:#16A34A}
    .badge{
      font-size:11px;padding:6px 10px;border-radius:999px;
      background:#FEF3C7;color:#92400E;border:1px solid rgba(245,158,11,.7);
    }
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
      background:#FFEFC1;border-color:#F97316;
      transform:translateY(-1px);
      box-shadow:0 10px 26px rgba(248,181,55,.6);
    }
    .file-ready{
      background:#DCFCE7;
      border-color:#16A34A;
      box-shadow:0 10px 26px rgba(34,197,94,.6);
    }
    .file input{display:none}
    .file-label-main{font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.06em}
    .file-label-sub{font-size:12px;color:var(--muted)}
    .free-note{
      margin-top:6px;font-size:12px;color:#166534;background:#FEF9C3;
      border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px;
      border:1px solid #FACC15;
    }
    .free-dot{width:8px;height:8px;border-radius:999px;background:#16A34A}
    fieldset{border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0;background:#FFFEF5;}
    legend{font-size:13px;padding:0 4px;color:#78350F}
    .row{display:flex;flex-wrap:wrap;gap:12px}
    .row > label{flex:1 1 150px;font-size:13px}
    .row input,.row select{
      width:100%;margin-top:3px;padding:7px 10px;border-radius:8px;
      border:1px solid #E5D3A1;font-size:13px;background:#FFFBEB;color:#1F2933;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#F97316;box-shadow:0 0 0 1px rgba(248,113,22,.6);
    }
    label{font-size:13px}
    .controls-note{font-size:11px;color:#9CA3AF;margin-top:4px}
    .section-title{font-size:1.1rem;margin-bottom:6px}
    .hidden{display:none}
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
      <span class="brand-mark"></span>
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
        <span>For cross-stitch, knitting, and quilting</span>
      </div>
      <h1>Turn art into stitchable patterns</h1>
      <p class="hero-tagline">
        PatternCraft.app converts your artwork into cross-stitch grids, knitting charts,
        and embroidery-ready files with one upload. Export a full ZIP you can print or take to your machine.
      </p>
      <div class="hero-cta-row">
        {% if user %}
          <button class="pill" onclick="document.getElementById('make').scrollIntoView({behavior:'smooth'})">
            Open the tool
          </button>
        {% else %}
          <a class="pill" href="/login?msg=Log+in+to+open+the+PatternCraft+tool.">
            Open the tool
          </a>
        {% endif %}
        <a class="pill pill-secondary" href="/pricing#how">
          See how it works
        </a>
      </div>
      <div class="hero-note">
        Step 1: <a href="/pricing#how">See how PatternCraft.app works</a> ·
        Step 2: <a href="/signup">Create your account</a> ·
        Step 3: Upload art and generate your pattern ZIP.
      </div>
      <div class="badge-row">
        <span class="badge">One pattern included with every account</span>
        <span class="badge">Designed for hobbyists & pattern sellers</span>
      </div>
    </div>

    <div class="card">
      <h2 class="section-title">Why makers use PatternCraft.app</h2>
      <p class="muted">
        A purpose-built pattern tool with stitchers in mind:
      </p>
      <ul class="muted" style="font-size:13px;margin-top:8px;padding-left:20px;">
        <li>Clean grids with bold 10×10 guides and symbol overlays</li>
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
      Create a PatternCraft.app account or log in to generate patterns. Every account includes one pattern on us.
      After that, plans on the pricing page keep you creating.
    </p>
    <div class="make-layout">
      <div class="make-main">
        <form method="POST" action="/api/convert" enctype="multipart/form-data">
          {% if user %}
          <label class="file">
            <input id="fileInput" type="file" name="file" accept="image/*" required onchange="pickFile(this)">
            <div>
              <div class="file-label-main">Upload picture here</div>
              <div class="file-label-sub">
                Drop in your artwork or tap to browse from your device.
              </div>
            </div>
          </label>
          {% else %}
          <a class="file" href="/login?msg=Log+in+or+create+your+free+account+to+upload+a+picture.">
            <div>
              <div class="file-label-main">Upload picture here</div>
              <div class="file-label-sub">
                Log in or create a free account to upload your artwork.
                Every signup includes <strong>1 free pattern</strong>.
              </div>
            </div>
          </a>
          {% endif %}

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

          <div style="margin-top:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
            <button class="pill" id="generateBtn" type="submit">Generate pattern ZIP</button>
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
    const generateBtn = document.getElementById('generateBtn');

    if (!inp.files || !inp.files[0]){
      if (wrapper) wrapper.classList.remove('file-ready');
      if (label) label.textContent = 'Upload picture here';
      if (generateBtn) generateBtn.classList.remove('pill-ready');
      return;
    }

    if (wrapper) wrapper.classList.add('file-ready');
    if (label) label.textContent = 'Image attached';
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


# ---------------------- INLINE HTML: SIGNUP / LOGIN / PRICING / PATTERNS / SUCCESS ----------------------


SIGNUP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    body{
      margin:0;background:#FFF7D6;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:#1F2933;
    }
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#FFFEFA;
      border-radius:14px;
      border:1px solid #F3E8C6;
      padding:20px;
      box-shadow:0 14px 32px rgba(180,137,52,.35);
    }
    h1{margin:0 0 10px;font-size:1.6rem}
    .muted{font-size:13px;color:#6B7280}
    label{display:block;font-size:13px;margin-top:12px}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #E5D3A1;font-size:14px;background:#FFFBEB;color:#1F2933;
    }
    input:focus{
      outline:none;border-color:#F97316;box-shadow:0 0 0 1px rgba(248,113,22,.6);
    }
    .pill{
      margin-top:16px;padding:11px 24px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#F97316,#EA580C);color:#fff;
      font-size:15px;font-weight:700;cursor:pointer;
      box-shadow:0 14px 32px rgba(248,113,22,.6);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 18px 40px rgba(248,113,22,.75);}
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;background:#FEE2E2;border-radius:8px;padding:6px 8px}
    a{color:#B45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#6B7280;padding-left:18px;margin-top:10px}
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
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    body{
      margin:0;background:#FFF7D6;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:#1F2933;
    }
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{
      background:#FFFEFA;
      border-radius:14px;
      border:1px solid #F3E8C6;
      padding:20px;
      box-shadow:0 14px 32px rgba(180,137,52,.35);
    }
    h1{margin:0 0 10px;font-size:1.6rem}
    .muted{font-size:13px;color:#6B7280}
    label{display:block;font-size:13px;margin-top:12px}
    input{
      width:100%;margin-top:4px;padding:9px 11px;border-radius:10px;
      border:1px solid #E5D3A1;font-size:14px;background:#FFFBEB;color:#1F2933;
    }
    input:focus{
      outline:none;border-color:#4F46E5;box-shadow:0 0 0 1px rgba(79,70,229,.6);
    }
    .pill{
      margin-top:16px;padding:11px 24px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#4C51BF,#4338CA);color:#fff;
      font-size:15px;font-weight:700;cursor:pointer;
      box-shadow:0 14px 32px rgba(79,70,229,.6);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 18px 40px rgba(79,70,229,.75);}
    .pill-secondary{
      margin-top:12px;
      padding:11px 24px;
      border-radius:999px;
      border:1px solid #F97316;
      background:linear-gradient(135deg,#FFFBEB,#FEF3C7);
      color:#B45309;
      font-size:15px;
      font-weight:700;
      cursor:pointer;
      text-decoration:none;
      display:inline-block;
      box-shadow:0 12px 28px rgba(248,181,55,.5);
    }
    .pill-secondary:hover{
      box-shadow:0 16px 36px rgba(248,181,55,.65);
    }
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;background:#FEE2E2;border-radius:8px;padding:6px 8px}
    a{color:#B45309;text-decoration:none;}
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
    {% if attempts_left is not none %}
      <p class="muted" style="margin-top:6px;">
        {% if attempts_left > 0 %}
          You have {{ attempts_left }} more attempt{{ 's' if attempts_left != 1 else '' }} before we suggest creating a new account.
        {% else %}
          If you’re having trouble logging in, you can create a new account with your email.
        {% endif %}
      </p>
    {% endif %}

    <div style="margin-top:16px;">
      <a class="pill-secondary" href="/signup">Create Free Account</a>
      <p class="muted" style="margin-top:6px;">
        Includes <strong>1 free pattern</strong> with every signup.
      </p>
    </div>
  </div>
</div>
</body>
</html>
"""

PRICING_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PatternCraft • Pricing</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>
:root{
  --fg:#1F2933; --muted:#6B7280; --accent:#E4006D; --line:#F3E8C6; --card:#FFFEFA;
  --radius:14px; --wrap:1100px;
}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{
  font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  color:var(--fg);
  background:#FFF7D6;
}
.wrap{max-width:var(--wrap);margin:0 auto;padding:20px 16px 32px}
header{position:sticky;top:0;background:#FFF7D6;border-bottom:1px solid var(--line);z-index:5}
.brand{font-weight:800;letter-spacing:.2px;display:flex;align-items:center;gap:8px}
.brand-mark{
  width:20px;height:20px;border-radius:6px;
  background:linear-gradient(135deg,#F97316,#FACC15);
  box-shadow:0 4px 10px rgba(248,181,55,.55);
}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.btn{
  display:inline-block;padding:11px 22px;border-radius:999px;
  border:1px solid var(--accent);background:var(--accent);color:#fff;
  text-decoration:none;font-weight:700;cursor:pointer;font-size:15px;
  box-shadow:0 12px 28px rgba(225,29,72,.45);
}
.btn:hover{box-shadow:0 16px 36px rgba(225,29,72,.6);}
.btn.ghost{background:transparent;color:var(--accent);box-shadow:none}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-top:16px}
.card{
  background:var(--card);
  border:1px solid var(--line);border-radius:var(--radius);padding:20px;
  box-shadow:0 12px 26px rgba(180,137,52,.3);
}
.card h3{margin:0 0 6px}
.price{font-size:28px;font-weight:800;margin:4px 0}
.small{font-size:14px;color:var(--muted)}
.list{margin:8px 0 12px;padding-left:20px;color:var(--muted);font-size:13px}
.badge{
  display:inline-block;background:#FDE68A;color:#92400E;border-radius:999px;
  padding:4px 12px;font-size:13px;font-weight:600;margin-bottom:6px
}
.notice{
  margin-top:10px;padding:10px 12px;border-radius:10px;
  background:#FEF3C7;border:1px solid #FACC15;color:#92400E;
  font-size:14px;
}
footer{border-top:1px solid var(--line);margin-top:24px}
@media (max-width:700px){ .cards{grid-template-columns:1fr} }

.cards .card form{
  margin-top:10px;
}
.cards .card .btn{
  width:100%;
  text-align:center;
}

/* steps section */
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
  box-shadow:0 10px 24px rgba(180,137,52,.25);
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
</head>
<body>
<header>
  <div class="wrap row" style="justify-content:space-between">
    <div class="brand">
      <span class="brand-mark"></span>
      <span>PatternCraft.app</span>
    </div>
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
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="pack10">
        <button class="btn" type="submit">Buy 10-pack</button>
      </form>
      <p class="small">Save big vs buying singles.</p>
    </div>

    <div class="card">
      <h3>3-Month Unlimited</h3>
      <div class="price">$75 / 3 months</div>
      <p class="small">Recurring every 3 months until canceled.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>Higher-resolution output</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_3m">
        <button class="btn" type="submit">Start 3-month plan</button>
      </form>
      <p class="small">Perfect for focused projects or seasons.</p>
    </div>

    <div class="card">
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
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_year">
        <button class="btn" type="submit">Choose annual unlimited</button>
      </form>
      <p class="small">Best value if you stitch more than 4 patterns a year.</p>
    </div>
  </div>
</section>

<section class="wrap" id="how">
  <h2>How PatternCraft.app works</h2>
  <p class="small">From photo to stitch-ready pattern in three simple steps.</p>
  <div class="steps">
    <div class="step-card">
      <div class="step-num">1</div>
      <h3>Choose a plan</h3>
      <p class="small">
        Start with a single pattern, a 10-pack, or an unlimited plan depending on how often you stitch.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3>Upload your image</h3>
      <p class="small">
        Drop in a photo, artwork, or logo. PatternCraft analyzes it and lets you adjust size and colors.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3>Download your pattern ZIP</h3>
      <p class="small">
        Get a ZIP with grid.png, legend.csv, meta.json, and optional PDF or embroidery files,
        ready to print or import into your tools.
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
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#FFF7D6;
      color:#1F2933;
    }
    .wrap{max-width:960px;margin:0 auto;padding:24px 16px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      font-weight:800;font-size:18px;letter-spacing:.08em;text-transform:uppercase;
      display:flex;align-items:center;gap:8px;
    }
    .brand-mark{
      width:20px;height:20px;border-radius:6px;
      background:linear-gradient(135deg,#F97316,#FACC15);
      box-shadow:0 4px 10px rgba(248,181,55,.55);
    }
    a{color:#B45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .top-links{font-size:13px;color:#6B7280}
    .top-links a{margin-left:8px;}
    h1{font-size:1.6rem;margin:0 0 10px}
    .muted{font-size:13px;color:#6B7280}
    .grid{display:grid;gap:14px;margin-top:16px}
    .card{
      background:#FFFEFA;
      border-radius:12px;border:1px solid #F3E8C6;
      padding:14px;display:flex;gap:12px;align-items:center;
      box-shadow:0 10px 24px rgba(180,137,52,.25);
    }
    .thumb{
      width:90px;height:90px;border-radius:10px;
      background:#FFFBEB;border:1px solid #F3E8C6;
      display:flex;align-items:center;justify-content:center;
      overflow:hidden;
    }
    .thumb img{max-width:100%;max-height:100%;display:block}
    .meta{flex:1 1 auto}
    .meta h2{margin:0 0 4px;font-size:15px}
    .meta p{margin:2px 0;font-size:12px;color:#6B7280}
    .actions{display:flex;flex-direction:column;gap:6px}
    .btn{
      display:inline-block;padding:8px 14px;border-radius:999px;
      border:1px solid #F97316;background:#F97316;color:#fff;
      font-size:13px;font-weight:700;text-decoration:none;text-align:center;
      box-shadow:0 10px 24px rgba(248,113,22,.5);
    }
    .btn.ghost{
      background:#FFFBEB;border-color:#F59E0B;color:#92400E;
      box-shadow:none;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <span class="brand-mark"></span>
      <span>PatternCraft.app</span>
    </div>
    <div class="top-links">
      <a href="/">Tool</a> · <a href="/pricing">Pricing</a> · <a href="/logout">Sign out</a>
    </div>
  </div>

  <h1>My patterns</h1>
  {% if not patterns %}
    <p class="muted">
      You haven’t generated any patterns yet. Go to the <a href="/">tool</a> to create your first pattern.
    </p>
  {% else %}
    <p class="muted">
      These are the patterns you’ve created with your account. You can preview the grid with its legend,
      print, or download the ZIP again.
    </p>
    <div class="grid">
      {% for p in patterns %}
        <div class="card">
          <div class="thumb">
            {% if p.preview %}
              <img src="/patterns/{{ p.id }}/preview.png" alt="Pattern preview">
            {% else %}
              <span class="muted" style="font-size:11px;">No preview</span>
            {% endif %}
          </div>
          <div class="meta">
            <h2>{{ p.ptype|upper }} pattern — {{ p.stitch_style }}</h2>
            <p>
              Created {{ p.created_at[:10] }} · {{ p.stitches_w }} × {{ p.stitches_h }} stitches
              {% if p.colors %} · {{ p.colors }} colors{% endif %}
            </p>
            {% if p.original_name %}
              <p class="muted">From: {{ p.original_name }}</p>
            {% endif %}
          </div>
          <div class="actions">
            <a class="btn" href="/patterns/{{ p.id }}">View &amp; print</a>
            <a class="btn ghost" href="/patterns/{{ p.id }}/download">Download ZIP</a>
          </div>
        </div>
      {% endfor %}
    </div>
  {% endif %}
</div>
</body>
</html>
"""

PATTERN_VIEW_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Saved pattern — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#FFF7D6;
      color:#1F2933;
    }
    .wrap{max-width:960px;margin:0 auto;padding:24px 16px 40px}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{
      font-weight:800;font-size:18px;letter-spacing:.08em;text-transform:uppercase;
      display:flex;align-items:center;gap:8px;
    }
    .brand-mark{
      width:20px;height:20px;border-radius:6px;
      background:linear-gradient(135deg,#F97316,#FACC15);
      box-shadow:0 4px 10px rgba(248,181,55,.55);
    }
    a{color:#B45309;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .top-links{font-size:13px;color:#6B7280}
    .top-links a{margin-left:8px;}
    h1{font-size:1.6rem;margin:0 0 10px}
    .muted{font-size:13px;color:#6B7280}
    .card{
      background:#FFFEFA;
      border-radius:14px;border:1px solid #F3E8C6;
      padding:18px;box-shadow:0 12px 30px rgba(180,137,52,.3)
    }
    .layout{
      display:grid;
      grid-template-columns:minmax(0,2fr) minmax(0,1.4fr);
      gap:18px;
      margin-top:14px;
      align-items:flex-start;
    }
    @media (max-width:800px){
      .layout{grid-template-columns:1fr}
    }
    .preview{
      border-radius:10px;
      border:1px solid #F3E8C6;
      background:#FFFBEB;
      padding:8px;
      text-align:center;
      max-height:70vh;
      overflow:auto;
    }
    .preview img{
      max-width:100%;
      height:auto;
    }
    .legend-box{
      border-radius:10px;
      border:1px solid #F3E8C6;
      background:#FFFBEB;
      padding:10px 12px;
      max-height:70vh;
      overflow:auto;
    }
    .legend-title{
      font-size:14px;
      font-weight:600;
      margin:0 0 6px;
    }
    .legend-sub{
      font-size:12px;
      color:#6B7280;
      margin:0 0 10px;
    }
    .legend-list{
      list-style:none;
      margin:0;
      padding:0;
      font-size:12px;
    }
    .legend-row{
      display:grid;
      grid-template-columns:18px 80px minmax(0,1fr) auto;
      gap:8px;
      align-items:center;
      padding:4px 0;
      border-bottom:1px solid #F3E8C6;
    }
    .legend-row:last-child{
      border-bottom:none;
    }
    .swatch{
      width:16px;
      height:16px;
      border-radius:4px;
      border:1px solid rgba(148,107,36,.8);
      display:inline-block;
    }
    .legend-hex{
      font-family:ui-monospace,Menlo,monospace;
      color:#1F2933;
    }
    .legend-count{
      color:#374151;
    }
    .legend-skeins{
      color:#6B7280;
      text-align:right;
      white-space:nowrap;
    }
    .actions{
      margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;
    }
    .btn{
      display:inline-block;padding:9px 18px;border-radius:999px;
      border:1px solid #F97316;background:#F97316;color:#fff;
      font-size:14px;font-weight:700;text-decoration:none;text-align:center;
      cursor:pointer;
      box-shadow:0 12px 28px rgba(248,113,22,.5);
    }
    .btn.ghost{
      background:#FFFBEB;border-color:#F59E0B;color:#92400E;
      box-shadow:none;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <span class="brand-mark"></span>
      <span>PatternCraft.app</span>
    </div>
    <div class="top-links">
      <a href="/">Tool</a> · <a href="/patterns">My patterns</a> · <a href="/logout">Sign out</a>
    </div>
  </div>

  <div class="card">
    <h1>Your saved pattern</h1>
    <p class="muted">
      {{ pattern.ptype|upper }} · {{ pattern.stitch_style }} · {{ pattern.stitches_w }} × {{ pattern.stitches_h }} stitches{% if pattern.colors %} · {{ pattern.colors }} colors{% endif %}
      {% if pattern.original_name %}<br>From: {{ pattern.original_name }}{% endif %}
    </p>

    <div class="layout">
      <div class="preview">
        {% if pattern.preview %}
          <img src="/patterns/{{ pattern.id }}/preview.png" alt="Pattern grid">
        {% else %}
          <p class="muted">No preview available for this pattern. You can still download the ZIP and print from there.</p>
        {% endif %}
      </div>

      <div class="legend-box">
        <p class="legend-title">Color legend</p>
        <p class="legend-sub">
          Swatches, counts, and estimated skeins from your generated legend.csv file.
        </p>
        {% if pattern.legend %}
          <ul class="legend-list">
            {% for c in pattern.legend %}
              <li class="legend-row">
                <span class="swatch" style="background: rgb({{ c.r }}, {{ c.g }}, {{ c.b }});"></span>
                <span class="legend-hex">{{ c.hex }}</span>
                <span class="legend-count">
                  {{ c.stitches }} sts ({{ '%.2f'|format(c.percent) }}%)
                </span>
                <span class="legend-skeins">
                  {{ '%.2f'|format(c.skeins_est) }} skeins
                </span>
              </li>
            {% endfor %}
          </ul>
        {% else %}
          <p class="muted">
            This pattern doesn’t have a parsed legend stored yet. You can still find the full legend inside the downloaded ZIP.
          </p>
        {% endif %}
      </div>
    </div>

    <div class="actions">
      <a class="btn" href="/patterns/{{ pattern.id }}/download">Download ZIP</a>
      <button class="btn ghost" type="button" onclick="window.print()">Print this view</button>
    </div>
  </div>
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
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    body{
      margin:0;
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#FFF7D6;
      color:#1F2933;
    }
    .wrap{
      max-width:520px;
      margin:0 auto;
      padding:32px 16px 40px;
    }
    .card{
      background:#FFFEFA;
      border-radius:14px;
      border:1px solid #F3E8C6;
      padding:24px;
      box-shadow:0 14px 32px rgba(180,137,52,.35);
    }
    h1{margin:0 0 10px;font-size:1.7rem}
    p{margin:6px 0;font-size:14px;color:#6B7280}
    a{
      color:#B45309;
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
      <p>Thank you{% if user %}, {{ user.email }}{% endif %}. Your PatternCraft.app plan will be updated shortly.</p>
      <p>You can go back to the tool and start generating patterns right away. If your account hasn’t updated yet, it will as soon as Stripe finishes processing your purchase and you update your membership settings.</p>
      <p style="margin-top:14px;">
        <a href="/">← Back to PatternCraft.app</a>
      </p>
    </div>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)


