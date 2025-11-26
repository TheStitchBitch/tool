from __future__ import annotations
import io
import json
import math
import os
import time
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

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Stripe configuration (LIVE secret key is set in Render env: STRIPE_SECRET_KEY)
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# Stripe price IDs (LIVE; from your latest CSV)
# Make sure these match the LIVE prices in your Stripe dashboard
STRIPE_PRICE_SINGLE = "price_1SXNyWCINTImVye2jayzoKKj"   # Single Pattern – $25
STRIPE_PRICE_PACK10 = "price_1SXNyRCINTImVye2m433u7pL"   # 10 Pattern Pack – $60
STRIPE_PRICE_3MO    = "price_1SXTFUCINTImVye2JwOxUN55"   # 3-Month Unlimited – $75 / 3 months
STRIPE_PRICE_ANNUAL = "price_1SXNyNCINTImVye2rcxl5LsO"   # Pro Annual – $99 / year

# Simple JSON “database” for users (lives on the app filesystem)
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# Config
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000  # max width/height in pixels

# Subscription durations (seconds)
THREE_MONTHS = 90 * 24 * 3600   # approximate window for 3‑month access
ONE_YEAR = 365 * 24 * 3600


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


def subscription_is_active(user: dict) -> Tuple[bool, bool]:
    """
    Returns (unlimited_active, updated_user_flag).
    unlimited_active = True if 3-mo or annual is currently within its window.
    updated_user_flag = True if we modified subscription due to expiry.
    """
    updated = False
    now = int(time.time())
    sub = (user.get("subscription") or "free").lower()
    started = int(user.get("subscription_started", 0) or 0)

    if sub == "unlimited_3m":
        if started and now - started <= THREE_MONTHS:
            return True, False
        # expired
        user["subscription"] = "free"
        updated = True
        return False, updated

    if sub == "unlimited_year":
        if started and now - started <= ONE_YEAR:
            return True, False
        # expired
        user["subscription"] = "free"
        updated = True
        return False, updated

    return False, False


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
    return render_template_string(HOMEPAGE_HTML, user=user)


@app.get("/pricing")
def pricing() -> str:
    user = get_current_user()
    reason = request.args.get("reason", "")
    message = ""
    if reason == "used_free":
        message = (
            "You’ve already used your PatternCraft.app pattern for this account. "
            "Choose a plan to keep generating patterns."
        )
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
        return redirect(
            url_for("login", msg="Log in or create an account before purchasing a plan.")
        )

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
            success_url=url_for("success", _external=True)
            + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=url_for("pricing", _external=True),
        )
    except Exception:
        return redirect(url_for("pricing", reason="checkout_error"))

    return redirect(checkout_session.url)


@app.get("/success")
def success():
    """
    After a successful Stripe Checkout, we:
    - look up the Checkout Session’s line item price,
    - update the user's subscription/credits accordingly.
    """
    user = get_current_user()
    updated = False
    session_id = request.args.get("session_id", "")

    if user and session_id and stripe.api_key:
        email = user.get("email")
        try:
            # Fetch line items to see which price was bought
            line_items = stripe.checkout.Session.list_line_items(session_id, limit=1)
            if line_items.data:
                price = line_items.data[0].price  # type: ignore[attr-defined]
                price_id = price.id

                users = load_users()
                stored = users.get(email)
                if stored:
                    # Single pattern => +1 credit
                    if price_id == STRIPE_PRICE_SINGLE:
                        stored["subscription"] = "single"
                        stored["credits"] = int(stored.get("credits", 0) or 0) + 1

                    # 10-pack => +10 credits
                    elif price_id == STRIPE_PRICE_PACK10:
                        stored["subscription"] = "pack10"
                        stored["credits"] = int(stored.get("credits", 0) or 0) + 10

                    # 3-month unlimited => time-based unlimited_3m
                    elif price_id == STRIPE_PRICE_3MO:
                        stored["subscription"] = "unlimited_3m"
                        stored["subscription_started"] = int(time.time())

                    # Annual unlimited => time-based unlimited_year
                    elif price_id == STRIPE_PRICE_ANNUAL:
                        stored["subscription"] = "unlimited_year"
                        stored["subscription_started"] = int(time.time())

                    users[email] = stored
                    save_users(users)
                    updated = True
        except Exception:
            # If anything fails, we still show success page; user can contact support if entitlements not updated
            pass

    return render_template_string(SUCCESS_HTML, user=user, updated=updated)


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
        "subscription": "free",   # free, single, pack10, unlimited_3m, unlimited_year
        "subscription_started": 0,
        "free_used": False,
        "credits": 0,            # used for credit-based plans
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
                    msg="We couldn’t match that email and password after several attempts. "
                        "Create a PatternCraft.app account to get started.",
                )
            )
        attempts_left = max(0, 3 - failures)
        return render_template_string(
            LOGIN_HTML,
            user=user,
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


# ---------------------- SAMPLE PATTERN ZIP ----------------------
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


# ---------------------- PATTERN GENERATOR (ACCOUNT + MEMBERSHIP GATED) ----------------------
@app.post("/api/convert")
def convert():
    # Require an account
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to generate patterns."))

    users = load_users()
    user = users.get(email)
    if not user:
        session.pop("user_email", None)
        return redirect(
            url_for("signup", msg="Create your PatternCraft.app account to continue.")
        )

    # Membership logic:
    # - unlimited_3m / unlimited_year: time-limited unlimited usage
    # - if credits > 0: consume 1 credit per convert
    # - else: free tier, 1 pattern per account
    unlimited_active, updated = subscription_is_active(user)
    if updated:
        users[email] = user
        save_users(users)

    credits = int(user.get("credits", 0) or 0)
    mark_free_used = False
    consume_credit = False

    if unlimited_active:
        pass  # unlimited use
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

            total_stitches = sum(counts.values()) or 1
            lines = ["hex,r,g,b,stitches,percent,skeins_est"]
            for (r, g, b), c in sorted(
                counts.items(), key=lambda kv: kv[1], reverse=True
            ):
                skeins = skeins_per_color(
                    c, cloth_count, strands, waste_pct / 100.0
                )
                lines.append(
                    f"{to_hex((r,g,b))},{r},{g},{b},{c},"
                    f"{(100*c/total_stitches):.2f},{skeins:.2f}"
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

    # update membership usage
    if consume_credit and credits > 0:
        user["credits"] = max(0, credits - 1)
    if mark_free_used:
        user["free_used"] = True
    if consume_credit or mark_free_used:
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
      display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2.4fr);
      gap:20px;margin-bottom:28px;align-items:stretch;
    }
    .hero-tagline{color:var(--muted);max-width:420px;}
    .muted{color:var(--muted);font-size:13px}
    .pill{
      padding:11px 20px;border-radius:999px;
      background:linear-gradient(135deg,var(--pill),#f97316);
      color:#fff;border:none;cursor:pointer;
      font-size:14px;font-weight:600;letter-spacing:.02em;
      box-shadow:0 7px 18px rgba(248,113,22,.35);
      transition:transform .08s,box-shadow .08s,background-color .08s;
      display:inline-block;
      text-decoration:none;
      text-align:center;
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
    .pill-ready{
      background:#16a34a;
      box-shadow:0 7px 18px rgba(22,163,74,.35);
    }
    .pill-ready:hover{
      box-shadow:0 10px 24px rgba(22,163,74,.45);
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

    /* "Video ad" slideshow styling (PC.app) */
    .video-shell{
      display:flex;
      flex-direction:column;
      gap:10px;
      padding:18px;
      background:linear-gradient(145deg,#fef3c7,#fce7f3,#e0f2fe);
      border-radius:20px;
      border:1px solid rgba(148,163,184,.35);
      box-shadow:0 14px 35px rgba(15,23,42,.22);
    }
    .video-label{
      font-size:12px;
      text-transform:uppercase;
      letter-spacing:.12em;
      color:#6b21a8;
      font-weight:700;
    }
    .video-frame{
      position:relative;
      border-radius:16px;
      overflow:hidden;
      background:#0f172a;
      height:230px;
      box-shadow:0 10px 30px rgba(15,23,42,.35);
    }
    .video-slide{
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding:16px;
      opacity:0;
      transform:translateX(10px);
      transition:opacity .6s ease,transform .6s ease;
      color:#0f172a;
    }
    .video-slide.active{
      opacity:1;
      transform:translateX(0);
    }
    .video-before,
    .video-after{
      flex:1;
      border-radius:14px;
      padding:10px;
      background:#fff;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
      min-width:0;
    }
    .video-before-title,
    .video-after-title{
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.12em;
      color:#6b7280;
      margin-bottom:6px;
    }
    .video-thumb{
      flex:1;
      border-radius:10px;
      background-size:cover;
      background-position:center;
      position:relative;
      overflow:hidden;
    }
    .video-thumb::before{
      content:"";
      position:absolute;
      inset:0;
      opacity:.22;
      background-image:
        linear-gradient(to right,rgba(15,23,42,.4) 1px,transparent 1px),
        linear-gradient(to bottom,rgba(15,23,42,.4) 1px,transparent 1px);
      background-size:12px 12px;
      mix-blend-mode:multiply;
    }
    .video-thumb.photo-1{
      background-image:linear-gradient(135deg,#fee2e2,#fce7f3,#e0f2fe);
    }
    .video-thumb.photo-2{
      background-image:linear-gradient(135deg,#fef3c7,#bbf7d0,#e0f2fe);
    }
    .video-thumb.photo-3{
      background-image:linear-gradient(135deg,#e0f2fe,#ddd6fe,#fce7f3);
    }
    .video-thumb.photo-4{
      background-image:linear-gradient(135deg,#fee2e2,#fef3c7,#dcfce7);
    }
    .video-thumb.pattern{
      background-image:
        linear-gradient(135deg,#fefce8,#e0f2fe),
        repeating-linear-gradient(90deg,transparent 0,transparent 10px,rgba(148,163,184,.7) 10px,rgba(148,163,184,.7) 11px),
        repeating-linear-gradient(180deg,transparent 0,transparent 10px,rgba(148,163,184,.7) 10px,rgba(148,163,184,.7) 11px);
      background-blend-mode:soft-light;
    }
    .video-caption{
      font-size:12px;
      color:#4b5563;
    }
    .video-arrow{
      width:40px;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:20px;
      color:#f97316;
      font-weight:800;
    }
    .video-tag{
      font-size:11px;
      padding:3px 7px;
      border-radius:999px;
      background:rgba(59,130,246,.08);
      color:#1d4ed8;
      display:inline-block;
      margin-top:6px;
    }

    .video-pricing-grid{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:6px;
      margin-top:6px;
    }
    .video-price-pill{
      font-size:11px;
      border-radius:999px;
      padding:6px 10px;
      background:rgba(255,255,255,.9);
      border:1px solid rgba(148,163,184,.4);
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
    }

    .video-slide.brand{
      justify-content:center;
    }
    .video-brand-inner{
      text-align:center;
      max-width:260px;
      margin:0 auto;
      color:#0f172a;
    }
    .video-brand-logo{
      font-weight:800;
      letter-spacing:.16em;
      text-transform:uppercase;
      font-size:11px;
      color:#6b21a8;
      margin-bottom:4px;
    }
    .video-brand-title{
      font-size:18px;
      font-weight:800;
      margin-bottom:4px;
    }
    .video-brand-sub{
      font-size:11px;
      color:#6b7280;
      text-transform:uppercase;
      letter-spacing:.16em;
      margin-bottom:6px;
    }
    .video-brand-copy{
      font-size:12px;
      color:#4b5563;
      margin:4px 0 8px;
    }
    .video-brand-tags{
      display:flex;
      flex-wrap:wrap;
      gap:6px;
      justify-content:center;
      margin-top:4px;
    }
    .video-brand-tags span{
      font-size:11px;
      padding:3px 8px;
      border-radius:999px;
      background:rgba(15,23,42,.06);
      color:#111827;
    }

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
    .file-ready{
      background:#dcfce7;
      border-color:#16a34a;
      box-shadow:0 6px 15px rgba(22,163,74,.35);
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
    @media (max-width:860px){
      .hero{grid-template-columns:1fr}
      .make-layout{grid-template-columns:1fr}
      .video-frame{height:220px;}
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

    <!-- “Ad-style” PC.app slideshow: product → upload → options → checkout → pattern -->
    <div class="video-shell">
      <div class="video-label">PC.app — Picture → Pattern walk-through</div>
      <div class="video-frame" id="videoFrame">
        <!-- Slide 1: choose a plan -->
        <div class="video-slide active">
          <div class="video-before">
            <div class="video-before-title">Step 1 · Pick a PatternCraft plan</div>
            <div class="video-thumb photo-1"></div>
            <span class="video-tag">Single, 10-pack, 3-Month Unlimited, or Pro Annual.</span>
          </div>
          <div class="video-arrow">→</div>
          <div class="video-after">
            <div class="video-after-title">Plans at a glance</div>
            <div class="video-pricing-grid">
              <div class="video-price-pill">Single Pattern · $25</div>
              <div class="video-price-pill">10 Pattern Pack · $60</div>
              <div class="video-price-pill">3-Month Unlimited · $75 / 3 mo</div>
              <div class="video-price-pill">Pro Annual · $99 / year</div>
            </div>
            <span class="video-tag">PC.app runs on PatternCraft.app — you choose how often you stitch.</span>
          </div>
        </div>

        <!-- Slide 2: upload photo -->
        <div class="video-slide">
          <div class="video-before">
            <div class="video-before-title">Step 2 · Upload your photo</div>
            <div class="video-thumb photo-2"></div>
            <span class="video-tag">Drag in a quilt picture, illustration, or logo.</span>
          </div>
          <div class="video-arrow">→</div>
          <div class="video-after">
            <div class="video-after-title">PC.app reads the image</div>
            <div class="video-thumb pattern"></div>
            <span class="video-tag">We analyze color blocks and detail for a clean, stitchable chart.</span>
          </div>
        </div>

        <!-- Slide 3: tool options -->
        <div class="video-slide">
          <div class="video-before">
            <div class="video-before-title">Step 3 · Choose your tool options</div>
            <div class="video-thumb photo-3"></div>
            <span class="video-tag">Cross-stitch, knitting, or embroidery — plus cloth count and colors.</span>
          </div>
          <div class="video-arrow">→</div>
          <div class="video-after">
            <div class="video-after-title">Dial in the pattern</div>
            <div class="video-thumb pattern"></div>
            <span class="video-tag">Set stitch width, max colors, symbols, PDF export, and more.</span>
          </div>
        </div>

        <!-- Slide 4: checkout -->
        <div class="video-slide">
          <div class="video-before">
            <div class="video-before-title">Step 4 · Checkout securely</div>
            <div class="video-thumb photo-4"></div>
            <span class="video-tag">Stripe-powered checkout with card and wallet payments.</span>
          </div>
          <div class="video-arrow">→</div>
          <div class="video-after">
            <div class="video-after-title">Access unlocks instantly</div>
            <div class="video-thumb pattern"></div>
            <span class="video-tag">Your account is upgraded and ready for PC.app pattern runs.</span>
          </div>
        </div>

        <!-- Slide 5: pattern download -->
        <div class="video-slide brand">
          <div class="video-brand-inner">
            <div class="video-brand-logo">PC.app</div>
            <div class="video-brand-sub">by PatternCraft.app</div>
            <div class="video-brand-title">Step 5 · Download your pattern ZIP</div>
            <p class="video-brand-copy">
              PC.app exports a full-resolution, watermark-free ZIP: grid.png, legend.csv,
              meta.json, and optional PDF or embroidery files. Print it, kit it, or sell your pattern.
            </p>
            <div class="video-brand-tags">
              <span>Product selection</span>
              <span>Photo upload</span>
              <span>Tool options</span>
              <span>Checkout</span>
              <span>Pattern generation</span>
            </div>
          </div>
        </div>
      </div>
      <div class="video-caption">
        A short PC.app loop: pick a plan, upload a quilt photo, tweak options, check out, and download a full PatternCraft.app ZIP ready to stitch.
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

  // PC.app slideshow: 5 slides × ~3s ≈ 15 seconds for a full loop
  (function(){
    const slides = Array.from(document.querySelectorAll('.video-slide'));
    if (!slides.length) return;
    let i = 0;
    setInterval(() => {
      const current = slides[i];
      i = (i + 1) % slides.length;
      const next = slides[i];
      if (current) current.classList.remove('active');
      if (next) next.classList.add('active');
    }, 3000);
  })();
</script>
</body>
</html>
"""

# ---------------------- INLINE HTML: SIGNUP / LOGIN / PRICING / SUCCESS ----------------------
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
      New to PatternCraft.app? <a href="/signup">Create an account</a>.
    </p>
  </div>
</div>
</body>
</html>
"""

PRICING_HTML = r"""
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
      <p class="small">Unlimited patterns, billed every 3 months.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>Higher-resolution output</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_3m">
        <button class="btn ghost" type="submit">Start 3-month plan</button>
      </form>
      <p class="small">Renews every 3 months until you cancel in Stripe.</p>
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
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_year">
        <button class="btn ghost" type="submit">Go Pro annual</button>
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
      <h3>Pick your plan</h3>
      <p class="small">
        Start with a single pattern, grab a 10-pack, or unlock unlimited patterns with 3-Month Unlimited or Pro Annual.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3>Upload your image</h3>
      <p class="small">
        Use a quilt photo, illustration, logo, or artwork. We analyze it for stitchable detail and color.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3>Download your pattern ZIP</h3>
      <p class="small">
        Get grid.png, legend.csv, meta.json, and optional PDF or embroidery files — ready to print or import.
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
      font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      background:#F7F4EF;
      color:#111827;
    }
    .wrap{
      max-width:520px;
      margin:0 auto;
      padding:32px 16px 40px;
    }
    .card{
      background:#fff;
      border-radius:14px;
      border:1px solid #e5e7eb;
      padding:24px;
      box-shadow:0 10px 30px rgba(15,23,42,.15);
    }
    h1{margin:0 0 10px;font-size:1.7rem}
    p{margin:6px 0;font-size:14px;color:#4b5563}
    a{
      color:#2563eb;
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
      <p>Thank you{% if user %}, {{ user.email }}{% endif %}. Your PatternCraft.app plan has been recorded.</p>
      {% if updated %}
        <p>Your account has been updated with new PatternCraft.app access. You can start generating patterns right away.</p>
      {% else %}
        <p>Your payment succeeded. If you don’t see your access updated yet, refresh the page or contact support with your receipt.</p>
      {% endif %}
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

