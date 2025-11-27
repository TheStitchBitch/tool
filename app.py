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

# Stripe configuration – make sure STRIPE_SECRET_KEY is set on Render
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# Live Stripe price IDs (from your latest CSV export)
# Single Pattern – $25 one-time
STRIPE_PRICE_SINGLE = "price_1SXNyWCINTImVye2jayzoKKj"
# 10 Pattern Pack – $60 one-time
STRIPE_PRICE_PACK10 = "price_1SXNyRCINTImVye2m433u7pL"
# Pro Annual Unlimited – $99/year subscription
STRIPE_PRICE_ANNUAL = "price_1SXNyNCINTImVye2rcxl5LsO"
# 3-Month Unlimited – $75 every 3 months subscription
STRIPE_PRICE_3MO = "price_1SXTFUCINTImVye2JwOxUN55"

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


def compute_plan_label(user: Optional[dict]) -> str:
    """
    Human-readable plan label for topbar, based on subscription,
    credits, and free trial.
    """
    if not user:
        return ""
    now = time.time()
    sub = user.get("subscription", "free")
    credits = int(user.get("credits", 0) or 0)
    free_used = bool(user.get("free_used"))

    if sub == "unlimited_3m":
        exp = float(user.get("sub_expires_at") or 0.0)
        if exp > now:
            return "3‑Month Unlimited"
    if sub == "unlimited_year":
        exp = float(user.get("sub_expires_at") or 0.0)
        if not exp or exp > now:
            return "Pro Annual Unlimited"

    if credits > 0:
        if credits == 1:
            return "1 pattern credit"
        return f"{credits} pattern credits"

    if not free_used:
        return "Free (1 pattern included)"

    return "Free"


def has_active_unlimited(user: dict) -> bool:
    """
    True if user currently has an active unlimited plan.
    """
    now = time.time()
    sub = user.get("subscription", "free")
    exp = float(user.get("sub_expires_at") or 0.0)
    if sub == "unlimited_3m" and exp > now:
        return True
    if sub == "unlimited_year" and (not exp or exp > now):
        return True
    return False


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


# ---------------------- FAVICON (EMBLEM) ----------------------
def generate_favicon_ico() -> bytes:
    """
    Generate a small icon emblem for PatternCraft.app.
    Simple pastel grid with a central 'P' – created on the fly
    so it’s original to this app.
    """
    size = 64
    img = Image.new("RGBA", (size, size), (248, 244, 240, 255))
    draw = ImageDraw.Draw(img)
    cell = 8
    colors = [
        (244, 114, 182, 255),  # pink
        (96, 165, 250, 255),   # blue
        (252, 211, 77, 255),   # yellow
        (52, 211, 153, 255),   # teal
    ]
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if (x // cell + y // cell) % 2 == 0:
                c = colors[(x // cell + 2 * (y // cell)) % len(colors)]
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=c)

    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    draw.text(
        (size // 2, size // 2),
        "P",
        font=font,
        fill=(15, 23, 42, 255),
        anchor="mm",
    )

    buf = io.BytesIO()
    # Save as ICO with a 32×32 size inside
    img.save(buf, format="ICO", sizes=[(32, 32)])
    return buf.getvalue()


@app.get("/favicon.ico")
def favicon():
    ico_bytes = generate_favicon_ico()
    return send_file(io.BytesIO(ico_bytes), mimetype="image/x-icon")


# ---------------------- BASIC ROUTES / ERRORS ----------------------
@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


@app.errorhandler(Exception)
def on_error(_e):
    # For production you might want better logging here.
    return make_response(jsonify({"error": "server_error"}), 500)


@app.get("/")
def index() -> str:
    user = get_current_user()
    plan_label = compute_plan_label(user)
    return render_template_string(HOMEPAGE_HTML, user=user, plan_label=plan_label)


@app.get("/pricing")
def pricing() -> str:
    user = get_current_user()
    message = ""
    reason = request.args.get("reason", "")
    if reason == "used_free":
        message = (
            "You’ve already used your included PatternCraft.app pattern on this account. "
            "Choose a plan to keep generating patterns."
        )
    elif reason == "no_credits":
        message = (
            "You’ve used all available credits on this account. "
            "Pick a pack or unlimited plan to continue."
        )
    elif reason == "checkout_error":
        message = "We couldn’t start checkout. Please try again or contact support."
    return render_template_string(PRICING_HTML, message=message)


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
            url_for(
                "login",
                msg="Log in or create an account before purchasing a plan.",
            )
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
    Stripe redirects here after a successful Checkout.
    We look up the Checkout Session, figure out which price was purchased,
    and update the user's account (credits or subscription).
    """
    session_id = request.args.get("session_id")
    user = get_current_user()
    status = "ok"

    if not session_id or not stripe.api_key:
        return render_template_string(SUCCESS_HTML, user=user, status="unknown")

    try:
        checkout_session = stripe.checkout.Session.retrieve(
            session_id, expand=["line_items", "customer_details"]
        )
    except Exception:
        return render_template_string(SUCCESS_HTML, user=user, status="error")

    client_email = checkout_session.get("client_reference_id") or (
        checkout_session.get("customer_details") or {}
    ).get("email")

    line_items = checkout_session.get("line_items")
    price_id: Optional[str] = None
    if line_items and getattr(line_items, "data", None):
        item = line_items.data[0]
        price = getattr(item, "price", None)
        if price is not None:
            price_id = price.id

    if not client_email or not price_id:
        return render_template_string(SUCCESS_HTML, user=user, status="partial")

    users = load_users()
    u = users.get(client_email)
    if not u:
        # No stored account for this email; show generic success.
        return render_template_string(SUCCESS_HTML, user=user, status="no_user")

    now_ts = time.time()
    changed = False

    # Credit-based plans
    if price_id == STRIPE_PRICE_SINGLE:
        u["credits"] = int(u.get("credits", 0) or 0) + 1
        u.setdefault("subscription", "free")
        changed = True
    elif price_id == STRIPE_PRICE_PACK10:
        u["credits"] = int(u.get("credits", 0) or 0) + 10
        u.setdefault("subscription", "free")
        changed = True
    # Unlimited subscriptions – extend from current expiry if already active
    elif price_id == STRIPE_PRICE_3MO:
        base = max(now_ts, float(u.get("sub_expires_at") or 0.0))
        period = 90 * 24 * 3600  # ~3 months
        u["subscription"] = "unlimited_3m"
        u["sub_expires_at"] = base + period
        changed = True
    elif price_id == STRIPE_PRICE_ANNUAL:
        base = max(now_ts, float(u.get("sub_expires_at") or 0.0))
        period = 365 * 24 * 3600
        u["subscription"] = "unlimited_year"
        u["sub_expires_at"] = base + period
        changed = True

    if changed:
        users[client_email] = u
        save_users(users)
        # If currently logged-in user matches, refresh it
        if user and user.get("email") == client_email:
            user = u

    return render_template_string(SUCCESS_HTML, user=user, status="ok")


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
        "subscription": "free",  # free | unlimited_3m | unlimited_year
        "sub_expires_at": 0.0,
        "free_used": False,
        "credits": 0,  # used for single + 10-pack plans
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
                    msg=(
                        "We couldn’t match that email and password after several attempts. "
                        "Create a PatternCraft.app account to get started."
                    ),
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
    w, h = 40, 40
    base = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(base)
    colors = [
        (239, 68, 68),  # red
        (249, 115, 22),  # orange
        (234, 179, 8),  # gold
        (34, 197, 94),  # green
        (59, 130, 246),  # blue
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
            "notes": (
                "Sample quilt-style color grid generated by PatternCraft.app with "
                "grid.png and legend.csv."
            ),
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
        return redirect(url_for("signup", msg="Create your PatternCraft.app account to continue."))

    subscription = user.get("subscription", "free")
    credits = int(user.get("credits", 0) or 0)
    mark_free_used = False
    consume_credit = False

    # Membership logic:
    # - unlimited_3m / unlimited_year: unlimited usage while active
    # - if credits > 0: consume 1 credit per convert
    # - else: free tier, 1 pattern per account
    if has_active_unlimited(user):
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
  <link rel="icon" href="/favicon.ico">
  <style>
    :root{
      --bg:#F7F4EF;
      --fg:#0f172a;
      --muted:#6b7280;
      --line:#e5e7eb;
      --accent:#4C7CF3;
      --accent-strong:#1d4ed8;
      --accent-soft:#e3ebff;
      --pill:#f97316;
      --card:#ffffff;
      --shadow:0 18px 45px rgba(15,23,42,.16);
      --radius:18px;
      --wrap:1120px;
    }
    *{box-sizing:border-box;}
    html,body{margin:0;padding:0;}
    body{
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#fde68a 0,#f1f5f9 35%,transparent 55%),
        radial-gradient(circle at top right,#bfdbfe 0,#f9fafb 40%,transparent 60%),
        linear-gradient(to bottom,#f3f4f6,#fefce8);
    }
    a{color:var(--accent-strong);text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:var(--wrap);margin:0 auto;padding:20px 16px 40px;}

    header{
      position:sticky;
      top:0;
      z-index:20;
      backdrop-filter:blur(16px);
      background:linear-gradient(to bottom,rgba(255,255,255,.97),rgba(255,255,255,.9));
      border-bottom:1px solid rgba(226,232,240,.85);
    }
    .topbar{
      max-width:var(--wrap);
      margin:0 auto;
      padding:10px 16px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      font-size:13px;
    }
    .brand{
      font-weight:800;
      font-size:19px;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:#0f172a;
    }
    .top-links{
      display:flex;
      align-items:center;
      gap:10px;
      flex-wrap:wrap;
      color:#4b5563;
    }
    .top-links a{text-decoration:none;}
    .pill-nav{
      padding:6px 12px;
      border-radius:999px;
      border:1px solid rgba(148,163,184,.7);
      background:#fff;
      font-weight:600;
    }
    .pill-nav.primary{
      border-color:var(--accent);
      background:var(--accent);
      color:#fff;
      box-shadow:0 8px 20px rgba(37,99,235,.35);
    }
    .pill-nav.primary:hover{
      box-shadow:0 11px 26px rgba(37,99,235,.5);
      text-decoration:none;
    }

    main{margin-top:16px;}

    .hero{
      display:grid;
      grid-template-columns:minmax(0,1.4fr) minmax(280px,1.1fr);
      gap:20px;
      align-items:flex-start;
    }
    .hero-primary{
      background:var(--card);
      border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.4);
      box-shadow:var(--shadow);
      padding:20px 20px 18px;
    }
    .chip{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:3px 9px;
      border-radius:999px;
      background:#ecfdf5;
      border:1px solid #bbf7d0;
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.09em;
      color:#166534;
      margin-bottom:8px;
    }
    .chip-dot{width:7px;height:7px;border-radius:999px;background:#22c55e;}
    h1{
      margin:0 0 6px;
      font-size:2.2rem;
      letter-spacing:-.02em;
    }
    .hero-tagline{
      margin:0;
      font-size:14px;
      color:var(--muted);
      max-width:520px;
    }
    .hero-cta-row{
      margin-top:14px;
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      align-items:center;
    }
    .pill{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:10px 18px;
      border-radius:999px;
      border:none;
      background:linear-gradient(135deg,var(--pill),#ea580c);
      color:#fff;
      font-size:14px;
      font-weight:600;
      letter-spacing:.02em;
      cursor:pointer;
      box-shadow:0 10px 26px rgba(248,113,22,.4);
      transition:transform .08s,box-shadow .08s,background .08s;
      text-decoration:none;
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 14px 32px rgba(248,113,22,.55);
      text-decoration:none;
    }
    .pill.secondary{
      background:#fff;
      border:1px solid rgba(148,163,184,.7);
      color:#0f172a;
      box-shadow:none;
    }
    .pill.secondary:hover{
      background:#f9fafb;
      box-shadow:0 7px 20px rgba(148,163,184,.35);
    }
    .pill-ready{
      background:linear-gradient(135deg,#16a34a,#22c55e);
      box-shadow:0 10px 26px rgba(22,163,74,.45);
    }
    .pill-ready:hover{
      box-shadow:0 13px 32px rgba(22,163,74,.6);
    }
    .hero-note{
      margin-top:8px;
      font-size:12px;
      color:#4b5563;
    }
    .badge-row{
      margin-top:10px;
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }
    .badge{
      font-size:11px;
      padding:4px 9px;
      border-radius:999px;
      background:#e5edff;
      color:#1d4ed8;
      border:1px solid #bfdbfe;
    }

    .hero-side{
      display:flex;
      flex-direction:column;
      gap:10px;
    }
    .side-card{
      background:rgba(255,255,255,.96);
      border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.45);
      box-shadow:0 14px 32px rgba(148,163,184,.28);
      padding:14px 14px 12px;
    }
    .side-title{
      margin:0 0 4px;
      font-size:14px;
      font-weight:700;
      letter-spacing:.04em;
      text-transform:uppercase;
      color:#4b5563;
    }
    .side-body{
      margin:0;
      font-size:13px;
      color:var(--muted);
    }
    .side-list{
      margin:6px 0 0;
      padding-left:18px;
      font-size:13px;
      color:#374151;
    }
    .side-list li{margin-bottom:3px;}

    .card-main{
      margin-top:22px;
      background:var(--card);
      border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.4);
      box-shadow:var(--shadow);
      padding:18px 18px 16px;
    }
    .section-title{
      margin:0 0 4px;
      font-size:1.1rem;
    }
    .muted{color:var(--muted);font-size:13px;}

    .make-layout{
      margin-top:12px;
      display:grid;
      gap:16px;
      grid-template-columns:minmax(0,1.3fr);
    }
    .file{
      border:2px dashed var(--accent);
      border-radius:18px;
      padding:16px;
      display:flex;
      align-items:center;
      gap:12px;
      cursor:pointer;
      background:var(--accent-soft);
      transition:background .15s,border-color .15s,transform .1s,box-shadow .1s;
    }
    .file:hover{
      background:#dbe4ff;
      border-color:#365ed1;
      transform:translateY(-1px);
      box-shadow:0 7px 20px rgba(37,99,235,.35);
    }
    .file-ready{
      background:#dcfce7;
      border-color:#16a34a;
      box-shadow:0 7px 20px rgba(22,163,74,.4);
    }
    .file input{display:none;}
    .file-label-main{
      font-weight:800;
      font-size:14px;
      letter-spacing:.08em;
      text-transform:uppercase;
    }
    .file-label-sub{
      font-size:12px;
      color:var(--muted);
    }
    .free-note{
      margin-top:6px;
      font-size:12px;
      color:#065f46;
      background:#d1fae5;
      border-radius:999px;
      padding:6px 10px;
      display:inline-flex;
      align-items:center;
      gap:6px;
    }
    .free-dot{
      width:8px;height:8px;border-radius:999px;background:#10b981;
    }

    fieldset{
      border:1px solid var(--line);
      border-radius:12px;
      padding:10px 10px 8px;
      margin:10px 0;
    }
    legend{
      font-size:13px;
      padding:0 4px;
      color:#4b5563;
    }
    .row{
      display:flex;
      flex-wrap:wrap;
      gap:12px;
    }
    .row > label{
      flex:1 1 150px;
      font-size:13px;
    }
    .row input,
    .row select{
      width:100%;
      margin-top:3px;
      padding:6px 8px;
      border-radius:8px;
      border:1px solid #cbd5f5;
      font-size:13px;
    }
    .row input:focus,
    .row select:focus{
      outline:none;
      border-color:#4f46e5;
      box-shadow:0 0 0 1px rgba(79,70,229,.35);
    }
    label{font-size:13px;}
    .controls-note{
      font-size:11px;
      color:#94a3b8;
      margin-top:4px;
    }
    .hidden{display:none;}

    footer{
      margin-top:26px;
      padding-top:10px;
      border-top:1px solid rgba(209,213,219,.9);
      font-size:12px;
      color:#9ca3af;
      display:flex;
      justify-content:space-between;
      align-items:center;
      flex-wrap:wrap;
      gap:8px;
    }

    @media (max-width:860px){
      .hero{
        grid-template-columns:1fr;
      }
    }
  </style>
</head>
<body>
<header>
  <div class="topbar">
    <div class="brand">PatternCraft.app</div>
    <div class="top-links">
      <a class="pill-nav" href="/pricing">Pricing</a>
      {% if user %}
        <span>
          Signed in as {{ user.email }}
          {% if plan_label %} ({{ plan_label }}){% endif %}
        </span>
        <a class="pill-nav" href="/logout">Sign out</a>
      {% else %}
        <a class="pill-nav" href="/login">Log in</a>
        <a class="pill-nav primary" href="/signup">Create account</a>
      {% endif %}
    </div>
  </div>
</header>

<main class="wrap">
  <section class="hero">
    <div class="hero-primary">
      <div class="chip">
        <span class="chip-dot"></span>
        <span>For cross-stitch, knitting, and quilting</span>
      </div>
      <h1>Turn art into stitchable patterns</h1>
      <p class="hero-tagline">
        PatternCraft.app converts your artwork into cross-stitch grids, knitting charts,
        and embroidery-ready line paths with one upload. Download a full pattern ZIP you can print
        or take to your machine.
      </p>
      <div class="hero-cta-row">
        {% if user %}
          <button class="pill" type="button"
                  onclick="document.getElementById('make').scrollIntoView({behavior:'smooth'})">
            Open the tool
          </button>
        {% else %}
          <a class="pill" href="/login?msg=Log+in+to+open+the+PatternCraft+tool.">
            Open the tool
          </a>
        {% endif %}
        <a class="pill secondary" href="/pricing#plans">
          See plans
        </a>
      </div>
      <div class="hero-note">
        Step 1: Pick a plan · Step 2: Upload your image · Step 3: Download your pattern ZIP.
      </div>
      <div class="badge-row">
        <span class="badge">One pattern included with every account</span>
        <span class="badge">Designed for hobbyists & pattern sellers</span>
      </div>
    </div>

    <div class="hero-side">
      <div class="side-card">
        <h2 class="side-title">Why makers use PatternCraft.app</h2>
        <p class="side-body">
          A purpose-built pattern tool with stitchers in mind:
        </p>
        <ul class="side-list">
          <li>Clean grids with bold 10×10 guides and symbol overlays</li>
          <li>Color legends with hex and RGB values for accurate palettes</li>
          <li>Fabric size estimates based on stitch count and cloth count</li>
          <li>Knitting charts that respect row proportions</li>
          <li>Embroidery line outputs ready for your machine software</li>
        </ul>
      </div>
    </div>
  </section>

  <section id="make" class="card-main">
    <h2 class="section-title">Make a pattern</h2>
    <p class="muted">
      Create a PatternCraft.app account or log in to generate patterns. Every account
      includes one pattern on us. After that, plans on the pricing page keep you creating.
    </p>

    <div class="make-layout">
      <form method="POST" action="/api/convert" enctype="multipart/form-data">
        <label class="file">
          <input id="fileInput" type="file" name="file" accept="image/*"
                 required onchange="pickFile(this)">
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
          <p class="controls-note">
            Defaults work well for most art. Increase stitch width or colors for more detail.
          </p>
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
            Generates a simple run-stitch path from your image. For advanced digitizing,
            continue in your embroidery software.
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
            Download includes grid.png, legend.csv (with color codes), meta.json,
            and optional pattern.pdf or embroidery files.
          </span>
        </div>
      </form>
    </div>
  </section>

  <footer>
    <span>© PatternCraft.app</span>
    <span><a href="/pricing">View pricing</a></span>
  </footer>
</main>

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

# ---------------------- INLINE HTML: SIGNUP / LOGIN / PRICING / SUCCESS ----------------------
SIGNUP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <style>
    body{
      margin:0;
      background:#F7F4EF;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:#111827;
    }
    .wrap{
      max-width:520px;
      margin:0 auto;
      padding:32px 16px 40px;
    }
    .card{
      background:#fff;
      border-radius:18px;
      border:1px solid #e5e7eb;
      padding:22px;
      box-shadow:0 18px 45px rgba(15,23,42,.18);
    }
    h1{margin:0 0 10px;font-size:1.6rem;}
    .muted{font-size:13px;color:#6b7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input[type="email"],input[type="password"]{
      width:100%;
      margin-top:4px;
      padding:8px 10px;
      border-radius:10px;
      border:1px solid #cbd5e1;
      font-size:14px;
    }
    input:focus{
      outline:none;
      border-color:#4f46e5;
      box-shadow:0 0 0 1px rgba(79,70,229,.35);
    }
    .pill{
      margin-top:16px;
      padding:9px 18px;
      border-radius:999px;
      border:none;
      background:linear-gradient(135deg,#f97316,#ea580c);
      color:#fff;
      font-size:14px;
      font-weight:600;
      cursor:pointer;
      box-shadow:0 10px 26px rgba(248,113,22,.4);
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 14px 32px rgba(248,113,22,.55);
    }
    .msg{margin-top:10px;font-size:13px;color:#b91c1c;}
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4b5563;padding-left:18px;margin-top:10px;}
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
  <link rel="icon" href="/favicon.ico">
  <style>
    body{
      margin:0;
      background:#F7F4EF;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:#111827;
    }
    .wrap{
      max-width:520px;
      margin:0 auto;
      padding:32px 16px 40px;
    }
    .card{
      background:#fff;
      border-radius:18px;
      border:1px solid #e5e7eb;
      padding:22px;
      box-shadow:0 18px 45px rgba(15,23,42,.18);
    }
    h1{margin:0 0 10px;font-size:1.6rem;}
    .muted{font-size:13px;color:#6b7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input{
      width:100%;
      margin-top:4px;
      padding:8px 10px;
      border-radius:10px;
      border:1px solid #cbd5e1;
      font-size:14px;
    }
    input:focus{
      outline:none;
      border-color:#4f46e5;
      box-shadow:0 0 0 1px rgba(79,70,229,.35);
    }
    .pill{
      margin-top:16px;
      padding:9px 18px;
      border-radius:999px;
      border:none;
      background:linear-gradient(135deg,#4c51bf,#4338ca);
      color:#fff;
      font-size:14px;
      font-weight:600;
      cursor:pointer;
      box-shadow:0 10px 26px rgba(79,70,229,.4);
    }
    .pill:hover{
      transform:translateY(-1px);
      box-shadow:0 14px 32px rgba(79,70,229,.55);
    }
    .msg{margin-top:10px;font-size:13px;color:#b91c1c;}
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
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app • Pricing</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <style>
    :root{
      --bg:#F7F4EF;
      --fg:#0f172a;
      --muted:#6b7280;
      --line:#e5e7eb;
      --card:#ffffff;
      --accent:#4C7CF3;
      --accent-soft:#e3ebff;
      --accent-strong:#1d4ed8;
      --danger:#e11d48;
      --radius:18px;
      --shadow:0 18px 45px rgba(15,23,42,.16);
      --wrap:1120px;
    }
    *{box-sizing:border-box;}
    html,body{margin:0;padding:0;}
    body{
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#fde68a 0,#f1f5f9 35%,transparent 55%),
        radial-gradient(circle at top right,#bfdbfe 0,#f9fafb 40%,transparent 60%),
        linear-gradient(to bottom,#f3f4f6,#fefce8);
    }
    a{color:var(--accent-strong);text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:var(--wrap);margin:0 auto;padding:16px 16px 32px;}

    header{
      position:sticky;
      top:0;
      z-index:10;
      backdrop-filter:blur(16px);
      background:linear-gradient(to bottom,rgba(255,255,255,.95),rgba(255,255,255,.88));
      border-bottom:1px solid rgba(226,232,240,.9);
    }
    .topbar{
      max-width:var(--wrap);
      margin:0 auto;
      padding:8px 16px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
    }
    .brand{
      font-weight:800;
      font-size:19px;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:#0f172a;
    }
    .nav{
      display:flex;
      align-items:center;
      gap:10px;
      flex-wrap:wrap;
      font-size:13px;
    }
    .nav-pill{
      padding:7px 14px;
      border-radius:999px;
      border:1px solid transparent;
      font-weight:600;
      font-size:13px;
      color:#0f172a;
      background:transparent;
      text-decoration:none;
    }
    .nav-pill.primary{
      border-color:var(--accent);
      background:var(--accent);
      color:#fff;
      box-shadow:0 8px 20px rgba(37,99,235,.35);
    }
    .nav-pill.primary:hover{
      box-shadow:0 11px 26px rgba(37,99,235,.5);
      text-decoration:none;
    }
    .nav-pill.ghost{
      border-color:rgba(148,163,184,.7);
      background:#ffffff;
    }
    .nav-pill.ghost:hover{
      background:#f9fafb;
      text-decoration:none;
    }

    main{margin-top:10px;}

    .hero{
      display:grid;
      grid-template-columns:minmax(0,1.4fr) minmax(280px,1.1fr);
      gap:18px;
      align-items:flex-start;
    }
    .hero-text h1{
      font-size:2.1rem;
      margin:0 0 4px;
    }
    .hero-text p{
      margin:0;
      font-size:14px;
      color:var(--muted);
      max-width:480px;
    }
    .badge{
      display:inline-flex;
      align-items:center;
      gap:8px;
      margin-bottom:8px;
      padding:3px 9px;
      border-radius:999px;
      background:#ecfdf5;
      border:1px solid #bbf7d0;
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.09em;
      color:#166534;
    }
    .badge-dot{
      width:7px;height:7px;border-radius:999px;background:#22c55e;
    }
    .hero-secondary{
      margin-top:8px;
      font-size:13px;
      color:#4b5563;
    }

    .notice{
      margin-top:10px;
      padding:8px 10px;
      border-radius:10px;
      border:1px solid #fed7aa;
      background:rgba(255,247,237,.95);
      color:#9a3412;
      font-size:13px;
    }

    .plans{
      margin-top:22px;
    }
    .plans-header{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      flex-wrap:wrap;
      margin-bottom:6px;
    }
    .plans-header h2{
      margin:0;
      font-size:1.1rem;
    }
    .plans-header p{
      margin:0;
      font-size:13px;
      color:var(--muted);
    }

    .plans-grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
      gap:14px;
      margin-top:10px;
    }
    .plan-card{
      background:var(--card);
      border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.4);
      box-shadow:0 12px 30px rgba(148,163,184,.24);
      padding:14px 14px 14px;
      display:flex;
      flex-direction:column;
      gap:6px;
      position:relative;
      overflow:hidden;
    }
    .plan-card.highlight{
      border-color:var(--accent);
      box-shadow:0 16px 38px rgba(37,99,235,.32);
    }
    .plan-label{
      font-size:12px;
      font-weight:700;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:#4b5563;
    }
    .plan-pill{
      position:absolute;
      top:8px;right:8px;
      font-size:11px;
      padding:3px 9px;
      border-radius:999px;
      background:#eff6ff;
      color:#1d4ed8;
      border:1px solid #bfdbfe;
      font-weight:600;
    }
    .plan-title{
      font-size:1.15rem;
      margin:2px 0 0;
    }
    .price-row{
      display:flex;
      align-items:baseline;
      gap:6px;
      margin-top:2px;
      margin-bottom:2px;
    }
    .price{
      font-size:26px;
      font-weight:800;
    }
    .price-period{
      font-size:13px;
      color:var(--muted);
    }
    .plan-sub{
      font-size:13px;
      color:var(--muted);
    }
    .plan-list{
      margin:6px 0 8px;
      padding-left:18px;
      font-size:13px;
      color:#374151;
    }
    .plan-list li{margin-bottom:2px;}
    .plan-cta{
      margin-top:auto;
      display:flex;
      flex-direction:column;
      gap:4px;
    }
    .btn{
      display:inline-flex;
      justify-content:center;
      align-items:center;
      width:100%;
      padding:8px 12px;
      border-radius:999px;
      border:1px solid var(--accent);
      background:var(--accent);
      color:#fff;
      font-weight:600;
      font-size:14px;
      cursor:pointer;
      text-decoration:none;
      box-shadow:0 9px 22px rgba(37,99,235,.42);
      transition:transform .08s,box-shadow .08s,background-color .08s;
    }
    .btn:hover{
      transform:translateY(-1px);
      box-shadow:0 13px 28px rgba(37,99,235,.55);
    }
    .btn.ghost{
      background:#fff;
      color:var(--accent-strong);
      box-shadow:none;
    }
    .btn.ghost:hover{
      background:#eff6ff;
      box-shadow:0 7px 18px rgba(148,163,184,.32);
    }
    .plan-footnote{
      font-size:12px;
      color:var(--muted);
    }

    .how-block{
      margin-top:8px;
      padding:10px 10px;
      border-radius:var(--radius);
      background:rgba(255,255,255,.9);
      border:1px solid rgba(209,213,219,.9);
      box-shadow:0 8px 22px rgba(148,163,184,.22);
    }
    .how-title{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:8px;
      margin-bottom:6px;
    }
    .how-title h2{
      margin:0;
      font-size:1.0rem;
    }
    .how-note{
      font-size:12px;
      color:var(--muted);
      max-width:260px;
      text-align:right;
    }
    .steps{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
      gap:8px;
      margin-top:4px;
    }
    .step-card{
      border-radius:10px;
      border:1px solid #e5e7eb;
      background:#f9fafb;
      padding:8px 8px 10px;
      font-size:13px;
    }
    .step-num{
      width:20px;height:20px;
      border-radius:999px;
      background:var(--accent);
      color:#fff;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      font-size:11px;
      font-weight:700;
      margin-bottom:3px;
    }
    .step-title{
      font-weight:600;
      margin:0 0 1px;
    }
    .step-text{
      margin:0;
      color:var(--muted);
      font-size:13px;
    }

    footer{
      margin-top:24px;
      padding-top:12px;
      border-top:1px solid rgba(209,213,219,.9);
      font-size:12px;
      color:#9ca3af;
    }
    footer .row{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:10px;
      flex-wrap:wrap;
    }
    @media (max-width:800px){
      .hero{
        grid-template-columns:1fr;
      }
      .how-note{
        max-width:none;
        text-align:left;
      }
    }
  </style>
</head>
<body>
<header>
  <div class="topbar">
    <div class="brand">PatternCraft.app</div>
    <nav class="nav">
      <a class="nav-pill ghost" href="/">Tool</a>
      <a class="nav-pill primary" href="/pricing">Pricing</a>
    </nav>
  </div>
</header>

<main class="wrap">
  <section class="hero" id="plans">
    <div class="hero-text">
      <div class="badge">
        <span class="badge-dot"></span>
        <span>Simple, transparent pricing</span>
      </div>
      <h1>Choose the plan that fits your stitching</h1>
      <p>
        Start with a single pattern, save with a pack, or go unlimited with a subscription.
        Every PatternCraft.app account comes with one free pattern to try the tool.
      </p>
      <p class="hero-secondary">
        You can upgrade at any time. Plans are tied to your PatternCraft.app login, not your device,
        so you can generate patterns from anywhere.
      </p>

      {% if message %}
      <div class="notice">
        {{ message }}
      </div>
      {% endif %}
    </div>

    <div>
      <div class="how-block">
        <div class="how-title">
          <h2>How billing works</h2>
          <div class="how-note">
            One‑time plans never renew automatically. Subscription plans renew until you cancel in your Stripe receipt or account.
          </div>
        </div>
        <div class="steps">
          <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">Pick your plan</div>
            <p class="step-text">
              Choose a single pattern, a 10‑pack of credits, or an unlimited plan.
            </p>
          </div>
          <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">Pay with Stripe</div>
            <p class="step-text">
              You’ll be taken to a secure Stripe Checkout page. Once payment clears, your PatternCraft.app account is updated.
            </p>
          </div>
          <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">Generate your patterns</div>
            <p class="step-text">
              Log in at PatternCraft.app, upload your art, choose stitch type, and download clean, high‑resolution pattern ZIPs.
            </p>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="plans">
    <div class="plans-header">
      <div>
        <h2>Plans</h2>
        <p>All plans include full‑resolution grids, legends with color codes, and export to PNG, CSV, and optional PDF.</p>
      </div>
    </div>

    <div class="plans-grid">
      <!-- Single Pattern -->
      <article class="plan-card">
        <div class="plan-label">Starter</div>
        <h3 class="plan-title">Single Pattern</h3>
        <div class="price-row">
          <div class="price">$25</div>
          <div class="price-period">one‑time</div>
        </div>
        <p class="plan-sub">Single pattern and legend. Use whenever.</p>
        <ul class="plan-list">
          <li>1 professional pattern conversion</li>
          <li>Detailed color legend with hex + RGB values</li>
          <li>High‑resolution grid output</li>
          <li>Use your pattern whenever you like</li>
        </ul>
        <div class="plan-cta">
          <form method="POST" action="/checkout">
            <input type="hidden" name="plan" value="single">
            <button class="btn" type="submit">Buy single</button>
          </form>
          <div class="plan-footnote">
            Best for one‑off projects or trying PatternCraft.app.
          </div>
        </div>
      </article>

      <!-- 10-Pattern Pack -->
      <article class="plan-card">
        <div class="plan-label">Credit pack</div>
        <h3 class="plan-title">10‑Pattern Pack</h3>
        <div class="price-row">
          <div class="price">$60</div>
          <div class="price-period">one‑time</div>
        </div>
        <p class="plan-sub">Great for consistent hobby use.</p>
        <ul class="plan-list">
          <li>10 pattern conversions tied to your account</li>
          <li>Credits never expire</li>
          <li>Includes all export formats</li>
          <li>Premium palette options and legends</li>
        </ul>
        <div class="plan-cta">
          <form method="POST" action="/checkout">
            <input type="hidden" name="plan" value="pack10">
            <button class="btn ghost" type="submit">Buy 10‑pack</button>
          </form>
          <div class="plan-footnote">
            Save more than 75% compared to buying 10 singles.
          </div>
        </div>
      </article>

      <!-- 3-Month Unlimited -->
      <article class="plan-card highlight">
        <div class="plan-label">Subscription</div>
        <span class="plan-pill">Great for busy seasons</span>
        <h3 class="plan-title">3‑Month Unlimited</h3>
        <div class="price-row">
          <div class="price">$75</div>
          <div class="price-period">every 3 months</div>
        </div>
        <p class="plan-sub">
          Unlimited pattern conversions, billed once per quarter.
        </p>
        <ul class="plan-list">
          <li>Unlimited pattern conversions for 3‑month periods</li>
          <li>Renews every 3 months until you cancel</li>
          <li>Higher‑resolution output for large projects</li>
          <li>Advanced color tools and palettes</li>
          <li>Priority processing in the queue</li>
        </ul>
        <div class="plan-cta">
          <form method="POST" action="/checkout">
            <input type="hidden" name="plan" value="unlimited_3m">
            <button class="btn" type="submit">Start 3‑Month Unlimited</button>
          </form>
          <div class="plan-footnote">
            Ideal for bursts of making, holiday seasons, or shop launches.
          </div>
        </div>
      </article>

      <!-- Annual Pro Unlimited -->
      <article class="plan-card">
        <div class="plan-label">Pro</div>
        <span class="plan-pill">Most popular</span>
        <h3 class="plan-title">Pro Annual Unlimited</h3>
        <div class="price-row">
          <div class="price">$99</div>
          <div class="price-period">per year</div>
        </div>
        <p class="plan-sub">
          Unlimited patterns all year, designed for serious makers and pattern sellers.
        </p>
        <ul class="plan-list">
          <li>Unlimited pattern conversions</li>
          <li>4× resolution for large or detailed designs</li>
          <li>Advanced color tools and palette exports</li>
          <li>Priority processing</li>
          <li>All export formats + templates</li>
          <li>Early access to new PatternCraft.app tools</li>
        </ul>
        <div class="plan-cta">
          <form method="POST" action="/checkout">
            <input type="hidden" name="plan" value="unlimited_year">
            <button class="btn ghost" type="submit">Go Pro Annual</button>
          </form>
          <div class="plan-footnote">
            Best value if you stitch or sell more than 4 patterns a year.
          </div>
        </div>
      </article>
    </div>
  </section>

  <footer>
    <div class="row">
      <span>© PatternCraft.app</span>
      <span><a href="/">Back to tool</a></span>
    </div>
  </footer>
</main>
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
  <link rel="icon" href="/favicon.ico">
  <style>
    body{
      margin:0;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
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
      border-radius:18px;
      border:1px solid #e5e7eb;
      padding:24px;
      box-shadow:0 18px 45px rgba(15,23,42,.16);
    }
    h1{margin:0 0 10px;font-size:1.7rem;}
    p{margin:6px 0;font-size:14px;color:#4b5563;}
    a{
      color:#2563eb;
      text-decoration:none;
      font-weight:600;
    }
    a:hover{text-decoration:underline;}
    .status-note{
      font-size:13px;
      margin-top:4px;
      color:#9ca3af;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Payment received</h1>
      <p>
        Thank you{% if user %}, {{ user.email }}{% endif %}.
        Your PatternCraft.app plan and credits have been updated.
      </p>
      {% if status != "ok" %}
        <p class="status-note">
          We had a small issue reading all the details from Stripe (status: {{ status }}),
          but your payment was processed. If something looks off in your plan or credits,
          contact support with your email and Stripe receipt.
        </p>
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

