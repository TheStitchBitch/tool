from __future__ import annotations

import io
import json
import math
import os
import zipfile
from typing import Dict, Tuple, List, Optional

import stripe
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

# Stripe configuration – secret key must be set in your Render env (STRIPE_SECRET_KEY)
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# Stripe price IDs (LIVE) – update these if you change prices in Stripe
STRIPE_PRICE_SINGLE = "price_1SXNyWCINTImVye2jayzoKKj"   # Single Pattern – $25 one-time
STRIPE_PRICE_PACK10 = "price_1SXNyRCINTImVye2m433u7pL"   # 10 Pattern Pack – $60 one-time
STRIPE_PRICE_3MO = "price_1SXTFUCINTImVye2JwOxUN55"      # 3‑Month Unlimited – $75 every 3 months (subscription)
STRIPE_PRICE_ANNUAL = "price_1SXNyNCINTImVye2rcxl5LsO"   # Pro Annual – $99 per year (subscription)

# Simple JSON “database” for users (email -> record)
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


def assign_symbols(colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], str]:
    """Deterministic symbol per palette color."""
    glyphs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+*#@&%=?/\\^~<>□■●▲◆★✚")
    return {c: glyphs[i % len(glyphs)] for i, c in enumerate(colors)}


def draw_symbols_on_grid(
    base: Image.Image, cell_px: int, sym_map: Dict[Tuple[int, int, int], str]
) -> Image.Image:
    """Overlay symbol per stitch, then grid (paid exports; no watermark)."""
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


# ---------------------- BASIC ROUTES & ERROR HANDLERS ----------------------
@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "file_too_large", "limit_mb": 25}), 413


@app.errorhandler(Exception)
def on_error(e):
    print("SERVER ERROR:", repr(e))
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
    except Exception as e:
        print("Stripe checkout error:", repr(e))
        return redirect(url_for("pricing", reason="checkout_error"))

    return redirect(checkout_session.url)


@app.get("/success")
def success():
    user = get_current_user()
    session_id = request.args.get("session_id")

    if session_id and stripe.api_key:
        try:
            chk = stripe.checkout.Session.retrieve(
                session_id, expand=["line_items.data.price"]
            )
            email = chk.get("client_reference_id") or (
                chk.get("customer_details") or {}
            ).get("email") or chk.get("customer_email")
            if email:
                users = load_users()
                u = users.get(email) or {
                    "email": email,
                    "password_hash": "",
                    "subscription": "free",
                    "free_used": False,
                    "credits": 0,
                }

                for item in chk["line_items"]["data"]:
                    price = item["price"]
                    price_id = price["id"]
                    if price_id == STRIPE_PRICE_SINGLE:
                        u["credits"] = int(u.get("credits", 0) or 0) + 1
                    elif price_id == STRIPE_PRICE_PACK10:
                        u["credits"] = int(u.get("credits", 0) or 0) + 10
                    elif price_id == STRIPE_PRICE_3MO:
                        u["subscription"] = "unlimited_3m"
                        u["subscription_id"] = chk.get("subscription") or u.get(
                            "subscription_id"
                        )
                    elif price_id == STRIPE_PRICE_ANNUAL:
                        u["subscription"] = "unlimited_year"
                        u["subscription_id"] = chk.get("subscription") or u.get(
                            "subscription_id"
                        )

                users[email] = u
                save_users(users)
        except Exception as e:
            print("Stripe success handling error:", repr(e))

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
    if email in users and users[email].get("password_hash"):
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
        "credits": 0,
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


# ---------------------- SAMPLE PATTERN ZIP ----------------------
@app.get("/sample-pattern.zip")
def sample_pattern_zip():
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
                draw.rectangle(
                    (px, py, px + patch - 1, py + patch - 1), fill=base_color
                )
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
                    draw.line(
                        (px, py + y, px + patch - 1, py + y),
                        fill=color,
                    )

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

    tips = [
        "Try printing the grid at 100% scale so each square stays easy to read.",
        "If the sample feels too colorful, reduce the number of colors in your own uploads.",
    ]

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
            "tips": tips,
        }
        z.writestr("meta.json", json.dumps(meta, indent=2))

        buf_png = io.BytesIO()
        grid_img.save(buf_png, format="PNG")
        z.writestr("grid.png", buf_png.getvalue())
        z.writestr("tips.txt", "\n".join(tips))

    out_zip.seek(0)
    return send_file(
        out_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name="patterncraft_sample_quilt.zip",
    )


# ---------------------- MEMBERSHIP CHECK ----------------------
def _subscription_is_active(user: dict) -> bool:
    if user.get("subscription") not in ("unlimited_3m", "unlimited_year"):
        return False
    sub_id = user.get("subscription_id")
    if not sub_id or not stripe.api_key:
        return True
    try:
        sub = stripe.Subscription.retrieve(sub_id)
        return sub["status"] in ("active", "trialing", "past_due")
    except Exception as e:
        print("Stripe subscription check failed:", repr(e))
        return True


# ---------------------- PATTERN GENERATOR ----------------------
@app.post("/api/convert")
def convert():
    email = session.get("user_email")
    if not email:
        return redirect(url_for("login", msg="Log in to generate patterns."))

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

    subscription = user.get("subscription", "free")
    credits = int(user.get("credits", 0) or 0)

    mark_free_used = False
    consume_credit = False

    if subscription in ("unlimited_3m", "unlimited_year") and _subscription_is_active(
        user
    ):
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

    tips: List[str] = []
    if ptype == "cross":
        tips.append("Cross-stitch: start from the center of the chart and work outward to keep your fabric balanced.")
    if ptype == "knit":
        tips.append("Knitting: treat each square as one stitch; check your gauge so the finished size matches the meta.json estimate.")
    if ptype == "emb":
        tips.append("Embroidery: this is a simple run-stitch path. Use it as a starting point and refine in your digitizing software.")

    if max_colors > 30:
        tips.append("You selected a large palette. If stitching feels too busy, try re-running with fewer colors (8–20).")
    elif max_colors <= 8:
        tips.append("With a small palette, consider bold color contrast so your design still reads clearly from a distance.")

    if cloth_count >= 18:
        tips.append("High-count fabric: use good lighting and consider magnification for eye comfort.")
    elif cloth_count <= 12:
        tips.append("Lower-count fabric is forgiving and great for beginners or large wall pieces.")

    tips.append("Legend: use the hex code column if you want to match colors in digital tools or paint software.")

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
                "tips": tips,
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
            meta = {
                "type": "emb",
                "stitch_style": "run",
                "points": len(pts),
                "pyembroidery": HAS_PYEMB,
                "tips": tips,
            }
            z.writestr("meta.json", json.dumps(meta, indent=2))
        else:
            return jsonify({"error": "unknown_ptype"}), 400

        if tips:
            z.writestr("tips.txt", "\n".join(tips))

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
HOMEPAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root{
      --bg:#F7F4EF;--fg:#111827;--muted:#6b6b6b;
      --line:#e8e4de;--radius:16px;--shadow:0 18px 40px rgba(15,23,42,.22);
      --accent:#4C7CF3;--accent-soft:#e3ebff;--accent-strong:#173d99;
      --pill:#f97316;--pill-soft:#fff7ed;
    }
    *{box-sizing:border-box;margin:0;padding:0;}
    body{
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;
      color:var(--fg);
      background:
        radial-gradient(circle at top left,#fde68a 0,#f1f5f9 40%,transparent 60%),
        radial-gradient(circle at top right,#bfdbfe 0,#f9fafb 45%,transparent 65%),
        linear-gradient(to bottom,#f3f4f6,#fefce8);
    }
    a{color:#2563eb;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1120px;margin:0 auto;padding:20px 16px 40px}
    h1{font-size:2.5rem;margin-bottom:6px}
    h2{margin-bottom:8px;font-size:1.25rem}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:18px;
    }
    .brand{font-weight:800;font-size:20px;letter-spacing:.12em;text-transform:uppercase;}
    .top-links{font-size:13px;color:#4b5563}
    .top-links a{margin-left:8px;}
    .tag-pill{
      display:inline-flex;align-items:center;gap:6px;
      padding:4px 10px;border-radius:999px;
      background:rgba(255,255,255,.9);border:1px solid rgba(148,163,184,.5);
      font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.08em;
    }
    .tag-dot{width:8px;height:8px;border-radius:999px;background:#22c55e}
    .hero{
      display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1.1fr);
      gap:22px;margin-bottom:28px;align-items:stretch;
    }
    .hero-left{padding:18px 0;}
    .hero-tagline{color:var(--muted);max-width:460px;margin-top:6px;}
    .muted{color:var(--muted);font-size:13px}
    .hero-cta-row{
      display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;align-items:center;
    }
    .pill{
      padding:11px 20px;border-radius:999px;
      background:linear-gradient(135deg,var(--pill),#ea580c);
      color:#fff;border:none;cursor:pointer;
      font-size:14px;font-weight:600;letter-spacing:.02em;
      box-shadow:0 12px 26px rgba(248,113,22,.42);
      transition:transform .09s,box-shadow .09s,background-color .09s;
      display:inline-block;text-decoration:none;text-align:center;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 16px 32px rgba(248,113,22,.5);}
    .pill-secondary{
      background:#fff;color:var(--fg);
      border:1px solid rgba(148,163,184,.6);
      box-shadow:0 6px 16px rgba(15,23,42,.12);
    }
    .pill-secondary:hover{
      box-shadow:0 10px 22px rgba(15,23,42,.18);
    }
    .pill-ready{
      background:#16a34a;
      box-shadow:0 12px 26px rgba(22,163,74,.4);
    }
    .hero-note{font-size:12px;color:#475569;margin-top:10px;}
    .badge-row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
    .badge{
      font-size:11px;padding:4px 8px;border-radius:999px;
      background:#e5edff;color:#1d4ed8;border:1px solid rgba(129,140,248,.5);
    }
    .card{
      background:#fff;border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.2);
      box-shadow:var(--shadow);
      padding:18px;
    }

    /* Right-hand process card – no black boxes, pastel walkthrough */
    .hero-visual{
      background:linear-gradient(135deg,#eef2ff,#fef3c7);
      border-radius:var(--radius);
      padding:14px 14px 16px;
      display:flex;
      flex-direction:column;
      gap:10px;
    }
    .process-title{
      font-size:11px;
      font-weight:700;
      letter-spacing:.16em;
      text-transform:uppercase;
      color:#4b5563;
    }
    .process-sub{
      font-size:12px;
      color:#6b7280;
    }
    .process-steps{
      list-style:none;
      padding:0;
      margin:10px 0 4px;
      display:grid;
      gap:8px;
    }
    .process-step{
      display:grid;
      grid-template-columns:auto 1fr;
      gap:6px 10px;
      align-items:flex-start;
    }
    .process-step-num{
      width:20px;
      height:20px;
      border-radius:999px;
      background:#4C7CF3;
      color:#fff;
      font-size:11px;
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:700;
      margin-top:2px;
    }
    .process-step-main{
      font-weight:600;
      font-size:13px;
      color:#111827;
    }
    .process-step-sub{
      grid-column:2 / 3;
      font-size:11px;
      color:#4b5563;
    }
    .process-mini{
      grid-column:2 / 3;
      margin-top:4px;
      border-radius:10px;
      padding:6px 8px;
      background:#ffffff;
      border:1px dashed rgba(148,163,184,.7);
      font-size:10px;
      display:flex;
      flex-wrap:wrap;
      gap:6px;
    }
    .mini-tag{
      padding:2px 6px;
      border-radius:999px;
      background:#eff6ff;
      color:#1d4ed8;
    }
    .mini-grid{
      width:56px;
      height:40px;
      border-radius:8px;
      background-image:
        linear-gradient(to right, rgba(148,163,184,.6) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(148,163,184,.6) 1px, transparent 1px);
      background-size:6px 6px;
      position:relative;
      overflow:hidden;
    }
    .mini-grid::after{
      content:"";
      position:absolute;
      inset:6px;
      border-radius:6px;
      border:1px solid rgba(248,113,22,.8);
    }
    .process-footer{
      margin-top:4px;
      font-size:10px;
      color:#6b7280;
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
      box-shadow:0 10px 28px rgba(37,99,235,.38);
    }
    .file-ready{
      background:#dcfce7;
      border-color:#16a34a;
      box-shadow:0 10px 26px rgba(22,163,74,.45);
    }
    .file input{display:none}
    .file-label-main{font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.08em}
    .file-label-sub{font-size:12px;color:var(--muted)}
    .free-note{
      margin-top:6px;font-size:12px;color:#065f46;background:#d1fae5;
      border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px;
    }
    .free-dot{width:8px;height:8px;border-radius:999px;background:#10b981}
    fieldset{border:1px solid var(--line);border-radius:12px;padding:10px 10px 12px;margin:10px 0}
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
    .how-card{
      margin-top:22px;
      display:grid;
      grid-template-columns:minmax(0,1.1fr) minmax(0,1.1fr);
      gap:18px;
    }
    .how-box{
      background:#fff7ed;
      border-radius:14px;
      border:1px solid #fed7aa;
      padding:14px 16px;
      font-size:13px;
    }
    .how-title{font-weight:700;margin-bottom:6px;font-size:14px;}
    .how-list{margin-left:18px;margin-top:4px;}
    .how-list li{margin-bottom:4px;}
    @media (max-width:880px){
      .hero{grid-template-columns:1fr}
      .hero-visual{order:-1}
      .make-layout{grid-template-columns:1fr}
      .how-card{grid-template-columns:1fr}
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
        · Signed in as {{ user.email }} ({{ user.subscription }} plan{% if user.credits %}, {{ user.credits }} credits{% endif %})
        · <a href="/logout">Sign out</a>
      {% else %}
        · <a href="/login">Log in</a> · <a href="/signup">Create account</a>
      {% endif %}
    </div>
  </div>

  <div class="hero">
    <div class="hero-left">
      <div class="tag-pill">
        <span class="tag-dot"></span>
        <span>Picture‑to‑pattern in minutes</span>
      </div>
      <h1>Turn art into stitchable patterns</h1>
      <p class="hero-tagline">
        PatternCraft.app converts your artwork into cross‑stitch grids, knitting charts,
        and embroidery‑ready files with one upload. Export a full ZIP you can print or take to your machine.
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
          See pricing & workflow
        </a>
      </div>
      <div class="hero-note">
        Step 1: <a href="/pricing#how">Choose a plan</a> ·
        Step 2: <a href="/signup">Create your account</a> ·
        Step 3: Upload art and download your pattern ZIP.
      </div>
      <div class="badge-row">
        <span class="badge">One pattern included with every account</span>
        <span class="badge">Color legend with hex codes & stitch counts</span>
      </div>
    </div>

    <div class="card hero-visual">
      <div class="process-title">From photo to pattern</div>
      <div class="process-sub">
        A quick visual of what happens when you use PatternCraft.app.
      </div>
      <ol class="process-steps">
        <li class="process-step">
          <div class="process-step-num">1</div>
          <div class="process-step-main">Select a plan</div>
          <div class="process-step-sub">Single pattern, 10‑pack, 3‑month unlimited, or Pro annual.</div>
          <div class="process-mini">
            <span class="mini-tag">Single</span>
            <span class="mini-tag">10‑Pack</span>
            <span class="mini-tag">3‑Month Unlimited</span>
            <span class="mini-tag">Pro Annual</span>
          </div>
        </li>
        <li class="process-step">
          <div class="process-step-num">2</div>
          <div class="process-step-main">Upload your picture</div>
          <div class="process-step-sub">
            A quilt mockup, illustration, logo, or photo. PatternCraft reads the colors and shapes.
          </div>
          <div class="process-mini">
            <span class="mini-tag">JPG</span>
            <span class="mini-tag">PNG</span>
            <span class="mini-tag">Art & photos</span>
          </div>
        </li>
        <li class="process-step">
          <div class="process-step-num">3</div>
          <div class="process-step-main">Choose stitch options</div>
          <div class="process-step-sub">
            Cross‑stitch, knitting, or embroidery. Set stitch width, cloth count, max colors, and stitch style label.
          </div>
          <div class="process-mini">
            <span class="mini-tag">Cross‑stitch</span>
            <span class="mini-tag">Knitting</span>
            <span class="mini-tag">Embroidery</span>
          </div>
        </li>
        <li class="process-step">
          <div class="process-step-num">4</div>
          <div class="process-step-main">Stripe checkout</div>
          <div class="process-step-sub">
            Secure, Stripe‑hosted payment. Your credits or subscription attach to your PatternCraft.app account.
          </div>
        </li>
        <li class="process-step">
          <div class="process-step-num">5</div>
          <div class="process-step-main">Download your pattern ZIP</div>
          <div class="process-step-sub">
            Includes grid.png, legend.csv (with hex codes), meta.json, and tips.txt — plus optional PDF or embroidery files.
          </div>
          <div class="process-mini">
            <div class="mini-grid"></div>
            <span>+ legend.csv, meta.json, tips.txt</span>
          </div>
        </li>
      </ol>
      <div class="process-footer">
        Paid exports are clean and full resolution — no watermarks, ready for printing or importing into your workflow.
      </div>
    </div>
  </div>

  <div class="how-card">
    <div class="how-box">
      <div class="how-title">Which product should I choose?</div>
      <ul class="how-list">
        <li><strong>Single Pattern</strong> — one pattern ZIP added to your account. Great for trying the tool.</li>
        <li><strong>10‑Pattern Pack</strong> — adds 10 credits. Each time you generate a pattern, 1 credit is used.</li>
        <li><strong>3‑Month Unlimited</strong> — subscription billed every 3 months while active. Unlimited patterns.</li>
        <li><strong>Pro Annual</strong> — yearly subscription for unlimited patterns and the best long‑term value.</li>
      </ul>
    </div>
    <div class="how-box">
      <div class="how-title">How do I use the tool?</div>
      <ul class="how-list">
        <li>Create an account or log in, then upload a photo or artwork.</li>
        <li>Pick <em>Cross‑stitch</em>, <em>Knitting</em>, or <em>Embroidery</em> and choose a stitch‑style label from the dropdown.</li>
        <li>Set stitch width, cloth count, and maximum colors. The legend.csv file includes hex color codes and stitch counts.</li>
        <li>Click <strong>Generate pattern ZIP</strong>. Your download contains the grid image, color legend, meta.json and tips.txt.</li>
      </ul>
    </div>
  </div>

  <div id="make" class="card" style="margin-top:24px;">
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
            <p class="controls-note">
              The legend.csv file includes hex color codes, RGB values, stitch counts, and estimated skeins per color.
            </p>
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
              Your download includes grid.png, legend.csv, meta.json, tips.txt, and optional pattern.pdf or embroidery files.
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
        ['half','Half stitches (mark these in your legend)'],
        ['back','Backstitch overlay details']
      ];
    } else if(type === 'knit'){
      opts = [
        ['stockinette','Stockinette'],
        ['garter','Garter'],
        ['seed','Seed'],
        ['rib1','Rib 1×1']
      ];
    } else {
      opts = [['run','Run stitch path']];
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
SIGNUP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#F7F4EF;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{background:#fff;border-radius:16px;border:1px solid #e5e7eb;padding:22px;box-shadow:0 16px 40px rgba(15,23,42,.22)}
    h1{margin:0 0 10px;font-size:1.7rem}
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
      box-shadow:0 10px 26px rgba(248,113,22,.42);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 32px rgba(248,113,22,.5);}
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

LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Log in — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{margin:0;background:#F7F4EF;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter;color:#111827}
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px}
    .card{background:#fff;border-radius:16px;border:1px solid #e5e7eb;padding:22px;box-shadow:0 16px 40px rgba(15,23,42,.22)}
    h1{margin:0 0 10px;font-size:1.7rem}
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
      box-shadow:0 10px 26px rgba(79,70,229,.42);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 32px rgba(79,70,229,.5);}
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
  font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
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
.guide{
  margin-top:22px;
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:16px;
}
.guide-card{
  background:#f9fafb;
  border-radius:var(--radius);
  border:1px solid #e5e7eb;
  padding:14px 16px;
}
.guide-title{font-weight:700;margin-bottom:6px;font-size:15px;}
.guide-list{margin-left:18px;margin-top:4px;font-size:14px;color:var(--muted);}
.guide-list li{margin-bottom:4px;}
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
        <li>Detailed legend with hex codes & stitch counts</li>
        <li>High‑resolution grid output</li>
        <li>Use your pattern whenever you like</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="single">
        <button class="btn" type="submit">Buy single</button>
      </form>
      <p class="small">Best for one‑off projects or trying PatternCraft.app.</p>
    </div>

    <!-- 10-Pattern Pack -->
    <div class="card">
      <h3>10‑Pattern Pack</h3>
      <div class="price">$60</div>
      <p class="small">Great for consistent hobby use.</p>
      <ul class="list">
        <li>10 pattern conversions (credits)</li>
        <li>Credits never expire</li>
        <li>Includes all export formats</li>
        <li>Premium palette options</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="pack10">
        <button class="btn ghost" type="submit">Buy 10‑pack</button>
      </form>
      <p class="small">Save big vs buying singles.</p>
    </div>

    <!-- 3-Month Unlimited -->
    <div class="card">
      <h3>3‑Month Unlimited</h3>
      <div class="price">$75</div>
      <p class="small">$75 billed every 3 months until you cancel.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>High‑quality grid output</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_3m">
        <button class="btn ghost" type="submit">Start 3‑month plan</button>
      </form>
      <p class="small">Perfect for focused projects or seasons.</p>
    </div>

    <!-- Annual Pro Unlimited -->
    <div class="card">
      <h3>Pro Annual Unlimited</h3>
      <div class="price">$99 / year</div>
      <p class="small">Unlimited patterns all year.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>High‑quality grid output</li>
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
  <p class="small">From photo to stitch‑ready pattern in three simple steps.</p>
  <div class="steps">
    <div class="step-card">
      <div class="step-num">1</div>
      <h3>Upload your image</h3>
      <p class="small">
        Start with a photo, artwork, or quilt design. PatternCraft analyzes it for stitchable detail.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3>Choose size, colors & stitch style</h3>
      <p class="small">
        Set stitch width, cloth count, and palette size. Choose cross‑stitch, knitting, or embroidery and pick a stitch‑style label.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3>Download your pattern ZIP</h3>
      <p class="small">
        Download a ZIP with grid.png, legend.csv (hex codes, RGB, stitch counts, skein estimates),
        meta.json, and tips.txt, plus optional PDF or embroidery files.
      </p>
    </div>
  </div>

  <div class="guide" style="margin-top:18px;">
    <div class="guide-card">
      <div class="guide-title">What each product unlocks</div>
      <ul class="guide-list">
        <li><strong>Single Pattern</strong> — adds 1 credit to your account. Generating a pattern uses 1 credit.</li>
        <li><strong>10‑Pattern Pack</strong> — adds 10 credits. Credits stay on your account until used, no expiration.</li>
        <li><strong>3‑Month Unlimited</strong> — while your subscription is active, you can generate as many patterns as you like.</li>
        <li><strong>Pro Annual Unlimited</strong> — same as unlimited, but billed once per year for heavy users and pattern sellers.</li>
      </ul>
    </div>
    <div class="guide-card">
      <div class="guide-title">What’s inside every pattern ZIP</div>
      <ul class="guide-list">
        <li><strong>grid.png</strong> — clean grid with bold 10×10 guides and optional symbol overlay.</li>
        <li><strong>legend.csv</strong> — hex codes, RGB values, stitches per color, percentage of total, and estimated skeins.</li>
        <li><strong>meta.json</strong> — stitch counts, estimated finished size, stitch‑style label, and notes.</li>
        <li><strong>tips.txt</strong> — a short list of tips tailored to your choices (fabric count, palette size, and pattern type).</li>
      </ul>
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

SUCCESS_HTML = """
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
      border-radius:16px;
      border:1px solid #e5e7eb;
      padding:24px;
      box-shadow:0 16px 40px rgba(15,23,42,.22);
    }
    h1{margin:0 0 10px;font-size:1.8rem}
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
      <p>Thank you{% if user %}, {{ user.email }}{% endif %}. Your PatternCraft.app plan has been updated.</p>
      <p>You can go back to the tool and start generating patterns right away. If anything looks off with your plan, contact support and include your Stripe receipt.</p>
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
