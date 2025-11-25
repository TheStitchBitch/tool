from __future__ import annotations
import io
import json
import math
import os
import zipfile
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta, timezone

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
app.permanent_session_lifetime = timedelta(days=365)

# ---------------------- STRIPE CONFIG ----------------------
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

# LIVE Stripe price IDs
STRIPE_PRICE_SINGLE = "price_1SXNyWCINTImVye2jayzoKKj"      # Single Pattern – $25
STRIPE_PRICE_PACK10 = "price_1SXNyRCINTImVye2m433u7pL"      # 10 Pattern Pack – $60
STRIPE_PRICE_3MO    = "price_1SXNyFCINTImVye2awgikQHW"      # 3 Month Unlimited – $75/month
STRIPE_PRICE_ANNUAL = "price_1SXNyNCINTImVye2rcxl5LsO"      # Pro Annual – $99/year

# Simple JSON “database” for users
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# Upload config
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
ALLOWED_MIME = {"image/png", "image/jpeg", "image/svg+xml", "application/dxf"}

CELL_PX = 12
MAX_DIM = 8000


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


def is_membership_active(user: Optional[dict]) -> bool:
    """
    True if user has an unlimited (3‑month or annual) plan that is still within its expiry window.
    """
    if not user:
        return False
    subscription = user.get("subscription")
    if subscription not in ("unlimited_3m", "unlimited_year"):
        return False
    expires_str = user.get("membership_expires")
    if not expires_str:
        return False
    try:
        expires = datetime.fromisoformat(expires_str)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) <= expires
    except Exception:
        return False


# ---------------------- IMAGE / PATTERN HELPERS ----------------------
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
def on_error(e):
    app.logger.exception("Unhandled error: %s", e)
    return make_response(jsonify({"error": "server_error"}), 500)


@app.get("/")
def index() -> str:
    user = get_current_user()
    pro_active = is_membership_active(user)
    return render_template_string(HOMEPAGE_HTML, user=user, pro_active=pro_active)


@app.get("/pricing")
def pricing() -> str:
    user = get_current_user()
    reason = request.args.get("reason", "")
    message = ""
    if reason == "used_free":
        message = "You’ve already used your PatternCraft.app included pattern for this account. Choose a plan to keep generating patterns."
    elif reason == "no_credits":
        message = "You’ve used all credits on this account. Pick a pack or unlimited plan to continue."
    elif reason == "checkout_error":
        message = "We couldn’t start checkout. Please try again or contact support."
    return render_template_string(PRICING_HTML, user=user, message=message)


# ---------------------- STRIPE CHECKOUT ----------------------
@app.post("/checkout")
def create_checkout():
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
    except Exception as e:
        app.logger.exception("Stripe checkout error: %s", e)
        return redirect(url_for("pricing", reason="checkout_error"))

    return redirect(checkout_session.url)


@app.get("/success")
def success():
    user = get_current_user()
    session_id = request.args.get("session_id")
    if user and session_id and stripe.api_key:
        try:
            checkout_session = stripe.checkout.Session.retrieve(
                session_id,
                expand=["line_items"],
            )
            line_items = checkout_session.get("line_items", {}).get("data", [])
            price_id = None
            if line_items:
                item0 = line_items[0]
                price = item0.get("price") or {}
                if isinstance(price, str):
                    price_id = price
                else:
                    price_id = price.get("id")
            if price_id:
                users = load_users()
                stored = users.get(user["email"])
                if stored:
                    now = datetime.now(timezone.utc)
                    credits = int(stored.get("credits", 0) or 0)
                    if price_id == STRIPE_PRICE_SINGLE:
                        stored["subscription"] = "single"
                        stored["credits"] = credits + 1
                    elif price_id == STRIPE_PRICE_PACK10:
                        stored["subscription"] = "pack10"
                        stored["credits"] = credits + 10
                    elif price_id == STRIPE_PRICE_3MO:
                        stored["subscription"] = "unlimited_3m"
                        expires = now + timedelta(days=90)
                        stored["membership_expires"] = expires.isoformat()
                    elif price_id == STRIPE_PRICE_ANNUAL:
                        stored["subscription"] = "unlimited_year"
                        expires = now + timedelta(days=365)
                        stored["membership_expires"] = expires.isoformat()
                    users[stored["email"]] = stored
                    save_users(users)
                    user = stored
        except Exception as e:
            app.logger.exception("Error finalizing success membership: %s", e)
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
        "subscription": "free",         # free, single, pack10, unlimited_3m, unlimited_year
        "free_used": False,             # 1 free pattern per account
        "credits": 0,                   # credits for Single / Pack10 plans
        "membership_expires": None,     # ISO timestamp for unlimited plans
    }
    save_users(users)
    session["user_email"] = email
    session["login_failures"] = 0
    session.permanent = True
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
    session.permanent = True
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

    pro_active = is_membership_active(user)  # 3‑Month Unlimited or Annual, not expired

    mark_free_used = False
    consume_credit = False

    # Membership logic:
    # - If pro_active (3‑Month or Annual): unlimited usage + pro resolution
    # - Else if credits > 0: consume 1 credit per convert
    # - Else free tier, 1 pattern per account
    if subscription in ("unlimited_3m", "unlimited_year") and pro_active:
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

            # Pro plans (3‑Month / Annual) get 4× cell resolution
            cell_px = CELL_PX * (4 if pro_active else 1)

            grid_img = draw_grid(quant, cell_px=cell_px)
            pdf_bytes: Optional[bytes] = None
            if want_symbols or want_pdf:
                pal = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
                sym_map = assign_symbols(pal)
                sym_img = draw_symbols_on_grid(quant, cell_px=cell_px, sym_map=sym_map)
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

            note_base = (
                "Knitting preview compresses row height; verify gauge."
                if ptype == "knit"
                else "Cross-stitch grid with 10×10 guides."
            )
            note = note_base
            if pro_active:
                note += " Pro tools: 4× cell resolution and advanced color palette."

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

    # Update usage
    if consume_credit and credits > 0:
        user["credits"] = max(0, credits - 1)
    if mark_free_used:
        user["free_used"] = True
    if consume_credit or mark_free_used:
        users[email] = user
        save_users(users)

    out_zip.seek(0)
    # Note: no watermark is applied to this ZIP; all tiers get clean pattern exports.
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
  <title>PatternCraft.app — Turn art into stitchable patterns</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root{
      --bg:#F3F4F6;--fg:#111827;--muted:#6B7280;
      --line:#E5E7EB;--radius:14px;--shadow:0 18px 40px rgba(15,23,42,.14);
      --accent:#4F46E5;--accent-soft:#EEF2FF;--accent-strong:#4338CA;
      --pill:#F97316;
    }
    *{box-sizing:border-box;margin:0;padding:0;}
    body{
      font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:var(--fg);
      background:radial-gradient(circle at top left,#EEF2FF 0,#F9FAFB 45%,transparent 70%),
                 radial-gradient(circle at top right,#FDE68A 0,#F9FAFB 40%,transparent 70%),
                 linear-gradient(to bottom,#F9FAFB,#E5E7EB);
      min-height:100vh;
    }
    a{color:#2563EB;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .wrap{max-width:1040px;margin:0 auto;padding:24px 16px 48px;}
    h1{font-size:2.5rem;margin-bottom:8px;letter-spacing:-.02em;}
    h2{font-size:1.2rem;margin-bottom:6px;}
    .topbar{
      display:flex;align-items:center;justify-content:space-between;
      margin-bottom:20px;
    }
    .brand{
      font-weight:800;font-size:20px;letter-spacing:.08em;
      text-transform:uppercase;
    }
    .top-links{font-size:13px;color:var(--muted);}
    .top-links a{margin-left:8px;}
    .top-links span.plan-tag{
      margin-left:6px;
      padding:3px 8px;
      border-radius:999px;
      background:#DBEAFE;
      color:#1D4ED8;
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.08em;
    }
    .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:4px 10px;border-radius:999px;
      background:rgba(255,255,255,.9);border:1px solid rgba(148,163,184,.45);
      font-size:11px;color:#4B5563;text-transform:uppercase;letter-spacing:.08em;
    }
    .chip-dot{width:8px;height:8px;border-radius:999px;background:#22C55E;}
    .card{
      background:#FFFFFF;border-radius:var(--radius);
      border:1px solid rgba(148,163,184,.25);
      box-shadow:var(--shadow);
      padding:20px;
    }
    .hero{
      display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2.1fr);
      gap:22px;margin-bottom:28px;align-items:center;
    }
    .hero-tagline{color:var(--muted);max-width:440px;font-size:14px;}
    .muted{color:var(--muted);font-size:13px;}
    .pill{
      padding:10px 18px;border-radius:999px;
      background:linear-gradient(135deg,var(--pill),#EA580C);
      color:#fff;border:none;cursor:pointer;
      font-size:14px;font-weight:600;letter-spacing:.02em;
      box-shadow:0 10px 22px rgba(248,113,22,.35);
      transition:transform .08s,box-shadow .08s,background-color .08s;
      display:inline-block;
      text-decoration:none;
      text-align:center;
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 14px 30px rgba(248,113,22,.45);}
    .pill-secondary{
      background:#FFFFFF;color:var(--fg);
      border:1px solid rgba(148,163,184,.7);
      box-shadow:none;
    }
    .pill-secondary:hover{
      box-shadow:0 6px 16px rgba(148,163,184,.45);
    }
    .pill-ready{
      background:#16A34A;
      box-shadow:0 10px 22px rgba(22,163,74,.35);
    }
    .pill-ready:hover{
      box-shadow:0 14px 30px rgba(22,163,74,.45);
    }
    .hero-cta-row{
      display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;align-items:center;
    }
    .hero-note{font-size:12px;color:#4B5563;margin-top:8px;}
    .badge-row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;}
    .badge{
      font-size:11px;padding:4px 8px;border-radius:999px;
      background:#E5EDFF;color:#1D4ED8;border:1px solid rgba(129,140,248,.65);
    }
    .badge-pro{
      background:#DCFCE7;color:#166534;border-color:#4ADE80;
    }
    .why-list{padding-left:18px;margin-top:4px;}
    .why-list li{margin:4px 0;font-size:13px;color:#374151;}
    .section-title{font-size:1.1rem;margin-bottom:6px;}
    .make-layout{display:grid;gap:18px;grid-template-columns:minmax(0,1.2fr);}
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
      background:#E0E7FF;border-color:#4338CA;
      transform:translateY(-1px);
      box-shadow:0 8px 18px rgba(67,56,202,.35);
    }
    .file-ready{
      background:#DCFCE7;
      border-color:#16A34A;
      box-shadow:0 8px 18px rgba(22,163,74,.35);
    }
    .file input{display:none;}
    .file-label-main{
      font-weight:800;font-size:14px;text-transform:uppercase;letter-spacing:.08em;
    }
    .file-label-sub{font-size:12px;color:var(--muted);}
    .free-note{
      margin-top:6px;font-size:12px;color:#065F46;background:#D1FAE5;
      border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:6px;
    }
    .free-dot{width:8px;height:8px;border-radius:999px;background:#10B981;}
    fieldset{border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0;}
    legend{font-size:13px;padding:0 4px;color:#374151;}
    .row{display:flex;flex-wrap:wrap;gap:12px;}
    .row > label{flex:1 1 150px;font-size:13px;}
    .row input,.row select{
      width:100%;margin-top:3px;padding:6px 8px;border-radius:8px;
      border:1px solid #CBD5F5;font-size:13px;background:#F9FAFB;
    }
    .row input:focus,.row select:focus{
      outline:none;border-color:#4F46E5;box-shadow:0 0 0 1px rgba(79,70,229,.35);
      background:#FFFFFF;
    }
    label{font-size:13px;}
    .controls-note{font-size:11px;color:#9CA3AF;margin-top:4px;}
    .hidden{display:none;}
    @media (max-width:860px){
      .hero{grid-template-columns:1fr;}
      .make-layout{grid-template-columns:1fr;}
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
        · Signed in as {{ user.email }}
        <span class="plan-tag">{{ user.subscription }}</span>
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
        {% if pro_active %}
          <span class="badge badge-pro">Pro tools active (4× resolution)</span>
        {% endif %}
      </div>
    </div>

    <div class="card">
      <h2 class="section-title">Why PatternCraft.app</h2>
      <ul class="why-list">
        <li>Clean grids with bold 10×10 guides and symbol overlays</li>
        <li>Floss estimates per color so you can kit projects confidently</li>
        <li>Knitting charts that respect row proportions and gauge</li>
        <li>Optional embroidery outputs ready for your own software</li>
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
          <label class="file">
            <input id="fileInput" type="file" name="file" accept="image/*" required onchange="pickFile(this)">
            <div>
              <div class="file-label-main">UPLOAD ARTWORK</div>
              <div class="file-label-sub">
                Drop in your illustration, logo, or photo — we’ll turn it into a stitchable pattern.
              </div>
            </div>
          </label>

          <div style="display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap;">
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

          <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
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
      if (label) label.textContent = 'UPLOAD ARTWORK';
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

SIGNUP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Create your account — PatternCraft.app</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{
      margin:0;background:#F3F4F6;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:#111827;
    }
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px;}
    .card{
      background:#FFFFFF;border-radius:14px;border:1px solid #E5E7EB;
      padding:22px;box-shadow:0 18px 40px rgba(15,23,42,.16);
    }
    h1{margin:0 0 10px;font-size:1.6rem;}
    .muted{font-size:13px;color:#6B7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input[type="email"],input[type="password"]{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #CBD5E1;font-size:14px;background:#F9FAFB;
    }
    input:focus{
      outline:none;border-color:#4F46E5;box-shadow:0 0 0 1px rgba(79,70,229,.35);
      background:#FFFFFF;
    }
    .pill{
      margin-top:14px;padding:9px 18px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#F97316,#EA580C);color:#FFFFFF;
      font-size:14px;font-weight:600;cursor:pointer;
      box-shadow:0 7px 18px rgba(248,113,22,.35);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(248,113,22,.45);}
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;}
    a{color:#2563EB;text-decoration:none;}
    a:hover{text-decoration:underline;}
    ul{font-size:13px;color:#4B5563;padding-left:18px;margin-top:10px;}
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
    body{
      margin:0;background:#F3F4F6;
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      color:#111827;
    }
    .wrap{max-width:520px;margin:0 auto;padding:32px 16px 40px;}
    .card{
      background:#FFFFFF;border-radius:14px;border:1px solid #E5E7EB;
      padding:22px;box-shadow:0 18px 40px rgba(15,23,42,.16);
    }
    h1{margin:0 0 10px;font-size:1.6rem;}
    .muted{font-size:13px;color:#6B7280;}
    label{display:block;font-size:13px;margin-top:12px;}
    input{
      width:100%;margin-top:4px;padding:8px 10px;border-radius:10px;
      border:1px solid #CBD5E1;font-size:14px;background:#F9FAFB;
    }
    input:focus{
      outline:none;border-color:#4F46E5;box-shadow:0 0 0 1px rgba(79,70,229,.35);
      background:#FFFFFF;
    }
    .pill{
      margin-top:14px;padding:9px 18px;border-radius:999px;
      border:none;background:linear-gradient(135deg,#4C51BF,#4338CA);color:#FFFFFF;
      font-size:14px;font-weight:600;cursor:pointer;
      box-shadow:0 7px 18px rgba(79,70,229,.35);
    }
    .pill:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(79,70,229,.45);}
    .msg{margin-top:10px;font-size:13px;color:#B91C1C;}
    a{color:#2563EB;text-decoration:none;}
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
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PatternCraft • Pricing</title>
<style>
:root{
  --fg:#111827; --muted:#6B7280; --accent:#E4006D; --line:#E5E7EB; --card:#F9FAFB;
  --radius:14px; --wrap:1100px;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{
  font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
  color:var(--fg);background:#FFFFFF;
}
.wrap{max-width:var(--wrap);margin:0 auto;padding:20px 16px 32px;}
header{
  position:sticky;top:0;background:#FFFFFF;border-bottom:1px solid var(--line);z-index:5;
}
.brand{font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-size:14px;}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.btn{
  display:inline-block;padding:9px 16px;border-radius:999px;
  border:1px solid var(--accent);background:var(--accent);color:#FFFFFF;
  text-decoration:none;font-weight:600;cursor:pointer;font-size:13px;
}
.btn.ghost{background:transparent;color:var(--accent);}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:18px;}
.card{
  background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:16px;
}
.card h3{margin:0 0 4px;font-size:1.05rem;}
.price{font-size:26px;font-weight:800;margin:4px 0;}
.small{font-size:13px;color:var(--muted);}
.list{margin:8px 0 12px;padding-left:18px;font-size:13px;}
.badge{
  display:inline-block;background:#FFE4F2;color:#BE185D;border-radius:999px;
  padding:4px 12px;font-size:12px;font-weight:600;margin-bottom:6px;text-transform:uppercase;
}
.notice{
  margin-top:10px;padding:10px 12px;border-radius:10px;
  background:#FFF7ED;border:1px solid #FED7AA;color:#9A3412;
  font-size:13px;
}
footer{border-top:1px solid var(--line);margin-top:24px;padding-top:12px;}
@media (max-width:700px){ .cards{grid-template-columns:1fr;} }

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
  padding:14px;
}
.step-num{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:24px;height:24px;
  border-radius:999px;
  background:var(--accent);
  color:#FFFFFF;
  font-size:13px;
  font-weight:700;
  margin-bottom:8px;
}
</style>
</head>
<body>
<header>
  <div class="wrap row" style="justify-content:space-between">
    <div class="brand">PATTERNCRAFT.APP</div>
    <nav class="row">
      <a class="btn ghost" href="/">Tool</a>
      <a class="btn" href="/pricing">Pricing</a>
    </nav>
  </div>
</header>

<section class="wrap">
  <div class="badge">Simple, transparent pricing</div>
  <h1 style="font-size:1.7rem;margin:4px 0 4px;">Choose the plan that fits your stitching</h1>
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
      <h3>10 Pattern Pack</h3>
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
      <p class="small">Save compared with buying single patterns.</p>
    </div>

    <div class="card">
      <h3>3 Month Unlimited</h3>
      <div class="price">$75 / month</div>
      <p class="small">Short-term unlimited access.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>Highest-quality output (4× resolution)</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_3m">
        <button class="btn ghost" type="submit">Start 3-month</button>
      </form>
      <p class="small">Perfect for focused projects or peak stitching seasons.</p>
    </div>

    <div class="card">
      <h3>Pro Annual Unlimited</h3>
      <div class="price">$99 / year</div>
      <p class="small">Unlimited patterns all year.</p>
      <ul class="list">
        <li>Unlimited pattern conversions</li>
        <li>Highest-quality output (4× resolution)</li>
        <li>Advanced color tools</li>
        <li>Priority processing</li>
        <li>All export formats + templates</li>
      </ul>
      <form method="POST" action="/checkout">
        <input type="hidden" name="plan" value="unlimited_year">
        <button class="btn ghost" type="submit">Go Pro annual</button>
      </form>
      <p class="small">Best value if you stitch more than four patterns a year.</p>
    </div>
  </div>
</section>

<section class="wrap" id="how">
  <h2 style="font-size:1.25rem;margin-bottom:4px;">How PatternCraft.app works</h2>
  <p class="small">From photo to stitch-ready pattern in three simple steps.</p>
  <div class="steps">
    <div class="step-card">
      <div class="step-num">1</div>
      <h3 style="font-size:1rem;margin:0 0 4px;">Upload your image</h3>
      <p class="small">
        Start with a photo, artwork, or logo. PatternCraft analyzes it for stitchable detail.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3 style="font-size:1rem;margin:0 0 4px;">Choose size & colors</h3>
      <p class="small">
        Set stitch width, cloth count, and palette size. Preview your grid with bold 10×10 guides.
      </p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3 style="font-size:1rem;margin:0 0 4px;">Download your pattern ZIP</h3>
      <p class="small">
        Download a ZIP with grid.png, legend.csv, meta.json, and optional PDF or embroidery files,
        ready to print or import into your workflow.
      </p>
    </div>
  </div>
</section>

<footer class="wrap small">
  <div class="row" style="justify-content:space-between">
    <div class="small">© PatternCraft.app</div>
    <div><a href="/" class="btn ghost">Back to tool</a></div>
  </div>
</footer>
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
      font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      background:#F3F4F6;
      color:#111827;
    }
    .wrap{
      max-width:520px;
      margin:0 auto;
      padding:32px 16px 40px;
    }
    .card{
      background:#FFFFFF;
      border-radius:14px;
      border:1px solid #E5E7EB;
      padding:24px;
      box-shadow:0 18px 40px rgba(15,23,42,.16);
    }
    h1{margin:0 0 10px;font-size:1.7rem;}
    p{margin:6px 0;font-size:14px;color:#4B5563;}
    a{
      color:#2563EB;
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
      <p>You can go back to the tool and start generating patterns right away using your new plan.</p>
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
