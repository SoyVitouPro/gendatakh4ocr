import os
import random
import threading
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
import glob
import re

# ==== Configuration ====
# Define multiple text files as sources and the number of lines to use from each.
data_base_path = "/data/Vitou/gen_data/text_data"
data_sources = {
    "khmer_name.txt": 20000,
}
background_dir = "./background"

# Dynamically collect all fonts in ./fonts (recursively) and validate loadability
fonts_dir = "./fonts/"
candidate_fonts = []
for root, dirs, files in os.walk(fonts_dir):
    for f in files:
        if f.lower().endswith((".ttf", ".otf")):
            candidate_fonts.append(os.path.join(root, f))
assert candidate_fonts, "❌ No font files found in ./fonts!"

def validate_fonts(paths):
    valid = []
    for p in paths:
        try:
            _ = ImageFont.truetype(p, 32)
            valid.append(p)
        except Exception:
            continue
    return valid

font_paths = validate_fonts(candidate_fonts)
assert font_paths, "❌ No valid font files could be loaded from ./fonts!"

output_base = "image_generated"
variants_per_word = 2
batch_size = 2000
thread_count = 20
margin = 4

# Perlin-like noise settings (tunable)
perlin_scale = 20           # larger -> smoother noise
perlin_octaves = 3
perlin_persistence = 0.5
perlin_lacunarity = 2.0
perlin_threshold = 0.3      # fallback; we now prefer keep ratio
perlin_keep = 0.99          # keep ~80% of text (erase ~20%)
perlin_blur_radius = 9.0    # base blur radius (global, guided by noise)
perlin_blur_weight = 0.7    # 0..1 how much the noise blends the blur

# Extra blur focused on text
text_blur_radius = 2.3
text_blur_weight = 0.8

# Partial erasure controls (never fully erase strokes)
perlin_min_alpha = 260       # minimum alpha (0-255) for text after erasure
perlin_softness = 0.04      # softness band around threshold in noise units (0..1)

# Small random rotation for text (degrees)
text_rotation_max = 1.0

non_khmer_pattern = re.compile(r'[A-Za-z0-9<>@#$%^&*+=/\\|{}\[\]~`!,:;"\'().-]')

def is_pure_khmer(text):
    return not non_khmer_pattern.search(text)

def load_lines_with_source(file_path, num_lines_to_load):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    lines = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_lines_to_load:
                    break
                if line.strip():
                    lines.append(line.strip())
    except FileNotFoundError:
        print(f"⚠️ Warning: Data source not found, skipping: {file_path}")
        return []
    return [(line, base_name) for line in lines]

all_words = []
total_loaded = 0
print("🔄 Loading data from sources...")
for file_name, num_lines in data_sources.items():
    file_path = os.path.join(data_base_path, file_name)
    loaded_data = load_lines_with_source(file_path, num_lines)
    all_words.extend(loaded_data)
    if loaded_data:
        print(f"  - Loaded {len(loaded_data)} lines from {file_name}")
        total_loaded += len(loaded_data)

print(f"✅ Loaded {total_loaded} total lines from {len(data_sources)} sources.")

def add_gaussian_noise(img, mean=0, std=10):
    np_img = np.array(img).astype(np.int16)
    noise = np.random.normal(mean, std, np_img.shape).astype(np.int16)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def add_salt_pepper_noise(img, amount=0.01, s_vs_p=0.5):
    np_img = np.array(img)
    total = np_img.size // 3
    for _ in range(int(amount * total * s_vs_p)):
        i, j = random.randint(0, np_img.shape[0]-1), random.randint(0, np_img.shape[1]-1)
        np_img[i, j] = [255, 255, 255]
    for _ in range(int(amount * total * (1 - s_vs_p))):
        i, j = random.randint(0, np_img.shape[0]-1), random.randint(0, np_img.shape[1]-1)
        np_img[i, j] = [0, 0, 0]
    return Image.fromarray(np_img)

def add_speckle_noise(img):
    np_img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.randn(*np_img.shape)
    noisy = np_img + np_img * noise * 0.5
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def add_blur(img, radius=1):
    return img.filter(ImageFilter.GaussianBlur(radius))

def corrupt_text_mask(mask, x, top, w, h):
    """Randomly remove small parts of the text mask to simulate missing ink/erasure.
    Returns (mask, stats) where stats contains density and counts.
    """
    m = mask.copy()
    d = ImageDraw.Draw(m)

    area = max(1, w * h)
    density = random.uniform(0.0003, 0.001)
    approx_hole_area = 6
    num_holes = max(3, min(150, int((area * density) / approx_hole_area)))

    hole_count = 0
    for _ in range(num_holes):
        r = random.randint(1, 2)
        cx = random.randint(x, x + max(1, w) - 1)
        cy = random.randint(top, top + max(1, h) - 1)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)
        hole_count += 1

    scratch_count = 0
    for _ in range(random.randint(0, 1)):
        length = random.randint(max(3, w // 20), max(6, w // 12))
        thickness = 1
        cx = random.randint(x, x + max(1, w) - 1)
        cy = random.randint(top, top + max(1, h) - 1)
        dx = random.choice([-length, length])
        dy = random.randint(-2, 2)
        d.line((cx, cy, cx + dx, cy + dy), fill=0, width=thickness)
        scratch_count += 1

    rect_count = 0
    for _ in range(random.randint(0, 2)):
        rw = random.randint(2, max(3, w // 24))
        rh = random.randint(2, max(3, h // 14))
        cx = random.randint(x, x + max(1, w) - 1)
        cy = random.randint(top, top + max(1, h) - 1)
        d.rectangle((cx, cy, cx + rw, cy + rh), fill=0)
        rect_count += 1

    stats = {
        "density": density,
        "holes": hole_count,
        "scratches": scratch_count,
        "rects": rect_count,
    }

    return m, stats

def resize_down_up(img, min_scale=0.45, max_scale=0.85):
    w, h = img.size
    scale = random.uniform(min_scale, max_scale)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    method_down = random.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST])
    method_up = random.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST])
    small = img.resize((nw, nh), resample=method_down)
    return small.resize((w, h), resample=method_up)

def add_poisson_noise(img):
    np_img = np.array(img).astype(np.float32)
    # Scale to [0,1]
    scaled = np_img / 255.0
    noisy = np.random.poisson(scaled * 255.0).astype(np.float32) / 255.0
    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def add_motion_blur(img, degree=None, angle=None):
    # Average several shifted copies along a direction
    if degree is None:
        degree = random.randint(3, 7)
    if angle is None:
        angle = random.uniform(0, 180)
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    acc = Image.new("RGB", img.size, (0, 0, 0))
    for i in range(degree):
        offx = int(round((i - degree // 2) * dx))
        offy = int(round((i - degree // 2) * dy))
        shifted = ImageChops.offset(img, offx, offy)
        acc = ImageChops.add(acc, shifted, scale=1.0)
    # Normalize by degree
    arr = np.array(acc).astype(np.float32) / max(1, degree)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def jpeg_recompress(img, q=None):
    if q is None:
        q = random.randint(15, 28)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True, subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def add_vignette(img, strength=0.25):
    w, h = img.size
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    mask = 1.0 - strength * (dist / max_dist)
    mask = np.clip(mask, 0.6, 1.0)
    arr = np.array(img).astype(np.float32)
    arr *= mask[..., None]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_film_grain(img, std=5):
    np_img = np.array(img).astype(np.int16)
    noise = np.random.normal(0, std, np_img.shape).astype(np.int16)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def adjust_brightness_contrast(img, b_delta=None, c_factor=None):
    if b_delta is None:
        b_delta = random.uniform(-0.05, 0.05)
    if c_factor is None:
        c_factor = random.uniform(0.9, 1.1)
    # Brightness
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(1.0 + b_delta)
    # Contrast
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(c_factor)
    return img

# (Simplified) We won't use per-text noise anymore.

def mask_keep_ratio(original_mask, corrupted_mask):
    # Retained for compatibility, but we won't corrupt masks anymore.
    orig = np.array(original_mask)
    corr = np.array(corrupted_mask)
    orig_count = max(1, int((orig > 0).sum()))
    corr_count = int((corr > 0).sum())
    return corr_count / float(orig_count)

def ensure_min_mask_keep(original_mask, corrupted_mask, min_keep=0.7):
    # No-op in simplified pipeline
    return corrupted_mask

def safe_slug(s):
    # Replace spaces and forbidden filename chars with underscores
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)

def trim_filename(name, max_len=180):
    # Ensure filename (without path) does not exceed filesystem limits
    if len(name) <= max_len:
        return name
    # Keep start and end, drop the middle
    keep_head = max_len // 2 - 3
    keep_tail = max_len - keep_head - 3
    return name[:keep_head] + "..." + name[-keep_tail:]

# ---- Perlin-like noise (value noise) ----
def _interp(a, b, t):
    return a * (1 - t) + b * t

def _generate_base_noise(w, h, rng):
    return rng.random((h, w)).astype(np.float32)

def _upsample(arr, new_w, new_h):
    # Use PIL bilinear for decent quality
    img = Image.fromarray(np.uint8(np.clip(arr * 255, 0, 255)))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0

def generate_perlin_like_noise(width, height, scale=64, octaves=3, persistence=0.5, lacunarity=2.0, seed=None):
    if seed is None:
        seed = random.randint(0, 10_000_000)
    rng = np.random.default_rng(seed)
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0 / max(1, scale)
    max_amp = 0.0
    for _ in range(octaves):
        gw = max(2, int(math.ceil(width * frequency)))
        gh = max(2, int(math.ceil(height * frequency)))
        layer = _generate_base_noise(gw, gh, rng)
        layer_up = _upsample(layer, width, height)
        noise += layer_up * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    if max_amp > 0:
        noise /= max_amp
    return np.clip(noise, 0.0, 1.0)

# ---- Advanced text effects ----
def mask_morphology(mask, mode=None):
    # mode: None, 'dilate', 'erode'
    if mode == 'dilate':
        return mask.filter(ImageFilter.MaxFilter(size=3))
    if mode == 'erode':
        return mask.filter(ImageFilter.MinFilter(size=3))
    return mask

def mask_component_dropout(mask, drop_prob=0.08, max_keep_ratio=0.95):
    # Drops small connected components in the mask with probability.
    # Returns (new_mask, stats)
    arr = (np.array(mask) > 0).astype(np.uint8)
    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=np.uint8)

    def neighbors(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

    comp_id = 1
    comps = []
    for i in range(h):
        for j in range(w):
            if arr[i, j] and not visited[i, j]:
                # BFS
                stack = [(i, j)]
                visited[i, j] = 1
                pixels = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    for nr, nc in neighbors(r, c):
                        if arr[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = 1
                            stack.append((nr, nc))
                            pixels.append((nr, nc))
                comps.append(pixels)
                comp_id += 1

    # Decide drops: only drop small components to preserve readability
    total_on = int(arr.sum())
    drop_count = 0
    if total_on > 0:
        max_drop_area = int(0.20 * total_on)  # do not drop the main body
        for pixels in comps:
            area = len(pixels)
            if area <= max_drop_area and random.random() < drop_prob:
                for r, c in pixels:
                    arr[r, c] = 0
                drop_count += 1

    new_mask = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
    keep_ratio = max(0.0, min(1.0, float(arr.sum()) / float(max(1, total_on)))) if total_on > 0 else 1.0
    stats = {"components": len(comps), "dropped": drop_count, "keep_ratio": keep_ratio}
    return new_mask, stats

# elastic_warp_mask removed in simplified pipeline

def random_ink_color():
    base = random.randint(0, 40)
    # Slight color cast for realism
    r_off = random.randint(0, 10)
    g_off = random.randint(0, 10)
    b_off = random.randint(0, 10)
    return (base + r_off, base + g_off, base + b_off)

def apply_ink_layer(bg_img, mask, color=(0, 0, 0), coverage=(0.92, 1.0)):
    # Modulate mask with per-pixel coverage noise
    m = np.array(mask).astype(np.float32) / 255.0
    cov = np.random.uniform(coverage[0], coverage[1], m.shape).astype(np.float32)
    a = np.clip(m * cov, 0.0, 1.0)
    alpha = (a * 255.0).astype(np.uint8)
    ink = Image.new("RGB", bg_img.size, color)
    return Image.composite(ink, bg_img, Image.fromarray(alpha, mode='L'))

def add_lamination_glare(img, angle=None, width=None, strength=None, position=None):
    # Add a diagonal light band with soft edges
    w, h = img.size
    if angle is None:
        angle = random.uniform(20, 70)
    if width is None:
        width = random.uniform(0.15, 0.35) * max(w, h)
    if strength is None:
        strength = random.uniform(0.08, 0.18)
    if position is None:
        position = random.uniform(-0.3, 1.3)

    # Create coordinate grid
    yy, xx = np.mgrid[0:h, 0:w]
    theta = np.deg2rad(angle)
    # Line normal
    nx, ny = np.cos(theta), np.sin(theta)
    # Distance from band center
    center = position * (w * nx + h * ny)
    dist = (xx * nx + yy * ny) - center
    band = np.exp(-(dist ** 2) / (2 * (width ** 2)))
    band = (band * strength)

    arr = np.array(img).astype(np.float32)
    arr = arr + (255.0 * band[..., None])
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), {"angle": round(angle, 1), "width": int(width), "strength": round(strength, 3), "pos": round(position, 2)}

def add_crossing_obstruction(img):
    # Draw pen strokes or simple stamp-like shapes with low opacity
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    kind = random.choice(["pen", "pen", "pen", "stamp"])  # mostly pen
    params = {"kind": kind}
    if kind == "pen":
        strokes = random.randint(1, 2)
        params["strokes"] = strokes
        for _ in range(strokes):
            color = random.choice([(30, 30, 30, 90), (10, 60, 140, 90), (150, 20, 20, 90)])
            width = random.randint(2, 3)
            params.setdefault("widths", []).append(width)
            # Random diagonal-ish line across
            x0 = random.randint(-w // 5, w // 5)
            y0 = random.randint(0, h)
            x1 = x0 + w + random.randint(-w // 5, w // 5)
            y1 = y0 + random.randint(-h // 5, h // 5)
            d.line((x0, y0, x1, y1), fill=color, width=width)
    else:  # stamp
        r = random.randint(min(w, h) // 10, min(w, h) // 6)
        cx = random.randint(r, w - r)
        cy = random.randint(r, h - r)
        params.update({"r": r, "cx": cx, "cy": cy})
        # Outer circle
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(160, 20, 20, 110), width=2)
        # Inner text bar
        d.rectangle((cx - int(0.7 * r), cy - 4, cx + int(0.7 * r), cy + 4), fill=(160, 20, 20, 110))

    combined = img.convert("RGBA")
    combined = Image.alpha_composite(combined, overlay).convert("RGB")
    return combined, params

backgrounds = []
for root, dirs, files in os.walk(background_dir):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = os.path.join(root, f)
            backgrounds.append(full_path)
assert backgrounds, "❌ No background images found!"

batch_lock = threading.Lock()
batch_indices = list(range(0, len(all_words), batch_size))

def generate_batch():
    while True:
        with batch_lock:
            if not batch_indices:
                return
            start_idx = batch_indices.pop(0)
        end_idx = min(start_idx + batch_size, len(all_words))
        chunk_words = all_words[start_idx:end_idx]

        # Simplified: no source subfolder; write directly under output_base
        words = [w for w, _src in chunk_words]
        chunk_folder = f"image_{start_idx}_{end_idx}"
        chunk_path = os.path.join(output_base, chunk_folder)
        os.makedirs(chunk_path, exist_ok=True)
        annotation_lines = []

        for word in words:
                timestamp = datetime.now().strftime("date_%d_%m_%y_time_%H_%M_%S_%f")[:-3]
                word_folder = f"image_folder_{timestamp}"
                word_path = os.path.join(chunk_path, word_folder)
                os.makedirs(word_path, exist_ok=True)
                for variant_idx in range(variants_per_word):
                    # Choose font and size
                    font_path = random.choice(font_paths)
                    font_size = random.choice([
                        random.randint(40, 60),
                        random.randint(60, 100),
                        random.randint(100, 160)
                    ])
                    # Try multiple fonts if a chosen one fails to load
                    font = None
                    for _retry in range(3):
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                            break
                        except Exception:
                            font_path = random.choice(font_paths)
                            continue
                    if font is None:
                        continue

                    font_slug = safe_slug(os.path.splitext(os.path.basename(font_path))[0])

                    # Measure text box
                    tmp_img = Image.new("RGB", (4, 4), (255, 255, 255))
                    tmp_draw = ImageDraw.Draw(tmp_img)
                    bbox = tmp_draw.textbbox((0, 0), word, font=font)
                    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if width <= 0 or height <= 0:
                        continue

                    # Random margins
                    margin_left = random.randint(2, 12)
                    margin_right = random.randint(2, 12)
                    margin_top = random.randint(2, 12)
                    margin_bottom = random.randint(2, 12)
                    img_size = (
                        width + margin_left + margin_right,
                        height + margin_top + margin_bottom,
                    )

                    # Background image
                    bg_path = random.choice(backgrounds)
                    base_img = Image.open(bg_path).convert("RGB").resize(img_size)

                    # Prepare text mask at target position
                    x, y = margin_left, margin_top - bbox[1]
                    text_mask = Image.new("L", img_size, 0)
                    mask_draw = ImageDraw.Draw(text_mask)
                    mask_draw.text((x, y), word, fill=255, font=font)
                    # Apply small rotation to text only
                    rot_angle = random.uniform(-text_rotation_max, text_rotation_max)
                    text_mask = text_mask.rotate(rot_angle, resample=Image.BILINEAR, expand=False)

                    # Generate perlin-like noise
                    noise = generate_perlin_like_noise(
                        img_size[0], img_size[1],
                        scale=perlin_scale,
                        octaves=perlin_octaves,
                        persistence=perlin_persistence,
                        lacunarity=perlin_lacunarity,
                    )
                    # Apply erasing only on one of the two variants; the other stays intact
                    orig_mask_np = np.array(text_mask, dtype=np.uint8)
                    apply_erasing = (variant_idx == 0)
                    if apply_erasing:
                        noise_vals = noise[orig_mask_np > 0]
                        if noise_vals.size > 0:
                            thr = float(np.quantile(noise_vals, max(0.0, min(1.0, 1.0 - perlin_keep))))
                        else:
                            thr = perlin_threshold
                        # Soft threshold band
                        s = float(perlin_softness)
                        low = max(0.0, thr - s)
                        high = min(1.0, thr + s)
                        # Build graded alpha: below low -> min_alpha, above high -> 255, linear in between
                        denom = max(1e-6, high - low)
                        alpha = perlin_min_alpha + (np.clip((noise - low) / denom, 0.0, 1.0) * (255 - perlin_min_alpha))
                        # Apply only where text exists; elsewhere alpha=0
                        mask_np = (alpha * (orig_mask_np > 0)).astype(np.uint8)
                    else:
                        # No erasing: keep full-strength mask (opaque text)
                        mask_np = (orig_mask_np > 0).astype(np.uint8) * 255
                    eroded_mask = Image.fromarray(mask_np, mode="L")

                    # Composite black text via the eroded mask
                    black_layer = Image.new("RGB", img_size, (0, 0, 0))
                    final_img = Image.composite(black_layer, base_img, eroded_mask)

                    # Global blur controlled by noise (more blur where noise is higher)
                    blurred = add_blur(final_img, radius=perlin_blur_radius)
                    blur_alpha = np.uint8(np.clip(noise * (perlin_blur_weight * 255.0), 0, 255))
                    blur_alpha_img = Image.fromarray(blur_alpha, mode="L")
                    final_img = Image.composite(blurred, final_img, blur_alpha_img)

                    # Extra blur on text only
                    text_blurred = add_blur(final_img, radius=text_blur_radius)
                    text_alpha = (mask_np.astype(np.float32) * float(text_blur_weight)).astype(np.uint8)
                    final_img = Image.composite(text_blurred, final_img, Image.fromarray(text_alpha, mode="L"))

                    # Save simple filename
                    filename = trim_filename(f"{font_slug}_{variant_idx + 1:02d}.jpg")
                    img_rel_path = os.path.join(output_base, chunk_folder, word_folder, filename)

                    # Save with low JPEG quality
                    final_img.save(
                        os.path.join(word_path, filename),
                        format="JPEG",
                        quality=random.randint(15, 22),
                        optimize=True,
                        progressive=True,
                        subsampling=2,
                    )
                    annotation_lines.append(f"{img_rel_path}\t{word}")

        with open(os.path.join(chunk_path, "annotations.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(annotation_lines))
        print(f"[{chunk_path}] ✅ Generated {len(words)} words")

if __name__ == "__main__":
    os.makedirs(output_base, exist_ok=True)
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        for _ in range(thread_count):
            executor.submit(generate_batch)
