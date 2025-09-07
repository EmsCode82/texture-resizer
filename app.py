from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageOps, ImageFilter
import io, os, uuid, requests
import re
from flask_cors import CORS
import numpy as np  # Added for PBR generation
from scipy.ndimage import convolve
from zipfile import ZipFile, ZIP_DEFLATED

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}) 

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def nearest_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def load_image_from_request():
         import logging
         logger = logging.getLogger(__name__)
         logger.debug("Starting image load from request")

         if 'file' in request.files:
             f = request.files['file']
             logger.debug(f"Processing uploaded file: {f.filename}, content_length: {f.content_length}")
             allowed_types = {'.png', '.jpg', '.jpeg', '.tga'}
             ext = os.path.splitext(f.filename.lower())[1]
             logger.debug(f"File extension: {ext}")
             if not ext:
                 logger.error("No file extension detected")
                 return None, "No file extension detected"
             if ext not in allowed_types:
                 logger.error(f"Unsupported file type: {ext}")
                 return None, f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
             if f.content_length and f.content_length > 10 * 1024 * 1024:  # 10MB limit
                 logger.error(f"File size {f.content_length} exceeds 10MB limit")
                 return None, "File size exceeds 10MB limit"
             try:
                 if f.content_length == 0:
                     logger.warning("File has zero content length, attempting to read stream")
                     f.stream.seek(0)  # Reset stream position
                 img = Image.open(f.stream).convert('RGBA')
                 width, height = img.size
                 if width < 64 or height < 64:
                     logger.error(f"Image too small: {width}x{height}")
                     return None, "Image too small: minimum 64x64"
                 if width > 8192 or height > 8192:
                     logger.error(f"Image too large: {width}x{height}")
                     return None, "Image too large: maximum 8192x8192"
                 if width == 0 or height == 0:
                     logger.error("Image has zero dimensions")
                     return None, "Invalid image: zero dimensions"
                 logger.debug(f"Image loaded successfully, size: {width}x{height}")
                 return img, f.filename
             except Exception as e:
                 logger.error(f"Invalid image file: {str(e)}")
                 return None, f"Invalid image file: {str(e)}"

         url = None
         if request.is_json:
             url = (request.json or {}).get('imageUrl')
         if not url:
             url = request.form.get('imageUrl')
         if not url:
             logger.error("No imageUrl or file provided")
             return None, "Provide an uploaded file (field 'file') or 'imageUrl'."

         try:
             logger.debug(f"Fetching image from URL: {url}")
             resp = requests.get(url, timeout=20)
             resp.raise_for_status()
             img = Image.open(io.BytesIO(resp.content)).convert('RGBA')
             width, height = img.size
             name = os.path.basename(url.split("?")[0])
             ext = os.path.splitext(name.lower())[1]
             logger.debug(f"URL file extension: {ext}")
             if not ext:
                 logger.error("No file extension detected in URL")
                 return None, "No file extension detected in URL"
             if ext not in allowed_types:
                 logger.error(f"Unsupported URL file type: {ext}")
                 return None, f"Unsupported URL file type. Allowed: {', '.join(allowed_types)}"
             if width < 64 or height < 64:
                 logger.error(f"Image too small: {width}x{height}")
                 return None, "Image too small: minimum 64x64"
             if width > 8192 or height > 8192:
                 logger.error(f"Image too large: {width}x{height}")
                 return None, "Image too large: maximum 8192x8192"
             logger.debug(f"Image from URL loaded successfully, size: {width}x{height}")
             return img, name
         except Exception as e:
             logger.error(f"Failed to download image: {str(e)}")
             return None, f"Failed to download image: {str(e)}"

def parse_ratio(r: str):
    parts = r.split(":")
    if len(parts) != 2:
        return None
    try:
        a = int(parts[0]); b = int(parts[1])
        if a <= 0 or b <= 0:
            return None
        return (a, b)
    except:
        return None

def to_pow2(n: int):
    lower = 1 << (n.bit_length() - 1)
    upper = lower << 1
    return lower if (n - lower) <= (upper - n) else upper

def size_from_ratio_long(ratio_tuple, long_side):
    a, b = ratio_tuple
    if a >= b:
        w = long_side
        h = round(long_side * b / a)
    else:
        h = long_side
        w = round(long_side * a / b)
    return (w, h)

def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]+', '-', (s or '').strip())[:60]

def save_variant(img: Image.Image, w: int, h: int, fmt: str,
                 pack=None, race=None, label=None, original_name=None, suffix="", compress=False):
    fmt = (fmt or "png").lower()
    if fmt not in ("png", "tga"):
        fmt = "png"
    ext = "tga" if fmt == "tga" else "png"

    # sanitize provided metadata and original filename
    safe_name = sanitize(os.path.splitext(original_name or "")[0])
    parts = [safe_name, sanitize(pack), sanitize(race), sanitize(label), suffix, f"{w}x{h}"]
    base = "_".join([x for x in parts if x]) or f"resized_{w}x{h}"

    # always append unique id to prevent overwriting
    fname = f"{base}_{uuid.uuid4().hex}.{ext}"
    path = os.path.join(OUTPUT_DIR, fname)

    # Save with compression for PNGs
    if ext == "png" and compress:
        img.save(path, "PNG", optimize=True)
    else:
        img.save(path, "TGA" if ext == "tga" else "PNG")
    size_bytes = os.path.getsize(path)
    return fname, path, size_bytes

def generate_normal_map(base_img: Image.Image):
    gray = base_img.convert('L')
    array = np.array(gray, dtype=float)
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply filters with convolution
    dx = convolve(array, sobel_x, mode='constant', cval=0.0)
    dy = convolve(array, sobel_y, mode='constant', cval=0.0)
    
    # Compute normals (unchanged)
    dz = np.ones_like(dx) * 10.0  # Strength factor
    normal = np.dstack((-dx, -dy, dz))
    norm = np.linalg.norm(normal, axis=2)[:,:,np.newaxis]
    normal = (normal / norm) * 0.5 + 0.5
    normal = np.clip(normal * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(normal, 'RGB')  # Normals are RGB

# New: Generate roughness map (simple invert grayscale)
def generate_roughness_map(base_img: Image.Image):
    gray = base_img.convert('L')
    inverted = ImageOps.invert(gray)
    return inverted.convert('RGB')  # Roughness often grayscale, but RGB for consistency

def generate_batch_variants(img: Image.Image, sizes, formats, mode,
                            pack, race, label, original_name=None, pbr=False, compress=False):
    """Builds outputs for multiple sizes/formats (PNG/TGA) and returns the results dict."""
    results = {f: {} for f in formats}
    if pbr:
        results['pbr'] = {}  # Sub-dict for PBR maps

    for target in sizes:
        # Build base square per size (unchanged)
        if mode == "stretch":
            base_img = img.resize((target, target), Image.Resampling.LANCZOS)
        elif mode == "crop":
            scale = max(target / img.width, target / img.height)
            new_w, new_h = int(img.width * scale), int(img.height * scale)
            scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x1 = (new_w - target) // 2
            y1 = (new_h - target) // 2
            base_img = scaled.crop((x1, y1, x1 + target, y1 + target))
        else:  # fit (pad with transparent pixels)
            fitted = ImageOps.contain(img, (target, target), Image.Resampling.LANCZOS)
            base_img = Image.new("RGBA", (target, target), (0, 0, 0, 0))
            x = (target - fitted.width) // 2
            y = (target - fitted.height) // 2
            base_img.paste(fitted, (x, y))

        for fmt in formats:  # fmt in {"png","tga"}
            fname, _, size_bytes = save_variant(
                base_img, target, target, fmt,
                pack=pack, race=race, label=label, original_name=original_name, suffix="base", compress=compress
            )
            results[fmt][str(target)] = {
                "url": f"{request.host_url.rstrip('/')}/files/{fname}",
                "bytes": size_bytes,
                "mb": round(size_bytes / (1024 * 1024), 3),
            }
        
        if pbr:
            # Generate PBR maps from resized base
            normal_img = generate_normal_map(base_img)
            roughness_img = generate_roughness_map(base_img)
            
            # Save normal (RGB, PNG only for now)
            n_fname, _, n_bytes = save_variant(
                normal_img, target, target, "png",
                pack=pack, race=race, label=label, original_name=original_name, suffix="normal", compress=compress
            )
            
            # Save roughness (RGB, PNG)
            r_fname, _, r_bytes = save_variant(
                roughness_img, target, target, "png",
                pack=pack, race=race, label=label, original_name=original_name, suffix="roughness", compress=compress
            )
            
            results['pbr'][str(target)] = {
                "normal": {
                    "url": f"{request.host_url.rstrip('/')}/files/{n_fname}",
                    "bytes": n_bytes,
                    "mb": round(n_bytes / (1024 * 1024), 3),
                },
                "roughness": {
                    "url": f"{request.host_url.rstrip('/')}/files/{r_fname}",
                    "bytes": r_bytes,
                    "mb": round(r_bytes / (1024 * 1024), 3),
                }
            }

    return results

@app.post("/resize")
def resize_endpoint():
    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err

    size_param  = request.args.get("size")
    download    = request.args.get("download", "0") == "1"
    width       = request.args.get("width")
    height      = request.args.get("height")
    ratio       = request.args.get("ratio")
    long_       = request.args.get("long")
    mode        = request.args.get("mode", "fit")
    pow2        = request.args.get("pow2", "0") == "1"
    fmt         = (request.args.get("format") or "png").lower()
    pack        = request.args.get("pack")
    race        = request.args.get("race")
    label       = request.args.get("label")

    if fmt not in ("png", "tga"):
        return jsonify({"error": "Only png or tga are supported for asset outputs."}), 400

    target_w = target_h = None
    # --- in /resize ---
    if width and height:
        try:
            target_w, target_h = int(width), int(height)
        except ValueError:
            return jsonify({"error": "width/height must be integers"}), 400
    elif ratio and long_:
        rt = parse_ratio(ratio)
        if not rt:
            return jsonify({"error": "Invalid ratio. Use ratio=16:9"}), 400
        try:
            long_side = int(long_)
        except ValueError:
            return jsonify({"error": "long must be an integer"}), 400
        target_w, target_h = size_from_ratio_long(rt, long_side)
    elif size_param:
        if size_param == "pow2":
            s = nearest_pow2(max(img.width, img.height))
            target_w = target_h = s
        else:
            try:
                s = int(size_param)
            except ValueError:
                return jsonify({"error": "size must be an integer or 'pow2'"}), 400
            target_w = target_h = s
    else:
        return jsonify({"error": "Provide size OR width+height OR ratio+long"}), 400

    if pow2:
        target_w = to_pow2(target_w)
        target_h = to_pow2(target_h)

    if mode == "stretch":
        out = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    elif mode == "crop":
        scale = max(target_w / img.width, target_h / img.height)
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x1 = (new_w - target_w) // 2
        y1 = (new_h - target_h) // 2
        out = scaled.crop((x1, y1, x1 + target_w, y1 + target_h))
    else:  # fit
        fitted = ImageOps.contain(img, (target_w, target_h), Image.Resampling.LANCZOS)
        out = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        x = (target_w - fitted.width) // 2
        y = (target_h - fitted.height) // 2
        out.paste(fitted, (x, y))

    compress = request.args.get("compress", "0") == "1"

    fname, save_path, size_bytes = save_variant(
        out, target_w, target_h, fmt, pack=pack, race=race, label=label, original_name=original_name, compress=compress
    )

    if download:
        return send_file(save_path, as_attachment=True, download_name=fname)

    public_url = f"{request.host_url.rstrip('/')}/files/{fname}"
    return jsonify({
        "public_url": public_url,
        "optimized_path": os.path.abspath(save_path),
        "width": out.width,
        "height": out.height,
        "format": fmt,
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024*1024), 3)
    })    

@app.get("/")
def index():
    return """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Assetgineer Texture Service</title></head>
  <body style="font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; line-height:1.6;">
    <h1>Assetgineer Texture Service</h1>

    <form method="post" enctype="multipart/form-data">
      <p><label>Upload image: <input type="file" name="file"></label></p>
      <p><label>OR Image URL: <input type="url" name="imageUrl" placeholder="https://example.com/image.png" style="width:100%;"></label></p>

      <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px;">
        <button formaction="/resize?size=512&download=1&compress=1">Resize to 512 (Compressed)</button>
        <button formaction="/resize?size=1024&download=1&compress=1">Resize to 1024 (Compressed)</button>
        <button formaction="/resize?size=2048&download=1&compress=1">Resize to 2048 (Compressed)</button>
        <button formaction="/resize?size=4096&download=1&compress=1">Resize to 4096 (Compressed)</button>
        <button formaction="/resize?size=pow2&download=1&compress=1">Resize to POW2 (Compressed)</button>
        <button formaction="/batch?compress=1" formmethod="post" formtarget="_blank">Batch (Compressed)</button>
        <button formaction="/profile/gameasset?compress=1" formmethod="post" formtarget="_blank" style="background:#222; color:#fff;">GameAsset Pack (Compressed)</button>
        <button formaction="/profile/gameasset?pbr=1&compress=1" formmethod="post" formtarget="_blank" style="background:#444; color:#fff;">GameAsset Pack with PBR (Compressed)</button>
        <button formaction="/validate" formmethod="post" formtarget="_blank" style="background:#007bff; color:#fff;">Validate Image</button>
        <button formaction="/package" formmethod="post" formtarget="_blank" style="background:#28a745; color:#fff;">Download Pack as Zip</button>
      </div>
    </form>

    <hr style="margin:28px 0;">

    <h2>ℹ️ What This Tool Does</h2>

    <p>
    This tool takes <b>any common image format</b> you upload (PNG, JPG, JPEG, GIF, BMP, WebP, etc.) 
    and automatically prepares it for use in games or digital projects. No matter the input, 
    your images are normalized to <code>RGBA</code> and exported in <b>PNG</b> or <b>TGA</b> format.
    It also generates basic PBR maps (normal and roughness) and validates textures for common issues.
    Use the <code>Compressed</code> options to reduce file sizes without quality loss.
    </p>

    <h3>How It Works</h3>
    <ul>
    <li><b>Upload a file</b> (any format) or paste an image link.</li>
    <li><b>Pick a size</b> — for example 512, 1024, 2048, or 4096 pixels.</li>
    <li>The tool will <b>resize and clean up</b> your image automatically.</li>
    <li>Optionally generate PBR maps (normal from edges, roughness from inverted grayscale).</li>
    <li>Validate for issues like seams, alpha problems, or artifacts.</li>
    <li>You can also create a whole <b>pack of images</b> in different sizes with one click, with compression available.</li>
    </ul>

    <h3>Why It’s Useful</h3>
    <ul>
    <li>Accepts almost any image format as input, even if your source isn’t already PNG.</li>
    <li>Makes sure outputs are game-ready in <code>PNG</code> and <code>TGA</code> formats only.</li>
    <li>Automatically keeps transparency (so backgrounds stay clear).</li>
    <li>Organizes filenames with labels you provide (like pack, race, or item name).</li>
    <li>Generates basic PBR maps (normal, roughness) for modern rendering in Unity/Unreal.</li>
    <li>Validates textures for quality issues, helping ensure marketplace compliance.</li>
    <li>Offers compression to reduce file sizes, improving download and storage efficiency.</li>
    </ul>

    <h3>Example</h3>
    <ol>
    <li>You upload <code>dragon.jpg</code> or <code>dragon.png</code>.</li>
    <li>You press <b>GameAsset Pack with PBR (Compressed)</b>.</li>
    <li>You instantly get 4 versions: 512, 1024, 2048, and 4096.</li>
    <li>Each version includes compressed base PNG/TGA plus normal and roughness maps, with direct download links.</li>
    <li>Press <b>Validate Image</b> to check for issues like seams or noise.</li>
    </ol>

    <hr style="margin:28px 0;">

    <h2>📘 Service Documentation</h2>

    <h3>Overview</h3>
    <p>
    This service is a <b>Flask-based image processing pipeline</b> for preparing 
    <b>game-ready textures</b>. It accepts <b>any common input format</b> 
    (PNG, JPG, JPEG, GIF, BMP, WebP, etc.), normalizes them to <code>RGBA</code>, 
    and exports clean, consistent <code>PNG</code> or <code>TGA</code> files. 
    It supports power-of-two constraints, batch generation, optional metadata tagging,
    basic PBR map generation (normal and roughness approximations), quality validation,
    and optional compression for PNG outputs.
    </p>

    <h3>1. Input Handling</h3>
    <ul>
    <li><b>File Upload</b>: via form field <code>file</code></li>
    <li><b>Image URL</b>: JSON <code>{"imageUrl": "https://..."}</code> or form field <code>imageUrl</code></li>
    <li>All inputs automatically normalized to <code>RGBA</code> (keeps transparency)</li>
    <li>Original filename is preserved in the output, combined with metadata and a unique ID</li>
    </ul>

    <h3>2. Resizing Endpoint <code>/resize</code></h3>
    <p>Choose exactly one sizing method:</p>
    <ul>
    <li><code>?size=N</code> → exact square (512, 1024, 2048, 4096)</li>
    <li><code>?size=pow2</code> → next power-of-two of the image’s largest side</li>
    <li><code>?width=W&height=H</code> → explicit rectangle</li>
    <li><code>?ratio=A:B&long=N</code> → maintain ratio, scale longest side</li>
    </ul>
    <p>Options:</p>
    <ul>
    <li><code>&mode=fit|crop|stretch</code> (default: fit)</li>
    <li><code>&pow2=1</code> → round each side individually to nearest power-of-two</li>
    <li><code>&pack=&race=&label=</code> → optional metadata in filenames</li>
    <li><code>&download=1</code> → return file directly instead of JSON</li>
    <li><code>&compress=1</code> → optimize PNG files for smaller size</li>
    </ul>

    <h3>3. Batch Endpoint <code>/batch</code></h3>
    <p>Generate multiple variants at once:</p>
    <ul>
    <li>Default sizes: 512, 1024, 2048, 4096</li>
    <li>Default formats: PNG (can also include TGA)</li>
    <li>All variants share the same mode/flags</li>
    <li>Add <code>&pbr=1</code> for normal/roughness maps</li>
    <li>Add <code>&compress=1</code> for optimized PNGs</li>
    </ul>
    <p>Returns JSON: format → size → {URL, file size in bytes/MB}; plus 'pbr' key if enabled</p>

    <h3>4. GameAsset Profile <code>/profile/gameasset</code></h3>
    <ul>
    <li>Convenience wrapper around <code>/batch</code></li>
    <li>Defaults: sizes = 512,1024,2048,4096 | formats = PNG,TGA | mode = fit</li>
    <li>Add <code>&pbr=1</code> for PBR maps</li>
    <li>Add <code>&compress=1</code> for optimized PNGs</li>
    <li>One-click export of complete game-ready texture packs</li>
    </ul>

    <h3>5. PBR Generation</h3>
    <p>Opt-in with <code>&pbr=1</code> on /batch or /profile/gameasset. Generates:</p>
    <ul>
    <li>Normal map: Approximated from edges using Sobel filters (RGB PNG).</li>
    <li>Roughness map: Simple inverted grayscale (RGB PNG).</li>
    <li>Note: These are basic approximations; for production, use dedicated tools.</li>
    <li>PNG outputs can be compressed with <code>&compress=1</code></li>
    </ul>

    <h3>6. Validate Endpoint <code>/validate</code></h3>
    <p>Checks input image for common texture issues. Same input as other endpoints.</p>
    <ul>
    <li>Returns JSON: {ok: true/false, issues: [...], warnings: [...]}</li>
    <li>Checks: Alpha problems, potential seams (for tiling), high variance (artifacts/noise).</li>
    </ul>

    <h3>7. Output</h3>
    <ul>
    <li>All outputs saved under <code>output/</code> folder</li>
    <li>Filenames include original name, metadata, suffix (e.g., _base, _normal), resolution, and unique ID</li>
    <li>Example: <code>griffonwoman_medieval_elf_base_1024x1024_abcd1234.png</code></li>
    <li>Publicly accessible via <code>/files/&lt;filename&gt;</code></li>
    </ul>

    <h3>Typical Workflow</h3>
    <ol>
    <li>Upload <code>griffonwoman.jpg</code> (2000×1600)</li>
    <li>Call: <code>POST /profile/gameasset?pack=medieval&race=elf&label=sword&pbr=1&compress=1</code></li>
    <li>Receive compressed PNG+TGA base outputs plus normal/roughness for 512, 1024, 2048, 4096</li>
    <li>Call: <code>POST /validate</code> to check for issues</li>
    <li>Each file has metadata tags, direct URL, and file size info</li>
    </ol>

    <hr style="margin:28px 0;">

    <h3>API</h3>
    <pre style="background:#f7f7f7; padding:12px; border-radius:8px;">
    POST /resize?size=512|1024|2048|4096|pow2
        &mode=fit|crop|stretch
        &format=png|tga
        &pack=&race=&label=
        &download=0|1
        &compress=0|1

    POST /batch?sizes=512,1024,2048,4096&formats=png,tga&mode=fit&pbr=0|1&compress=0|1

    POST /profile/gameasset   (PNG+TGA, sizes 512–4096)&pbr=0|1&compress=0|1
    POST /validate            (Checks for issues; same body)

    Body for all:
    - multipart/form-data with file field "file", or
    - JSON: {"imageUrl":"https://yourcdn.com/image.png"}
    </pre>
  </body>
</html>
"""

@app.get("/favicon.ico")
def favicon():
    return "", 204  # empty response, no error


@app.get("/files/<path:filename>")
def files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.post("/batch")
def batch():
    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err

    sizes_param   = request.args.get("sizes")
    formats_param = request.args.get("formats", "png")
    mode          = request.args.get("mode", "fit")
    pow2          = request.args.get("pow2", "0") == "1"
    pack          = request.args.get("pack")
    race          = request.args.get("race")
    label         = request.args.get("label")
    pbr           = request.args.get("pbr", "0") == "1"  # New: Opt-in PBR

    if sizes_param:
        try:
            sizes = [int(s) for s in sizes_param.split(",") if s.strip()]
        except ValueError:
            return jsonify({"error": "Invalid sizes. Example: sizes=512,1024,2048"}), 400
    else:
        sizes = [512, 1024, 2048, 4096]
    if pow2:
        sizes = [to_pow2(s) for s in sizes]

    valid_ext = {"png", "tga"}
    formats = [f for f in (s.strip().lower() for s in formats_param.split(",")) if f in valid_ext]
    if not formats:
        return jsonify({"error": "No valid formats. Use formats=png,tga"}), 400

    compress = request.args.get("compress", "0") == "1"

    results = generate_batch_variants(img, sizes, formats, mode, pack, race, label, original_name=original_name, pbr=pbr, compress=compress)
    return jsonify({"ok": True, "results": results})

@app.post("/profile/gameasset")
def profile_gameasset():
    # get image + original name (or an error string)
    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err

    sizes_param   = request.args.get("sizes", "512,1024,2048,4096")
    formats_param = request.args.get("formats", "png,tga")
    mode          = request.args.get("mode", "fit")
    pow2          = request.args.get("pow2", "0") == "1"
    pack          = request.args.get("pack")
    race          = request.args.get("race")
    label         = request.args.get("label")
    pbr           = request.args.get("pbr", "0") == "1"  # New: Opt-in PBR

    # sizes (with validation)
    try:
        sizes = [int(s) for s in sizes_param.split(",") if s.strip()]
    except ValueError:
        return jsonify({"error": "Invalid sizes. Example: sizes=512,1024,2048"}), 400
    if pow2:
        sizes = [to_pow2(s) for s in sizes]

    # formats (PNG/TGA only, with validation)
    valid_ext = {"png", "tga"}
    formats = [f for f in (s.strip().lower() for s in formats_param.split(",")) if f and f in valid_ext]
    if not formats:
        return jsonify({"error": "No valid formats. Use formats=png,tga"}), 400

    compress = request.args.get("compress", "0") == "1"

    results = generate_batch_variants(
        img, sizes, formats, mode, pack, race, label, original_name=original_name, pbr=pbr, compress=compress
    )

    return jsonify({"ok": True, "results": results})

# New Endpoint: /validate
@app.post("/validate")
def validate_endpoint():
    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400

    issues = []
    array = np.array(img)

    # Alpha issues
    if img.mode == 'RGBA':
        alpha = array[:, :, 3]
        if np.all(alpha == 255):
            issues.append("Texture has alpha channel but is fully opaque – consider converting to RGB for optimization.")
        elif np.any(alpha < 255) and np.mean(alpha[alpha < 255]) < 128:
            issues.append("Semi-transparent areas detected – ensure engine supports alpha blending.")

    # Seams check (for tileable textures: compare edges)
    height, width = array.shape[:2]
    if height > 1 and width > 1:
        left = array[:, 0, :3]  # RGB
        right = array[:, -1, :3]
        if np.mean(np.abs(left - right)) > 20:  # Adjustable threshold
            issues.append("Potential vertical seam: Left and right edges differ significantly – may not tile seamlessly.")

        top = array[0, :, :3]
        bottom = array[-1, :, :3]
        if np.mean(np.abs(top - bottom)) > 20:
            issues.append("Potential horizontal seam: Top and bottom edges differ significantly – may not tile seamlessly.")

    # Basic artifacts: High variance in edges (compression noise proxy)
    gray = np.array(img.convert('L'))
    variance = np.var(gray)
    if variance > 10000:  # Heuristic; tune based on tests
        issues.append("High image variance detected – possible compression artifacts or noise; consider smoothing.")

    return jsonify({
        "ok": len(issues) == 0,
        "issues": issues,
        "warnings": [] if issues else ["Texture passes basic validation."]
    })

@app.post("/package")
def package_endpoint():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    img, original_name_or_err = load_image_from_request()
    if img is None:
        logger.error(f"Validation failed: {original_name_or_err}")
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err
    logger.debug(f"Image validated successfully: {original_name}")

    # Validate parameters
    pack = request.args.get("pack")
    race = request.args.get("race")
    label = request.args.get("label")
    pbr = request.args.get("pbr", "1") == "1"
    compress = request.args.get("compress", "0") == "1"
    if pack and not re.match(r"^[a-zA-Z0-9_-]+$", pack):
        return jsonify({"error": "Invalid pack name. Use alphanumeric, hyphen, or underscore."}), 400
    if race and not re.match(r"^[a-zA-Z0-9_-]+$", race):
        return jsonify({"error": "Invalid race name. Use alphanumeric, hyphen, or underscore."}), 400
    if label and not re.match(r"^[a-zA-Z0-9_-]+$", label):
        return jsonify({"error": "Invalid label. Use alphanumeric, hyphen, or underscore."}), 400
    # Additional length validation
    for param, name in [(pack, "pack"), (race, "race"), (label, "label")]:
        if param and len(param) > 30:
            return jsonify({"error": f"{name} too long: max 30 characters"}), 400

    # Validate sizes and formats
    sizes_param = request.args.get("sizes", "512,1024,2048,4096")
    try:
        sizes = [int(s) for s in sizes_param.split(",") if s.strip()]
        if not all(64 <= s <= 8192 for s in sizes):
            return jsonify({"error": "Sizes must be between 64 and 8192"}), 400
    except ValueError:
        return jsonify({"error": "Invalid sizes. Example: sizes=512,1024"}), 400
    formats_param = request.args.get("formats", "png,tga")
    formats = [f.strip().lower() for f in formats_param.split(",")]
    if not all(f in {"png", "tga"} for f in formats):
        return jsonify({"error": "Formats must be png or tga"}), 400

    # Generate variants and store file paths
    logger.debug(f"Generating batch variants with pbr={pbr}, compress={compress}")
    results = generate_batch_variants(img, sizes, formats, "fit", pack, race, label, original_name, pbr=pbr, compress=compress)
    temp_files = []
    
    # Collect base image paths
    for fmt in results:
        if fmt == 'pbr':
            continue
        for size in results[fmt]:
            file_info = results[fmt][size]
            file_path = file_info["url"].replace(f"{request.host_url.rstrip('/')}/files/", "")
            temp_files.append(os.path.join(OUTPUT_DIR, file_path))
            logger.debug(f"Added to temp_files: {file_path} (format: {fmt}, size: {size})")
    
    # Collect PBR map paths
    if pbr and 'pbr' in results:
        for size in results['pbr']:
            for map_type in ['normal', 'roughness']:
                file_info = results['pbr'][size][map_type]
                file_path = file_info["url"].replace(f"{request.host_url.rstrip('/')}/files/", "")
                temp_files.append(os.path.join(OUTPUT_DIR, file_path))
                logger.debug(f"Added to temp_files: {file_path} (pbr map: {map_type}, size: {size})")
    else:
        logger.debug("No PBR files generated")

    # Create README
    readme_content = f"Asset Pack: {original_name}\nGenerated by Assetgineer\nSizes: {','.join(map(str, sizes))}\nFormats: {','.join(formats)}{' + PBR' if pbr else ''}\nCompression: {'Yes' if compress else 'No'}\nGenerated on: 07/Sep/2025"
    readme_path = os.path.join(OUTPUT_DIR, f"README_{uuid.uuid4().hex}.txt")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    temp_files.append(readme_path)
    logger.debug(f"Created README: {readme_path}")

    # Create zip in memory
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zipf:
        for file_path in temp_files:
            if os.path.exists(file_path):
                arcname = os.path.join("Textures" if "_base_" in file_path else "PBR", os.path.basename(file_path)) if file_path != readme_path else "README.txt"
                zipf.write(file_path, arcname)
                logger.debug(f"Added to zip: {file_path} as {arcname}")
            else:
                logger.warning(f"File not found for zipping: {file_path}")
    
    # Clean up temporary files
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up: {file_path}")

    # Serve the zip from memory
    zip_buffer.seek(0)
    zip_name = f"asset_pack_{sanitize(original_name)}_{uuid.uuid4().hex}.zip"
    logger.debug(f"Serving zip: {zip_name}")
    response = send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_name
    )
    logger.debug("Zip served successfully")
    return response

# Add a static file serving endpoint (unchanged)
@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)