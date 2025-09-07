from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageOps, ImageFilter
import io, os, uuid, requests
import re
from flask_cors import CORS
import numpy as np
from scipy.ndimage import convolve
from zipfile import ZipFile, ZIP_DEFLATED
import multiprocessing
from PIL import Image, UnidentifiedImageError, ImageOps, ImageFilter, ImageEnhance
import socket
import gc

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
        if f.content_length and f.content_length > 10 * 1024 * 1024:
            size_mb = round(f.content_length / (1024 * 1024), 2)
            logger.error(f"File size {f.content_length} bytes ({size_mb} MB) exceeds 10MB limit")
            return None, f"File size ({size_mb} MB) exceeds 10MB limit"
        try:
            if f.content_length == 0:
                logger.warning("File has zero content length, attempting to read stream")
                f.stream.seek(0)
            img = Image.open(f.stream)
            img.verify()
            f.stream.seek(0)
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
        except PIL.UnidentifiedImageError:
            logger.error("Corrupted or unreadable image file")
            return None, "Corrupted or unreadable image file"
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
        img = Image.open(io.BytesIO(resp.content))
        img.verify()
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {str(e)}")
        return None, f"Failed to download image from {url}: {str(e)}"
    except UnidentifiedImageError:
        logger.error("Corrupted or unreadable image from URL")
        return None, "Corrupted or unreadable image from URL"
    except Exception as e:
        logger.error(f"Failed to process image from {url}: {str(e)}")
        return None, f"Failed to process image from {url}: {str(e)}"    

def load_image_from_url(url):
    import logging
    logger = logging.getLogger(__name__)
    allowed_types = {'.png', '.jpg', '.jpeg', '.tga'}
    original_getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        """
        This is a wrapper for socket.getaddrinfo that forces IPv4 resolution.
        It takes all the same arguments as the original function but ignores the
        'family' argument and substitutes socket.AF_INET.
        """
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    # Apply the monkey patch
    socket.getaddrinfo = ipv4_getaddrinfo

    try:
        logger.debug(f"Fetching image from URL: {url}")
        resp = requests.get(url, timeout=30)  # Increased for safety
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.verify()
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {str(e)}")
        return None, f"Failed to download image from {url}: {str(e)}"
    except UnidentifiedImageError:
        logger.error("Corrupted or unreadable image from URL")
        return None, "Corrupted or unreadable image from URL"
    except Exception as e:
        logger.error(f"Failed to process image from {url}: {str(e)}")
        return None, f"Failed to process image from {url}: {str(e)}"
    finally:
        # Restore original to avoid side effects
        socket.getaddrinfo = original_getaddrinfo

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

def save_variant(img, original_name, size, label='base', file_format='png', pbr_type=None):
    """
    Saves an image variant with a unique name incorporating its properties.
    This version ensures the filename is always unique across all variants and sizes.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        size_str = f"{size[0]}x{size[1]}"
        
        # Construct a descriptive, unique filename
        base_name = os.path.splitext(original_name)[0]
        pbr_suffix = f"_{pbr_type}" if pbr_type else ""
        
        # CRITICAL: The label (e.g., 'earthy') and a unique ID are now part of the filename.
        # This prevents all name collisions at the source.
        unique_id = uuid.uuid4().hex[:12]
        fname = f"{base_name}_{label}{pbr_suffix}_{size_str}_{unique_id}.{file_format}"

        fpath = os.path.join(OUTPUT_DIR, fname)
        
        # Handle TGA saving with correct alpha channel detection
        if file_format == 'tga':
            if img.mode in ['RGBA', 'LA']:
                img.save(fpath, format='TGA', tga_24_bpp=False) # 32-bit with alpha
            else:
                img.save(fpath, format='TGA', tga_24_bpp=True)  # 24-bit without alpha
        else: # Default to PNG
            img.save(fpath, format='PNG')
            
        logger.debug(f"Successfully saved variant to {fpath}")
        return {'status': 'success', 'path': fpath}
    except Exception as e:
        logger.error(f"Failed to save variant {original_name} ({label}, {size_str}): {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

def generate_normal_map(base_img: Image.Image):
    gray = base_img.convert('L')
    array = np.array(gray, dtype=float)
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    dx = convolve(array, sobel_x, mode='constant', cval=0.0)
    dy = convolve(array, sobel_y, mode='constant', cval=0.0)
    
    dz = np.ones_like(dx) * 10.0
    normal = np.dstack((-dx, -dy, dz))
    norm = np.linalg.norm(normal, axis=2)[:,:,np.newaxis]
    normal = (normal / norm) * 0.5 + 0.5
    normal = np.clip(normal * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(normal, 'RGB')

def generate_roughness_map(base_img: Image.Image):
    gray = base_img.convert('L')
    inverted = ImageOps.invert(gray)
    return inverted.convert('RGB')

def process_variant(args):
    import logging
    logger = logging.getLogger(__name__)
    img, size, fmt, mode, pack, race, label, original_name, pbr, compress, host_url = args
    # Handle scalar or tuple: if scalar, square; else (w, h)
    if isinstance(size, tuple):
        w, h = size
    else:
        w = h = size  # Square fallback
    if mode == "stretch":
        base_img = img.resize((w, h), Image.Resampling.LANCZOS)
    elif mode == "crop":
        scale = max(w / img.width, h / img.height)
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        base_img = scaled.crop((x1, y1, x1 + w, y1 + h))
    else:  # fit
        fitted = ImageOps.contain(img, (w, h), Image.Resampling.LANCZOS)
        base_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        x = (w - fitted.width) // 2
        y = (h - fitted.height) // 2
        base_img.paste(fitted, (x, y))

    fname, path, size_bytes = save_variant(
        base_img, w, h, fmt,  # Pass w, h
        pack=pack, race=race, label=label, original_name=original_name, suffix="base", compress=compress
    )
    size_str = f"{w}x{h}"
    result = {
        size_str: {
            "url": f"{host_url.rstrip('/')}/files/{fname}",
            "bytes": size_bytes,
            "mb": round(size_bytes / (1024 * 1024), 3),
        }
    }

    if pbr:
        normal_img = generate_normal_map(base_img)
        roughness_img = generate_roughness_map(base_img)
        n_fname, n_path, n_bytes = save_variant(
            normal_img, w, h, "png",  # Pass w, h
            pack=pack, race=race, label=label, original_name=original_name, suffix="normal", compress=compress
        )
        r_fname, r_path, r_bytes = save_variant(
            roughness_img, w, h, "png",  # Pass w, h
            pack=pack, race=race, label=label, original_name=original_name, suffix="roughness", compress=compress
        )
        result['pbr'] = {
            size_str: {  # Use size_str
                "normal": {"url": f"{host_url.rstrip('/')}/files/{n_fname}", "bytes": n_bytes, "mb": round(n_bytes / (1024 * 1024), 3)},
                "roughness": {"url": f"{host_url.rstrip('/')}/files/{r_fname}", "bytes": r_bytes, "mb": round(r_bytes / (1024 * 1024), 3)}
            }
        }
        if pbr:
            gc.collect()
    return fmt, result

def generate_batch_variants(img, sizes, formats, mode, pack, race, label, original_name=None, pbr=False, compress=False):
    import logging
    logger = logging.getLogger(__name__)
    from multiprocessing import Pool
    with Pool(processes=min(multiprocessing.cpu_count() - 1, len(sizes) * len(formats))) as pool:
        args = [(img, size, fmt, mode, pack, race, label, original_name, pbr and fmt == 'png', compress, request.host_url) for size in sizes for fmt in formats]
        results = pool.map(process_variant, args)
        gc.collect()
        if pbr:
            gc.collect()
    final_results = {f: {} for f in formats}
    pbr_results = {}  # Top-level PBR dict
    if pbr:
        # Initialize with scalar str(size) for compatibility, but update will use actual size_str from result
        pbr_results = {str(size): {} for size in sizes}
    for fmt, result in results:
        if fmt != 'pbr':
            final_results[fmt].update(result)
        if 'pbr' in result:
            size_str = list(result.keys())[0]  # e.g., '512x512' or '512x256'
            # Ensure pbr_results has the key (create if missing for non-square)
            if size_str not in pbr_results:
                pbr_results[size_str] = {}
            pbr_results[size_str].update(result['pbr'][size_str])
    if pbr:
        final_results['pbr'] = pbr_results
    return final_results

@app.post("/resize")
def resize_endpoint():
    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err

    size_param = request.args.get("size")
    download = request.args.get("download", "0") == "1"
    width = request.args.get("width")
    height = request.args.get("height")
    ratio = request.args.get("ratio")
    long_ = request.args.get("long")
    mode = request.args.get("mode", "fit")
    pow2 = request.args.get("pow2", "0") == "1"
    fmt = (request.args.get("format") or "png").lower()
    pack = request.args.get("pack")
    race = request.args.get("race")
    label = request.args.get("label")

    if fmt not in ("png", "tga"):
        return jsonify({"error": "Only png or tga are supported for asset outputs."}), 400
    if pack and len(pack) > 50:
        return jsonify({"error": f"Pack name exceeds 50 characters: {len(pack)}."}), 400
    if pack and not re.match(r"^[a-zA-Z0-9_-]+$", pack):
        return jsonify({"error": "Invalid pack name. Use alphanumeric, hyphen, or underscore."}), 400
    if race and len(race) > 50:
        return jsonify({"error": f"Race name exceeds 50 characters: {len(race)}."}), 400
    if race and not re.match(r"^[a-zA-Z0-9_-]+$", race):
        return jsonify({"error": "Invalid race name. Use alphanumeric, hyphen, or underscore."}), 400
    if label and len(label) > 50:
        return jsonify({"error": f"Label exceeds 50 characters: {len(label)}."}), 400
    if label and not re.match(r"^[a-zA-Z0-9_-]+$", label):
        return jsonify({"error": "Invalid label. Use alphanumeric, hyphen, or underscore."}), 400

    target_w = target_h = None
    if width and height:
        try:
            target_w, target_h = int(width), int(height)
            if target_w < 64 or target_h < 64:
                return jsonify({"error": "Width/height must be at least 64."}), 400
            if target_w > 8192 or target_h > 8192:
                return jsonify({"error": "Width/height must not exceed 8192."}), 400
        except ValueError:
            return jsonify({"error": "width/height must be integers"}), 400
    elif ratio and long_:
        rt = parse_ratio(ratio)
        if not rt:
            return jsonify({"error": "Invalid ratio. Use ratio=16:9"}), 400
        try:
            long_side = int(long_)
            if long_side < 64 or long_side > 8192:
                return jsonify({"error": "Long side must be between 64 and 8192."}), 400
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
                if s < 64 or s > 8192:
                    return jsonify({"error": "Size must be between 64 and 8192."}), 400
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
    else:
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
      <p><label>Upload image: <input type="file" name="file" accept=".png,.jpg,.jpeg,.tga" required></label></p>
      <p><label>Pack name: <input type="text" name="pack" required pattern="[a-zA-Z0-9_-]+" title="Use alphanumeric, hyphen, or underscore"></label></p>
      <p><label>Race (optional): <input type="text" name="race" pattern="[a-zA-Z0-9_-]+" title="Use alphanumeric, hyphen, or underscore"></label></p>
      <p><label>Label (optional): <input type="text" name="label" pattern="[a-zA-Z0-9_-]+" title="Use alphanumeric, hyphen, or underscore"></label></p>
      <p><label>OR Image URL: <input type="url" name="imageUrl" placeholder="https://example.com/image.png" style="width:100%;"></label></p>
      <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px;">
        <button formaction="/resize?size=512" formmethod="post" formtarget="_blank">Resize to 512</button>
        <button formaction="/resize?size=1024" formmethod="post" formtarget="_blank">Resize to 1024</button>
        <button formaction="/resize?size=2048" formmethod="post" formtarget="_blank">Resize to 2048</button>
        <button formaction="/resize?size=4096" formmethod="post" formtarget="_blank">Resize to 4096</button>
        <button formaction="/resize?size=pow2" formmethod="post" formtarget="_blank">Resize to POW2</button>
        <button formaction="/batch" formmethod="post" formtarget="_blank">Batch</button>
        <button formaction="/profile/gameasset" formmethod="post" formtarget="_blank" style="background:#222; color:#fff;">GameAsset Pack</button>
        <button formaction="/profile/gameasset?pbr=1" formmethod="post" formtarget="_blank" style="background:#444; color:#fff;">GameAsset Pack with PBR</button>
        <button formaction="/validate" formmethod="post" formtarget="_blank" style="background:#007bff; color:#fff;">Validate Image</button>
        <button formaction="/package?pbr=1" formmethod="post" formtarget="_blank" style="background:#28a745; color:#fff;">Download Pack as Zip</button>
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
    <li>You press <b>GameAsset Pack with PBR</b>.</li>
    <li>You instantly get 4 versions: 512, 1024, 2048, and 4096.</li>
    <li>Each version includes base PNG/TGA plus normal and roughness maps, with direct download links.</li>
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
    return "", 204

@app.get("/files/<path:filename>")
def files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.post("/batch")
def batch():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err

    sizes_param = request.args.get("sizes")
    pack = request.form.get("pack") or request.args.get("pack")
    race = request.form.get("race") or request.args.get("race")
    label = request.form.get("label") or request.args.get("label")
    formats_param = request.args.get("formats", "png")
    mode = request.args.get("mode", "fit")
    pow2 = request.args.get("pow2", "0") == "1"
    pbr = request.args.get("pbr", "0") == "1"
    if not pack:
        return jsonify({"error": "Pack name is required."}), 400
    if len(pack) > 50:
        return jsonify({"error": f"Pack name exceeds 50 characters: {len(pack)}."}), 400
    if not re.match(r"^[a-zA-Z0-9_-]+$", pack):
        return jsonify({"error": "Invalid pack name. Use alphanumeric, hyphen, or underscore."}), 400
    if race and len(race) > 50:
        return jsonify({"error": f"Race name exceeds 50 characters: {len(race)}."}), 400
    if race and not re.match(r"^[a-zA-Z0-9_-]+$", race):
        return jsonify({"error": "Invalid race name. Use alphanumeric, hyphen, or underscore."}), 400
    if label and len(label) > 50:
        return jsonify({"error": f"Label exceeds 50 characters: {len(label)}."}), 400
    if label and not re.match(r"^[a-zA-Z0-9_-]+$", label):
        return jsonify({"error": "Invalid label. Use alphanumeric, hyphen, or underscore."}), 400
    if sizes_param:
        try:
            sizes = [int(s) for s in sizes_param.split(",") if s.strip()]
            if not all(64 <= s <= 8192 for s in sizes):
                return jsonify({"error": "Sizes must be between 64 and 8192."}), 400
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
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err

    sizes_param = request.args.get("sizes", "512,1024,2048,4096")
    pack = request.form.get("pack") or request.args.get("pack")
    race = request.form.get("race") or request.args.get("race")
    label = request.form.get("label") or request.args.get("label")
    formats_param = request.args.get("formats", "png,tga")
    mode = request.args.get("mode", "fit")
    pow2 = request.args.get("pow2", "0") == "1"
    pbr = request.args.get("pbr", "0") == "1"
    if not pack:
        return jsonify({"error": "Pack name is required."}), 400
    if len(pack) > 50:
        return jsonify({"error": f"Pack name exceeds 50 characters: {len(pack)}."}), 400
    if not re.match(r"^[a-zA-Z0-9_-]+$", pack):
        return jsonify({"error": "Invalid pack name. Use alphanumeric, hyphen, or underscore."}), 400
    if race and len(race) > 50:
        return jsonify({"error": f"Race name exceeds 50 characters: {len(race)}."}), 400
    if race and not re.match(r"^[a-zA-Z0-9_-]+$", race):
        return jsonify({"error": "Invalid race name. Use alphanumeric, hyphen, or underscore."}), 400
    if label and len(label) > 50:
        return jsonify({"error": f"Label exceeds 50 characters: {len(label)}."}), 400
    if label and not re.match(r"^[a-zA-Z0-9_-]+$", label):
        return jsonify({"error": "Invalid label. Use alphanumeric, hyphen, or underscore."}), 400
    try:
        sizes = [int(s) for s in sizes_param.split(",") if s.strip()]
        if not all(64 <= s <= 8192 for s in sizes):
            return jsonify({"error": "Sizes must be between 64 and 8192."}), 400
    except ValueError:
        return jsonify({"error": "Invalid sizes. Example: sizes=512,1024,2048"}), 400
    if pow2:
        sizes = [to_pow2(s) for s in sizes]

    valid_ext = {"png", "tga"}
    formats = [f for f in (s.strip().lower() for s in formats_param.split(",")) if f and f in valid_ext]
    if not formats:
        return jsonify({"error": "No valid formats. Use formats=png,tga"}), 400

    compress = request.args.get("compress", "0") == "1"

    results = generate_batch_variants(
        img, sizes, formats, mode, pack, race, label, original_name=original_name, pbr=pbr, compress=compress
    )

    return jsonify({"ok": True, "results": results})

@app.post("/validate")
def validate_endpoint():
    img, original_name_or_err = load_image_from_request()
    if isinstance(original_name_or_err, str) and original_name_or_err.startswith("Failed"):
        return jsonify({"error": original_name_or_err}), 400
    if img is None:
        return jsonify({"error": original_name_or_err}), 400

    issues = []
    array = np.array(img)

    if img.mode == 'RGBA':
        alpha = array[:, :, 3]
        if np.all(alpha == 255):
            issues.append("Texture has alpha channel but is fully opaque – consider converting to RGB for optimization.")
        elif np.any(alpha < 255) and np.mean(alpha[alpha < 255]) < 128:
            issues.append("Semi-transparent areas detected – ensure engine supports alpha blending.")
        transparent_ratio = np.sum(alpha == 0) / alpha.size
        if transparent_ratio > 0.9:
            issues.append("Over 90% of the image is transparent – ensure this is intentional.")

    height, width = array.shape[:2]
    if height > 1 and width > 1:
        left = array[:, 0, :3]
        right = array[:, -1, :3]
        if np.mean(np.abs(left - right)) > 20:
            issues.append("Potential vertical seam: Left and right edges differ significantly – may not tile seamlessly.")
        top = array[0, :, :3]
        bottom = array[-1, :, :3]
        if np.mean(np.abs(top - bottom)) > 20:
            issues.append("Potential horizontal seam: Top and bottom edges differ significantly – may not tile seamlessly.")

    gray = np.array(img.convert('L'))
    variance = np.var(gray)
    if variance > 10000:
        issues.append("High image variance detected – possible compression artifacts or noise; consider smoothing.")

    if img.width != to_pow2(img.width) or img.height != to_pow2(img.height):
        issues.append("Non-power-of-two dimensions detected – some game engines prefer power-of-two textures.")

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

    pack = request.form.get("pack") or request.args.get("pack")
    race = request.form.get("race") or request.args.get("race")
    label = request.form.get("label") or request.args.get("label")
    pbr = request.args.get("pbr", "1") == "1"
    compress = request.args.get("compress", "0") == "1"
    if not pack:
        logger.error("Pack name is required")
        return jsonify({"error": "Pack name is required."}), 400
    if len(pack) > 50:
        logger.error(f"Pack name too long: {len(pack)} characters")
        return jsonify({"error": f"Pack name exceeds 50 characters: {len(pack)}."}), 400
    if not re.match(r"^[a-zA-Z0-9_-]+$", pack):
        return jsonify({"error": "Invalid pack name. Use alphanumeric, hyphen, or underscore."}), 400
    if race and len(race) > 50:
        logger.error(f"Race name too long: {len(race)} characters")
        return jsonify({"error": f"Race name exceeds 50 characters: {len(race)}."}), 400
    if race and not re.match(r"^[a-zA-Z0-9_-]+$", race):
        return jsonify({"error": "Invalid race name. Use alphanumeric, hyphen, or underscore."}), 400
    if label and len(label) > 50:
        logger.error(f"Label too long: {len(label)} characters")
        return jsonify({"error": f"Label exceeds 50 characters: {len(label)}."}), 400
    if label and not re.match(r"^[a-zA-Z0-9_-]+$", label):
        return jsonify({"error": "Invalid label. Use alphanumeric, hyphen, or underscore."}), 400

    logger.debug(f"Generating batch variants with pbr={pbr}, compress={compress}")
    results = generate_batch_variants(img, [512, 1024, 2048, 4096], ["png", "tga"], "fit", pack, race, label, original_name, pbr=pbr, compress=compress)
    temp_files = []
    
    # Extract base files from all formats (png, tga)
    for fmt in ['png', 'tga']:
        if fmt in results:
            for size_key in results[fmt]:  # e.g., '512x512'
                file_info = results[fmt][size_key]
                base_url = file_info.get("url")
                if base_url:
                    # Robust fname extraction
                    if 'files/' in base_url:
                        fname = base_url.split('files/')[-1]
                    else:
                        fname = os.path.basename(base_url)
                    file_path = os.path.join(OUTPUT_DIR, fname)
                    if os.path.exists(file_path):
                        temp_files.append(file_path)
                        logger.debug(f"Added base: {file_path} (fmt: {fmt}, key: {size_key})")

    # Extract PBR from top-level 'pbr' key
    if pbr and 'pbr' in results:
        for size_key in results['pbr']:  # e.g., '512x512'
            for map_type in ['normal', 'roughness']:
                if map_type in results['pbr'][size_key]:
                    file_info = results['pbr'][size_key][map_type]
                    pbr_url = file_info.get("url")
                    if pbr_url:
                        if 'files/' in pbr_url:
                            fname = pbr_url.split('files/')[-1]
                        else:
                            fname = os.path.basename(pbr_url)
                        file_path = os.path.join(OUTPUT_DIR, fname)
                        if os.path.exists(file_path):
                            temp_files.append(file_path)
                            logger.debug(f"Added PBR: {file_path} (type: {map_type}, key: {size_key})")
    else:
        logger.debug("No PBR generated or 'pbr' key missing")

    readme_content = f"Asset Pack: {original_name}\nGenerated by Assetgineer\nSizes: 512, 1024, 2048, 4096\nFormats: PNG, TGA{' + PBR' if pbr else ''}\nCompression: {'Yes' if compress else 'No'}\nGenerated on: 07/Sep/2025"
    readme_path = os.path.join(OUTPUT_DIR, f"README_{uuid.uuid4().hex}.txt")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    temp_files.append(readme_path)
    logger.debug(f"Created README: {readme_path}")

    logger.debug(f"Total temp_files before zip: {len(temp_files)}")

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zipf:
        for file_path in temp_files:
            if os.path.exists(file_path):
                arcname = os.path.join("Textures" if "_base_" in file_path else "PBR", os.path.basename(file_path)) if file_path != readme_path else "README.txt"
                zipf.write(file_path, arcname)
                logger.debug(f"Added to zip: {file_path} as {arcname}")
            else:
                logger.warning(f"File not found for zipping: {file_path}")
    
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up: {file_path}")

    if zip_buffer.tell() == 0:
        logger.error("Zip buffer is empty, no files processed")
        return jsonify({"error": "Failed to generate zip file"}), 500
    zip_buffer.seek(0)
    zip_name = f"asset_pack_{sanitize(original_name)}_{uuid.uuid4().hex}.zip"
    logger.debug(f"Serving zip: {zip_name}")
    response = send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_name,
        conditional=True
    )
    response.headers['Content-Disposition'] = f'attachment; filename={zip_name}'
    logger.debug("Zip served successfully")
    return response

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True, download_name=filename)

# In-memory status storage (prototype only; use Redis in production)
statuses = {}  # {pack_id: {'status': 'processing', 'progress': 0}}

@app.post("/generate_pack")
def generate_pack():
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("Processing /generate_pack request")    
    api_key = request.headers.get('X-API-Key')
    if api_key != 'mock_api_key_123':
        return jsonify({'error': 'Unauthorized: Invalid API key'}), 401

    if not request.is_json:
        return jsonify({'error': 'JSON payload required'}), 400

    data = request.get_json()
    image_urls = data.get('imageUrls', [])
    aspect_ratio = data.get('aspectRatio', '1:1')
    sizes = data.get('sizes', [512, 1024])
    formats = data.get('formats', ['png'])
    variants_data = data.get('variants', {'count': 1, 'options': ['default']})
    pbr_maps = data.get('pbrMaps', False)
    package_name = data.get('packageName', 'asset-pack')

    if not image_urls: return jsonify({'error': 'At least one imageUrl required'}), 400
    
    all_temp_files = []
    file_manifest = []

    try:
        for url in image_urls:
            img, original_name = load_image_from_url(url)
            if img is None:
                raise ValueError(f"Failed to load image from URL: {url}")

            for variant_name in variants_data.get('options', ['default']):
                mod_img = img
                if variant_name == 'earthy': mod_img = ImageEnhance.Color(img).enhance(0.7)
                elif variant_name == 'vibrant': mod_img = ImageEnhance.Color(img).enhance(1.3)
                elif variant_name == 'muted': mod_img = ImageEnhance.Color(img).enhance(0.5)

                for size_val in sizes:
                    resized_img = mod_img.resize((size_val, size_val), Image.LANCZOS)
                    
                    for fmt in formats:
                        result = save_variant(resized_img, original_name, (size_val, size_val), label=variant_name, file_format=fmt)
                        if result['status'] != 'success':
                            raise IOError(f"Failed to save base texture: {result.get('message', 'Unknown error')}")
                        all_temp_files.append(result['path'])
                        file_manifest.append({'name': os.path.basename(result['path']), 'type': 'Texture', 'variant': variant_name})

                    if pbr_maps:
                        # Generate PBR maps using existing functions
                        normal_img = generate_normal_map(resized_img)
                        roughness_img = generate_roughness_map(resized_img)
                        pbr_results = {'normal': normal_img, 'roughness': roughness_img}
                        
                        for pbr_type, pbr_img in pbr_results.items():
                            for fmt in formats:
                                result = save_variant(pbr_img, original_name, (size_val, size_val), label=variant_name, file_format=fmt, pbr_type=pbr_type)
                                if result['status'] != 'success':
                                    raise IOError(f"Failed to save PBR map: {result.get('message', 'Unknown error')}")
                                all_temp_files.append(result['path'])
                                file_manifest.append({'name': os.path.basename(result['path']), 'type': pbr_type.capitalize(), 'variant': variant_name})
        
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zipf:
            for file_path in all_temp_files:
                filename = os.path.basename(file_path)
                variant_subfolder = 'default'
                for v_opt in variants_data.get('options', ['default']):
                    if f"_{v_opt}_" in filename:
                        variant_subfolder = v_opt
                        break
                
                arc_dir = "PBR" if "_normal_" in filename or "_roughness_" in filename else "Textures"
                arcname = os.path.join(arc_dir, variant_subfolder, filename)
                zipf.write(file_path, arcname)
            
            readme_content = f"Asset Pack: {package_name}\nGenerated by Assetgineer."
            zipf.writestr("README.txt", readme_content)

        zip_buffer.seek(0)
        zip_name = f"pack_{uuid.uuid4().hex[:12]}.zip"
        with open(os.path.join(OUTPUT_DIR, zip_name), 'wb') as f:
            f.write(zip_buffer.getvalue())

        return jsonify({
            'status': 'success',
            'zipUrl': f"/files/{zip_name}",
            'packageName': package_name,
            'fileCount': len(all_temp_files),
            'manifest': file_manifest
        }), 200

    except Exception as e:
        logger.error(f"Critical error during pack generation: {e}", exc_info=True)
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500
    finally:
        for f in all_temp_files:
            try:
                if os.path.exists(f): os.remove(f)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {f}: {e}")

@app.get("/status/<pack_id>")
def get_status(pack_id):
    import logging
    logger = logging.getLogger(__name__)
    status = statuses.get(pack_id, {'status': 'not_found', 'progress': 0})
    logger.debug(f"Status check for pack_id {pack_id}: {status}")
    return jsonify(status)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)