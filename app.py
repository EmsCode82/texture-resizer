from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageOps
import io, os, uuid, requests
import re
from flask_cors import CORS

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
    if 'file' in request.files:
        f = request.files['file']
        img = Image.open(f.stream).convert('RGBA')
        return img, f.filename  # keep original filename

    url = None
    if request.is_json:
        url = (request.json or {}).get('imageUrl')
    if not url:
        url = request.form.get('imageUrl')
    if not url:
        return None, "Provide an uploaded file (field 'file') or 'imageUrl'."

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGBA')
        name = os.path.basename(url.split("?")[0])  # try to keep original name from URL
        return img, name
    except Exception as e:
        return None, f"Failed to download image: {e}"

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
                 pack=None, race=None, label=None, original_name=None):
    fmt = (fmt or "png").lower()
    if fmt not in ("png", "tga"):
        fmt = "png"
    ext = "tga" if fmt == "tga" else "png"

    # sanitize provided metadata and original filename
    safe_name = sanitize(os.path.splitext(original_name or "")[0])
    parts = [safe_name, sanitize(pack), sanitize(race), sanitize(label), f"{w}x{h}"]
    base = "_".join([x for x in parts if x]) or f"resized_{w}x{h}"

    # always append unique id to prevent overwriting
    fname = f"{base}_{uuid.uuid4().hex}.{ext}"
    path = os.path.join(OUTPUT_DIR, fname)

    img.save(path, "TGA" if ext == "tga" else "PNG")
    size_bytes = os.path.getsize(path)
    return fname, path, size_bytes

def generate_batch_variants(img: Image.Image, sizes, formats, mode,
                            pack, race, label, original_name=None):
    """Builds outputs for multiple sizes/formats (PNG/TGA) and returns the results dict."""
    results = {f: {} for f in formats}

    for target in sizes:
        # Build base square per size
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
                pack=pack, race=race, label=label, original_name=original_name
            )
            results[fmt][str(target)] = {
                "url": f"{request.host_url.rstrip('/')}/files/{fname}",
                "bytes": size_bytes,
                "mb": round(size_bytes / (1024 * 1024), 3),
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

    fname, save_path, size_bytes = save_variant(
        out, target_w, target_h, fmt, pack=pack, race=race, label=label, original_name=original_name
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
  <head><meta charset="utf-8"><title>Texture Resizer</title></head>
  <body style="font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; line-height:1.6;">
    <h1>Local Texture Resizer</h1>

    <form method="post" enctype="multipart/form-data">
      <p><label>Upload image: <input type="file" name="file"></label></p>
      <p><label>OR Image URL: <input type="url" name="imageUrl" placeholder="https://example.com/image.png" style="width:100%;"></label></p>

      <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px;">
        <button formaction="/resize?size=512&download=1">Resize to 512</button>
        <button formaction="/resize?size=1024&download=1">Resize to 1024</button>
        <button formaction="/resize?size=2048&download=1">Resize to 2048</button>
        <button formaction="/resize?size=4096&download=1">Resize to 4096</button>
        <button formaction="/resize?size=pow2&download=1">Resize to POW2</button>
        <button formaction="/batch" formmethod="post" formtarget="_blank">Batch (512/1024/2048/4096)</button>
        <button formaction="/profile/gameasset" formmethod="post" formtarget="_blank" style="background:#222; color:#fff;">GameAsset Pack (PNG+TGA 512–4096)</button>
      </div>
    </form>

    <hr style="margin:28px 0;">

    <h2>ℹ️ What This Tool Does</h2>

    <p>
    This tool takes <b>any common image format</b> you upload (PNG, JPG, JPEG, GIF, BMP, WebP, etc.) 
    and automatically prepares it for use in games or digital projects. No matter the input, 
    your images are normalized to <code>RGBA</code> and exported in <b>PNG</b> or <b>TGA</b> format.
    </p>

    <h3>How It Works</h3>
    <ul>
    <li><b>Upload a file</b> (any format) or paste an image link.</li>
    <li><b>Pick a size</b> — for example 512, 1024, 2048, or 4096 pixels.</li>
    <li>The tool will <b>resize and clean up</b> your image automatically.</li>
    <li>You can also create a whole <b>pack of images</b> in different sizes with one click.</li>
    </ul>

    <h3>Why It’s Useful</h3>
    <ul>
    <li>Accepts almost any image format as input, even if your source isn’t already PNG.</li>
    <li>Makes sure outputs are game-ready in <code>PNG</code> and <code>TGA</code> formats only.</li>
    <li>Automatically keeps transparency (so backgrounds stay clear).</li>
    <li>Organizes filenames with labels you provide (like pack, race, or item name).</li>
    </ul>

    <h3>Example</h3>
    <ol>
    <li>You upload <code>dragon.jpg</code> or <code>dragon.png</code>.</li>
    <li>You press <b>GameAsset Pack</b>.</li>
    <li>You instantly get 4 versions: 512, 1024, 2048, and 4096.</li>
    <li>Each version is exported as PNG and TGA, with a direct download link.</li>
    </ol>

    <hr style="margin:28px 0;">

    <h2>📘 Service Documentation</h2>

    <h3>Overview</h3>
    <p>
    This service is a <b>Flask-based image processing pipeline</b> for preparing 
    <b>game-ready textures</b>. It accepts <b>any common input format</b> 
    (PNG, JPG, JPEG, GIF, BMP, WebP, etc.), normalizes them to <code>RGBA</code>, 
    and exports clean, consistent <code>PNG</code> or <code>TGA</code> files. 
    It supports power-of-two constraints, batch generation, and optional metadata tagging.
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
    </ul>

    <h3>3. Batch Endpoint <code>/batch</code></h3>
    <p>Generate multiple variants at once:</p>
    <ul>
    <li>Default sizes: 512, 1024, 2048, 4096</li>
    <li>Default formats: PNG (can also include TGA)</li>
    <li>All variants share the same mode/flags</li>
    </ul>
    <p>Returns JSON: format → size → {URL, file size in bytes/MB}</p>

    <h3>4. GameAsset Profile <code>/profile/gameasset</code></h3>
    <ul>
    <li>Convenience wrapper around <code>/batch</code></li>
    <li>Defaults: sizes = 512,1024,2048,4096 | formats = PNG,TGA | mode = fit</li>
    <li>One-click export of complete game-ready texture packs</li>
    </ul>

    <h3>5. Output</h3>
    <ul>
    <li>All outputs saved under <code>output/</code> folder</li>
    <li>Filenames include original name, metadata, resolution, and unique ID</li>
    <li>Example: <code>griffonwoman_medieval_elf_1024x1024_abcd1234.png</code></li>
    <li>Publicly accessible via <code>/files/&lt;filename&gt;</code></li>
    </ul>

    <h3>Typical Workflow</h3>
    <ol>
    <li>Upload <code>griffonwoman.jpg</code> (2000×1600)</li>
    <li>Call: <code>POST /profile/gameasset?pack=medieval&race=elf&label=sword</code></li>
    <li>Receive PNG+TGA outputs for 512, 1024, 2048, 4096</li>
    <li>Each with original name included, metadata tags, direct URL, and file size info</li>
    </ol>

    <hr style="margin:28px 0;">

    <h3>API</h3>
    <pre style="background:#f7f7f7; padding:12px; border-radius:8px;">
    POST /resize?size=512|1024|2048|4096|pow2
        &mode=fit|crop|stretch
        &format=png|tga
        &pack=&race=&label=
        &download=0|1

    POST /batch?sizes=512,1024,2048,4096&formats=png,tga&mode=fit

    POST /profile/gameasset   (PNG+TGA, sizes 512–4096)
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

    results = generate_batch_variants(img, sizes, formats, mode, pack, race, label, original_name=original_name)
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

    results = generate_batch_variants(
        img, sizes, formats, mode, pack, race, label, original_name=original_name
    )
    return jsonify({"ok": True, "results": results})


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)