# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError, ImageOps, ImageEnhance
import io
import os
import re
import gc
import json
import uuid
import base64
import logging
import requests
import numpy as np
import send_from_directory

from zipfile import ZipFile, ZIP_DEFLATED
from scipy.ndimage import convolve, distance_transform_edt

# -----------------------------------------------------------------------------
# Basic setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

OUTPUT_DIR = "output"
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

variants_data = {
    "options": ["default", "earthy", "vibrant", "muted"]
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("assetgineer")

# -----------------------------------------------------------------------------
# Stability config
# -----------------------------------------------------------------------------
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_HOST = "https://api.stability.ai"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_dimensions(base_size: int, ratio_str: str | None):
    """Return (w,h) from base_size and 'W:H' ratio; fallback square."""
    if not ratio_str:
        return (base_size, base_size)
    try:
        w, h = map(int, str(ratio_str).split(":"))
        if w <= 0 or h <= 0:
            return (base_size, base_size)
        if w >= h:
            return (base_size, max(1, int(base_size * h / w)))
        else:
            return (max(1, int(base_size * w / h)), base_size)
    except Exception:
        return (base_size, base_size)

def to_multiple_of(n: int, m: int) -> int:
    return max(m, (n // m) * m)

def ensure_model_dims(img: Image.Image,
                      multiple: int = 64,
                      min_side: int = 512,
                      max_side: int = 1024) -> Image.Image:
    """Resize to keep AR, clamp to [min_side..max_side], and round to multiples of `multiple`."""
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
    elif min(w, h) < min_side:
        scale = min_side / float(min(w, h))
    if scale != 1.0:
        w = int(w * scale)
        h = int(h * scale)
    w = to_multiple_of(w, multiple)
    h = to_multiple_of(h, multiple)
    w = max(min_side, min(max_side, w))
    h = max(min_side, min(max_side, h))
    if (w, h) != img.size:
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    return img

def binary_mask_from_rgba(mask_rgba: Image.Image, invert: bool = False) -> Image.Image:
    """
    Return a single-channel 'L' mask where 255 = painted area, 0 = keep.
    We use alpha>0 as "painted", regardless of color (red/white/etc).
    """
    arr = np.array(mask_rgba.convert("RGBA"))
    painted = arr[..., 3] > 0  # any nontransparent pixel
    mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    mask[painted] = 255
    if invert:
        mask = 255 - mask
    return Image.fromarray(mask, "L")

def log_json_snippet(label: str, obj: dict | list, limit: int = 400):
    try:
        s = json.dumps(obj)[:limit]
    except Exception:
        s = str(obj)[:limit]
    logger.info(f"{label}: {s}")

# -----------------------------------------------------------------------------
# Image utilities (PBR/pack)
# -----------------------------------------------------------------------------
def load_image_from_url(url):
    try:
        logger.debug(f"Fetching image from URL: {url}")
        headers = {"User-Agent": "Assetgineer-Service/1.0"}
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        name = os.path.basename(url.split("?")[0]) or f"image_{uuid.uuid4().hex[:6]}"
        logger.debug(f"Loaded image size: {img.size}")
        return img, name
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None, None
    except (UnidentifiedImageError, IOError) as e:
        logger.error(f"Unreadable image from {url}: {e}")
        return None, None

def generate_normal_map(image: Image.Image, edge_padding_px: int = 16) -> Image.Image:
    img_rgba = image.convert("RGBA")
    r, g, b, a = img_rgba.split()
    alpha_np = np.array(a, dtype=np.uint8)
    mask = alpha_np > 0

    gray = np.array(Image.merge("RGB", (r, g, b)).convert("L"), dtype=np.float32)

    if mask.any():
        fg_mean = gray[mask].mean()
        gray_bg_filled = gray.copy()
        gray_bg_filled[~mask] = fg_mean
    else:
        gray_bg_filled = gray

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    dz_dx = convolve(gray_bg_filled, sobel_x)
    dz_dy = convolve(gray_bg_filled, sobel_y)

    intensity = 1.0
    dx = dz_dx * intensity
    dy = dz_dy * intensity
    norm = np.sqrt(dx * dx + dy * dy + 1.0)

    nx = ((dx / norm + 1.0) * 0.5 * 255.0).astype(np.uint8)
    ny = ((dy / norm + 1.0) * 0.5 * 255.0).astype(np.uint8)
    nz = ((1.0 / norm + 1.0) * 0.5 * 255.0).astype(np.uint8)

    normal = np.dstack([nx, ny, nz, alpha_np])
    normal[..., 0][~mask] = 128
    normal[..., 1][~mask] = 128
    normal[..., 2][~mask] = 255

    n_img = Image.fromarray(normal, "RGBA")
    n_img = dilate_rgba(n_img, padding=edge_padding_px)
    return n_img

def generate_roughness_map(image: Image.Image) -> Image.Image:
    return ImageOps.invert(image.convert("L"))

def dilate_rgba(img: Image.Image, padding: int = 16) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    mask = alpha > 0
    inv = ~mask
    if not inv.any():
        return img

    _, (iy, ix) = distance_transform_edt(inv, return_indices=True)
    filled_rgb = rgb.copy()
    filled_rgb[inv] = rgb[iy[inv], ix[inv]]

    dist = distance_transform_edt(inv)
    limited_rgb = rgb.copy()
    reach = inv & (dist <= padding)
    limited_rgb[reach] = filled_rgb[reach]

    out = np.dstack([limited_rgb, alpha])
    return Image.fromarray(out, "RGBA")

def dilate_gray_using_alpha(gray_img: Image.Image, alpha_img: Image.Image, padding: int = 16) -> Image.Image:
    gray = gray_img.convert("L")
    a = alpha_img.convert("L")
    merged = Image.merge("RGBA", (gray, gray, gray, a))
    dilated_rgba = dilate_rgba(merged, padding=padding)
    r, _, _, _ = dilated_rgba.split()
    return r

def save_variant(img, original_name, size_str, label="base", file_format="png", pbr_type=None, ratio_str="1:1"):
    try:
        base_name, _ = os.path.splitext(original_name or "image")
        base_name = re.sub(r"[^a-zA-Z0-9_-]", "", base_name) or "image"

        pbr_suffix = f"_{pbr_type}" if pbr_type else ""
        unique_id = uuid.uuid4().hex[:8]
        ratio_txt = str(ratio_str or "1:1").replace(":", "x")

        ext = (file_format or "png").lower()
        if ext == "jpg":
            ext = "jpeg"
        pil_fmt = {"png": "PNG", "tga": "TGA", "jpeg": "JPEG", "webp": "WEBP"}.get(ext, ext.upper())

        image_to_save = img
        if pbr_type == "roughness":
            if image_to_save.mode != "L":
                image_to_save = image_to_save.convert("L")
        else:
            if ext in ("jpeg",):
                image_to_save = image_to_save.convert("RGB")
            elif ext in ("png", "tga", "webp"):
                image_to_save = image_to_save.convert("RGBA")
            else:
                image_to_save = image_to_save.convert("RGB")

        fname = f"{base_name}_{label}_{ratio_txt}{pbr_suffix}_{size_str}_{unique_id}.{ext}"
        fpath = os.path.join(OUTPUT_DIR, fname)

        save_kwargs = {}
        if pil_fmt == "PNG":
            save_kwargs.update({"optimize": True})
        if pil_fmt == "JPEG":
            save_kwargs.update({"quality": 95})

        image_to_save.save(fpath, format=pil_fmt, **save_kwargs)
        return {"status": "success", "path": fpath, "arcname": fname}
    except Exception as e:
        logger.error(f"Failed to save variant {original_name} ({label}, {size_str}): {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    
def nearest_pow2(n: int) -> int:
    p = 1
    while p < n: p <<= 1
    return p

def parse_ratio(r: str):
    parts = r.split(":")
    if len(parts) != 2: return None
    try:
        a = int(parts[0]); b = int(parts[1])
        if a <= 0 or b <= 0: return None
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
        w = long_side; h = round(long_side * b / a)
    else:
        h = long_side; w = round(long_side * a / b)
    return (w, h)

def load_image_from_request():
    """Accepts multipart 'file' OR JSON/form {'imageUrl': ...} just like the old app."""
    # Try file first
    if 'file' in request.files and request.files['file'].filename:
        f = request.files['file']
        try:
            img = Image.open(f.stream).convert('RGBA')
            return img, f.filename
        except Exception as e:
            return None, f"Invalid image file: {e}"

    # Try URL from JSON or form
    url = None
    if request.is_json:
        url = (request.json or {}).get('imageUrl')
    if not url:
        url = request.form.get('imageUrl')
    if not url:
        return None, "Provide an uploaded file (field 'file') or 'imageUrl'."

    img, name = load_image_from_url(url)
    if img is None:
        return None, "Failed to download or read image."
    return img, name

def _fit_crop_stretch(img: Image.Image, target_w: int, target_h: int, mode: str) -> Image.Image:
    if mode == "stretch":
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    elif mode == "crop":
        scale = max(target_w / img.width, target_h / img.height)
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x1 = (new_w - target_w) // 2
        y1 = (new_h - target_h) // 2
        return scaled.crop((x1, y1, x1 + target_w, y1 + target_h))
    else:  # fit
        fitted = ImageOps.contain(img, (target_w, target_h), Image.Resampling.LANCZOS)
        out = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        x = (target_w - fitted.width) // 2
        y = (target_h - fitted.height) // 2
        out.paste(fitted, (x, y))
        return out

@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Assetgineer Texture Service</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{--b:#111;--t:#fff;--muted:#666;}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:40px auto;padding:0 16px;line-height:1.55}
    .grid{display:grid;gap:12px}
    @media(min-width:720px){.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:1fr 1fr 1fr}}
    .card{border:1px solid #eee;border-radius:12px;padding:14px}
    label{font-weight:600;margin:8px 0 6px;display:block}
    input,select,button{width:100%;padding:10px;border:1px solid #ddd;border-radius:10px}
    button{background:var(--b);color:var(--t);cursor:pointer}
    button.secondary{background:#f5f5f5;color:#111;border-color:#eee}
    .row{display:flex;gap:12px;flex-wrap:wrap}
    .muted{color:var(--muted)}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
    .links{display:flex;gap:10px}
    .pill{display:inline-block;padding:4px 8px;border-radius:999px;background:#f3f3f3;border:1px solid #e9e9e9}
    .ok{background:#f7f7f7;border-radius:12px;padding:12px;margin-top:10px}
    .error{color:#b00020;margin-top:8px}
  </style>
</head>
<body>
  <h1>Assetgineer Texture Service</h1>
  <p class="muted">Upload a file or paste an image URL. Then run a custom resize or export a full PNG/TGA pack (PoT sizes). This page only calls your existing endpoints; it won’t affect Base44.</p>

  <form id="ioForm" method="post" enctype="multipart/form-data" class="grid g2 card">
    <div>
      <label>Upload image</label>
      <input type="file" name="file" accept=".png,.jpg,.jpeg,.gif,.bmp,.webp,.tga">
    </div>
    <div>
      <label>OR Image URL</label>
      <input type="url" name="imageUrl" id="imageUrl" placeholder="https://example.com/image.png">
    </div>
    <div>
      <label>Pack <span class="muted">(optional)</span></label>
      <input type="text" name="pack" id="pack" placeholder="Characters">
    </div>
    <div>
      <label>Race <span class="muted">(optional)</span></label>
      <input type="text" name="race" id="race" placeholder="Human">
    </div>
    <div>
      <label>Label <span class="muted">(optional)</span></label>
      <input type="text" name="label" id="label" placeholder="DesertRogue">
    </div>
  </form>

  <!-- Quick buttons (download-ready) -->
  <div class="card">
    <h3>Quick download</h3>
    <div class="row">
      <button form="ioForm" formaction="/resize?size=512&download=1" formmethod="post" formtarget="_blank">Resize 512</button>
      <button form="ioForm" formaction="/resize?size=1024&download=1" formmethod="post" formtarget="_blank">Resize 1024</button>
      <button form="ioForm" formaction="/resize?size=2048&download=1" formmethod="post" formtarget="_blank">Resize 2048</button>
      <button form="ioForm" formaction="/resize?size=4096&download=1" formmethod="post" formtarget="_blank">Resize 4096</button>
      <button class="secondary" form="ioForm" formaction="/resize?size=pow2&download=1" formmethod="post" formtarget="_blank">Resize POW2</button>
    </div>
    <p class="muted">These open the processed file directly.</p>
  </div>

  <!-- Custom resize with controls -->
  <div class="card grid g3">
    <div>
      <label>Mode</label>
      <select id="r_mode">
        <option value="fit" selected>fit (pad transparency)</option>
        <option value="crop">crop (fill frame)</option>
        <option value="stretch">stretch (force)</option>
      </select>
    </div>
    <div>
      <label>Format</label>
      <select id="r_format">
        <option value="png" selected>png</option>
        <option value="tga">tga</option>
      </select>
    </div>
    <div>
      <label class="row" style="align-items:center;gap:8px"><input type="checkbox" id="r_pow2"> <span>Round each side to nearest PoT</span></label>
    </div>

    <div>
      <label>Square size</label>
      <input id="r_size" type="number" min="64" max="8192" step="64" placeholder="e.g. 1024">
    </div>
    <div>
      <label>Width × Height</label>
      <input id="r_wh" placeholder="e.g. 2048x1024">
    </div>
    <div>
      <label>Ratio & long edge</label>
      <input id="r_ratio_long" placeholder="e.g. 16:9 @ 2048  (type: 16:9@2048)">
    </div>

    <div class="row" style="grid-column:1/-1">
      <button id="btnResizeJson">Run Custom Resize (show JSON)</button>
      <button class="secondary" id="btnResizeDownload">Run Custom Resize (download)</button>
    </div>
    <div id="resizeOut" class="ok" style="display:none"></div>
    <div id="resizeErr" class="error"></div>
    <p class="muted" style="grid-column:1/-1">Tip: provide exactly one sizing style (Square OR Width×Height OR Ratio@Long). If multiple are set, square takes priority, then width×height, then ratio.</p>
  </div>

  <!-- Pack exporter -->
  <div class="card grid g3">
    <div>
      <label>Sizes (comma PoT)</label>
      <input id="p_sizes" value="512,1024,2048,4096">
    </div>
    <div>
      <label>Formats</label>
      <div class="row">
        <label class="pill"><input type="checkbox" id="p_png" checked> PNG</label>
        <label class="pill"><input type="checkbox" id="p_tga" checked> TGA</label>
      </div>
    </div>
    <div>
      <label>Mode</label>
      <select id="p_mode">
        <option value="fit" selected>fit</option>
        <option value="crop">crop</option>
        <option value="stretch">stretch</option>
      </select>
    </div>
    <div class="row" style="align-items:center;gap:8px">
      <label class="pill"><input type="checkbox" id="p_pow2"> pow2 (round each size)</label>
    </div>

    <div class="row" style="grid-column:1/-1">
      <button id="btnPack">Export Pack (JSON links)</button>
      <button class="secondary" id="btnBatch">Batch (defaults; JSON)</button>
    </div>
    <div id="packOut" class="ok" style="display:none"></div>
    <div id="packErr" class="error"></div>
  </div>

  <div class="card">
    <h3>API quick ref</h3>
    <pre>
POST /resize?size=512|1024|2048|4096|pow2
     &mode=fit|crop|stretch
     &format=png|tga
     &pack=&race=&label=
     &download=0|1

POST /batch?sizes=512,1024,2048,4096&formats=png,tga&mode=fit
POST /profile/gameasset   (PNG+TGA, sizes 512–4096)

Body for all:
- multipart/form-data with file "file", or
- JSON: {"imageUrl": "https://…"}
    </pre>
  </div>

<script>
function val(id){return document.getElementById(id).value.trim();}
function checked(id){return document.getElementById(id).checked;}
function qs(obj){return new URLSearchParams(Object.entries(obj).filter(([_,v])=>v!==undefined && v!==null && v!=="")).toString();}
function tail(u){try{const p=new URL(u);return p.pathname.split("/").pop()}catch{return u}}

async function postWithBody(url, bodyForm){
  // If a file was chosen, submit multipart form directly; else send JSON {imageUrl}
  const formEl = document.getElementById('ioForm');
  const fileField = formEl.querySelector('input[type=file]');
  const hasFile = fileField && fileField.files && fileField.files.length>0;
  const pack = val('pack'), race = val('race'), label = val('label');
  const urlObj = new URL(url, window.location.origin);
  if (pack) urlObj.searchParams.set('pack', pack);
  if (race) urlObj.searchParams.set('race', race);
  if (label) urlObj.searchParams.set('label', label);

  let res;
  if (hasFile){
    const fd = new FormData(formEl);
    res = await fetch(urlObj.toString(), { method: 'POST', body: fd });
  } else {
    const imageUrl = val('imageUrl');
    if (!imageUrl) throw new Error('Provide an image URL or choose a file.');
    res = await fetch(urlObj.toString(), {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ imageUrl })
    });
  }
  if (!res.ok) throw new Error('Service error ' + res.status);
  return res;
}

// ----- Custom Resize -----
document.getElementById('btnResizeJson').addEventListener('click', async (e)=>{
  e.preventDefault();
  const out = document.getElementById('resizeOut');
  const err = document.getElementById('resizeErr');
  out.style.display='none'; out.textContent=''; err.textContent='';

  try{
    const mode = val('r_mode');
    const format = val('r_format');
    const pow2 = checked('r_pow2') ? '1' : '0';

    // precedence: size -> widthxheight -> ratio@long
    let endpoint = '/resize?mode='+encodeURIComponent(mode)+'&format='+encodeURIComponent(format)+'&pow2='+pow2;
    const sz = val('r_size');
    const wh = val('r_wh');
    const rl = val('r_ratio_long');

    if (sz){
      endpoint += '&size='+encodeURIComponent(sz);
    } else if (wh && wh.includes('x')){
      const [w,h] = wh.toLowerCase().split('x').map(s=>s.trim());
      endpoint += '&width='+encodeURIComponent(w)+'&height='+encodeURIComponent(h);
    } else if (rl && rl.includes('@')){
      const [ratio,long] = rl.split('@').map(s=>s.trim());
      endpoint += '&ratio='+encodeURIComponent(ratio)+'&long='+encodeURIComponent(long);
    } else {
      throw new Error('Provide one sizing style: size OR widthxheight OR ratio@long');
    }

    const res = await postWithBody(endpoint, true);
    const json = await res.json();

    out.style.display='block';
    out.innerHTML = '<div><b>Result</b></div><div class="mono">'+JSON.stringify(json, null, 2)+'</div>';
  }catch(ex){ err.textContent = ex.message || String(ex); }
});

document.getElementById('btnResizeDownload').addEventListener('click', async (e)=>{
  e.preventDefault();
  const err = document.getElementById('resizeErr'); err.textContent='';
  try{
    const mode = val('r_mode');
    const format = val('r_format');
    const pow2 = checked('r_pow2') ? '1' : '0';
    let endpoint = '/resize?download=1&mode='+encodeURIComponent(mode)+'&format='+encodeURIComponent(format)+'&pow2='+pow2;

    const sz = val('r_size');
    const wh = val('r_wh');
    const rl = val('r_ratio_long');

    if (sz){
      endpoint += '&size='+encodeURIComponent(sz);
    } else if (wh && wh.includes('x')){
      const [w,h] = wh.toLowerCase().split('x').map(s=>s.trim());
      endpoint += '&width='+encodeURIComponent(w)+'&height='+encodeURIComponent(h);
    } else if (rl && rl.includes('@')){
      const [ratio,long] = rl.split('@').map(s=>s.trim());
      endpoint += '&ratio='+encodeURIComponent(ratio)+'&long='+encodeURIComponent(long);
    } else {
      throw new Error('Provide one sizing style: size OR widthxheight OR ratio@long');
    }

    // submit a hidden form to trigger download reliably
    const form = document.createElement('form');
    form.method = 'post';
    form.action = endpoint;
    form.enctype = 'multipart/form-data';
    form.target = '_blank';
    // include the same fields as ioForm (file/url/pack/race/label)
    const io = document.getElementById('ioForm');
    const clone = io.cloneNode(true);
    // move chosen file input value (cannot clone FileList); append original instead
    const fileInput = io.querySelector('input[type=file]');
    if (fileInput && fileInput.files.length){
      const fi = document.createElement('input');
      fi.type='file'; fi.name='file';
      // NOTE: browsers don’t allow programmatically setting FileList; this still works if no file is needed.
      // If you need guaranteed file submit for download flow, use JSON + server returns file by URL.
    }
    // copy imageUrl/pack/race/label as hidden inputs for safety
    ['imageUrl','pack','race','label'].forEach(id=>{
      const v = document.getElementById(id).value;
      if (v){
        const hid = document.createElement('input');
        hid.type='hidden'; hid.name=id; hid.value=v;
        form.appendChild(hid);
      }
    });
    document.body.appendChild(form);
    form.submit();
    setTimeout(()=>document.body.removeChild(form), 2000);
  }catch(ex){ err.textContent = ex.message || String(ex); }
});

// ----- Pack Exporter -----
function renderPack(json, mountId){
  const m = document.getElementById(mountId);
  m.style.display='block';
  const groups = [];
  if (json.results){
    const order = ['png','tga','pbr'];
    for (const key of order){
      if (!json.results[key]) continue;
      const dict = json.results[key];
      if (key==='pbr'){
        // PBR (if ever added back) would render maps here
        continue;
      }
      const entries = Object.entries(dict).sort((a,b)=>Number(a[0].split('x')[0]) - Number(b[0].split('x')[0]));
      groups.push('<h4>'+key.toUpperCase()+'</h4>');
      for (const [sizeKey, info] of entries){
        groups.push(
          '<div class="row" style="justify-content:space-between;align-items:center;margin:6px 0;">' +
            '<div><span class="mono">'+sizeKey+'</span></div>' +
            '<div class="links">' +
              '<a href="'+info.url+'" target="_blank" rel="noreferrer">Open</a>' +
              '<button onclick="navigator.clipboard.writeText(\\''+info.url+'\\')">Copy URL</button>' +
            '</div>' +
          '</div>'
        );
      }
    }
  }
  m.innerHTML = '<h3>Texture Pack Ready</h3>' + groups.join('') || '<div>No results.</div>';
}

document.getElementById('btnPack').addEventListener('click', async (e)=>{
  e.preventDefault();
  const out = document.getElementById('packOut');
  const err = document.getElementById('packErr');
  out.style.display='none'; out.textContent=''; err.textContent='';

  try{
    const sizes = val('p_sizes') || '512,1024,2048,4096';
    const fmts = [checked('p_png')?'png':null, checked('p_tga')?'tga':null].filter(Boolean).join(',') || 'png,tga';
    const mode = val('p_mode');
    const pow2 = checked('p_pow2') ? '1' : '0';
    const endpoint = '/profile/gameasset?' + new URLSearchParams({sizes: sizes, formats: fmts, mode: mode, pow2: pow2}).toString();

    const res = await postWithBody(endpoint, true);
    const json = await res.json();
    renderPack(json, 'packOut');
  }catch(ex){ err.textContent = ex.message || String(ex); }
});

document.getElementById('btnBatch').addEventListener('click', async (e)=>{
  e.preventDefault();
  const out = document.getElementById('packOut');
  const err = document.getElementById('packErr');
  out.style.display='none'; out.textContent=''; err.textContent='';

  try{
    const sizes = val('p_sizes') || '512,1024,2048,4096';
    const fmts = [checked('p_png')?'png':null, checked('p_tga')?'tga':null].filter(Boolean).join(',') || 'png,tga';
    const mode = val('p_mode');
    const pow2 = checked('p_pow2') ? '1' : '0';
    const endpoint = '/batch?' + new URLSearchParams({sizes: sizes, formats: fmts, mode: mode, pow2: pow2}).toString();

    const res = await postWithBody(endpoint, true);
    const json = await res.json();
    renderPack(json, 'packOut');
  }catch(ex){ err.textContent = ex.message || String(ex); }
});
</script>
</body>
</html>
    """

@app.post("/resize")
def resize_endpoint():
    img, original_name_or_err = load_image_from_request()
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
    fmt = (request.args.get("format") or "png").lower()
    if fmt not in ("png", "tga", "jpeg", "jpg", "webp"):
        return jsonify({"error": "format must be one of png,tga,jpeg,webp"}), 400

    # Determine target size
    if width and height:
        try:
            target_w, target_h = int(width), int(height)
        except ValueError:
            return jsonify({"error": "width/height must be integers"}), 400
    elif ratio and long_:
        rt = parse_ratio(ratio)
        if not rt: return jsonify({"error": "Invalid ratio. Use ratio=16:9"}), 400
        try:
            long_side = int(long_)
        except ValueError:
            return jsonify({"error": "long must be an integer"}), 400
        target_w, target_h = size_from_ratio_long(rt, long_side)
    elif size_param:
        if size_param == "pow2":
            s = nearest_pow2(max(img.width, img.height))
        else:
            try:
                s = int(size_param)
            except ValueError:
                return jsonify({"error": "size must be an integer or 'pow2'"}), 400
        target_w = target_h = s
    else:
        return jsonify({"error": "Provide size OR width+height OR ratio+long"}), 400

    if request.args.get("pow2", "0") == "1":
        target_w = to_pow2(target_w); target_h = to_pow2(target_h)

    out = _fit_crop_stretch(img, target_w, target_h, mode)
    size_str = f"{target_w}x{target_h}"
    res = save_variant(out, original_name, size_str, label=request.args.get("label") or "base",
                       file_format=fmt, ratio_str=request.args.get("ratio"))
    if res["status"] != "success":
        return jsonify({"error": res.get("message", "Save failed")}), 500

    path = res["path"]; fname = res["arcname"]
    if download:
        return send_file(path, as_attachment=True, download_name=fname)

    bytes_ = os.path.getsize(path)
    return jsonify({
        "public_url": f"{request.host_url.rstrip('/')}/files/{fname}",
        "optimized_path": os.path.abspath(path),
        "width": out.width,
        "height": out.height,
        "format": fmt,
        "file_size_bytes": bytes_,
        "file_size_mb": round(bytes_ / (1024*1024), 3)
    })

@app.post("/batch")
def batch():
    img, original_name_or_err = load_image_from_request()
    if img is None:
        return jsonify({"error": original_name_or_err}), 400
    original_name = original_name_or_err

    sizes_param = request.args.get("sizes") or "512,1024,2048,4096"
    formats_param = request.args.get("formats", "png")
    mode = request.args.get("mode", "fit")
    pow2 = request.args.get("pow2", "0") == "1"
    label = request.form.get("label") or request.args.get("label") or "base"

    try:
        sizes = [int(s.strip()) for s in sizes_param.split(",") if s.strip()]
    except:
        return jsonify({"error": "Invalid sizes. Example: sizes=512,1024,2048"}), 400
    if pow2:
        sizes = [to_pow2(s) for s in sizes]

    valid_ext = {"png","tga","jpeg","webp"}
    formats = [f.strip().lower() for f in formats_param.split(",") if f.strip().lower() in valid_ext]
    if not formats:
        return jsonify({"error": "No valid formats. Use formats=png,tga,jpeg,webp"}), 400

    results = {fmt:{} for fmt in formats}
    for s in sizes:
        w = h = s
        out = _fit_crop_stretch(img, w, h, mode)
        size_str = f"{w}x{h}"
        for fmt in formats:
            res = save_variant(out, original_name, size_str, label=label, file_format=fmt)
            if res["status"] != "success":
                return jsonify({"error": res.get("message","Save failed")}), 500
            bytes_ = os.path.getsize(res["path"])
            results[fmt][size_str] = {
                "url": f"{request.host_url.rstrip('/')}/files/{res['arcname']}",
                "bytes": bytes_,
                "mb": round(bytes_/(1024*1024),3)
            }
    return jsonify({"ok": True, "results": results})


@app.post("/profile/gameasset")
def profile_gameasset():
    # Convenience wrapper: sizes 512..4096, formats png+tga
    request_args = request.args.to_dict(flat=True)
    request_args.setdefault("sizes", "512,1024,2048,4096")
    request_args.setdefault("formats", "png,tga")
    with app.test_request_context(
        "/batch", method="POST", json=request.get_json(silent=True), data=(request.form or None), query_string=request_args
    ):
        return batch()


@app.post("/validate")
def validate_endpoint():
    img, original_name_or_err = load_image_from_request()
    if img is None:
        return jsonify({"error": original_name_or_err}), 400

    issues = []
    array = np.array(img)
    if img.mode == 'RGBA':
        alpha = array[:, :, 3]
        if np.all(alpha == 255):
            issues.append("Alpha channel is fully opaque – consider RGB for optimization.")
        elif np.any(alpha < 255) and np.mean(alpha[alpha < 255]) < 128:
            issues.append("Semi-transparent areas detected – ensure engine supports alpha blending.")
        transparent_ratio = np.sum(alpha == 0) / alpha.size
        if transparent_ratio > 0.9:
            issues.append("Over 90% transparent – ensure this is intentional.")

    h, w = array.shape[:2]
    if h > 1 and w > 1:
        left = array[:, 0, :3]; right = array[:, -1, :3]
        top  = array[0, :, :3]; bottom = array[-1, :, :3]
        if np.mean(np.abs(left - right)) > 20:
            issues.append("Potential vertical seam – may not tile seamlessly.")
        if np.mean(np.abs(top - bottom)) > 20:
            issues.append("Potential horizontal seam – may not tile seamlessly.")

    gray = np.array(img.convert('L'))
    variance = np.var(gray)
    if variance > 10000:
        issues.append("High variance – possible artifacts/noise.")

    if img.width != to_pow2(img.width) or img.height != to_pow2(img.height):
        issues.append("Non-power-of-two dimensions detected.")

    return jsonify({"ok": len(issues)==0, "issues": issues, "warnings": [] if issues else ["Texture passes basic validation."]})


@app.post("/package")
def package_endpoint():
    # Build a quick pack zip from /profile/gameasset results.
    # Reuse /batch to generate files, then zip them.
    img, name_or_err = load_image_from_request()
    if img is None: return jsonify({"error": name_or_err}), 400

    # Generate defaults via in-memory call to /batch
    with app.test_request_context(
        "/batch?sizes=512,1024,2048,4096&formats=png,tga&mode=fit",
        method="POST", json=request.get_json(silent=True), data=(request.form or None)
    ):
        resp = batch()
    if resp.status_code != 200:
        return resp
    payload = resp.get_json()
    if not payload.get("ok"):
        return jsonify({"error":"Batch failed"}), 500

    # Collect files
    file_paths = []
    for fmt, sizes in payload["results"].items():
        for _, info in sizes.items():
            url = info["url"]
            fname = url.split("/files/")[-1]
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(path):
                file_paths.append(path)

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zipf:
        for p in file_paths:
            arc_dir = "Textures"
            zipf.write(p, os.path.join(arc_dir, os.path.basename(p)))
    zip_buffer.seek(0)

    zip_name = f"asset_pack_{uuid.uuid4().hex[:8]}.zip"
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name=zip_name)

# -----------------------------------------------------------------------------
# Asset pack endpoint
# -----------------------------------------------------------------------------
@app.post("/generate_pack")
def generate_pack():
    logger.info("POST /generate_pack")
    api_key = request.headers.get("X-API-Key")
    if api_key != "mock_api_key_123":
        return jsonify({"error": "Unauthorized: Invalid API key"}), 401
    if not request.is_json:
        return jsonify({"error": "JSON payload required"}), 400

    data = request.get_json(force=True)
    image_urls = data.get("imageUrls", [])
    package_name = data.get("packageName", f"pack_{uuid.uuid4().hex[:6]}")
    sizes = data.get("sizes", [1024])
    formats = data.get("formats", ["png"])
    pbr_maps = bool(data.get("pbrMaps", False))
    aspect_ratio = data.get("aspectRatio", None)

    edge_padding_px = int(data.get("edgePadding", 16))
    dilate_base = bool(data.get("dilateBase", False))
    base_edge_padding_px = int(data.get("baseEdgePadding", edge_padding_px))
    dilate_roughness = bool(data.get("dilateRoughness", False))
    roughness_edge_padding_px = int(data.get("roughnessEdgePadding", edge_padding_px))

    available_variants = set(variants_data.get("options", ["default"]))
    requested_variants = data.get("variants", {}).get("options", None)
    if isinstance(requested_variants, list):
        selected_variant_names = [v for v in requested_variants if v in available_variants]
        if not selected_variant_names:
            logger.info("No valid variants in payload; using defaults.")
            selected_variant_names = list(available_variants)
    else:
        selected_variant_names = list(available_variants)

    if not image_urls:
        return jsonify({"error": "imageUrls array is required"}), 400

    zip_buffer = io.BytesIO()
    all_temp_files = []

    try:
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zip_file:
            for url in image_urls:
                img, original_name = load_image_from_url(url)
                if img is None:
                    continue

                for variant_name in selected_variant_names:
                    mod_img = img
                    if variant_name == "earthy":
                        mod_img = ImageEnhance.Color(img).enhance(0.7)
                    elif variant_name == "vibrant":
                        mod_img = ImageEnhance.Color(img).enhance(1.3)
                    elif variant_name == "muted":
                        mod_img = ImageEnhance.Color(img).enhance(0.5)

                    for size_val in sizes:
                        dimensions = get_dimensions(size_val, aspect_ratio)
                        size_str_for_filename = f"{dimensions[0]}x{dimensions[1]}"
                        resized_img = mod_img.resize(dimensions, Image.Resampling.LANCZOS)

                        base_to_save = resized_img
                        if dilate_base:
                            base_to_save = dilate_rgba(resized_img.convert("RGBA"), padding=base_edge_padding_px)

                        for fmt in formats:
                            result = save_variant(
                                base_to_save, original_name, size_str_for_filename,
                                label=variant_name, file_format=fmt, ratio_str=aspect_ratio
                            )
                            if result["status"] == "success":
                                arcname = os.path.join("Textures", variant_name, result["arcname"])
                                zip_file.write(result["path"], arcname)
                                all_temp_files.append(result["path"])
                            else:
                                raise IOError(f"Failed to save base texture: {result.get('message')}")

                        if pbr_maps:
                            normal_img = generate_normal_map(resized_img, edge_padding_px=edge_padding_px)
                            rough_img = generate_roughness_map(resized_img)
                            if dilate_roughness:
                                alpha_from_resized = resized_img.convert("RGBA").split()[-1]
                                rough_img = dilate_gray_using_alpha(
                                    rough_img, alpha_from_resized, padding=roughness_edge_padding_px
                                )

                            for pbr_type, pbr_img in {"normal": normal_img, "roughness": rough_img}.items():
                                for fmt in formats:
                                    result = save_variant(
                                        pbr_img, original_name, size_str_for_filename,
                                        label=variant_name, file_format=fmt, pbr_type=pbr_type, ratio_str=aspect_ratio
                                    )
                                    if result["status"] == "success":
                                        arcname = os.path.join("PBR", pbr_type.capitalize(), variant_name, result["arcname"])
                                        zip_file.write(result["path"], arcname)
                                        all_temp_files.append(result["path"])
                                    else:
                                        raise IOError(f"Failed to save PBR map ({pbr_type}): {result.get('message')}")

                        del resized_img, base_to_save
                        gc.collect()

                del img, mod_img
                gc.collect()

        zip_buffer.seek(0)
        zip_filename = f"{package_name}.zip"
        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=zip_filename
        )

    except Exception as e:
        logger.error(f"Error during pack generation: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate asset pack", "details": str(e)}), 500
    finally:
        logger.info(f"Cleaning up {len(all_temp_files)} temp files")
        for fpath in all_temp_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception as e:
                logger.error(f"Cleanup error {fpath}: {e}")

# -----------------------------------------------------------------------------
# File serving & upload (same-origin URLs)
# -----------------------------------------------------------------------------
@app.route("/files/<path:filename>")
def serve_file(filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(output_path):
        return send_file(output_path)
    temp_path = os.path.join(TEMP_UPLOAD_DIR, filename)
    if os.path.exists(temp_path):
        return send_file(temp_path)
    return jsonify({"error": "File not found"}), 404

@app.post("/upload_file")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    filename = f"temp_{uuid.uuid4().hex[:12]}_{file.filename}"
    save_path = os.path.join(TEMP_UPLOAD_DIR, filename)
    file.save(save_path)

    file_url = f"{request.host_url}files/{filename}"
    logger.info(f"Uploaded file: {file_url}")
    return jsonify({"file_url": file_url})

# -----------------------------------------------------------------------------
# Image generation (Stability AI)
# -----------------------------------------------------------------------------
@app.post("/generate_image")
def generate_image():
    if not STABILITY_API_KEY:
        logger.error("STABILITY_API_KEY not configured")
        return jsonify({"error": "Image generation service is not configured."}), 503

    try:
        data = request.get_json(force=True)
        operation = data.get("operation", "text-to-image")
        prompt = data.get("prompt") or ""
        negative_prompt = data.get("negative_prompt")
        aspect_ratio = data.get("aspect_ratio", "1:1")
        denoising_strength = float(data.get("denoising_strength", 0.75))  # maps to image_strength
        cfg_scale = float(data.get("cfg_scale", 7))
        steps = int(data.get("steps", 30))
        mask_invert = bool(data.get("mask_invert", False))

        logger.info(f"op={operation} cfg={cfg_scale} steps={steps} strength={denoising_strength}")

        # -----------------------------
        # Text-to-Image (JSON payload)
        # -----------------------------
        if operation == "text-to-image":
            w, h = get_dimensions(1024, aspect_ratio)
            w = to_multiple_of(w, 64)
            h = to_multiple_of(h, 64)

            payload = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": cfg_scale,
                "width": w,
                "height": h,
                "samples": 1,
                "steps": steps
            }
            if negative_prompt:
                payload["text_prompts"].append({"text": negative_prompt, "weight": -1})

            api_url = f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/text-to-image"
            log_json_snippet("t2i payload", payload)
            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {STABILITY_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json=payload
            )

        # -----------------------------
        # Enhance / Inpaint (multipart)
        # -----------------------------
        elif operation in ["enhance", "inpaint"]:
            init_image_url = data.get("init_image_url")
            if not init_image_url:
                return jsonify({"error": "Base image URL is required."}), 400

            # Download base image
            r = requests.get(init_image_url, timeout=30)
            r.raise_for_status()
            init_rgba = Image.open(io.BytesIO(r.content)).convert("RGBA")
            init_rgba = ensure_model_dims(init_rgba)  # normalize dims for model

            # If inpaint, load mask (base64 or url) and normalize to L (white=change)
            mask_L = None
            if operation == "inpaint":
                mask_image_base64 = data.get("mask_image_base64")
                mask_image_url = data.get("mask_image_url")
                if not (mask_image_base64 or mask_image_url):
                    return jsonify({"error": "Mask image is required (mask_image_base64 or mask_image_url)."}), 400

                if mask_image_base64:
                    b64 = mask_image_base64.split(",", 1)[1] if mask_image_base64.startswith("data:") else mask_image_base64
                    mask_rgba = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
                else:
                    mr = requests.get(mask_image_url, timeout=30)
                    mr.raise_for_status()
                    mask_rgba = Image.open(io.BytesIO(mr.content)).convert("RGBA")

                if mask_rgba.size != init_rgba.size:
                    mask_rgba = mask_rgba.resize(init_rgba.size, Image.Resampling.NEAREST)

                mask_L = binary_mask_from_rgba(mask_rgba, invert=mask_invert)

            # Write temps
            temp_dir = os.path.join(TEMP_UPLOAD_DIR, f"req_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)
            init_path = os.path.join(temp_dir, "init.png")
            init_rgba.save(init_path, "PNG")

            files = {"init_image": open(init_path, "rb")}
            data_fields = {
                # Stability wants text_prompts as a JSON string in multipart
                "text_prompts": json.dumps(
                    [{"text": prompt}] + ([{"text": negative_prompt, "weight": -1}] if negative_prompt else [])
                ),
                "cfg_scale": str(cfg_scale),
                "samples": "1",
                "steps": str(steps),
                "image_strength": str(denoising_strength)
            }

            if operation == "inpaint":
                mask_path = os.path.join(temp_dir, "mask.png")
                mask_L.save(mask_path, "PNG")
                files["mask_image"] = open(mask_path, "rb")
                data_fields["mask_source"] = "MASK_IMAGE_WHITE"
                api_url = f"{STABILITY_HOST}/v1/generation/stable-inpainting-512-v2-0/image-to-image/masking"
            else:
                api_url = f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/image-to-image"

            log_json_snippet("img2img meta", {
                "op": operation,
                "cfg_scale": cfg_scale,
                "steps": steps,
                "image_strength": denoising_strength,
                "init_size": init_rgba.size,
            })
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json"},
                files=files,
                data=data_fields
            )

            # cleanup
            try:
                files["init_image"].close()
            except Exception:
                pass
            try:
                if "mask_image" in files:
                    files["mask_image"].close()
            except Exception:
                pass
            try:
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
            except Exception:
                pass

        else:
            return jsonify({"error": f"Unknown operation: {operation}"}), 400

        # Handle Stability response
        if response.status_code != 200:
            try:
                err = response.json()
            except Exception:
                err = {"message": response.text}
            logger.error(f"Stability error {response.status_code}: {str(err)[:500]}")
            return jsonify({"error": "Failed to generate image.", "details": err}), response.status_code

        resp_json = response.json()
        if not resp_json.get("artifacts"):
            return jsonify({"error": "No image artifacts returned."}), 502

        b64img = resp_json["artifacts"][0]["base64"]
        return jsonify({"url": f"data:image/png;base64,{b64img}"})

    except Exception as e:
        logger.error(f"/generate_image internal error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error."}), 500

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
