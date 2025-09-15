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
