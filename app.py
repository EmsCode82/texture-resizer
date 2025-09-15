import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError, ImageOps, ImageEnhance
import io
import os
import uuid
import requests
import re
import base64
import json
from flask import send_file
from flask_cors import CORS
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import logging

# Import Supabase client
from supabase import create_client, Client

# --- Basic Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
OUTPUT_DIR = "output"
variants_data = {
    'options': ['default', 'earthy', 'vibrant', 'muted']
}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Supabase Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "public-assets")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None
else:
    logger.warning("Supabase URL or Key not found. Supabase features for image storage will be disabled.")

# Safe helper for Supabase responses
def _sb_ok(res):
    try:
        code = getattr(res, "status_code", None)
        if code is None:
            return not isinstance(res, dict) or ("error" not in res and "message" not in res)
        return 200 <= code < 300
    except Exception:
        return True

# --- Stability AI Configuration ---
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_HOST = "https://api.stability.ai"

# --- Helper Function for Aspect Ratios ---
def get_dimensions(base_size, ratio_str):
    if not ratio_str:
        return (base_size, base_size)
    try:
        w, h = map(int, str(ratio_str).split(':'))
        if w <= 0 or h <= 0: return (base_size, base_size)
        if w >= h: return (base_size, max(1, int(base_size * h / w)))
        else: return (max(1, int(base_size * w / h)), base_size)
    except Exception:
        return (base_size, base_size)

# --- Image Processing Functions ---
def load_image_from_url(url):
    logger.debug(f"Fetching image from URL: {url}")
    try:
        headers = {'User-Agent': 'Assetgineer-Service/1.0'}
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGBA')
        name = os.path.basename(url.split("?")[0]) or f"image_{uuid.uuid4().hex[:6]}"
        logger.debug(f"Image from URL loaded successfully, size: {img.size}")
        return img, name
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None, None
    except (UnidentifiedImageError, IOError) as e:
        logger.error(f"Corrupted or unreadable image from URL {url}: {e}")
        return None, None

def generate_normal_map(image: Image.Image, edge_padding_px: int = 16) -> Image.Image:
    img_rgba = image.convert('RGBA')
    r, g, b, a = img_rgba.split()
    alpha_np = np.array(a, dtype=np.uint8)
    mask = alpha_np > 0
    gray = np.array(Image.merge('RGB', (r, g, b)).convert('L'), dtype=np.float32)
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
    n_img = Image.fromarray(normal, 'RGBA')
    n_img = dilate_rgba(n_img, padding=edge_padding_px)
    return n_img

def generate_roughness_map(image: Image.Image) -> Image.Image:
    return ImageOps.invert(image.convert('L'))

def dilate_rgba(img: Image.Image, padding: int = 16) -> Image.Image:
    arr = np.array(img.convert('RGBA'))
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    mask = alpha > 0
    inv = ~mask
    if not inv.any(): return img
    _, (iy, ix) = distance_transform_edt(inv, return_indices=True)
    filled_rgb = rgb.copy()
    filled_rgb[inv] = rgb[iy[inv], ix[inv]]
    dist = distance_transform_edt(inv)
    limited_rgb = rgb.copy()
    reach = (inv) & (dist <= padding)
    limited_rgb[reach] = filled_rgb[reach]
    out = np.dstack([limited_rgb, alpha])
    return Image.fromarray(out, 'RGBA')

def dilate_gray_using_alpha(gray_img: Image.Image, alpha_img: Image.Image, padding: int = 16) -> Image.Image:
    gray = gray_img.convert('L')
    a = alpha_img.convert('L')
    merged = Image.merge('RGBA', (gray, gray, gray, a))
    dilated_rgba = dilate_rgba(merged, padding=padding)
    r, _, _, _ = dilated_rgba.split()
    return r

def save_variant(img, original_name, size_str, label='base', file_format='png', pbr_type=None, ratio_str='1:1'):
    logger.debug(f"Saving variant for asset pack: {original_name} ({label}, {size_str})")
    try:
        base_name, _ = os.path.splitext(original_name or "image")
        base_name = re.sub(r'[^a-zA-Z0-9_-]', '', base_name) or "image"
        pbr_suffix = f"_{pbr_type}" if pbr_type else ""
        unique_id = uuid.uuid4().hex[:8]
        ratio_txt = str(ratio_str or '1:1').replace(':', 'x')
        ext = (file_format or 'png').lower()
        if ext == 'jpg': ext = 'jpeg'
        pil_fmt = {'png': 'PNG', 'tga': 'TGA', 'jpeg': 'JPEG', 'webp': 'WEBP'}.get(ext, ext.upper())
        image_to_save = img
        if pbr_type == 'roughness':
            if image_to_save.mode != 'L': image_to_save = image_to_save.convert('L')
        else:
            if ext in ('jpeg',): image_to_save = image_to_save.convert('RGB')
            elif ext in ('png', 'tga', 'webp'): image_to_save = image_to_save.convert('RGBA')
            else: image_to_save = image_to_save.convert('RGB')
        fname = f"{base_name}_{label}_{ratio_txt}{pbr_suffix}_{size_str}_{unique_id}.{ext}"
        fpath = os.path.join(OUTPUT_DIR, fname)
        save_kwargs = {}
        if pil_fmt == 'PNG': save_kwargs.update({'optimize': True})
        if pil_fmt == 'JPEG': save_kwargs.update({'quality': 95})
        image_to_save.save(fpath, format=pil_fmt, **save_kwargs)
        logger.debug(f"Successfully saved variant to {fpath}")
        return {'status': 'success', 'path': fpath, 'arcname': fname}
    except Exception as e:
        logger.error(f"Failed to save variant {original_name} ({label}, {size_str}): {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

# --- Asset Pack Generation Endpoint ---
@app.post("/generate_pack")
def generate_pack():
    logger.info("Received request for /generate_pack")
    if not supabase: return jsonify({'error': 'Supabase service unavailable'}), 503
    
    data = request.get_json(force=True)
    image_urls = data.get('imageUrls', [])
    package_name = data.get('packageName', f"pack_{uuid.uuid4().hex[:6]}")
    sizes = data.get('sizes', [1024])
    formats = data.get('formats', ['png'])
    pbr_maps = bool(data.get('pbrMaps', False))
    aspect_ratio = data.get('aspectRatio', None)
    edge_padding_px = int(data.get('edgePadding', 16))
    dilate_base = bool(data.get('dilateBase', False))
    base_edge_padding_px = int(data.get('baseEdgePadding', edge_padding_px))
    dilate_roughness = bool(data.get('dilateRoughness', False))
    roughness_edge_padding_px = int(data.get('roughnessEdgePadding', edge_padding_px))
    available_variants = set(variants_data.get('options', ['default']))
    requested_variants = data.get('variants', {}).get('options', None)
    
    if isinstance(requested_variants, list):
        selected_variant_names = [v for v in requested_variants if v in available_variants]
        if not selected_variant_names: selected_variant_names = list(available_variants)
    else:
        selected_variant_names = list(available_variants)

    if not image_urls: return jsonify({'error': 'imageUrls array is required'}), 400

    zip_buffer = io.BytesIO()
    all_temp_files = []
    
    try:
        with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
            for url in image_urls:
                img, original_name = load_image_from_url(url)
                if img is None: continue
                for variant_name in selected_variant_names:
                    mod_img = img
                    if variant_name == 'earthy': mod_img = ImageEnhance.Color(img).enhance(0.7)
                    elif variant_name == 'vibrant': mod_img = ImageEnhance.Color(img).enhance(1.3)
                    elif variant_name == 'muted': mod_img = ImageEnhance.Color(img).enhance(0.5)
                    for size_val in sizes:
                        dimensions = get_dimensions(size_val, aspect_ratio)
                        size_str_for_filename = f"{dimensions[0]}x{dimensions[1]}"
                        resized_img = mod_img.resize(dimensions, Image.Resampling.LANCZOS)
                        base_to_save = resized_img
                        if dilate_base: base_to_save = dilate_rgba(resized_img.convert('RGBA'), padding=base_edge_padding_px)
                        for fmt in formats:
                            result = save_variant(base_to_save, original_name, size_str_for_filename, label=variant_name, file_format=fmt, ratio_str=aspect_ratio)
                            if result['status'] == 'success':
                                arcname = os.path.join('Textures', variant_name, result['arcname'])
                                zip_file.write(result['path'], arcname)
                                all_temp_files.append(result['path'])
                            else: raise IOError(f"Failed to save base texture: {result.get('message')}")
                        if pbr_maps:
                            normal_img = generate_normal_map(resized_img, edge_padding_px=edge_padding_px)
                            rough_img = generate_roughness_map(resized_img)
                            if dilate_roughness:
                                alpha_from_resized = resized_img.convert('RGBA').split()[-1]
                                rough_img = dilate_gray_using_alpha(rough_img, alpha_from_resized, padding=roughness_edge_padding_px)
                            pbr_results = {'normal': normal_img, 'roughness': rough_img}
                            for pbr_type, pbr_img in pbr_results.items():
                                for fmt in formats:
                                    result = save_variant(pbr_img, original_name, size_str_for_filename, label=variant_name, file_format=fmt, pbr_type=pbr_type, ratio_str=aspect_ratio)
                                    if result['status'] == 'success':
                                        arcname = os.path.join('PBR', pbr_type.capitalize(), variant_name, result['arcname'])
                                        zip_file.write(result['path'], arcname)
                                        all_temp_files.append(result['path'])
                                    else: raise IOError(f"Failed to save PBR map ({pbr_type}): {result.get('message')}")
                        del resized_img, base_to_save
                        gc.collect()
                del img, mod_img
                gc.collect()
        
        zip_buffer.seek(0)
        zip_filename = f"asset_packs/{package_name}_{uuid.uuid4().hex[:8]}.zip"
        
        logger.info(f"Uploading generated asset pack zip to Supabase: {zip_filename}")
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(zip_filename, zip_buffer.getvalue(), {"content-type": "application/zip"})

        if _sb_ok(res):
            public_zip_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(zip_filename)
            logger.info(f"Asset pack uploaded to Supabase: {public_zip_url}")
            return jsonify({'zip_url': public_zip_url})
        else:
            details = getattr(res, "json", lambda: {"message": f"Supabase upload failed: {getattr(res, 'status_code', 'unknown')}"})()
            logger.error(f"Failed to upload asset pack to Supabase: {details}")
            return jsonify({'error': 'Failed to upload asset pack to Supabase', 'details': details}), 500
            
    except Exception as e:
        logger.error(f"An error occurred during pack generation: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate asset pack', 'details': str(e)}), 500
    finally:
        logger.info(f"Cleaning up {len(all_temp_files)} temporary files.")
        for fpath in all_temp_files:
            try:
                if os.path.exists(fpath): os.remove(fpath)
            except Exception as e: logger.error(f"Error cleaning up file {fpath}: {e}")

# --- File Upload Endpoint ---
@app.post("/upload_file")
def upload_file():
    logger.info("Received request for /upload_file")
    if not supabase: return jsonify({'error': 'Supabase service unavailable'}), 503
    if 'file' not in request.files: return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected for uploading'}), 400
    if file:
        filename = f"temp_{uuid.uuid4().hex[:12]}_{file.filename}"
        try:
            file_bytes = file.read()
            res = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, file_bytes)
            if _sb_ok(res):
                public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
                logger.info(f"File uploaded to Supabase: {public_url}")
                return jsonify({'file_url': public_url})
            else:
                details = getattr(res, "json", lambda: {"message": "upload failed"})()
                logger.error(f"Supabase upload failed: {details}")
                return jsonify({'error': 'Failed to upload file to Supabase', 'details': details}), 500
        except Exception as e:
            logger.error(f"Error during Supabase file upload: {e}", exc_info=True)
            return jsonify({'error': 'Internal server error during upload', 'details': str(e)}), 500
    return jsonify({'error': 'File upload failed'}), 500

# --- Main Image Generation Endpoint ---
@app.post("/generate_image")
def generate_image():
    logger.info("Received request for /generate_image")
    if not STABILITY_API_KEY: return jsonify({"error": "Image generation service is not configured."}), 503
    if not supabase: return jsonify({"error": "Image generation service is not fully configured (Supabase)."}, 503)

    try:
        data = request.get_json(force=True)
        operation = data.get('operation', 'text-to-image')
        prompt = data.get('prompt')
        negative_prompt = data.get('negative_prompt', None)
        aspect_ratio = data.get('aspect_ratio', '1:1')
        denoising_strength = data.get('denoising_strength', 0.75)
        cfg_scale = data.get('cfg_scale', 7)
        steps = data.get('steps', 30)

        if operation == 'text-to-image':
            api_url = f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/text-to-image"
            width, height = get_dimensions(1024, aspect_ratio)
            payload = {"text_prompts": [{"text": prompt}], "cfg_scale": cfg_scale, "width": width, "height": height, "samples": 1, "steps": steps}
            if negative_prompt: payload["text_prompts"].append({"text": negative_prompt, "weight": -1})
            response = requests.post(api_url, headers={"Authorization": f"Bearer {STABILITY_API_KEY}", "Content-Type": "application/json"}, json=payload)
        
        elif operation in ['enhance', 'inpaint']:
            init_image_url = data.get('init_image_url')
            if not init_image_url: return jsonify({"error": "Base image URL is required."}), 400
            
            temp_local_req_dir = os.path.join(OUTPUT_DIR, f"temp_req_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_local_req_dir, exist_ok=True)
            
            init_image_path, mask_image_path = None, None
            payload_files = {}
            
            try:
                init_img_resp = requests.get(init_image_url, timeout=20)
                init_img_resp.raise_for_status()
                init_image_path = os.path.join(temp_local_req_dir, "init_image.png")
                with open(init_image_path, 'wb') as f: f.write(init_img_resp.content)
                payload_files["init_image"] = open(init_image_path, "rb")

                text_prompts_list = [{"text": prompt}]
                if negative_prompt: text_prompts_list.append({"text": negative_prompt, "weight": -1})
                
                payload_data = {"text_prompts": json.dumps(text_prompts_list), "cfg_scale": cfg_scale, "samples": 1, "steps": steps, "image_strength": denoising_strength}
                
                if operation == 'inpaint':
                    mask_image_base64 = data.get('mask_image_base64')
                    if not mask_image_base64: return jsonify({"error": "Mask image base64 is required for inpainting."}), 400

                    mask_image_path = os.path.join(temp_local_req_dir, "mask_image.png")
                    logger.info("Decoding base64 mask_image.")
                    b64_data = mask_image_base64
                    if b64_data.startswith("data:"): b64_data = b64_data.split(",", 1)[1]
                    with open(mask_image_path, 'wb') as f: f.write(base64.b64decode(b64_data))
                    
                    payload_files["mask_image"] = open(mask_image_path, "rb")
                    payload_data["mask_source"] = "MASK_IMAGE_BLACK"
                    api_url = f"{STABILITY_HOST}/v1/generation/stable-inpainting-512-v2-0/image-to-image/masking"
                else: # 'enhance' operation
                    api_url = f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/image-to-image"

                response = requests.post(api_url, headers={"Authorization": f"Bearer {STABILITY_API_KEY}"}, files=payload_files, data=payload_data)
            finally:
                if "init_image" in payload_files: payload_files["init_image"].close()
                if "mask_image" in payload_files: payload_files["mask_image"].close()
                if os.path.exists(temp_local_req_dir):
                    if init_image_path and os.path.exists(init_image_path): os.remove(init_image_path)
                    if mask_image_path and os.path.exists(mask_image_path): os.remove(mask_image_path)
                    try: os.rmdir(temp_local_req_dir)
                    except OSError as e: logger.warning(f"Could not remove temp directory {temp_local_req_dir}: {e}")
        else:
            return jsonify({"error": f"Invalid operation: {operation}"}), 400

        if response.status_code != 200:
            logger.error(f"Stability AI API error: {response.status_code} - {response.text}")
            return jsonify({"error": "Failed to generate image.", "details": response.text}), response.status_code

        response_json = response.json()
        if not response_json.get("artifacts"): return jsonify({"error": "No image artifacts found in Stability AI response."}), 500
            
        base64_image = response_json["artifacts"][0]["base64"]
        img_data_bytes = base64.b64decode(base64_image)
        output_filename = f"generated_images/gen_{uuid.uuid4().hex[:12]}.png"
        
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).upload(output_filename, img_data_bytes, {"content-type": "image/png"})
            if _sb_ok(res):
                final_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(output_filename)
                logger.info(f"Image generated and uploaded to Supabase: {final_url}")
                return jsonify({"url": final_url})
            else:
                details = getattr(res, "json", lambda: {"message": "upload failed"})()
                logger.error(f"Supabase upload of generated image failed: {details}")
                return jsonify({'error': 'Failed to save generated image to Supabase', 'details': details}), 500
        except Exception as e:
            logger.error(f"Error during Supabase upload of generated image: {e}", exc_info=True)
            return jsonify({'error': 'Internal server error saving generated image', 'details': str(e)}), 500

    except Exception as e:
        logger.error(f"Error in /generate_image: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error.', 'details': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))