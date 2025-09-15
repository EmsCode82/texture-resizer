from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError, ImageOps, ImageEnhance
import io
import os
import uuid
import requests
import re
import base64
from flask_cors import CORS
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import logging

# NEW: Import Supabase client
from supabase import create_client, Client

# --- Basic Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
OUTPUT_DIR = "output" # Still used for asset pack zipping (local temporary files)
variants_data = {
    'options': ['default', 'earthy', 'vibrant', 'muted']
}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use logger object directly

# --- NEW: Supabase Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "public-assets") # Default bucket name

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None # Ensure it's None if init fails
else:
    logger.warning("Supabase URL or Key not found. Supabase features for image storage will be disabled.")


# --- Stability AI Configuration ---
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_HOST = "https://api.stability.ai"

# --- Helper Function for Aspect Ratios ---
def get_dimensions(base_size, ratio_str):
    """Calculates width and height based on a base size and ratio string (e.g., '16:9'). Falls back to square."""
    if not ratio_str:
        return (base_size, base_size)
    try:
        w, h = map(int, str(ratio_str).split(':'))
        if w <= 0 or h <= 0:
            return (base_size, base_size)
        if w >= h:
            # Landscape or square
            return (base_size, max(1, int(base_size * h / w)))
        else:
            # Portrait
            return (max(1, int(base_size * w / h)), base_size)
    except Exception:
        # Fallback to square if ratio is invalid
        return (base_size, base_size)


# --- Image Processing Functions (Remain unchanged, as they operate on PIL Image objects) ---

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
    """
    Alpha-aware Sobel normal map generator.
    - Uses luminance only where alpha > 0
    - Sets background to flat (128,128,255) while preserving alpha
    - Applies RGBA edge dilation to prevent purple/halo fringing in mips
    """
    img_rgba = image.convert('RGBA')
    r, g, b, a = img_rgba.split()
    alpha_np = np.array(a, dtype=np.uint8)
    mask = alpha_np > 0

    gray = np.array(Image.merge('RGB', (r, g, b)).convert('L'), dtype=np.float32)

    # Fill background with foreground mean to avoid hard steps at silhouette
    if mask.any():
        fg_mean = gray[mask].mean()
        gray_bg_filled = gray.copy()
        gray_bg_filled[~mask] = fg_mean
    else:
        gray_bg_filled = gray

    # Sobel gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    dz_dx = convolve(gray_bg_filled, sobel_x)
    dz_dy = convolve(gray_bg_filled, sobel_y)

    # Tangent-space normal (X = dZ/dX, Y = dZ/dY, Z = 1)
    intensity = 1.0
    dx = dz_dx * intensity
    dy = dz_dy * intensity
    norm = np.sqrt(dx * dx + dy * dy + 1.0)

    nx = ((dx / norm + 1.0) * 0.5 * 255.0).astype(np.uint8)
    ny = ((dy / norm + 1.0) * 0.5 * 255.0).astype(np.uint8)
    nz = ((1.0 / norm + 1.0) * 0.5 * 255.0).astype(np.uint8)

    normal = np.dstack([nx, ny, nz, alpha_np])

    # Force background to flat normal color (keeps alpha untouched)
    normal[..., 0][~mask] = 128
    normal[..., 1][~mask] = 128
    normal[..., 2][~mask] = 255

    n_img = Image.fromarray(normal, 'RGBA')

    # Alpha-aware edge padding around silhouette
    n_img = dilate_rgba(n_img, padding=edge_padding_px)
    return n_img


def generate_roughness_map(image: Image.Image) -> Image.Image:
    """Simple placeholder roughness: invert luminance. Returns single-channel 'L'."""
    return ImageOps.invert(image.convert('L'))

def dilate_rgba(img: Image.Image, padding: int = 16) -> Image.Image:
    """
    Bleeds edge RGB colors outward into transparent pixels (alpha-aware) to prevent haloing.
    Alpha is preserved. 'padding' limits how far the bleed goes.
    """
    arr = np.array(img.convert('RGBA'))
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    # Foreground mask (where alpha > 0)
    mask = alpha > 0
    inv = ~mask  # transparent/background

    if not inv.any():
        return img  # nothing to dilate

    # nearest foreground pixel indices for every background pixel
    _, (iy, ix) = distance_transform_edt(inv, return_indices=True)

    # Fill background pixels with nearest foreground RGB
    filled_rgb = rgb.copy()
    filled_rgb[inv] = rgb[iy[inv], ix[inv]]

    # Limit dilation distance to 'padding' pixels
    dist = distance_transform_edt(inv)
    limited_rgb = rgb.copy()
    reach = (inv) & (dist <= padding)
    limited_rgb[reach] = filled_rgb[reach]

    out = np.dstack([limited_rgb, alpha])
    return Image.fromarray(out, 'RGBA')

def dilate_gray_using_alpha(gray_img: Image.Image, alpha_img: Image.Image, padding: int = 16) -> Image.Image:
    """
    Dilation for single-channel textures (e.g., Roughness) using the RGBA dilater.
    Uses alpha as the foreground mask, bleeds grayscale values into transparent edges,
    and returns a single-channel 'L' image.
    """
    gray = gray_img.convert('L')
    a = alpha_img.convert('L')

    # Pack to RGBA so we can reuse dilate_rgba (put gray into RGB)
    merged = Image.merge('RGBA', (gray, gray, gray, a))
    dilated_rgba = dilate_rgba(merged, padding=padding)

    # Extract the red channel back as grayscale
    r, _, _, _ = dilated_rgba.split()
    return r

def save_variant(img, original_name, size_str, label='base', file_format='png', pbr_type=None, ratio_str='1:1'):
    """Saves an image variant to disk with robust format handling and safe filenames for asset pack."""
    logger.debug(f"Saving variant for asset pack: {original_name} ({label}, {size_str})")
    try:
        base_name, _ = os.path.splitext(original_name or "image")
        base_name = re.sub(r'[^a-zA-Z0-9_-]', '', base_name) or "image"

        pbr_suffix = f"_{pbr_type}" if pbr_type else ""
        unique_id = uuid.uuid4().hex[:8]

        # Safe ratio text
        ratio_txt = str(ratio_str or '1:1').replace(':', 'x')

        # Normalize extension/format
        ext = (file_format or 'png').lower()
        if ext == 'jpg':
            ext = 'jpeg'
        pil_fmt = {
            'png': 'PNG',
            'tga': 'TGA',
            'jpeg': 'JPEG',
            'webp': 'WEBP'
        }.get(ext, ext.upper())

        # Choose correct image mode per type/format
        image_to_save = img
        if pbr_type == 'roughness':
            # keep single-channel for engine-friendly roughness
            if image_to_save.mode != 'L':
                image_to_save = image_to_save.convert('L')
        else:
            # color textures / normals: keep alpha when supported
            if ext in ('jpeg',):  # formats without alpha
                image_to_save = image_to_save.convert('RGB')
            elif ext in ('png', 'tga', 'webp'):
                image_to_save = image_to_save.convert('RGBA')
            else:
                # default safe: RGB
                image_to_save = image_to_save.convert('RGB')

        fname = f"{base_name}_{label}_{ratio_txt}{pbr_suffix}_{size_str}_{unique_id}.{ext}"
        fpath = os.path.join(OUTPUT_DIR, fname) # Asset pack files saved locally then zipped

        save_kwargs = {}
        if pil_fmt == 'PNG':
            save_kwargs.update({'optimize': True})
        if pil_fmt == 'JPEG':
            save_kwargs.update({'quality': 95})  # tweak as desired

        # Special case: TGA expects 'RGBA' (or 'RGB'); Pillow handles 'L' too, but keep modes consistent.
        image_to_save.save(fpath, format=pil_fmt, **save_kwargs)

        logger.debug(f"Successfully saved variant to {fpath}")
        return {'status': 'success', 'path': fpath, 'arcname': fname}
    except Exception as e:
        logger.error(f"Failed to save variant {original_name} ({label}, {size_str}): {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}


# --- Asset Pack Generation Endpoint (Existing Logic) ---

@app.post("/generate_pack")
def generate_pack():
    """Generates an asset pack with image variants and PBR maps, then zips them."""
    logger.info("Received request for /generate_pack")

    # --- Auth & payload sanity ---
    api_key = request.headers.get('X-API-Key')
    if api_key != 'mock_api_key_123': # Placeholder API key
        return jsonify({'error': 'Unauthorized: Invalid API key'}), 401
    if not request.is_json:
        return jsonify({'error': 'JSON payload required'}), 400

    # --- Payload parsing ---
    data = request.get_json(force=True)
    image_urls = data.get('imageUrls', [])
    package_name = data.get('packageName', f"pack_{uuid.uuid4().hex[:6]}")
    sizes = data.get('sizes', [1024])
    formats = data.get('formats', ['png'])
    pbr_maps = bool(data.get('pbrMaps', False))
    aspect_ratio = data.get('aspectRatio', None)

    # Normal-map edge padding (pixels)
    edge_padding_px = int(data.get('edgePadding', 16))

    # Optional dilation for Base and Roughness
    dilate_base = bool(data.get('dilateBase', False))
    base_edge_padding_px = int(data.get('baseEdgePadding', edge_padding_px))
    dilate_roughness = bool(data.get('dilateRoughness', False))
    roughness_edge_padding_px = int(data.get('roughnessEdgePadding', edge_padding_px))

    # Parse selected variants from payload; fall back to your default set
    available_variants = set(variants_data.get('options', ['default']))
    requested_variants = data.get('variants', {}).get('options', None)
    if isinstance(requested_variants, list):
        selected_variant_names = [v for v in requested_variants if v in available_variants]
        if not selected_variant_names:
            logger.info("No valid variants in payload; falling back to defaults.")
            selected_variant_names = list(available_variants)
    else:
        selected_variant_names = list(available_variants)

    if not image_urls:
        return jsonify({'error': 'imageUrls array is required'}), 400

    zip_buffer = io.BytesIO()
    all_temp_files = [] # For local files before zipping

    try:
        with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
            for url in image_urls:
                img, original_name = load_image_from_url(url)
                if img is None:
                    continue

                # Iterate ONLY the variants selected by the frontend
                for variant_name in selected_variant_names:
                    mod_img = img
                    if variant_name == 'earthy':
                        mod_img = ImageEnhance.Color(img).enhance(0.7)
                    elif variant_name == 'vibrant':
                        mod_img = ImageEnhance.Color(img).enhance(1.3)
                    elif variant_name == 'muted':
                        mod_img = ImageEnhance.Color(img).enhance(0.5)
                    # 'default' leaves mod_img as-is

                    for size_val in sizes:
                        # -- Compute dimensions from ratio and resize --
                        dimensions = get_dimensions(size_val, aspect_ratio)
                        size_str_for_filename = f"{dimensions[0]}x{dimensions[1]}"
                        resized_img = mod_img.resize(dimensions, Image.Resampling.LANCZOS)

                        # =========================
                        # Base Color export (with optional edge dilation)
                        # =========================
                        base_to_save = resized_img
                        if dilate_base:
                            base_to_save = dilate_rgba(resized_img.convert('RGBA'), padding=base_edge_padding_px)

                        for fmt in formats:
                            result = save_variant(
                                base_to_save,
                                original_name,
                                size_str_for_filename,
                                label=variant_name,
                                file_format=fmt,
                                ratio_str=aspect_ratio
                            )
                            if result['status'] == 'success':
                                arcname = os.path.join('Textures', variant_name, result['arcname'])
                                zip_file.write(result['path'], arcname)
                                all_temp_files.append(result['path'])
                            else:
                                raise IOError(f"Failed to save base texture: {result.get('message')}")

                        # =========================
                        # PBR Maps export
                        # =========================
                        if pbr_maps:
                            # --- Normal map (alpha-aware + edge padding) ---
                            normal_img = generate_normal_map(resized_img, edge_padding_px=edge_padding_px)

                            # --- Roughness map (optional alpha-aware dilation) ---
                            rough_img = generate_roughness_map(resized_img)  # returns 'L'
                            if dilate_roughness:
                                alpha_from_resized = resized_img.convert('RGBA').split()[-1]
                                rough_img = dilate_gray_using_alpha(
                                    rough_img,
                                    alpha_from_resized,
                                    padding=roughness_edge_padding_px
                                )

                            pbr_results = {
                                'normal': normal_img,
                                'roughness': rough_img
                            }

                            # Save each PBR layer
                            for pbr_type, pbr_img in pbr_results.items():
                                for fmt in formats:
                                    result = save_variant(
                                        pbr_img,
                                        original_name,
                                        size_str_for_filename,
                                        label=variant_name,
                                        file_format=fmt,
                                        pbr_type=pbr_type,
                                        ratio_str=aspect_ratio
                                    )
                                    if result['status'] == 'success':
                                        arcname = os.path.join('PBR', pbr_type.capitalize(), variant_name, result['arcname'])
                                        zip_file.write(result['path'], arcname)
                                        all_temp_files.append(result['path'])
                                    else:
                                        raise IOError(f"Failed to save PBR map ({pbr_type}): {result.get('message')}")

                        # cleanup per-size
                        del resized_img, base_to_save
                        gc.collect()

                # cleanup per-image
                del img, mod_img
                gc.collect()

        zip_buffer.seek(0)
        zip_filename = f"{package_name}.zip"

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )

    except Exception as e:
        logger.error(f"An error occurred during pack generation: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate asset pack', 'details': str(e)}), 500
    finally:
        logger.info(f"Cleaning up {len(all_temp_files)} temporary files.")
        for fpath in all_temp_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception as e:
                logger.error(f"Error cleaning up file {fpath}: {e}")

# --- Endpoint to Handle Temporary File Uploads (NEW Supabase version) ---
@app.post("/upload_file")
def upload_file():
    logger.info("Received request for /upload_file")
    if not supabase:
        logger.error("Supabase client not initialized. File upload to Supabase disabled.")
        return jsonify({'error': 'Supabase service unavailable'}), 503

    if 'file' not in request.files:
        logger.error("No file part in request to /upload_file")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        filename = f"temp_{uuid.uuid4().hex[:12]}_{file.filename}"
        try:
            file_bytes = file.read() # Read file content into bytes
            # Upload to Supabase
            res = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, file_bytes)
            if res.status_code == 200:
                public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
                logger.info(f"File uploaded to Supabase: {public_url}")
                return jsonify({'file_url': public_url})
            else:
                logger.error(f"Supabase upload failed: {res.json()}")
                return jsonify({'error': 'Failed to upload file to Supabase', 'details': res.json()}), 500
        except Exception as e:
            logger.error(f"Error during Supabase file upload: {e}", exc_info=True)
            return jsonify({'error': 'Internal server error during upload', 'details': str(e)}), 500

    return jsonify({'error': 'File upload failed'}), 500

# --- Main Image Generation Endpoint (NEW Supabase version) ---
@app.post("/generate_image")
def generate_image():
    """Handles text-to-image, image-to-image (enhance), and inpainting, saving to Supabase."""
    logger.info("Received request for /generate_image")
    if not STABILITY_API_KEY:
        logger.error("Stability AI API key is not configured.")
        return jsonify({"error": "Image generation service is not configured."}), 503
    if not supabase: # Check Supabase initialization
        logger.error("Supabase client not initialized. Image upload to Supabase disabled.")
        return jsonify({"error": "Image generation service is not fully configured (Supabase)."}, 503)

    try:
        data = request.get_json(force=True)
        operation = data.get('operation', 'text-to-image')
        prompt = data.get('prompt')
        negative_prompt = data.get('negative_prompt', None)
        aspect_ratio = data.get('aspect_ratio', '1:1')
        denoising_strength = data.get('denoising_strength', 0.75)
        cfg_scale = data.get('cfg_scale', 7)
        steps = data.get('steps', 30)

        # --- Text-to-Image Operation ---
        if operation == 'text-to-image':
            api_url = f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/text-to-image"
            width, height = get_dimensions(1024, aspect_ratio)
            payload = {
                "text_prompts": [{"text": prompt}], # Use json for text_prompts
                "cfg_scale": cfg_scale, 
                "width": width, 
                "height": height,
                "samples": 1, 
                "steps": steps,
            }
            if negative_prompt:
                payload["text_prompts"].append({"text": negative_prompt, "weight": -1}) # Add negative prompt with weight -1

            response = requests.post(
                api_url, 
                headers={"Authorization": f"Bearer {STABILITY_API_KEY}", "Content-Type": "application/json"}, 
                json=payload # Send as JSON
            )

        # --- Image-to-Image & Inpainting Operations ---
        elif operation in ['enhance', 'inpaint']:
            init_image_url = data.get('init_image_url')
            if not init_image_url:
                return jsonify({"error": "Base image URL is required."}), 400
            
            # Temporary local directory to store downloaded images for API call
            temp_local_req_dir = os.path.join(OUTPUT_DIR, f"temp_req_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_local_req_dir, exist_ok=True)
            
            init_image_path = None
            mask_image_path = None
            
            try:
                # Download the base image from its URL (likely Supabase)
                init_img_resp = requests.get(init_image_url, timeout=20)
                init_img_resp.raise_for_status()
                init_image_path = os.path.join(temp_local_req_dir, "init_image.png")
                with open(init_image_path, 'wb') as f:
                    f.write(init_img_resp.content)

                payload_files = {"init_image": open(init_image_path, "rb")}
                
                # Stability AI's image-to-image/masking expects text_prompts as JSON string
                text_prompts_list = [{"text": prompt}]
                if negative_prompt:
                    text_prompts_list.append({"text": negative_prompt, "weight": -1})
                
                payload_data = {
                    "text_prompts": json.dumps(text_prompts_list), # Send as JSON string for multipart
                    "cfg_scale": cfg_scale,
                    "samples": 1,
                    "steps": steps,
                    "image_strength": denoising_strength # Denoising strength common for image-to-image
                }
                
                if operation == 'inpaint':
                    mask_image_url = data.get('mask_image_url')
                    if not mask_image_url:
                        return jsonify({"error": "Mask image URL is required for inpainting."}), 400
                    
                    mask_img_resp = requests.get(mask_image_url, timeout=20)
                    mask_img_resp.raise_for_status()
                    mask_image_path = os.path.join(temp_local_req_dir, "mask_image.png")
                    with open(mask_image_path, 'wb') as f:
                        f.write(mask_img_resp.content)
                    
                    payload_files["mask_image"] = open(mask_image_path, "rb")
                    payload_data["mask_source"] = "MASK_IMAGE_BLACK" # Indicates mask color to use as the masked area
                    api_url = f"{STABILITY_HOST}/v1/generation/stable-inpainting-512-v2-0/image-to-image/masking" # Inpainting model
                else: # 'enhance' operation
                    api_url = f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/image-to-image" # General image-to-image model

                response = requests.post(
                    api_url, 
                    headers={"Authorization": f"Bearer {STABILITY_API_KEY}"}, 
                    files=payload_files,
                    data=payload_data
                )

            finally:
                # Close file handles and clean up temporary local files
                if "init_image" in payload_files:
                    payload_files["init_image"].close()
                if "mask_image" in payload_files:
                    payload_files["mask_image"].close()
                
                if os.path.exists(temp_local_req_dir):
                    if init_image_path and os.path.exists(init_image_path):
                        os.remove(init_image_path)
                    if mask_image_path and os.path.exists(mask_image_path):
                        os.remove(mask_image_path)
                    try:
                        os.rmdir(temp_local_req_dir) # Remove dir only if empty
                    except OSError as e:
                        logger.warning(f"Could not remove temp directory {temp_local_req_dir}: {e}")

        else:
            return jsonify({"error": f"Invalid operation: {operation}"}), 400

        # --- Process Stability AI Response and Upload to Supabase ---
        if response.status_code != 200:
            logger.error(f"Stability AI API error: {response.status_code} - {response.text}")
            return jsonify({"error": "Failed to generate image.", "details": response.text}), response.status_code

        response_json = response.json()
        if not response_json.get("artifacts"):
            return jsonify({"error": "No image artifacts found in Stability AI response."}), 500
            
        base64_image = response_json["artifacts"][0]["base64"]
        img_data_bytes = base64.b64decode(base64_image)
        
        output_filename = f"gen_{uuid.uuid4().hex[:12]}.png"
        
        # Upload generated image to Supabase
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).upload(output_filename, img_data_bytes)
            if res.status_code == 200:
                final_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(output_filename)
                logger.info(f"Image generated and uploaded to Supabase: {final_url}")
                return jsonify({"url": final_url})
            else:
                logger.error(f"Supabase upload of generated image failed: {res.json()}")
                return jsonify({'error': 'Failed to save generated image to Supabase', 'details': res.json()}), 500
        except Exception as e:
            logger.error(f"Error during Supabase upload of generated image: {e}", exc_info=True)
            return jsonify({'error': 'Internal server error saving generated image', 'details': str(e)}), 500

    except Exception as e:
        logger.error(f"Error in /generate_image: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error.', 'details': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))