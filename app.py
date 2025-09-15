from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError, ImageOps, ImageEnhance
import io
import os
import uuid
import requests
import re
import base64
import json
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
    logger.warning("Supabase URL or Key not found. Supabase features will be disabled.")

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
    """Calculates width and height based on a base size and ratio string (e.g., '16:9'). Falls back to square."""
    if not ratio_str:
        return (base_size, base_size)
    try:
        w, h = map(int, str(ratio_str).split(':'))
        if w <= 0 or h <= 0:
            return (base_size, base_size)
        if w >= h:
            return (base_size, max(1, int(base_size * h / w)))
        else:
            return (max(1, int(base_size * w / h)), base_size)
    except Exception:
        return (base_size, base_size)

# --- Image Processing Functions (keeping all your existing ones) ---
def load_image_from_url(url):
    try:
        logger.debug(f"Fetching image from URL: {url}")
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
    """Alpha-aware Sobel normal map generator."""
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
    """Simple placeholder roughness: invert luminance."""
    return ImageOps.invert(image.convert('L'))

def dilate_rgba(img: Image.Image, padding: int = 16) -> Image.Image:
    """Bleeds edge RGB colors outward into transparent pixels."""
    arr = np.array(img.convert('RGBA'))
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
    reach = (inv) & (dist <= padding)
    limited_rgb[reach] = filled_rgb[reach]

    out = np.dstack([limited_rgb, alpha])
    return Image.fromarray(out, 'RGBA')

def dilate_gray_using_alpha(gray_img: Image.Image, alpha_img: Image.Image, padding: int = 16) -> Image.Image:
    """Dilation for single-channel textures using alpha."""
    gray = gray_img.convert('L')
    a = alpha_img.convert('L')

    merged = Image.merge('RGBA', (gray, gray, gray, a))
    dilated_rgba = dilate_rgba(merged, padding=padding)

    r, _, _, _ = dilated_rgba.split()
    return r

def save_variant(img, original_name, size_str, label='base', file_format='png', pbr_type=None, ratio_str='1:1'):
    """Saves an image variant to disk."""
    try:
        base_name, _ = os.path.splitext(original_name or "image")
        base_name = re.sub(r'[^a-zA-Z0-9_-]', '', base_name) or "image"

        pbr_suffix = f"_{pbr_type}" if pbr_type else ""
        unique_id = uuid.uuid4().hex[:8]

        ratio_txt = str(ratio_str or '1:1').replace(':', 'x')

        ext = (file_format or 'png').lower()
        if ext == 'jpg':
            ext = 'jpeg'
        pil_fmt = {'png': 'PNG', 'jpeg': 'JPEG', 'webp': 'WEBP'}.get(ext, 'PNG')

        filename = f"{base_name}_{size_str}_{label}{pbr_suffix}_{ratio_txt}_{unique_id}.{ext}"
        path = os.path.join(OUTPUT_DIR, filename)

        if pil_fmt == 'JPEG':
            if img.mode in ('RGBA', 'LA'):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    bg.paste(img, mask=img.split()[3])
                else:
                    bg.paste(img, mask=img.split()[1])
                img = bg
            img.save(path, pil_fmt, quality=95, optimize=True)
        else:
            img.save(path, pil_fmt, optimize=True)

        return filename, path
    except Exception as e:
        logger.error(f"Error saving variant: {e}")
        return None, None

def apply_color_variant(image, variant='default'):
    """Applies color variants to an image."""
    if variant == 'default':
        return image

    img = image.convert('RGBA')
    
    if variant == 'earthy':
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.7)
        r, g, b, a = img.split()
        r = r.point(lambda x: min(255, int(x * 1.1 + 10)))
        g = g.point(lambda x: min(255, int(x * 0.95 + 5)))
        b = b.point(lambda x: min(255, int(x * 0.8)))
        img = Image.merge('RGBA', (r, g, b, a))
    elif variant == 'vibrant':
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.05)
    elif variant == 'muted':
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.6)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(0.9)
    
    return img

# --- API Endpoints ---

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Upload file endpoint - now uploads to Supabase if available."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read file content
        file_content = file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"

        if supabase:
            try:
                # Upload to Supabase
                res = supabase.storage.from_(SUPABASE_BUCKET).upload(
                    unique_filename, file_content
                )
                if _sb_ok(res):
                    public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)
                    return jsonify({"file_url": public_url})
                else:
                    logger.error(f"Supabase upload failed: {res}")
                    return jsonify({"error": "Upload to storage failed"}), 500
            except Exception as e:
                logger.error(f"Supabase upload error: {e}")
                return jsonify({"error": "Storage service error"}), 500
        else:
            # Fallback to local storage
            local_path = os.path.join(OUTPUT_DIR, unique_filename)
            with open(local_path, 'wb') as f:
                f.write(file_content)
            return jsonify({"file_url": f"/files/{unique_filename}"})

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "File upload failed"}), 500

@app.route('/generate_image', methods=['POST'])
def generate_image():
    """Generate image using Stability AI with support for text-to-image, enhance, and inpaint."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        prompt = data.get('prompt', '').strip()
        negative_prompt = data.get('negative_prompt', '').strip()
        aspect_ratio = data.get('aspect_ratio', '1:1')
        operation = data.get('operation', 'text-to-image')
        denoising_strength = float(data.get('denoising_strength', 0.75))
        cfg_scale = int(data.get('cfg_scale', 7))
        steps = int(data.get('steps', 30))

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        logger.info(f"Generating image: operation={operation}, prompt='{prompt[:50]}...'")

        # Get dimensions based on aspect ratio
        width, height = get_dimensions(1024, aspect_ratio)

        if operation == 'text-to-image':
            # Standard text-to-image generation
            payload = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": cfg_scale,
                "height": height,
                "width": width,
                "samples": 1,
                "steps": steps,
            }
            if negative_prompt:
                payload["text_prompts"].append({"text": negative_prompt, "weight": -1})

            response = requests.post(
                f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/text-to-image",
                headers={"Authorization": f"Bearer {STABILITY_API_KEY}"},
                json=payload
            )

        elif operation in ['enhance', 'inpaint']:
            # Image-to-image operations
            init_image_url = data.get('init_image_url')
            if not init_image_url:
                return jsonify({"error": "Base image URL is required for image-to-image operations"}), 400

            # Create temporary directory for this request
            temp_dir = os.path.join(OUTPUT_DIR, f"temp_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)

            try:
                # Download base image
                base_resp = requests.get(init_image_url, timeout=20)
                base_resp.raise_for_status()
                init_image_path = os.path.join(temp_dir, "init_image.png")
                with open(init_image_path, 'wb') as f:
                    f.write(base_resp.content)

                # Prepare multipart payload
                files = {"init_image": open(init_image_path, "rb")}
                
                text_prompts = [{"text": prompt}]
                if negative_prompt:
                    text_prompts.append({"text": negative_prompt, "weight": -1})

                payload_data = {
                    "text_prompts": json.dumps(text_prompts),
                    "cfg_scale": cfg_scale,
                    "samples": 1,
                    "steps": steps,
                    "image_strength": denoising_strength
                }

                if operation == 'inpaint':
                    # Handle mask - either from URL or base64
                    mask_image_url = data.get('mask_image_url')
                    mask_image_base64 = data.get('mask_image_base64')

                    if not mask_image_url and not mask_image_base64:
                        return jsonify({"error": "Mask image is required for inpainting"}), 400

                    mask_image_path = os.path.join(temp_dir, "mask_image.png")
                    
                    if mask_image_url:
                        # Download mask from URL
                        mask_resp = requests.get(mask_image_url, timeout=20)
                        mask_resp.raise_for_status()
                        with open(mask_image_path, 'wb') as f:
                            f.write(mask_resp.content)
                    else:
                        # Decode base64 mask
                        base64_data = mask_image_base64
                        if base64_data.startswith("data:"):
                            base64_data = base64_data.split(",", 1)[1]
                        with open(mask_image_path, 'wb') as f:
                            f.write(base64.b64decode(base64_data))

                    files["mask_image"] = open(mask_image_path, "rb")
                    payload_data["mask_source"] = "MASK_IMAGE_BLACK"
                    api_url = f"{STABILITY_HOST}/v1/generation/stable-inpainting-512-v2-0/image-to-image/masking"
                else:
                    api_url = f"{STABILITY_HOST}/v1/generation/stable-diffusion-v1-6/image-to-image"

                response = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {STABILITY_API_KEY}"},
                    files=files,
                    data=payload_data
                )

            finally:
                # Clean up files
                for file_obj in files.values():
                    try:
                        file_obj.close()
                    except:
                        pass
                # Remove temp directory
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass

        else:
            return jsonify({"error": f"Unknown operation: {operation}"}), 400

        # Handle Stability AI response
        if response.status_code != 200:
            error_msg = f"Stability AI API error: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - {response.text}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), response.status_code

        response_data = response.json()
        if not response_data.get('artifacts'):
            return jsonify({"error": "No image generated"}), 500

        # Get the generated image data
        image_data = base64.b64decode(response_data['artifacts'][0]['base64'])
        
        # Upload to Supabase or save locally
        unique_filename = f"generated_{uuid.uuid4().hex}.png"
        
        if supabase:
            try:
                res = supabase.storage.from_(SUPABASE_BUCKET).upload(
                    unique_filename, image_data
                )
                if _sb_ok(res):
                    public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)
                    return jsonify({"url": public_url})
                else:
                    logger.error(f"Failed to upload generated image to Supabase: {res}")
                    # Fallback to local
            except Exception as e:
                logger.error(f"Supabase upload error for generated image: {e}")
                # Fallback to local

        # Local fallback
        local_path = os.path.join(OUTPUT_DIR, unique_filename)
        with open(local_path, 'wb') as f:
            f.write(image_data)
        return jsonify({"url": f"/files/{unique_filename}"})

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return jsonify({"error": "Image generation failed"}), 500

@app.route('/generate_pack', methods=['POST'])
def generate_pack():
    """Generate asset pack with PBR maps."""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        config = data.get('config', {})

        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        logger.info(f"Generating asset pack for image: {image_url}")

        # Load the source image
        source_image, source_name = load_image_from_url(image_url)
        if not source_image:
            return jsonify({"error": "Failed to load source image"}), 400

        # Extract configuration
        sizes = config.get('sizes', ['1024'])
        formats = config.get('formats', ['png'])
        color_variants = config.get('color_variants', ['default'])
        pbr_maps = config.get('pbr_maps', [])
        padding = config.get('padding', 0)
        aspect_ratio = config.get('aspect_ratio', '1:1')

        # Create a unique directory for this pack
        pack_id = uuid.uuid4().hex[:12]
        pack_dir = os.path.join(OUTPUT_DIR, f"pack_{pack_id}")
        os.makedirs(pack_dir, exist_ok=True)

        generated_files = []

        try:
            for size_str in sizes:
                size = int(size_str)
                w, h = get_dimensions(size, aspect_ratio)
                
                for variant in color_variants:
                    # Apply color variant
                    variant_image = apply_color_variant(source_image, variant)
                    
                    # Resize image
                    resized = variant_image.resize((w, h), Image.LANCZOS)
                    
                    # Apply padding if specified
                    if padding > 0:
                        new_w = w + 2 * padding
                        new_h = h + 2 * padding
                        padded = Image.new('RGBA', (new_w, new_h), (0, 0, 0, 0))
                        padded.paste(resized, (padding, padding))
                        resized = padded

                    # Save base image in all requested formats
                    for fmt in formats:
                        filename, filepath = save_variant(
                            resized, source_name, size_str, variant, fmt, None, aspect_ratio
                        )
                        if filename:
                            generated_files.append((filename, filepath))

                    # Generate PBR maps if requested
                    for pbr_type in pbr_maps:
                        if pbr_type == 'normal':
                            pbr_image = generate_normal_map(resized)
                        elif pbr_type == 'roughness':
                            pbr_image = generate_roughness_map(resized)
                        else:
                            continue  # Skip unknown PBR types

                        for fmt in formats:
                            filename, filepath = save_variant(
                                pbr_image, source_name, size_str, variant, fmt, pbr_type, aspect_ratio
                            )
                            if filename:
                                generated_files.append((filename, filepath))

            # Create ZIP file
            zip_filename = f"asset_pack_{pack_id}.zip"
            zip_path = os.path.join(OUTPUT_DIR, zip_filename)
            
            with ZipFile(zip_path, 'w', ZIP_DEFLATED) as zipf:
                for filename, filepath in generated_files:
                    zipf.write(filepath, filename)

            # Upload ZIP to Supabase or serve locally
            if supabase:
                try:
                    with open(zip_path, 'rb') as zip_file:
                        zip_data = zip_file.read()
                    
                    res = supabase.storage.from_(SUPABASE_BUCKET).upload(
                        zip_filename, zip_data
                    )
                    if _sb_ok(res):
                        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(zip_filename)
                        return jsonify({
                            "download_url": public_url,
                            "file_count": len(generated_files),
                            "pack_id": pack_id
                        })
                except Exception as e:
                    logger.error(f"Failed to upload ZIP to Supabase: {e}")

            # Local fallback
            return jsonify({
                "download_url": f"/files/{zip_filename}",
                "file_count": len(generated_files),
                "pack_id": pack_id
            })

        finally:
            # Clean up temporary files
            try:
                import shutil
                for filename, filepath in generated_files:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                if os.path.exists(pack_dir):
                    shutil.rmtree(pack_dir)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    except Exception as e:
        logger.error(f"Asset pack generation error: {e}")
        return jsonify({"error": "Asset pack generation failed"}), 500

@app.route('/files/<filename>')
def serve_file(filename):
    """Serve files from local storage (fallback when Supabase unavailable)."""
    try:
        return send_file(os.path.join(OUTPUT_DIR, filename))
    except Exception as e:
        logger.error(f"File serving error: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/variants')
def get_variants():
    """Get available color variants."""
    return jsonify(variants_data)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)