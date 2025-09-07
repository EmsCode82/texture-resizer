from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError, ImageOps, ImageEnhance
import io
import os
import uuid
import requests
import re
from flask_cors import CORS
import numpy as np
from scipy.ndimage import convolve
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import logging

# --- Basic Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Helper Function for Aspect Ratios ---
def get_dimensions(base_size, ratio_str):
    """Calculates width and height based on a base size and ratio string (e.g., '16:9')."""
    try:
        w, h = map(int, ratio_str.split(':'))
        if w >= h:
            # Landscape or square
            return (base_size, int(base_size * h / w))
        else:
            # Portrait
            return (int(base_size * w / h), base_size)
    except (ValueError, ZeroDivisionError):
        # Fallback to square if ratio is invalid
        return (base_size, base_size)

# --- Image Processing Functions ---

def load_image_from_url(url):
    logger = logging.getLogger(__name__)
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

def generate_normal_map(image):
    gray_image = image.convert('L')
    np_image = np.array(gray_image, dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    dz_dx = convolve(np_image, sobel_x)
    dz_dy = convolve(np_image, sobel_y)
    
    intensity = 1.0
    normal_map = np.zeros((*np_image.shape, 3), dtype=np.uint8)
    dz_dx_norm = dz_dx * intensity
    dz_dy_norm = dz_dy * intensity
    
    norm = np.sqrt(dz_dx_norm**2 + dz_dy_norm**2 + 1)
    normal_map[..., 0] = (dz_dx_norm / norm + 1) / 2 * 255
    normal_map[..., 1] = (dz_dy_norm / norm + 1) / 2 * 255
    normal_map[..., 2] = (1 / norm + 1) / 2 * 255
    
    return Image.fromarray(normal_map, 'RGB').convert('RGBA')

def generate_roughness_map(image):
    return ImageOps.invert(image.convert('L')).convert('RGBA')

def save_variant(img, original_name, size_str, label='base', file_format='png', pbr_type=None, ratio_str='1x1'):
    """MODIFIED: Accepts size_str and ratio_str for filename"""
    logger = logging.getLogger(__name__)
    try:
        base_name, _ = os.path.splitext(original_name)
        base_name = re.sub(r'[^a-zA-Z0-9_-]', '', base_name)
        
        pbr_suffix = f"_{pbr_type}" if pbr_type else ""
        unique_id = uuid.uuid4().hex[:8]
        
        # Add ratio to filename to ensure uniqueness
        sanitized_ratio = ratio_str.replace(':', 'x')
        fname = f"{base_name}_{label}_{sanitized_ratio}{pbr_suffix}_{size_str}_{unique_id}.{file_format}"
        fpath = os.path.join(OUTPUT_DIR, fname)
        
        if file_format.lower() == 'tga':
            img.convert('RGBA').save(fpath, format='TGA')
        else:
            img.convert('RGBA').save(fpath, format='PNG')
            
        logger.debug(f"Successfully saved variant to {fpath}")
        return {'status': 'success', 'path': fpath, 'arcname': fname}
    except Exception as e:
        logger.error(f"Failed to save variant {original_name} ({label}, {size_str}): {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

# --- Main API Endpoint ---

@app.post("/generate_pack")
def generate_pack():
    """MODIFIED: Reads ratio and updates resize/save logic"""
    logger = logging.getLogger(__name__)
    logger.info("Received request for /generate_pack")
    
    api_key = request.headers.get('X-API-Key')
    if api_key != 'mock_api_key_123':
        return jsonify({'error': 'Unauthorized: Invalid API key'}), 401
    if not request.is_json:
        return jsonify({'error': 'JSON payload required'}), 400
    
    data = request.get_json()
    image_urls = data.get('imageUrls', [])
    package_name = data.get('packageName', 'asset-pack')
    sizes = data.get('sizes', [512, 1024])
    formats = data.get('formats', ['png'])
    pbr_maps = data.get('pbrMaps', False)
    variants_data = data.get('variants', {'options': ['default']})
    aspect_ratio = data.get('aspectRatio', '1:1') # NEW: Get aspect ratio

    if not image_urls:
        return jsonify({'error': 'imageUrls array is required'}), 400

    zip_buffer = io.BytesIO()
    all_temp_files = []

    try:
        with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
            for url in image_urls:
                img, original_name = load_image_from_url(url)
                if img is None: continue

                for variant_name in variants_data.get('options', ['default']):
                    mod_img = img
                    if variant_name == 'earthy': mod_img = ImageEnhance.Color(img).enhance(0.7)
                    elif variant_name == 'vibrant': mod_img = ImageEnhance.Color(img).enhance(1.3)
                    elif variant_name == 'muted': mod_img = ImageEnhance.Color(img).enhance(0.5)

                    for size_val in sizes:
                        # MODIFIED: Calculate dimensions based on ratio
                        dimensions = get_dimensions(size_val, aspect_ratio)
                        size_str_for_filename = f"{dimensions[0]}x{dimensions[1]}"
                        resized_img = mod_img.resize(dimensions, Image.Resampling.LANCZOS)
                        
                        # --- Save Base Texture ---
                        for fmt in formats:
                            result = save_variant(resized_img, original_name, size_str_for_filename, label=variant_name, file_format=fmt, ratio_str=aspect_ratio)
                            if result['status'] == 'success':
                                arcname = os.path.join('Textures', variant_name, result['arcname'])
                                zip_file.write(result['path'], arcname)
                                all_temp_files.append(result['path'])
                            else:
                                raise IOError(f"Failed to save base texture: {result.get('message')}")
                        
                        # --- Save PBR Maps ---
                        if pbr_maps:
                            pbr_results = {
                                'normal': generate_normal_map(resized_img),
                                'roughness': generate_roughness_map(resized_img)
                            }
                            for pbr_type, pbr_img in pbr_results.items():
                                for fmt in formats:
                                    result = save_variant(pbr_img, original_name, size_str_for_filename, label=variant_name, file_format=fmt, pbr_type=pbr_type, ratio_str=aspect_ratio)
                                    if result['status'] == 'success':
                                        arcname = os.path.join('PBR', pbr_type.capitalize(), variant_name, result['arcname'])
                                        zip_file.write(result['path'], arcname)
                                        all_temp_files.append(result['path'])
                                    else:
                                        raise IOError(f"Failed to save PBR map: {result.get('message')}")
                        
                        del resized_img
                        gc.collect()

                del img
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
                if os.path.exists(fpath): os.remove(fpath)
            except Exception as e:
                logger.error(f"Error cleaning up file {fpath}: {e}")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))