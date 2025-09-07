from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError, ImageOps, ImageFilter, ImageEnhance
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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    normal_map[..., 0] = (dz_dx_norm / norm + 1) / 2 * 255  # X (Red)
    normal_map[..., 1] = (dz_dy_norm / norm + 1) / 2 * 255  # Y (Green)
    normal_map[..., 2] = (1 / norm + 1) / 2 * 255          # Z (Blue)
    
    return Image.fromarray(normal_map, 'RGB').convert('RGBA')

def generate_roughness_map(image):
    return ImageOps.invert(image.convert('L')).convert('RGBA')

def save_variant(img, original_name, size, label='base', file_format='png', pbr_type=None):
    logger = logging.getLogger(__name__)
    try:
        size_str = f"{size[0]}x{size[1]}"
        base_name, _ = os.path.splitext(original_name)
        base_name = re.sub(r'[^a-zA-Z0-9_-]', '', base_name) # Sanitize name
        
        pbr_suffix = f"_{pbr_type}" if pbr_type else ""
        unique_id = uuid.uuid4().hex[:8]
        
        fname = f"{base_name}_{label}{pbr_suffix}_{size_str}_{unique_id}.{file_format}"
        fpath = os.path.join(OUTPUT_DIR, fname)
        
        if file_format.lower() == 'tga':
            save_img = img.convert('RGBA')
            save_img.save(fpath, format='TGA')
        else:
            img.convert('RGBA').save(fpath, format='PNG')
            
        logger.debug(f"Successfully saved variant to {fpath}")
        return {'status': 'success', 'path': fpath}
    except Exception as e:
        logger.error(f"Failed to save variant {original_name} ({label}, {size_str}): {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

# --- Main API Endpoint ---

@app.post("/generate_pack")
def generate_pack():
    logger = logging.getLogger(__name__)
    logger.info("Received request for /generate_pack")

    api_key = request.headers.get('X-API-Key')
    if api_key != 'mock_api_key_123':
        logger.warning("Unauthorized access attempt with invalid API key.")
        return jsonify({'error': 'Unauthorized: Invalid API key'}), 401

    if not request.is_json:
        logger.error("Request is not JSON.")
        return jsonify({'error': 'JSON payload required'}), 400

    data = request.get_json()
    image_urls = data.get('imageUrls', [])
    sizes = data.get('sizes', [512, 1024])
    formats = data.get('formats', ['png'])
    variants_data = data.get('variants', {'options': ['default']})
    pbr_maps = data.get('pbrMaps', False)
    package_name = data.get('packageName', 'asset-pack')
    package_name = re.sub(r'[^a-zA-Z0-9_-]', '', package_name) or "asset-pack"

    if not image_urls:
        return jsonify({'error': 'At least one imageUrl is required'}), 400
    
    all_temp_files = []
    file_manifest = []

    try:
        for url in image_urls:
            img, original_name = load_image_from_url(url)
            if img is None:
                raise ValueError(f"Failed to load image from URL: {url}")

            for variant_name in variants_data.get('options', ['default']):
                mod_img = img.copy()
                if variant_name == 'earthy': mod_img = ImageEnhance.Color(mod_img).enhance(0.7)
                elif variant_name == 'vibrant': mod_img = ImageEnhance.Color(mod_img).enhance(1.3)
                elif variant_name == 'muted': mod_img = ImageEnhance.Color(mod_img).enhance(0.5)

                for size_val in sizes:
                    resized_img = mod_img.resize((size_val, size_val), Image.Resampling.LANCZOS)
                    
                    for fmt in formats:
                        # Save base texture
                        result = save_variant(resized_img, original_name, (size_val, size_val), label=variant_name, file_format=fmt)
                        if result['status'] != 'success':
                            raise IOError(f"Failed to save base texture: {result.get('message', 'Unknown error')}")
                        all_temp_files.append(result['path'])
                        file_manifest.append({'name': os.path.basename(result['path']), 'type': 'Texture', 'variant': variant_name})
                    
                    if pbr_maps:
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
                    
                    gc.collect()

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zipf:
            for file_path in all_temp_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    variant_subfolder = 'default'
                    for v_opt in variants_data.get('options', ['default']):
                        if f"_{v_opt}_" in filename:
                            variant_subfolder = v_opt
                            break
                    
                    if "_normal_" in filename or "_roughness_" in filename:
                        arcname = os.path.join("PBR", variant_subfolder, filename)
                    else:
                        arcname = os.path.join("Textures", variant_subfolder, filename)
                    
                    zipf.write(file_path, arcname)
            
            readme_content = f"Asset Pack: {package_name}\nGenerated by Assetgineer\nVariants: {', '.join(variants_data.get('options', ['default']))}"
            zipf.writestr("README.txt", readme_content)

        zip_buffer.seek(0)
        
        logger.info(f"Successfully created ZIP file '{package_name}.zip' with {len(all_temp_files)} files.")
        
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=f"{package_name}.zip",
            mimetype='application/zip'
        )

    except Exception as e:
        logger.critical(f"A critical error occurred in generate_pack: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during package generation.", "details": str(e)}), 500
    finally:
        # Clean up temporary files
        for f in all_temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logger.debug(f"Cleaned up temp file: {f}")
                except OSError as e:
                    logger.error(f"Error removing temp file {f}: {e}")
        gc.collect()

@app.route('/files/<filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)