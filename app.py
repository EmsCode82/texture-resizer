from flask import Flask, request, jsonify, send_file, url_for
from PIL import Image, UnidentifiedImageError, ImageOps, ImageEnhance
import io
import os
import uuid
import requests
import re
from flask_cors import CORS
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import logging
import base64

# --- Basic Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
OUTPUT_DIR = "output"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
variants_data = {
    'options': ['default', 'earthy', 'vibrant', 'muted'] 
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Stability AI Configuration ---
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_API_HOST = "https://api.stability.ai"
# UPDATED: We'll use different endpoints based on the operation
TXT2IMG_ENGINE_ID = "stable-diffusion-xl-1024-v1-0"
INPAINT_ENGINE_ID = "stable-inpainting-v1-0" 

if not STABILITY_API_KEY:
    logging.warning("STABILITY_API_KEY environment variable not set. /generate_image will fail.")

# --- Helper & Image Processing Functions (No changes here) ---
def get_dimensions(base_size, ratio_str):
    if not ratio_str:
        return (base_size, base_size)
    try:
        w, h = map(int, str(ratio_str).split(':'))
        if w <= 0 or h <= 0: return (base_size, base_size)
        if w >= h: return (base_size, max(1, int(base_size * h / w)))
        else: return (max(1, int(base_size * w / h)), base_size)
    except Exception: return (base_size, base_size)

def load_image_from_url(url):
    logger = logging.getLogger(__name__)
    try:
        headers = {'User-Agent': 'Assetgineer-Service/1.0'}
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        return img
    except Exception as e:
        logger.error(f"Failed to load image from URL {url}: {e}")
        return None

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
        gray = gray_bg_filled
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = convolve(gray, sobel_x)
    grad_y = convolve(gray, sobel_y)
    normal_map_np = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    normal_map_np[..., 0] = np.clip((-grad_x / np.sqrt(grad_x**2 + grad_y**2 + 1)) * 127.5 + 127.5, 0, 255)
    normal_map_np[..., 1] = np.clip((grad_y / np.sqrt(grad_x**2 + grad_y**2 + 1)) * 127.5 + 127.5, 0, 255)
    normal_map_np[..., 2] = 255
    normal_map_np[~mask] = [128, 128, 255]
    normal_map = Image.fromarray(normal_map_np, 'RGB').convert('RGBA')
    r_n, g_n, b_n, _ = normal_map.split()
    final_rgba = Image.merge('RGBA', [r_n, g_n, b_n, a])
    dist_transform = distance_transform_edt(alpha_np == 0, return_distances=True, return_indices=False)
    edge_mask = (dist_transform > 0) & (dist_transform <= edge_padding_px)
    if np.any(edge_mask):
        final_array = np.array(final_rgba)
        r, g, b, alpha_channel = final_array[..., 0], final_array[..., 1], final_array[..., 2], final_array[..., 3]
        r[edge_mask] = 128
        g[edge_mask] = 128
        b[edge_mask] = 255
        alpha_channel[edge_mask] = 0
        final_rgba = Image.fromarray(np.stack([r, g, b, alpha_channel], axis=-1), 'RGBA')
    return final_rgba

def generate_metalness_map(image: Image.Image, threshold: int = 128) -> Image.Image:
    return image.convert('L').point(lambda x: 255 if x > threshold else 0).convert('RGBA')

def generate_roughness_map(image: Image.Image, invert: bool = False) -> Image.Image:
    img = image.convert('L')
    if invert:
        img = ImageOps.invert(img)
    return img.convert('RGBA')

def enhance_image(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(1.5)

# --- API Endpoints ---
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        file_url = url_for('static', filename=f'uploads/{filename}', _external=True)
        return jsonify({'file_url': file_url})
    return jsonify({'error': 'File upload failed'}), 500

@app.route('/generate_image', methods=['POST'])
def generate_image_route():
    logger = logging.getLogger(__name__)
    data = request.json
    operation = data.get('operation', 'text-to-image')
    
    # --- UPDATED: Inpainting Logic ---
    if operation == 'inpaint':
        logger.info("Starting inpainting operation...")
        if not STABILITY_API_KEY:
            return jsonify({'error': 'Stability API key not configured'}), 500
        
        init_image_url = data.get('init_image_url')
        mask_image_url = data.get('mask_image_url')
        prompt = data.get('prompt')

        if not all([init_image_url, mask_image_url, prompt]):
            return jsonify({'error': 'Missing init_image_url, mask_image_url, or prompt for inpainting'}), 400

        init_image_bytes = io.BytesIO()
        mask_image_bytes = io.BytesIO()
        try:
            init_img = load_image_from_url(init_image_url)
            mask_img = load_image_from_url(mask_image_url)
            
            # Ensure mask is single channel 'L' and binarized for Stability
            mask_img = mask_img.convert('L').point(lambda p: 255 if p > 128 else 0)
            
            init_img.save(init_image_bytes, format='PNG')
            mask_img.save(mask_image_bytes, format='PNG')
            
            init_image_bytes.seek(0)
            mask_image_bytes.seek(0)
        except Exception as e:
            logger.error(f"Failed to process images for inpainting: {e}")
            return jsonify({'error': 'Failed to load or process initial images'}), 500

        api_url = f"{STABILITY_API_HOST}/v1/generation/{INPAINT_ENGINE_ID}/image-to-image/masking"
        
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {STABILITY_API_KEY}"},
            files={
                'init_image': init_image_bytes,
                'mask_image': mask_image_bytes,
            },
            data={
                "mask_source": "MASK_IMAGE_BLACK",
                "text_prompts[0][text]": prompt,
                "text_prompts[0][weight]": 1,
                "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "samples": 1,
                "steps": 30,
            }
        )

    # --- UPDATED: Text-to-Image / Enhance Logic ---
    else:
        logger.info(f"Starting {operation} operation...")
        if not STABILITY_API_KEY:
            return jsonify({'error': 'Stability API key not configured'}), 500
        
        width, height = get_dimensions(1024, data.get('aspect_ratio'))
        api_url = f"{STABILITY_API_HOST}/v1/generation/{TXT2IMG_ENGINE_ID}/text-to-image"
        
        payload = {
            "text_prompts": [{"text": data.get('prompt', ''), "weight": 1.0}],
            "cfg_scale": 7,
            "height": height,
            "width": width,
            "samples": 1,
            "steps": 30,
        }
        if data.get('negative_prompt'):
            payload["text_prompts"].append({"text": data.get('negative_prompt'), "weight": -1.0})

        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {STABILITY_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
        )

    # --- Common Response Handling ---
    if response.status_code != 200:
        logger.error(f"Stability API error: {response.status_code} - {response.text}")
        return jsonify({"error": "Failed to generate image on server.", "details": response.text}), 500

    artifacts = response.json().get("artifacts", [])
    if not artifacts:
        return jsonify({"error": "No image artifacts returned from API"}), 500
    
    image_data = base64.b64decode(artifacts[0]["base64"])
    image = Image.open(io.BytesIO(image_data))
    
    filename = f"generated_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath, format='PNG')
    
    file_url = url_for('static', filename=f'uploads/{filename}', _external=True)
    return jsonify({'url': file_url})


@app.route('/process_maps', methods=['POST'])
def process_maps_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    try:
        image = Image.open(file.stream).convert('RGBA')
    except (UnidentifiedImageError, IOError):
        return jsonify({'error': 'Invalid or corrupted image file'}), 400

    try:
        normal_map = generate_normal_map(image)
        metalness_map = generate_metalness_map(image)
        roughness_map = generate_roughness_map(image)
        enhanced_image = enhance_image(image)
    except Exception as e:
        logging.error(f"Error during map generation: {e}")
        return jsonify({'error': 'An error occurred during map processing'}), 500

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'a', ZIP_DEFLATED, False) as zip_file:
        for map_img, name in [(normal_map, 'normal.png'), (metalness_map, 'metalness.png'), (roughness_map, 'roughness.png'), (enhanced_image, 'enhanced.png')]:
            img_buffer = io.BytesIO()
            map_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zip_file.writestr(name, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    
    gc.collect()
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='asset_maps.zip'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)