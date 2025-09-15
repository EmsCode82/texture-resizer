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
# NEW: Define a static folder for uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
variants_data = {
    'options': ['default', 'earthy', 'vibrant', 'muted'] 
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- NEW: Stability AI Configuration ---
# IMPORTANT: Add this to your environment variables on Railway
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_API_HOST = "https://api.stability.ai"
STABILITY_ENGINE_ID = "stable-diffusion-xl-1024-v1-0" # Or another preferred engine

if not STABILITY_API_KEY:
    logging.warning("STABILITY_API_KEY environment variable not set. /generate_image will fail.")

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
    
    mask = alpha > 0
    if not mask.any(): return img
    
    dist = distance_transform_edt(~mask, return_distances=True, return_indices=True)
    indices = dist[1]
    
    bleed_mask = (dist[0] > 0) & (dist[0] <= padding)
    
    rgb[bleed_mask] = rgb[tuple(indices[..., i][bleed_mask] for i in range(2))]
    
    new_arr = np.dstack((rgb, alpha))
    return Image.fromarray(new_arr, 'RGBA')

def create_variants(image: Image.Image, variant_type='default'):
    """
    Generates a dictionary of image variants based on the selected type.
    """
    variants = {}
    base_rgb = image.convert('RGB')
    
    if variant_type == 'earthy':
        # Desaturate and apply a sepia-like tint
        enhancer = ImageEnhance.Color(base_rgb)
        desaturated = enhancer.enhance(0.4)
        sepia = Image.new('RGB', desaturated.size, (255, 240, 192))
        variants['Earthy'] = Image.blend(desaturated, sepia, 0.2)
    elif variant_type == 'vibrant':
        # Increase saturation and contrast
        enhancer = ImageEnhance.Color(base_rgb)
        variants['Vibrant'] = enhancer.enhance(1.5)
    elif variant_type == 'muted':
        # Decrease saturation
        enhancer = ImageEnhance.Color(base_rgb)
        variants['Muted'] = enhancer.enhance(0.5)
        
    return variants

# --- NEW: Endpoint for uploading mask images ---
@app.post("/upload_image")
def upload_image():
    logger = logging.getLogger(__name__)
    if 'file' not in request.files:
        logger.error("No file part in request to /upload_image")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file in request to /upload_image")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = f"mask_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Return the absolute URL for the saved file
        file_url = url_for('static', filename=f'uploads/{filename}', _external=True)
        logger.info(f"Image uploaded and saved to {filepath}, URL: {file_url}")
        
        return jsonify({'file_url': file_url})

    return jsonify({'error': 'File upload failed'}), 500


# --- NEW: Endpoint for AI Image Generation ---
@app.post("/generate_image")
def generate_image():
    logger = logging.getLogger(__name__)
    logger.info("Received request for /generate_image")
    
    if not STABILITY_API_KEY:
        return jsonify({'error': 'Image generation service is not configured on the server.'}), 503

    data = request.get_json()
    prompt = data.get('prompt')
    operation = data.get('operation', 'text-to-image')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    # Prepare the payload for Stability AI
    payload = {
        "text_prompts[0][text]": prompt,
        "text_prompts[0][weight]": 1.0,
        "cfg_scale": 7,
        "steps": 40,
        "samples": 1,
    }

    if data.get('negative_prompt'):
        payload["text_prompts[1][text]"] = data.get('negative_prompt')
        payload["text_prompts[1][weight]"] = -1.0
        
    if data.get('aspect_ratio'):
        try:
            width, height = get_dimensions(1024, data.get('aspect_ratio'))
            payload["width"] = width
            payload["height"] = height
        except Exception:
            logger.warning("Invalid aspect ratio, falling back to default.")

    files = {}

    try:
        if operation == 'inpaint' and data.get('init_image_url') and data.get('mask_image_url'):
            logger.info("Preparing for inpainting operation...")
            init_image, _ = load_image_from_url(data.get('init_image_url'))
            mask_image, _ = load_image_from_url(data.get('mask_image_url'))

            if not init_image or not mask_image:
                return jsonify({'error': 'Could not load base image or mask for inpainting'}), 500
            
            init_image_bytes = io.BytesIO()
            init_image.save(init_image_bytes, format='PNG')
            files['init_image'] = init_image_bytes.getvalue()

            mask_image_bytes = io.BytesIO()
            mask_image.save(mask_image_bytes, format='PNG')
            files['mask_image'] = mask_image_bytes.getvalue()

            payload['mask_source'] = 'MASK_IMAGE_BLACK' # Tells Stability to use the mask image (white=inpaint)
            
            api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/image-to-image/masking"

        elif operation == 'enhance' and data.get('init_image_url'):
            logger.info("Preparing for image-to-image (enhance) operation...")
            init_image, _ = load_image_from_url(data.get('init_image_url'))
            if not init_image:
                return jsonify({'error': 'Could not load base image for enhancement'}), 500
            
            init_image_bytes = io.BytesIO()
            init_image.save(init_image_bytes, format='PNG')
            files['init_image'] = init_image_bytes.getvalue()

            payload['image_strength'] = data.get('strength', 0.7)
            api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/image-to-image"
        
        else: # text-to-image
            logger.info("Preparing for text-to-image operation...")
            api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/text-to-image"

        # --- Make the API call to Stability AI ---
        response = requests.post(
            api_endpoint,
            headers={"Authorization": f"Bearer {STABILITY_API_KEY}"},
            data=payload,
            files=files
        )
        response.raise_for_status() # Raise an error for bad status codes (4xx or 5xx)

        response_data = response.json()
        
        # Save the generated image and return its URL
        for i, image in enumerate(response_data["artifacts"]):
            image_b64 = image["base64"]
            
            # Save to a file
            filename = f"gen_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(image_b64))

            # Return the URL of the first generated image
            file_url = url_for('static', filename=f'uploads/{filename}', _external=True)
            logger.info(f"Successfully generated image, saved to {filepath}, URL: {file_url}")
            return jsonify({'url': file_url})

    except requests.exceptions.HTTPError as e:
        logger.error(f"Stability AI API error: {e.response.text}", exc_info=True)
        return jsonify({'error': 'Failed to generate image from AI service.', 'details': e.response.json()}), 502
    except Exception as e:
        logger.error(f"An unexpected error occurred in /generate_image: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500


# --- Main API Endpoint ---
@app.post("/generate_pack")
def generate_pack():
    logger = logging.getLogger(__name__)
    if 'image_url' not in request.form:
        logger.warning("Request to /generate_pack missing image_url")
        return "Missing image_url", 400
    
    image_url = request.form['image_url']
    variant_type = request.form.get('variant', 'default')
    
    base_image, name = load_image_from_url(image_url)
    if not base_image:
        logger.error(f"Failed to process image_url for /generate_pack: {image_url}")
        return "Could not process the provided image URL", 400

    output_maps = {
        'albedo': base_image,
        'normal': generate_normal_map(base_image),
        'roughness': generate_roughness_map(base_image)
    }

    color_variants = create_variants(base_image, variant_type)

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
        for map_name, img in output_maps.items():
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(f'{name}_{map_name}.png', img_buffer.getvalue())
        
        for variant_name, img in color_variants.items():
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(f'{name}_albedo_{variant_name}.png', img_buffer.getvalue())

    zip_buffer.seek(0)
    
    # Clean up
    del base_image, output_maps, color_variants
    gc.collect()

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f'{name}_asset_pack.zip',
        mimetype='application/zip'
    )
    
@app.get("/variants")
def get_variants():
    return jsonify(variants_data)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)