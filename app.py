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

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Stability AI Configuration ---
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_API_HOST = "https://api.stability.ai"
# UPDATED: We'll use different engines based on the operation
TXT2IMG_ENGINE_ID = "stable-diffusion-xl-1024-v1-0" # For text-to-image and enhance
INPAINT_ENGINE_ID = "stable-inpainting-v1-0" # For inpainting

if not STABILITY_API_KEY:
    logging.warning("STABILITY_API_KEY environment variable not set. Image generation endpoints will fail.")

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
        
        img = Image.open(io.BytesIO(resp.content))
        # Remove the 'name' return as it's not used consistently and caused confusion
        logger.debug(f"Image from URL loaded successfully, size: {img.size}")
        return img
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None
    except (UnidentifiedImageError, IOError) as e:
        logger.error(f"Corrupted or unreadable image from URL {url}: {e}")
        return None

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

def generate_metalness_map(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Generates a simple metalness map from the alpha channel or luminance."""
    # Using alpha as a proxy for metalness (non-transparent parts might be metallic)
    # This is a placeholder and should be replaced with more sophisticated logic
    if image.mode == 'RGBA':
        alpha_channel = image.split()[-1]
        # Invert alpha to get a white-on-black mask, then threshold
        metalness_map = ImageOps.invert(alpha_channel).point(lambda p: 255 if p > threshold else 0)
    else:
        # If no alpha, use luminance
        metalness_map = image.convert('L').point(lambda p: 255 if p > threshold else 0)
    return metalness_map.convert('RGBA')

def generate_roughness_map(image: Image.Image, invert: bool = False) -> Image.Image:
    """Simple placeholder roughness: invert luminance. Returns single-channel 'L' converted to RGBA."""
    img = image.convert('L')
    if invert:
        img = ImageOps.invert(img)
    return img.convert('RGBA')

def dilate_rgba(img: Image.Image, padding: int = 16) -> Image.Image:
    """
    Bleeds edge RGB colors outward into transparent pixels (alpha-aware) to prevent haloing.
    Alpha is preserved. 'padding' limits how far the bleed goes.
    """
    arr = np.array(img.convert('RGBA'))
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    mask = alpha > 0
    if not mask.any(): return img # No opaque pixels to bleed

    # Compute Euclidean distance transform on the inverse mask (transparent areas)
    # This gives distance to the nearest opaque pixel for each transparent pixel
    dist, indices = distance_transform_edt(~mask, return_distances=True, return_indices=True)

    # For transparent pixels within padding distance, copy RGB from nearest opaque pixel
    bleed_mask = (dist > 0) & (dist <= padding)

    # Use the indices from EDT to get the RGB values of the nearest opaque pixel
    # indices is an array of shape (2, H, W) where indices[0] are row indices and indices[1] are col indices
    rgb[bleed_mask] = rgb[indices[0][bleed_mask], indices[1][bleed_mask]]

    new_arr = np.dstack((rgb, alpha)) # Recombine RGB and original Alpha
    return Image.fromarray(new_arr, 'RGBA')

def enhance_image(image: Image.Image) -> Image.Image:
    """Applies some basic enhancements (sharpness, contrast, color) to an image."""
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2) # Slight sharpening
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1) # Slight contrast boost
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.05) # Slight color boost
    return image

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

# --- API Endpoints ---
@app.route('/upload_image', methods=['POST'])
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

@app.route('/generate_image', methods=['POST'])
def generate_image_route():
    logger = logging.getLogger(__name__)
    logger.info("Received request for /generate_image")

    if not STABILITY_API_KEY:
        return jsonify({'error': 'Image generation service is not configured on the server.'}), 503

    data = request.get_json()
    prompt = data.get('prompt')
    operation = data.get('operation', 'text-to-image')

    if not prompt and operation != 'inpaint': # Inpainting can have an empty prompt if only init_image is changed
         return jsonify({'error': 'Prompt is required for text-to-image/enhance operations'}), 400

    # Prepare the payload for Stability AI
    # Note: Stability AI's API expects text_prompts as a list of dictionaries,
    # and files as a separate 'files' argument in the requests.post call.
    # The 'data' argument in requests.post is for form fields, so we need to
    # flatten the text_prompts into that format.

    # Default parameters for all operations, can be overridden
    common_params = {
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE", # A good general preset for quality
        "samples": 1,
        "steps": 30, # Increased steps for better quality
    }

    stability_payload = {}
    files_to_send = {}
    api_endpoint = ""
    engine_id_to_use = ""

    text_prompts_list = []
    if prompt:
        text_prompts_list.append({"text": prompt, "weight": 1.0})
    if data.get('negative_prompt'):
        text_prompts_list.append({"text": data.get('negative_prompt'), "weight": -1.0})

    # Flatten text_prompts for multipart/form-data
    for i, tp in enumerate(text_prompts_list):
        stability_payload[f"text_prompts[{i}][text]"] = tp["text"]
        stability_payload[f"text_prompts[{i}][weight]"] = tp["weight"]


    try:
        if operation == 'inpaint':
            logger.info("Preparing for inpainting operation...")
            engine_id_to_use = INPAINT_ENGINE_ID
            api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{engine_id_to_use}/image-to-image/masking"

            init_image_url = data.get('init_image_url')
            mask_image_url = data.get('mask_image_url')

            if not all([init_image_url, mask_image_url]):
                return jsonify({'error': 'Missing init_image_url or mask_image_url for inpainting'}), 400

            # Load images
            init_image_pil = load_image_from_url(init_image_url)
            mask_image_pil = load_image_from_url(mask_image_url)

            if not init_image_pil or not mask_image_pil:
                return jsonify({'error': 'Failed to load base image or mask for inpainting from URLs'}), 500

            # Convert to bytes for multipart/form-data
            init_image_bytes = io.BytesIO()
            # Ensure init_image is RGB (Stability AI expects RGB for image-to-image)
            init_image_pil.convert("RGB").save(init_image_bytes, format='PNG')
            init_image_bytes.seek(0)
            files_to_send['init_image'] = ('init_image.png', init_image_bytes.getvalue(), 'image/png')

            mask_image_bytes = io.BytesIO()
            # Ensure mask is 'L' (grayscale) or '1' (binary) for Stability AI mask_image
            # A simple binary mask is often best for inpainting
            mask_image_pil.convert("L").point(lambda p: 255 if p > 128 else 0, mode='1').save(mask_image_bytes, format='PNG')
            mask_image_bytes.seek(0)
            files_to_send['mask_image'] = ('mask_image.png', mask_image_bytes.getvalue(), 'image/png')

            stability_payload["mask_source"] = "MASK_IMAGE_BLACK" # Use mask_image provided, where black areas are masked (inpainted)

            # Crucial parameters for inpainting
            stability_payload["image_strength"] = 0.7 # How much to preserve areas outside the mask (0.0-1.0)
                                                    # 0.0 = only mask area changes; 1.0 = entire image is regenerated based on prompt
                                                    # Defaulting to 0.7 to keep background but change masked area.
            stability_payload["mask_strength"] = 0.8 # How much to blend the original image with the generated mask
                                                    # 0.0 = mask is ignored; 1.0 = mask is strictly followed
                                                    # 0.8 ensures good adherence to the mask.


        elif operation == 'enhance' and data.get('init_image_url'):
            logger.info("Preparing for image-to-image (enhance) operation...")
            engine_id_to_use = TXT2IMG_ENGINE_ID
            api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{engine_id_to_use}/image-to-image"

            init_image_url = data.get('init_image_url')
            init_image_pil = load_image_from_url(init_image_url)
            if not init_image_pil:
                return jsonify({'error': 'Could not load base image for enhancement'}), 500

            init_image_bytes = io.BytesIO()
            init_image_pil.convert("RGB").save(init_image_bytes, format='PNG')
            init_image_bytes.seek(0)
            files_to_send['init_image'] = ('init_image.png', init_image_bytes.getvalue(), 'image/png')

            stability_payload['image_strength'] = data.get('strength', 0.3) # Lower strength for more creativity
            # No mask, so mask_source and mask_strength are not applicable

        else: # text-to-image (default operation)
            logger.info("Preparing for text-to-image operation...")
            engine_id_to_use = TXT2IMG_ENGINE_ID
            api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{engine_id_to_use}/text-to-image"

            # Set width/height for text2img as it's the primary generation mode
            width, height = get_dimensions(1024, data.get('aspect_ratio'))
            stability_payload["width"] = width
            stability_payload["height"] = height

        # Add common parameters to payload
        stability_payload.update(common_params)

        # Make the API call to Stability AI
        logger.debug(f"Calling Stability AI API at {api_endpoint} with payload: {stability_payload.keys()}")
        
        response = requests.post(
            api_endpoint,
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Accept": "application/json"
            },
            data=stability_payload, # Use data for form fields
            files=files_to_send # Use files for actual image files
        )
        response.raise_for_status() # Raise an error for bad status codes (4xx or 5xx)

        response_data = response.json()

        # Save the generated image and return its URL
        artifacts = response_data.get("artifacts", [])
        if not artifacts:
            return jsonify({"error": "No image artifacts returned from API"}), 500

        image_data = base64.b64decode(artifacts[0]["base64"])
        image_pil = Image.open(io.BytesIO(image_data))

        filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_pil.save(filepath, format='PNG')

        file_url = url_for('static', filename=f'uploads/{filename}', _external=True)
        logger.info(f"Successfully generated image, saved to {filepath}, URL: {file_url}")
        return jsonify({'url': file_url})

    except requests.exceptions.HTTPError as e:
        logger.error(f"Stability AI API HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
        error_details = e.response.json() if e.response.text else {'message': 'Unknown API error'}
        return jsonify({'error': 'Failed to generate image from AI service.', 'details': error_details}), 502
    except Exception as e:
        logger.error(f"An unexpected error occurred in /generate_image: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500


@app.route('/process_maps', methods=['POST'])
def process_maps_route():
    logger = logging.getLogger(__name__)
    if 'file' not in request.files:
        logger.warning("Request to /process_maps missing file part")
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("Request to /process_maps received empty filename")
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        image = Image.open(file.stream).convert('RGBA')
    except (UnidentifiedImageError, IOError):
        logger.error("Invalid or corrupted image file received for /process_maps")
        return jsonify({'error': 'Invalid or corrupted image file'}), 400

    try:
        normal_map = generate_normal_map(image)
        metalness_map = generate_metalness_map(image)
        roughness_map = generate_roughness_map(image)
        enhanced_image = enhance_image(image)
    except Exception as e:
        logger.error(f"Error during map generation: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during map processing'}), 500

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'a', ZIP_DEFLATED, False) as zip_file:
        for map_img, name in [(normal_map, 'normal.png'), (metalness_map, 'metalness.png'), (roughness_map, 'roughness.png'), (enhanced_image, 'enhanced.png')]:
            img_buffer = io.BytesIO()
            map_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zip_file.writestr(name, img_buffer.getvalue())

    zip_buffer.seek(0)

    gc.collect() # Trigger garbage collection

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='asset_maps.zip'
    )

@app.route('/generate_pack', methods=['POST'])
def generate_pack():
    logger = logging.getLogger(__name__)
    if 'image_url' not in request.form:
        logger.warning("Request to /generate_pack missing image_url")
        return "Missing image_url", 400

    image_url = request.form['image_url']
    variant_type = request.form.get('variant', 'default')

    # Removed the redundant 'name' return from load_image_from_url
    base_image_pil = load_image_from_url(image_url)
    if not base_image_pil:
        logger.error(f"Failed to process image_url for /generate_pack: {image_url}")
        return "Could not process the provided image URL", 400

    # Derive a name from the URL, if possible, or use a generic one
    name_match = re.search(r'/([\w-]+)\.\w+$', image_url)
    name = name_match.group(1) if name_match else f"asset_{uuid.uuid4().hex[:6]}"

    output_maps = {
        'albedo': base_image_pil,
        'normal': generate_normal_map(base_image_pil),
        'roughness': generate_roughness_map(base_image_pil),
        'metalness': generate_metalness_map(base_image_pil) # Re-added metalness map
    }

    color_variants = create_variants(base_image_pil, variant_type)

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
    del base_image_pil, output_maps, color_variants
    gc.collect()

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f'{name}_asset_pack.zip',
        mimetype='application/zip'
    )

@app.route("/variants")
def get_variants():
    return jsonify(variants_data)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)