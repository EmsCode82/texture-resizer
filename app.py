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
logger = logging.getLogger(__name__)

# --- NEW: Stability AI Configuration ---
# IMPORTANT: Add this to your environment variables on Railway
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_API_HOST = "https://api.stability.ai"
STABILITY_ENGINE_ID = "stable-diffusion-xl-1024-v1-0" # Or another preferred engine

if not STABILITY_API_KEY:
    logger.warning("STABILITY_API_KEY environment variable not set. /generate_image will fail.")

# --- Helper Function for Aspect Ratios ---
def get_dimensions(base_size, ratio_str):
    """Calculates width and height based on a base size and ratio string (e.g., '16:9'). Falls back to square."""
    if not ratio_str:
        return (base_size, base_size)
    try:
        w, h = map(int, str(ratio_str).split(':'))
        if w <= 0 or h <= 0:
            return (base_size, base_size)
        
        # Determine target dimensions based on primary image generation size (1024x1024 for SDXL)
        target_width = 1024
        target_height = 1024

        # Calculate dimensions while maintaining aspect ratio, scaling to fit 1024 square
        if w > h: # Landscape
            scaled_width = target_width
            scaled_height = int(target_width * h / w)
        else: # Portrait or Square
            scaled_height = target_height
            scaled_width = int(target_height * w / h)

        # Ensure dimensions are multiples of 8 or 64 as required by some models
        # For SDXL 1024x1024, generally multiples of 8 are fine.
        scaled_width = (scaled_width // 8) * 8
        scaled_height = (scaled_height // 8) * 8

        # Stability AI recommends 1024x1024 for SDXL, we'll try to get as close as possible
        # without exceeding limits or distorting aspect.
        # This function might need further refinement for exact SDXL compliant sizes
        # if the aspect ratio is extreme. For now, it aims to keep aspect ratio
        # while fitting within a 1024x1024 square, and ensuring divisibility by 8.

        # However, for the purpose of simplicity for the initial SDXL call,
        # we'll still pass the requested aspect ratio to the API and let it handle exact sizing.
        # This get_dimensions is more for UI preview or if we were doing local processing/cropping.
        # For direct Stability AI integration, we mainly use ratio_str as is.
        return (scaled_width, scaled_height) # Returning calculated dimensions, though API uses ratio string

    except Exception as e:
        logger.error(f"Error parsing aspect ratio {ratio_str}: {e}. Falling back to square.")
        return (base_size, base_size) # Fallback to square

# --- Image Processing Functions (Dilate RGBA is moved here as it's a helper for normal maps) ---

def dilate_rgba(img: Image.Image, padding: int = 16) -> Image.Image:
    """
    Bleeds edge RGB colors outward into transparent pixels (alpha-aware) to prevent haloing.
    Alpha is preserved. 'padding' limits how far the bleed goes.
    """
    arr = np.array(img.convert('RGBA'))
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    
    # Calculate EDT (Euclidean Distance Transform) from alpha channel
    # This gives distance to nearest non-transparent pixel
    distance = distance_transform_edt(alpha < 1, sampling=1)
    
    # Determine which pixels are inside the bleed region
    bleed_mask = (distance > 0) & (distance <= padding)
    
    # For each pixel in the bleed region, find the closest opaque pixel's RGB value
    # This is a simplified approach; a more robust one involves iterating or using scipy's map_coordinates
    # For a quick fix, we'll iterate.
    
    # Create an output array for RGB
    out_rgb = rgb.copy()

    # Find the coordinates of opaque pixels
    opaque_coords = np.argwhere(alpha >= 1)

    # For each pixel in the bleed region
    for y, x in np.argwhere(bleed_mask):
        # Find the closest opaque pixel
        if opaque_coords.size > 0:
            distances_to_opaque = np.linalg.norm(opaque_coords - np.array([y, x]), axis=1)
            closest_opaque_idx = np.argmin(distances_to_opaque)
            oy, ox = opaque_coords[closest_opaque_idx]
            out_rgb[y, x] = rgb[oy, ox]
        else:
            # If there are no opaque pixels, just leave as is or default (shouldn't happen with valid input)
            pass

    # Combine the new RGB with the original alpha
    out_arr = np.dstack([out_rgb, alpha])
    return Image.fromarray(out_arr, 'RGBA')


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
    intensity = 1.0 # Strength of the normal map effect
    dx = dz_dx * intensity
    dy = dz_dy * intensity
    norm = np.sqrt(dx * dx + dy * dy + 1.0) # Corrected normalization

    # Map to 0-255 range and invert Y for typical normal map conventions
    nx = ((dx / norm + 1.0) * 0.5 * 255.0).astype(np.uint8)
    ny = ((dy / norm + 1.0) * 0.5 * 255.0).astype(np.uint8) # Y is inverted here if needed
    nz = ((1.0 / norm + 1.0) * 0.5 * 255.0).astype(np.uint8) # Z always points out

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
    # Convert RGBA to L (Luminance)
    l_img = image.convert('L')
    
    # Invert luminance
    roughness_img = ImageOps.invert(l_img)
    
    # Apply alpha from original image to the roughness map if desired,
    # or just use the inverted luminance for the entire image.
    # For now, let's keep it simple as inverted luminance.
    return roughness_img


# --- Flask Endpoints ---

# NEW: Endpoint for serving uploaded static files
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# NEW: Endpoint for client to upload files (e.g., mask images)
@app.post("/upload_image")
def upload_image():
    logger.info("Received request to /upload_image")
    if 'file' not in request.files:
        logger.error("No file part in request.")
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file.")
        return jsonify({"message": "No selected file"}), 400
    if file:
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved to {filepath}")
        # Return a publicly accessible URL
        file_url = url_for('uploaded_file', filename=filename, _external=True)
        return jsonify({"file_url": file_url}), 200
    logger.error("File upload failed for unknown reason.")
    return jsonify({"message": "File upload failed"}), 500


@app.post("/generate_image")
def generate_image():
    logger.info("Received request to /generate_image")
    if not STABILITY_API_KEY:
        return jsonify({"message": "STABILITY_API_KEY not configured on server"}), 500
    
    data = request.get_json()
    prompt = data.get("prompt")
    negative_prompt = data.get("negative_prompt", "")
    aspect_ratio = data.get("aspect_ratio", "1:1")
    operation = data.get("operation", "text-to-image")
    init_image_url = data.get("init_image_url")
    mask_image_url = data.get("mask_image_url")
    strength = data.get("strength", 0.7) # Image strength for img2img

    if not prompt:
        return jsonify({"message": "Prompt is required"}), 400

    api_url = ""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_API_KEY}"
    }
    files = {}
    json_data = {}
    
    # Common parameters for all Stability AI generations
    json_data = {
        "text_prompts": [
            {"text": prompt, "weight": 1},
            {"text": negative_prompt, "weight": -1}, # Negative prompt
        ],
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE", # Recommended for SDXL
        "samples": 1,
        "steps": 30,
        "width": 1024, # SDXL Base resolution
        "height": 1024, # SDXL Base resolution
        "seed": 0, # Use 0 for random seed
        "sampler": "K_DPMPP_2M_SDE", # Good general purpose sampler
        "style_preset": "photographic" # Example style preset
    }

    if aspect_ratio != "1:1": # Adjust for non-square aspect ratios if needed (SDXL supports this natively now)
        width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
        if width_ratio > height_ratio:
            json_data["width"] = 1024
            json_data["height"] = (1024 * height_ratio // width_ratio // 8) * 8
        else:
            json_data["height"] = 1024
            json_data["width"] = (1024 * width_ratio // height_ratio // 8) * 8
        
        # Ensure minimum size for extreme aspect ratios
        json_data["width"] = max(768, json_data["width"])
        json_data["height"] = max(768, json_data["height"])


    if operation == "text-to-image":
        logger.info(f"Generating text-to-image for prompt: {prompt}")
        api_url = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/text-to-image"
    
    elif operation == "enhance" and init_image_url:
        logger.info(f"Generating image-to-image (enhance) for prompt: {prompt}, init_image: {init_image_url}")
        api_url = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/image-to-image"
        
        init_image, _ = load_image_from_url(init_image_url)
        if not init_image:
            return jsonify({"message": "Failed to load initial image for enhance operation"}), 400
        
        init_image_bytes = io.BytesIO()
        init_image.save(init_image_bytes, format='PNG')
        init_image_bytes.seek(0)
        
        files['init_image'] = ('init_image.png', init_image_bytes, 'image/png')
        json_data["image_strength"] = strength # How much to modify the image (0.1 = subtle, 0.9 = major changes)
        
    elif operation == "inpaint" and init_image_url and mask_image_url:
        logger.info(f"Generating inpaint for prompt: {prompt}, init_image: {init_image_url}, mask_image: {mask_image_url}")
        api_url = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/image-to-image/masking"
        
        init_image, _ = load_image_from_url(init_image_url)
        mask_image, _ = load_image_from_url(mask_image_url)

        if not init_image or not mask_image:
            return jsonify({"message": "Failed to load initial or mask image for inpaint operation"}), 400
        
        # Ensure mask is L-mode (grayscale) for Stability AI, convert if needed
        # Stability AI expects a black mask on a white background for inpainting
        # Our frontend mask is purple on transparent, so we need to convert it to a suitable format
        # If the frontend mask is opaque purple where it's drawn, and transparent elsewhere,
        # we need to make it white where opaque, black where transparent.
        # Simple conversion to L mode and then inversion might work, or direct pixel manipulation.

        # Let's assume the frontend provides a mask where opaque areas (brush strokes) are where changes should occur.
        # We need a mask where the area to be changed is WHITE, and the area to be preserved is BLACK.
        # If our mask_image is transparent where no mask, and opaque purple where masked:
        # 1. Convert mask to L (grayscale). Transparent areas will be black. Opaque areas (purple) will have some grayscale value.
        # 2. Convert to binary: white for masked areas, black for unmasked.
        #    A threshold is needed. Opaque brush strokes are > 0 alpha.
        
        mask_image_alpha = mask_image.getchannel('A') # Get alpha channel
        binary_mask_array = np.array(mask_image_alpha) # Alpha 0-255
        
        # Create an all-black image
        stability_mask = Image.new('L', mask_image.size, 0)
        # Set areas where alpha > 0 (i.e., painted areas) to white (255)
        stability_mask_array = np.array(stability_mask)
        stability_mask_array[binary_mask_array > 0] = 255
        stability_mask = Image.fromarray(stability_mask_array, 'L')


        init_image_bytes = io.BytesIO()
        init_image.save(init_image_bytes, format='PNG')
        init_image_bytes.seek(0)
        
        mask_image_bytes_processed = io.BytesIO()
        stability_mask.save(mask_image_bytes_processed, format='PNG') # Save the processed mask
        mask_image_bytes_processed.seek(0)
        
        files['init_image'] = ('init_image.png', init_image_bytes, 'image/png')
        files['mask_image'] = ('mask_image.png', mask_image_bytes_processed, 'image/png')
        
        json_data["image_strength"] = 0.7 # How much to preserve original image outside mask (0.0 to 1.0)
        json_data["mask_strength"] = 0.9 # How strongly the mask is applied (0.0 to 1.0)
        json_data["mask_source"] = "MASK_IMAGE_WHITE" # Crucial for inpainting: areas in white in mask_image are changed.
        # If your mask is designed as 'area to preserve is white', use MASK_IMAGE_BLACK.
        # Given your frontend, we want to change the painted (purple) area, which is made white in stability_mask.

    else:
        return jsonify({"message": f"Invalid operation or missing parameters: {operation}"}), 400

    logger.info(f"Making Stability AI API call to: {api_url}")
    try:
        # Stability AI requires form data for image uploads, not raw JSON
        # json_data needs to be stringified and added as a field in the multipart form
        form_data = {
            'text_prompts[0][text]': json_data['text_prompts'][0]['text'],
            'text_prompts[0][weight]': json_data['text_prompts'][0]['weight'],
            'cfg_scale': str(json_data['cfg_scale']),
            'clip_guidance_preset': json_data['clip_guidance_preset'],
            'samples': str(json_data['samples']),
            'steps': str(json_data['steps']),
            'width': str(json_data['width']),
            'height': str(json_data['height']),
            'seed': str(json_data['seed']),
            'sampler': json_data['sampler'],
            'style_preset': json_data['style_preset'],
        }
        if negative_prompt:
            form_data['text_prompts[1][text]'] = negative_prompt
            form_data['text_prompts[1][weight]'] = -1
        
        if operation == "enhance" or operation == "inpaint":
            form_data['image_strength'] = str(json_data['image_strength'])
            
        if operation == "inpaint":
            form_data['mask_strength'] = str(json_data['mask_strength'])
            form_data['mask_source'] = json_data['mask_source']

        response = requests.post(
            api_url,
            headers=headers,
            files=files,
            data=form_data
        )

        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        response_data = response.json()

        if not response_data.get("artifacts"):
            logger.error(f"Stability AI did not return any artifacts: {response_data}")
            return jsonify({"message": "Stability AI did not generate an image"}), 500

        # Assuming the first artifact is the image we want
        img_b64 = response_data["artifacts"][0]["base64"]
        img_data = base64.b64decode(img_b64)

        # Save to file
        image_id = uuid.uuid4().hex
        output_filename = f"generated_{image_id}.png"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_filepath, "wb") as f:
            f.write(img_data)
        
        # Generate URL
        # For simplicity, we'll return a relative path that the frontend can append to its backend URL
        # For a production setup, you'd want a proper static file server or CDN.
        generated_image_url = f"/backend_static_files/{output_filename}" 
        # Note: You'll need to configure your Flask app to serve files from OUTPUT_DIR
        # For Railway, this might involve a static path config in your app or a separate Nginx.
        # For local testing, you can add a route:
        # @app.route('/backend_static_files/<filename>')
        # def serve_generated_image(filename):
        #    return send_from_directory(OUTPUT_DIR, filename)
        logger.info(f"Generated image saved: {output_filepath}, URL: {generated_image_url}")
        return jsonify({"url": generated_image_url}), 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Stability AI API: {e}")
        error_detail = e.response.text if e.response is not None else str(e)
        return jsonify({"message": "Failed to generate image with Stability AI", "detail": error_detail}), e.response.status_code if e.response is not None else 500
    except Exception as e:
        logger.exception("An unexpected error occurred during image generation.")
        return jsonify({"message": f"Internal server error: {str(e)}"}), 500

@app.get("/")
def home():
    return "Assetgineer Backend is running!"

# Helper to serve generated images from the OUTPUT_DIR for development/local testing
@app.route('/backend_static_files/<path:filename>')
def serve_generated_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    from flask import send_from_directory
    app.run(debug=True, port=5000)
