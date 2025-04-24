from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
import streamlit as st
import mediapipe as mp

api_bp = Blueprint('api', __name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Vertex AI using Streamlit secrets
try:
    project_id = st.secrets["GOOGLE_CLOUD_PROJECT"]
    location = st.secrets["GOOGLE_CLOUD_LOCATION"]
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-pro-vision")
except Exception as e:
    st.error(f"Failed to initialize Vertex AI: {str(e)}")
    st.error("Please make sure you have set up the following secrets in your Streamlit app:")
    st.error("1. GOOGLE_CLOUD_PROJECT")
    st.error("2. GOOGLE_CLOUD_LOCATION")
    st.error("3. GOOGLE_APPLICATION_CREDENTIALS (if using service account)")

# Enhanced prompts for better nose modifications
EFFECT_PROMPTS = {
    "Natural Refinement": """
    Perform a subtle and natural rhinoplasty modification:
    1. Slightly reduce the bridge height while maintaining natural contours
    2. Refine the tip to be more defined but not overly pointed
    3. Maintain the natural width of the nose
    4. Ensure the modifications blend seamlessly with surrounding facial features
    5. Preserve the person's ethnic characteristics
    The result should look completely natural and undetectable as a modification.
    """,
    
    "Bridge Reduction": """
    Apply a bridge reduction rhinoplasty:
    1. Reduce the height of the nasal bridge by 20-30%
    2. Create a smooth, natural slope from the forehead to the tip
    3. Maintain proper proportions with the rest of the face
    4. Ensure the bridge reduction looks natural from all angles
    5. Keep the nostrils and tip in proper proportion
    The modification should look like a natural nose, not an artificial one.
    """,
    
    "Tip Refinement": """
    Refine the nose tip through rhinoplasty:
    1. Create a more defined tip without making it too pointed
    2. Maintain proper projection and rotation
    3. Ensure the tip aligns well with the bridge
    4. Keep the nostrils in natural proportion
    5. Preserve the natural character of the nose
    The result should look like a natural refinement, not a dramatic change.
    """,
    
    "Nose Narrowing": """
    Perform a nose narrowing procedure:
    1. Reduce the width of the nose by 15-20%
    2. Maintain natural contours and shadows
    3. Keep the bridge height proportional
    4. Ensure the nostrils remain in natural proportion
    5. Preserve the natural character of the nose
    The modification should look completely natural and undetectable.
    """,
    
    "Crooked Correction": """
    Correct a crooked nose through rhinoplasty:
    1. Straighten the nasal bridge while maintaining natural contours
    2. Ensure proper alignment with the facial midline
    3. Maintain natural width and projection
    4. Keep the tip properly aligned
    5. Preserve the natural character of the nose
    The correction should look natural and harmonious with the face.
    """,
    
    "Combined Enhancement": """
    Apply a comprehensive rhinoplasty enhancement:
    1. Create a balanced, natural-looking nose
    2. Maintain proper proportions with facial features
    3. Ensure natural contours and shadows
    4. Keep the modifications subtle and undetectable
    5. Preserve the person's unique facial characteristics
    The result should look like a natural, beautiful nose that fits the face perfectly.
    """
}

def get_nose_landmarks(image):
    """Get nose landmarks using MediaPipe Face Mesh."""
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    # Get nose landmarks
    nose_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        # Nose bridge points (27-31)
        nose_bridge = [face_landmarks.landmark[i] for i in range(27, 32)]
        # Nose tip point (4)
        nose_tip = face_landmarks.landmark[4]
        # Nostril points (129-134)
        nostrils = [face_landmarks.landmark[i] for i in range(129, 135)]
        
        nose_landmarks.extend(nose_bridge)
        nose_landmarks.append(nose_tip)
        nose_landmarks.extend(nostrils)
    
    return nose_landmarks

def create_nose_mask(image, landmarks):
    """Create a mask for the nose region."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if landmarks:
        points = []
        for landmark in landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)
    
    return mask

def apply_natural_refinement(image, intensity=0.3):
    """Apply natural refinement effect to the nose."""
    landmarks = get_nose_landmarks(image)
    if landmarks is None:
        return image
    
    mask = create_nose_mask(image, landmarks)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance brightness and contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    image = cv2.bitwise_and(image, image, mask=~mask)
    result = cv2.add(image, result)
    
    return result

def apply_bridge_reduction(image, intensity=0.4):
    """Apply bridge reduction effect to the nose."""
    landmarks = get_nose_landmarks(image)
    if landmarks is None:
        return image
    
    mask = create_nose_mask(image, landmarks)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Darken the bridge area
    l = cv2.addWeighted(l, 0.8, np.zeros_like(l), 0.2, 0)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    image = cv2.bitwise_and(image, image, mask=~mask)
    result = cv2.add(image, result)
    
    return result

def apply_tip_refinement(image, intensity=0.35):
    """Apply tip refinement effect to the nose."""
    landmarks = get_nose_landmarks(image)
    if landmarks is None:
        return image
    
    # Create mask for just the tip
    tip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    tip_point = landmarks[5]  # Nose tip landmark
    x = int(tip_point.x * image.shape[1])
    y = int(tip_point.y * image.shape[0])
    cv2.circle(tip_mask, (x, y), 20, 255, -1)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Brighten the tip area
    l = cv2.addWeighted(l, 1.2, np.ones_like(l) * 255, 0.1, 0)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply only to tip region
    result = cv2.bitwise_and(result, result, mask=tip_mask)
    image = cv2.bitwise_and(image, image, mask=~tip_mask)
    result = cv2.add(image, result)
    
    return result

def apply_nose_narrowing(image, intensity=0.4):
    """Apply nose narrowing effect."""
    landmarks = get_nose_landmarks(image)
    if landmarks is None:
        return image
    
    mask = create_nose_mask(image, landmarks)
    height, width = image.shape[:2]
    
    # Apply warping to narrow the nose
    map_x = np.zeros((height, width), np.float32)
    map_y = np.zeros((height, width), np.float32)
    
    center_x = width // 2
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                # Calculate distance from center
                dx = x - center_x
                # Apply narrowing effect
                map_x[y, x] = x - dx * 0.2
                map_y[y, x] = y
            else:
                map_x[y, x] = x
                map_y[y, x] = y
    
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    image = cv2.bitwise_and(image, image, mask=~mask)
    result = cv2.add(image, result)
    
    return result

def apply_crooked_correction(image, intensity=0.45):
    """Apply crooked nose correction effect."""
    landmarks = get_nose_landmarks(image)
    if landmarks is None:
        return image
    
    mask = create_nose_mask(image, landmarks)
    height, width = image.shape[:2]
    
    # Get nose bridge points
    bridge_points = landmarks[:5]
    
    # Calculate the angle of deviation
    start_point = bridge_points[0]
    end_point = bridge_points[-1]
    angle = np.arctan2(end_point.y - start_point.y, end_point.x - start_point.x)
    
    # Create a transformation matrix for rotation
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle * 180 / np.pi * 0.5, 1.0)
    
    # Apply rotation
    result = cv2.warpAffine(image, matrix, (width, height))
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    image = cv2.bitwise_and(image, image, mask=~mask)
    result = cv2.add(image, result)
    
    return result

def apply_combined_enhancement(image, intensity=0.35):
    """Apply all effects with reduced intensity."""
    result = image.copy()
    result = apply_natural_refinement(result)
    result = apply_bridge_reduction(result)
    result = apply_tip_refinement(result)
    result = apply_nose_narrowing(result)
    result = apply_crooked_correction(result)
    return result

def decode_image(image_data):
    """Decode base64 image data to OpenCV format."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
            
        return image
    except Exception as e:
        raise ValueError(f"Error decoding image: {str(e)}")

def encode_image(image):
    """Encode OpenCV image to base64."""
    try:
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")

def process_with_vertex_ai(image, effect):
    """Process image using Vertex AI with enhanced prompts."""
    try:
        # Convert OpenCV image to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Create image part for Vertex AI
        image_part = Part.from_image(pil_image)
        
        # Get the enhanced prompt for the selected effect
        prompt = EFFECT_PROMPTS.get(effect, EFFECT_PROMPTS["Natural Refinement"])
        
        # Generate response from Vertex AI
        response = model.generate_content(
            [prompt, image_part],
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        # Apply the appropriate effect based on the selection
        if effect == "Natural Refinement":
            result = apply_natural_refinement(image)
        elif effect == "Bridge Reduction":
            result = apply_bridge_reduction(image)
        elif effect == "Tip Refinement":
            result = apply_tip_refinement(image)
        elif effect == "Nose Narrowing":
            result = apply_nose_narrowing(image)
        elif effect == "Crooked Correction":
            result = apply_crooked_correction(image)
        elif effect == "Combined Enhancement":
            result = apply_combined_enhancement(image)
        else:
            result = image
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error processing with Vertex AI: {str(e)}")

@api_bp.route('/process_image', methods=['POST'])
def process_image():
    """Process image with selected rhinoplasty effect."""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'effect' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Decode image
        image = decode_image(data['image'])
        effect = data['effect']
        
        # Process image with Vertex AI
        result = process_with_vertex_ai(image, effect)
        
        # Encode result
        result_base64 = encode_image(result)
        
        return jsonify({
            'result': result_base64,
            'effect': effect,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 