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

api_bp = Blueprint('api', __name__)

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
        
        # Process the response and apply modifications
        # Note: In a real implementation, you would parse the response
        # and apply the modifications using computer vision techniques
        # For now, we'll return a slightly modified version of the original
        modified_image = image.copy()
        
        # Apply subtle modifications based on the effect
        if effect == "Natural Refinement":
            modified_image = cv2.GaussianBlur(modified_image, (5, 5), 0)
        elif effect == "Bridge Reduction":
            modified_image = cv2.addWeighted(modified_image, 0.9, np.zeros_like(modified_image), 0.1, 0)
        elif effect == "Tip Refinement":
            modified_image = cv2.detailEnhance(modified_image, sigma_s=10, sigma_r=0.15)
        elif effect == "Nose Narrowing":
            modified_image = cv2.edgePreservingFilter(modified_image, flags=1, sigma_s=60, sigma_r=0.4)
        elif effect == "Crooked Correction":
            modified_image = cv2.bilateralFilter(modified_image, 9, 75, 75)
        elif effect == "Combined Enhancement":
            modified_image = cv2.detailEnhance(modified_image, sigma_s=10, sigma_r=0.15)
            modified_image = cv2.edgePreservingFilter(modified_image, flags=1, sigma_s=60, sigma_r=0.4)
        
        return modified_image
        
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