import os
import base64
import json
import tempfile
from flask import Blueprint, request, jsonify
from google.cloud import aiplatform
from google.oauth2 import service_account
import streamlit as st
import io
from PIL import Image
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# Create a Blueprint for the API
api_bp = Blueprint('api', __name__)

def init_vertex_ai():
    """Initialize Vertex AI with project and location settings."""
    project_id = st.secrets.get("GOOGLE_CLOUD_PROJECT")
    location = st.secrets.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not found in Streamlit secrets")
    
    vertexai.init(project=project_id, location=location)
    return GenerativeModel("gemini-pro-vision")

# Process image with Vertex AI
def process_image_with_vertex_ai(image_base64, prompt):
    """Process an image with Vertex AI's image generation model"""
    try:
        # Initialize Vertex AI
        if not init_vertex_ai():
            return {"error": "Failed to initialize Vertex AI"}
        
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        
        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(image_data)
            image_path = temp_file.name
        
        # Create the model
        model = aiplatform.ImageGenerationModel.from_pretrained("imagegeneration@002")
        
        # Generate the image
        response = model.generate_images(
            prompt=prompt,
            base_image=image_path,
            number_of_images=1,
            seed=42
        )
        
        # Get the generated image
        generated_image = response.images[0]
        
        # Convert the image to base64
        image_base64 = base64.b64encode(generated_image).decode('utf-8')
        
        # Clean up temporary files
        os.unlink(image_path)
        
        return {"resultImage": f"data:image/jpeg;base64,{image_base64}"}
    except Exception as e:
        print(f"Error processing image with Vertex AI: {e}")
        return {"error": str(e)}

# API endpoint to process an image
@api_bp.route('/process-image', methods=['POST'])
def process_image():
    """Process an image using Vertex AI and return the results."""
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Get prompt from request or use default
        prompt = data.get('prompt', 'Apply rhinoplasty effect to make the nose more refined and proportional while maintaining natural appearance.')
        
        # Initialize Vertex AI
        model = init_vertex_ai()
        
        # Convert image to bytes for Vertex AI
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create image part for Vertex AI
        image_part = Part.from_data(data=img_byte_arr, mime_type='image/jpeg')
        
        # Generate content with Vertex AI
        generation_config = GenerationConfig(
            max_output_tokens=2048,
            temperature=0.4,
            top_p=0.8,
            top_k=40
        )
        
        response = model.generate_content(
            [prompt, image_part],
            generation_config=generation_config
        )
        
        if response.text:
            # For now, we'll return the original image
            # In a real implementation, you would process the response and modify the image
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            processed_image = base64.b64encode(img_byte_arr).decode('utf-8')
            
            return jsonify({
                'success': True,
                'processed_image': f"data:image/jpeg;base64,{processed_image}"
            })
        else:
            return jsonify({'error': 'No response from Vertex AI'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500 