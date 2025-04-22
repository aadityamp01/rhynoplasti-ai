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

# Create a Blueprint for the API
api_bp = Blueprint('api', __name__)

def init_vertex_ai():
    """Initialize Vertex AI with project and location settings."""
    project_id = st.secrets.get("GOOGLE_CLOUD_PROJECT")
    location = st.secrets.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not found in Streamlit secrets")
    
    aiplatform.init(project=project_id, location=location)
    return True

def process_image_with_vertex_ai(image_base64, prompt):
    """Process an image with Vertex AI's Imagen model."""
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
        
        # Create the Imagen model
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
        
        # Get the effect type
        effect = data.get('effect', 'Natural Refinement')
        
        # Define prompts for different effects
        prompts = {
            "Natural Refinement": "Apply subtle rhinoplasty to make the nose more refined and proportional. Focus on gentle contouring and slight reduction in size. Maintain natural facial features and skin texture. The result should look like a natural, post-surgery outcome.",
            "Bridge Reduction": "Perform rhinoplasty to reduce the height of the nose bridge. Create a more delicate and balanced profile while maintaining natural facial harmony. The bridge should appear slightly lower but still look natural and proportional.",
            "Tip Refinement": "Refine the nose tip through rhinoplasty to create better definition and projection. Make it more elegant while keeping it natural-looking. The tip should be slightly more defined but not overly sculpted.",
            "Nose Narrowing": "Apply rhinoplasty to narrow the nose width, creating better facial proportions. The nose should appear slightly narrower while maintaining natural appearance and avoiding any artificial look.",
            "Crooked Correction": "Correct any nose deviation through rhinoplasty to improve facial symmetry. The nose should appear straight and centered while maintaining natural appearance and avoiding any signs of artificial correction.",
            "Combined Enhancement": "Apply comprehensive rhinoplasty combining multiple refinements. The nose should appear more refined, proportional, and balanced while maintaining a completely natural appearance. All changes should look like the result of skilled surgical intervention."
        }
        
        # Get the appropriate prompt
        prompt = prompts.get(effect, prompts["Natural Refinement"])
        
        # Process the image
        result = process_image_with_vertex_ai(data['image'], prompt)
        
        if "error" in result:
            return jsonify({'error': result["error"]}), 500
            
        return jsonify({
            'success': True,
            'processed_image': result["resultImage"]
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500 