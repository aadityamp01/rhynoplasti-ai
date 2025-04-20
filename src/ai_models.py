import requests
import json
import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

class AIModelManager:
    def __init__(self):
        self.grok_api_key = None  # Will be set through environment variable
        self.available_models = {
            "grok": self._process_grok,
            "stable_diffusion": self._process_stable_diffusion,
            "dalle": self._process_dalle
        }
    
    def set_api_key(self, model_name, api_key):
        """Set API key for a specific model"""
        if model_name == "grok":
            self.grok_api_key = api_key
        # Add other model API keys as needed
    
    def process_image(self, image, model_name="grok", options=None):
        """Process image using specified AI model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not supported")
        
        return self.available_models[model_name](image, options)
    
    def _process_grok(self, image, options=None):
        """Process image using Grok API"""
        if not self.grok_api_key:
            raise ValueError("Grok API key not set")
        
        # Convert image to base64
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.grok_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "image": img_str,
            "task": "rhinoplasty_simulation",
            "options": options or {}
        }
        
        # Make API request to Grok
        response = requests.post(
            "https://api.grok.ai/v1/image/transform",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Grok API error: {response.text}")
        
        # Process response
        result = response.json()
        result_image = base64.b64decode(result["image"])
        return Image.open(BytesIO(result_image))
    
    def _process_stable_diffusion(self, image, options=None):
        """Process image using Stable Diffusion API"""
        # Implementation for Stable Diffusion
        pass
    
    def _process_dalle(self, image, options=None):
        """Process image using DALL-E API"""
        # Implementation for DALL-E
        pass

# Example usage:
"""
model_manager = AIModelManager()
model_manager.set_api_key("grok", "your-api-key-here")

# Process image
result = model_manager.process_image(
    image,
    model_name="grok",
    options={
        "rhinoplasty_type": "natural",
        "intensity": 0.7
    }
)
""" 