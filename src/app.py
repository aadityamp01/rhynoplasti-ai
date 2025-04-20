import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import tempfile
import time

try:
    import cv2
    import mediapipe as mp
    from diffusers import StableDiffusionInpaintPipeline
    import torch
    OPENCV_AVAILABLE = True
    STABLE_DIFFUSION_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing required libraries: {str(e)}")
    OPENCV_AVAILABLE = False
    STABLE_DIFFUSION_AVAILABLE = False

# Global variable to store the model
@st.cache_resource
def load_model():
    try:
        # Load the Stable Diffusion model
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None  # Disable safety checker for better performance
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        
        return pipe
    except Exception as e:
        st.error(f"Error loading Stable Diffusion model: {str(e)}")
        return None

def detect_nose_landmarks(image):
    if not OPENCV_AVAILABLE:
        st.error("OpenCV is not available. Please check the installation.")
        return None
        
    try:
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            # Process the image
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get nose landmarks (indices 1-10 are typically nose-related)
            nose_landmarks = results.multi_face_landmarks[0].landmark[1:11]
            
            # Create a mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Convert landmarks to pixel coordinates
            h, w = image.shape[:2]
            nose_points = np.array([[int(l.x * w), int(l.y * h)] for l in nose_landmarks])
            
            # Draw the nose region on the mask
            cv2.fillConvexPoly(mask, nose_points, 255)
            
            return mask
    except Exception as e:
        st.error(f"Error in nose landmark detection: {str(e)}")
        return None

def generate_rhinoplasty_image(image, mask):
    try:
        if not STABLE_DIFFUSION_AVAILABLE:
            st.error("Stable Diffusion is not available. Please check the installation.")
            return None
            
        # Get the model from cache
        pipe = load_model()
        if pipe is None:
            return None
            
        # Convert image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Create the prompt
        prompt = "a natural-looking nose after rhinoplasty, subtle refinement, realistic, high quality, detailed"
        negative_prompt = "deformed, ugly, unrealistic, cartoon, anime, illustration, painting, drawing"
        
        # Generate the image
        with st.spinner("Generating rhinoplasty result..."):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=20,  # Reduced for faster generation
                guidance_scale=7.5
            ).images[0]
        
        return result
    except Exception as e:
        st.error(f"Error in image generation: {str(e)}")
        return None

def main():
    st.title("AI Rhinoplasty Simulator")
    st.write("Upload a photo to see how you might look after rhinoplasty")
    
    if not OPENCV_AVAILABLE:
        st.error("OpenCV is not available. Please check the installation.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Display original image
            st.image(image, channels="BGR", caption="Original Image")
            
            # Detect nose landmarks and create mask
            mask = detect_nose_landmarks(image)
            
            if mask is not None:
                # Display mask
                st.image(mask, caption="Nose Region Mask")
                
                if st.button("Generate Rhinoplasty Result"):
                    # Generate the rhinoplasty image
                    result_image = generate_rhinoplasty_image(image, mask)
                    
                    if result_image:
                        # Display result
                        st.image(result_image, caption="After Rhinoplasty")
            else:
                st.error("No face detected in the image. Please try another photo.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 