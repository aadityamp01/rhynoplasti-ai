import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import base64
from PIL import Image
import io
import requests
import threading
from src.api.process_image import api_bp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Register the API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Set up Streamlit page config
st.set_page_config(
    page_title="AI Rhinoplasty Simulator",
    page_icon="👃",
    layout="wide"
)

# Rest of your app code...
def main():
    st.title("AI Rhinoplasty Simulator")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Effect selection
        selected_effect = st.selectbox(
            "Select Rhinoplasty Effect",
            list(RHINOPLASTY_OPTIONS.keys())
        )
        
        if st.button("Apply Effect"):
            with st.spinner("Processing..."):
                # Process image with API
                result = process_image_with_api(image, selected_effect)
                    
                # Display result
                st.image(result, caption=f"After {selected_effect}", use_column_width=True)
                
                # Add download button
                img_byte_arr = io.BytesIO()
                result.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                st.download_button(
                    label="Download Result",
                    data=img_byte_arr,
                    file_name="rhinoplasty_result.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main() 