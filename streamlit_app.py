import os
import sys
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# Set up Streamlit page config (must be the first Streamlit command)
st.set_page_config(
    page_title="AI Rhinoplasty Simulator",
    page_icon="👃",
    layout="wide"
)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions from src.app
from src.app import (
    detect_nose_landmarks,
    create_nose_mask,
    apply_natural_refinement,
    apply_bridge_reduction,
    apply_tip_refinement,
    apply_nose_narrowing,
    apply_crooked_correction,
    apply_combined_enhancement,
    process_image_with_api,
    RHINOPLASTY_OPTIONS
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .effect-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px;
    }
    .effect-button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

def process_image(image, effect):
    """Process image with selected rhinoplasty effect."""
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply selected effect
    if effect == "Natural Refinement":
        result = apply_natural_refinement(img)
    elif effect == "Bridge Reduction":
        result = apply_bridge_reduction(img)
    elif effect == "Tip Refinement":
        result = apply_tip_refinement(img)
    elif effect == "Nose Narrowing":
        result = apply_nose_narrowing(img)
    elif effect == "Crooked Correction":
        result = apply_crooked_correction(img)
    elif effect == "Combined Enhancement":
        result = apply_combined_enhancement(img)
    else:
        return image
    
    # Convert back to PIL Image
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

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
        
        # Show effect description
        st.write(RHINOPLASTY_OPTIONS[selected_effect]["description"])
        
        if st.button("Apply Effect"):
            with st.spinner("Processing..."):
                # Process image
                result = process_image(image, selected_effect)
                
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