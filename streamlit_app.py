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

# Check for required secrets
required_secrets = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"]
missing_secrets = [secret for secret in required_secrets if secret not in st.secrets]

if missing_secrets:
    st.error("⚠️ Missing required secrets!")
    st.error("Please add the following secrets to your Streamlit app:")
    for secret in missing_secrets:
        st.error(f"- {secret}")
    st.info("You can add secrets in the Streamlit Cloud dashboard under 'Secrets'")
    st.stop()

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
    .ai-badge {
        background-color: #4285F4;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        display: inline-block;
        margin-bottom: 10px;
    }
    .processing-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .secret-status {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def process_image(image, effect):
    """Process image with selected rhinoplasty effect."""
    try:
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
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return image

def main():
    st.title("AI Rhinoplasty Simulator")
    
    # Add Google Vertex AI badge and secret status
    st.markdown('<div class="ai-badge">Powered by Google Vertex AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="secret-status">✓ Google Cloud credentials configured</div>', unsafe_allow_html=True)
    
    # Add processing information
    st.markdown("""
        <div class="processing-info">
            <strong>Processing Information:</strong><br>
            • Using Google's Vertex AI for precise nose modifications<br>
            • AI-powered facial landmark detection<br>
            • Real-time image processing and enhancement<br>
            • Project ID: {project_id}<br>
            • Location: {location}
        </div>
    """.format(
        project_id=st.secrets["GOOGLE_CLOUD_PROJECT"],
        location=st.secrets["GOOGLE_CLOUD_LOCATION"]
    ), unsafe_allow_html=True)
    
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
            with st.spinner("Processing with Google Vertex AI..."):
                try:
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
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.error("Please check your Google Cloud credentials and try again.")

if __name__ == "__main__":
    main() 