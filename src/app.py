import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import cv2
import mediapipe as mp
from scipy.interpolate import griddata
import random
import base64
import json
import requests
from pathlib import Path
from io import BytesIO
import math

# Set page config
st.set_page_config(
    page_title="AI Rhinoplasty Simulator",
    page_icon="👃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .option-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .option-card:hover {
        border-color: #4CAF50;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    iframe {
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Define rhinoplasty options with icons
RHINOPLASTY_OPTIONS = {
    "Natural Refinement": {
        "description": "Subtle changes to create a more balanced nose while maintaining natural appearance",
        "icon": "✨",
        "intensity": 0.5
    },
    "Nose Bridge Reduction": {
        "description": "Reduce the height of the nose bridge for a more streamlined profile",
        "icon": "📉",
        "intensity": 0.7
    },
    "Tip Refinement": {
        "description": "Refine the nose tip to be more defined and elegant",
        "icon": "🎯",
        "intensity": 0.6
    },
    "Wide Nose Narrowing": {
        "description": "Narrow a wide nose for better facial harmony",
        "icon": "↔️",
        "intensity": 0.8
    },
    "Crooked Nose Correction": {
        "description": "Straighten a crooked nose for better symmetry",
        "icon": "📏",
        "intensity": 0.7
    },
    "Combined Enhancement": {
        "description": "Comprehensive nose reshaping with multiple refinements",
        "icon": "🌟",
        "intensity": 0.6
    }
}

# Define available AI models
AI_MODELS = {
    "ar": {
        "name": "AR Simulation",
        "description": "Advanced AR technology for realistic rhinoplasty simulation",
        "icon": "🤖"
    },
    "local": {
        "name": "Local Processing",
        "description": "Process images locally using computer vision techniques",
        "icon": "💻"
    }
}

def get_ar_html():
    """Get the HTML content for the AR integration"""
    html_path = Path(__file__).parent / "ar_integration.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Get the Banuba client token from Streamlit secrets
    banuba_token = st.secrets.get("BANUBA_CLIENT_TOKEN", "")
    
    # Add a script tag to set the global token variable
    token_script = f"""
    <script>
        window.BANUBA_CLIENT_TOKEN = "{banuba_token}";
    </script>
    """
    
    # Insert the token script before the AR Integration script
    html_content = html_content.replace(
        '<!-- AR Integration -->',
        f'<!-- AR Integration -->\n{token_script}'
    )
    
    return html_content

def detect_nose_landmarks(image):
    """
    Detect nose landmarks in the image using MediaPipe Face Mesh.
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        
    Returns:
        tuple: (mask, nose_region, landmarks)
            - mask: Binary mask of the nose region
            - nose_region: Cropped nose region
            - landmarks: List of nose landmarks
    """
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
            return None, None, None
        
        # Get all face landmarks
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Get nose landmarks (indices 1-10 are typically nose-related)
        nose_landmarks = face_landmarks[1:11]
        
        # Get additional landmarks for better nose shape detection
        nose_bridge = [face_landmarks[i] for i in [6, 197, 195, 5, 4, 1, 19, 94]]
        nose_tip = [face_landmarks[i] for i in [1, 2, 98, 97, 2, 326, 327, 331]]
        nose_side_left = [face_landmarks[i] for i in [129, 209, 49, 131, 134, 45, 4, 1, 19, 94]]
        nose_side_right = [face_landmarks[i] for i in [358, 429, 279, 359, 362, 275, 4, 1, 19, 94]]
        
        # Create a mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Convert landmarks to pixel coordinates
        h, w = image.shape[:2]
        nose_points = np.array([[int(l.x * w), int(l.y * h)] for l in nose_landmarks])
        
        # Draw the nose region on the mask
        cv2.fillConvexPoly(mask, nose_points, 255)
        
        # Get the nose region
        nose_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert additional landmarks to pixel coordinates
        nose_bridge_points = np.array([[int(l.x * w), int(l.y * h)] for l in nose_bridge])
        nose_tip_points = np.array([[int(l.x * w), int(l.y * h)] for l in nose_tip])
        nose_side_left_points = np.array([[int(l.x * w), int(l.y * h)] for l in nose_side_left])
        nose_side_right_points = np.array([[int(l.x * w), int(l.y * h)] for l in nose_side_right])
        
        # Return the full list of face landmarks for processing
        # This is what the rhinoplasty functions expect
        return mask, nose_region, face_landmarks

def simulate_rhinoplasty(image, mask, nose_landmarks, option):
    try:
        # Create a copy of the image
        result = image.copy()
        
        # Get nose bridge and tip points
        nose_bridge_points, nose_tip_points, nose_side_left_points, nose_side_right_points = nose_landmarks
        
        # Get the intensity for the selected option
        intensity = RHINOPLASTY_OPTIONS[option]["intensity"]
        
        # Apply different rhinoplasty simulations based on the selected option
        if option == "Natural Refinement":
            # Subtle changes to create a more balanced nose
            result = apply_natural_refinement(result, mask, nose_landmarks, intensity)
        elif option == "Nose Bridge Reduction":
            # Reduce the height of the nose bridge
            result = apply_bridge_reduction(result, mask, nose_landmarks, intensity)
        elif option == "Tip Refinement":
            # Refine the nose tip
            result = apply_tip_refinement(result, mask, nose_landmarks, intensity)
        elif option == "Wide Nose Narrowing":
            # Narrow a wide nose
            result = apply_nose_narrowing(result, mask, nose_landmarks, intensity)
        elif option == "Crooked Nose Correction":
            # Straighten a crooked nose
            result = apply_crooked_correction(result, mask, nose_landmarks, intensity)
        elif option == "Combined Enhancement":
            # Apply multiple refinements
            result = apply_combined_enhancement(result, mask, nose_landmarks, intensity)
        
        # Apply final touches for realism
        result = apply_final_touches(result, mask)
        
        return result
    except Exception as e:
        st.error(f"Error in rhinoplasty simulation: {str(e)}")
        return image

def apply_natural_refinement(image, mask, landmarks, intensity):
    """Apply subtle natural refinement to the nose."""
    # Check if we have enough landmarks
    if len(landmarks) < 34:
        print("Warning: Not enough landmarks detected for natural refinement")
        return image
    
    try:
        # Create a smooth mask for the nose area
        nose_mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Apply brightness adjustment
        brightness_factor = 1.0 + (0.1 * intensity)
        brightened = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        
        # Apply subtle smoothing
        smoothed = cv2.GaussianBlur(brightened, (3, 3), 0)
        
        # Blend the results
        result = cv2.addWeighted(image, 1 - intensity, smoothed, intensity, 0)
        return result
    except Exception as e:
        print(f"Error in natural refinement: {e}")
        return image

def apply_bridge_reduction(image, mask, landmarks, intensity):
    """Reduce the height of the nose bridge."""
    # Check if we have enough landmarks
    if len(landmarks) < 31:
        print("Warning: Not enough landmarks detected for bridge reduction")
        return image
    
    try:
        # Get bridge landmarks
        bridge_top = landmarks[27]  # Nose bridge top
        bridge_bottom = landmarks[30]  # Nose bridge bottom
        
        # Calculate vertical shift
        shift = int((bridge_bottom[1] - bridge_top[1]) * 0.2 * intensity)
        
        # Create displacement map for the bridge
        height, width = image.shape[:2]
        displacement_map = np.zeros((height, width, 2), dtype=np.float32)
        
        # Apply vertical shift to bridge area
        for y in range(bridge_top[1], bridge_bottom[1]):
            for x in range(bridge_top[0] - 10, bridge_top[0] + 10):
                if 0 <= x < width and 0 <= y < height:
                    displacement_map[y, x, 1] = -shift
        
        # Apply the displacement
        result = cv2.remap(image, 
                          np.arange(width).astype(np.float32),
                          np.arange(height).astype(np.float32),
                          displacement_map,
                          cv2.INTER_LINEAR)
        
        # Blend with original
        return cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
    except Exception as e:
        print(f"Error in bridge reduction: {e}")
        return image

def apply_tip_refinement(image, mask, landmarks, intensity):
    """Refine and lift the nose tip."""
    # Check if we have enough landmarks
    if len(landmarks) < 34:
        print("Warning: Not enough landmarks detected for tip refinement")
        return image
    
    try:
        # Get tip landmarks
        tip = landmarks[33]  # Nose tip
        
        # Create a circular mask for the tip
        tip_mask = np.zeros_like(mask)
        cv2.circle(tip_mask, (tip[0], tip[1]), 15, 255, -1)
        tip_mask = cv2.GaussianBlur(tip_mask, (21, 21), 0)
        
        # Brighten and lift the tip
        brightened = cv2.convertScaleAbs(image, alpha=1.0 + (0.2 * intensity), beta=0)
        
        # Apply subtle upward shift
        height, width = image.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, -5 * intensity]])
        shifted = cv2.warpAffine(brightened, M, (width, height))
        
        # Blend results
        return cv2.addWeighted(image, 1 - intensity, shifted, intensity, 0)
    except Exception as e:
        print(f"Error in tip refinement: {e}")
        return image

def apply_nose_narrowing(image, mask, landmarks, intensity):
    """Narrow the width of the nose."""
    # Check if we have enough landmarks
    if len(landmarks) < 359:
        print("Warning: Not enough landmarks detected for nose narrowing")
        return image
    
    try:
        # Get nose width landmarks
        left = landmarks[129]  # Left nostril
        right = landmarks[358]  # Right nostril
        
        # Create horizontal displacement map
        height, width = image.shape[:2]
        displacement_map = np.zeros((height, width, 2), dtype=np.float32)
        
        # Calculate horizontal shift
        shift = int((right[0] - left[0]) * 0.15 * intensity)
        
        # Apply horizontal shift to nose sides
        for x in range(left[0], right[0]):
            for y in range(left[1] - 20, left[1] + 20):
                if 0 <= x < width and 0 <= y < height:
                    if x < (left[0] + right[0]) // 2:
                        displacement_map[y, x, 0] = shift
                    else:
                        displacement_map[y, x, 0] = -shift
        
        # Apply the displacement
        result = cv2.remap(image,
                          np.arange(width).astype(np.float32),
                          np.arange(height).astype(np.float32),
                          displacement_map,
                          cv2.INTER_LINEAR)
        
        # Blend with original
        return cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
    except Exception as e:
        print(f"Error in nose narrowing: {e}")
        return image

def apply_crooked_correction(image, mask, landmarks, intensity):
    """Correct crooked nose by straightening the bridge."""
    # Check if we have enough landmarks
    if len(landmarks) < 31:
        print("Warning: Not enough landmarks detected for crooked nose correction")
        return image
    
    try:
        # Get bridge landmarks
        bridge_top = landmarks[27]
        bridge_bottom = landmarks[30]
        
        # Calculate angle to straighten
        dx = bridge_bottom[0] - bridge_top[0]
        dy = bridge_bottom[1] - bridge_top[1]
        angle = math.atan2(dx, dy)
        
        # Create a mask for the bridge area
        bridge_mask = np.zeros_like(mask)
        cv2.line(bridge_mask, bridge_top, bridge_bottom, 255, 10)
        
        # Apply rotation to straighten
        height, width = image.shape[:2]
        center = (bridge_top[0], bridge_top[1])
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle * 180 / math.pi, 1.0)
        
        # Apply the rotation only to the bridge area
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        result = image.copy()
        result[bridge_mask > 0] = rotated[bridge_mask > 0]
        
        # Blend with original based on intensity
        result = cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
        
        return result
    except Exception as e:
        print(f"Error in crooked nose correction: {e}")
        return image

def apply_combined_enhancement(image, mask, landmarks, intensity):
    """Apply a combination of all enhancements."""
    # Check if we have enough landmarks
    if len(landmarks) < 359:
        print("Warning: Not enough landmarks detected for combined enhancement")
        return image
    
    try:
        # Apply each effect with reduced intensity
        reduced_intensity = intensity * 0.7
        
        result = apply_natural_refinement(image, mask, landmarks, reduced_intensity)
        result = apply_bridge_reduction(result, mask, landmarks, reduced_intensity)
        result = apply_tip_refinement(result, mask, landmarks, reduced_intensity)
        result = apply_nose_narrowing(result, mask, landmarks, reduced_intensity)
        result = apply_crooked_correction(result, mask, landmarks, reduced_intensity)
        
        return result
    except Exception as e:
        print(f"Error in combined enhancement: {e}")
        return image

def apply_final_touches(image, mask):
    # Create a copy of the image
    result = image.copy()
    
    # Apply a slight blur to the entire image to smooth out any artifacts
    result = cv2.GaussianBlur(result, (3, 3), 0)
    
    # Apply a slight sharpening to enhance details
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    result = cv2.filter2D(result, -1, kernel)
    
    # Apply a slight color correction to match the skin tone
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.95  # Decrease saturation by 5%
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply a slight noise reduction
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    
    return result

def check_banuba_token():
    """Check if the Banuba client token is properly set in Streamlit secrets"""
    try:
        # Check if the token exists in secrets
        token_exists = "BANUBA_CLIENT_TOKEN" in st.secrets
        
        # Get the token (without exposing it)
        token = st.secrets.get("BANUBA_CLIENT_TOKEN", "")
        token_valid = bool(token and token.strip())
        
        # Display status
        if token_exists and token_valid:
            st.success("✅ Banuba client token is properly configured")
            return True
        elif token_exists:
            st.error("❌ Banuba client token is empty. Please add a valid token to your Streamlit secrets.")
            return False
        else:
            st.error("❌ Banuba client token is missing. Please add BANUBA_CLIENT_TOKEN to your Streamlit secrets.")
            return False
    except Exception as e:
        st.error(f"❌ Error checking Banuba token: {str(e)}")
        return False

def check_effect_files():
    """Check if the required effect files exist"""
    # We no longer need to check for effect files since we're using custom canvas-based effects
    return True

def process_image(image, option, intensity):
    """
    Process the uploaded image with the selected rhinoplasty option.
    
    Args:
        image (PIL.Image): The uploaded image
        option (str): The selected rhinoplasty option
        intensity (float): The intensity of the effect (0.0 to 1.0)
    
    Returns:
        PIL.Image: The processed image
    """
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect nose landmarks
    mask, nose_region, nose_landmarks = detect_nose_landmarks(image_cv)
    
    if mask is None:
        raise ValueError("No face detected in the image. Please try another photo.")
    
    # Apply the selected rhinoplasty option
    try:
        if option == "Natural Refinement":
            result = apply_natural_refinement(image_cv, mask, nose_landmarks, intensity)
        elif option == "Nose Bridge Reduction":
            result = apply_bridge_reduction(image_cv, mask, nose_landmarks, intensity)
        elif option == "Tip Refinement":
            result = apply_tip_refinement(image_cv, mask, nose_landmarks, intensity)
        elif option == "Wide Nose Narrowing":
            result = apply_nose_narrowing(image_cv, mask, nose_landmarks, intensity)
        elif option == "Crooked Nose Correction":
            result = apply_crooked_correction(image_cv, mask, nose_landmarks, intensity)
        else:  # Combined Enhancement
            result = apply_combined_enhancement(image_cv, mask, nose_landmarks, intensity)
        
        # Convert back to PIL Image
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    except Exception as e:
        print(f"Error processing image: {e}")
        # Return the original image if processing fails
        return image

def main():
    # Header
    st.title("Rhinoplasty AI Simulator")
    
    # Check Banuba token and effect files
    token_valid = check_banuba_token()
    effects_valid = check_effect_files()
    
    # Tabs for different features
    tab1, tab2 = st.tabs(["Image Upload", "AR Simulation"])
    
    with tab1:
        st.header("Upload Your Photo")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Rhinoplasty options
            st.subheader("Choose Rhinoplasty Option")
            option = st.selectbox(
                "Select the type of rhinoplasty simulation",
                ["Natural Refinement", "Nose Bridge Reduction", "Tip Refinement", 
                 "Wide Nose Narrowing", "Crooked Nose Correction", "Combined Enhancement"]
            )
            
            # Intensity slider
            intensity = st.slider("Adjust the intensity of the effect", 0.0, 1.0, 0.5)
            
            if st.button("Apply Rhinoplasty"):
                with st.spinner("Processing..."):
                    # Process the image
                    result = process_image(image, option, intensity)
                    
                    # Display the result
                    st.image(result, caption="Rhinoplasty Result", use_container_width=True)
                    
                    # Add download button
                    buf = BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button(
                        label="Download Result",
                        data=buf.getvalue(),
                        file_name="rhinoplasty_result.png",
                        mime="image/png"
                    )
    
    with tab2:
        st.header("AR Rhinoplasty Simulation")
        
        if not token_valid:
            st.error("Please configure the Banuba client token in your Streamlit secrets to use the AR simulation.")
            return
        
        if not effects_valid:
            st.error("Please download the required effect files from Banuba and place them in the 'effects' directory.")
            return
        
        # Display the AR simulation
        st.components.v1.html(get_ar_html(), height=600)
        

if __name__ == "__main__":
    main() 