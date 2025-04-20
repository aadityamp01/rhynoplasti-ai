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
    
    # Replace the placeholder in the HTML with the actual token
    html_content = html_content.replace("'YOUR_CLIENT_TOKEN'", f"'{banuba_token}'")
    
    return html_content

def detect_nose_landmarks(image):
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
            
            return mask, nose_region, (nose_bridge_points, nose_tip_points, nose_side_left_points, nose_side_right_points)
    except Exception as e:
        st.error(f"Error in nose landmark detection: {str(e)}")
        return None, None, None

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
            result = apply_natural_refinement(result, mask, nose_bridge_points, nose_tip_points, intensity)
        elif option == "Nose Bridge Reduction":
            # Reduce the height of the nose bridge
            result = apply_bridge_reduction(result, mask, nose_bridge_points, intensity)
        elif option == "Tip Refinement":
            # Refine the nose tip
            result = apply_tip_refinement(result, mask, nose_tip_points, intensity)
        elif option == "Wide Nose Narrowing":
            # Narrow a wide nose
            result = apply_nose_narrowing(result, mask, nose_bridge_points, nose_tip_points, nose_side_left_points, nose_side_right_points, intensity)
        elif option == "Crooked Nose Correction":
            # Straighten a crooked nose
            result = apply_crooked_correction(result, mask, nose_bridge_points, intensity)
        elif option == "Combined Enhancement":
            # Apply multiple refinements
            result = apply_combined_enhancement(result, mask, nose_bridge_points, nose_tip_points, nose_side_left_points, nose_side_right_points, intensity)
        
        # Apply final touches for realism
        result = apply_final_touches(result, mask)
        
        return result
    except Exception as e:
        st.error(f"Error in rhinoplasty simulation: {str(e)}")
        return image

def apply_natural_refinement(image, mask, nose_bridge_points, nose_tip_points, intensity=0.5):
    # Create a copy of the image
    result = image.copy()
    
    # Create a mask for the nose
    nose_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(nose_mask, [nose_bridge_points, nose_tip_points], 255)
    
    # Apply a slight blur to the nose region to simulate smoothing
    nose_region = cv2.bitwise_and(result, result, mask=nose_mask)
    blurred_nose = cv2.GaussianBlur(nose_region, (5, 5), 0)
    
    # Apply a slight brightness increase to the nose region
    hsv = cv2.cvtColor(blurred_nose, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + 0.05 * intensity)  # Increase brightness by 5% * intensity
    brightened_nose = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(nose_mask))
    result = cv2.add(result, brightened_nose)
    
    # Apply a slight contour adjustment to simulate a more defined nose
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(nose_mask, kernel, iterations=1)
    contour_mask = cv2.bitwise_xor(dilated_mask, nose_mask)
    
    # Apply a slight darkening to the contour to create definition
    contour_region = cv2.bitwise_and(result, result, mask=contour_mask)
    hsv_contour = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * (1 - 0.05 * intensity)  # Decrease brightness by 5% * intensity
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    # Apply a slight nose tip refinement
    tip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(tip_mask, [nose_tip_points], 255)
    
    # Apply a slight blur to the nose tip region
    tip_region = cv2.bitwise_and(result, result, mask=tip_mask)
    blurred_tip = cv2.GaussianBlur(tip_region, (5, 5), 0)
    
    # Apply a slight brightness increase to the nose tip region
    hsv_tip = cv2.cvtColor(blurred_tip, cv2.COLOR_BGR2HSV)
    hsv_tip[:, :, 2] = hsv_tip[:, :, 2] * (1 + 0.1 * intensity)  # Increase brightness by 10% * intensity
    brightened_tip = cv2.cvtColor(hsv_tip, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose tip region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(tip_mask))
    result = cv2.add(result, brightened_tip)
    
    return result

def apply_bridge_reduction(image, mask, nose_bridge_points, intensity=0.7):
    # Create a copy of the image
    result = image.copy()
    
    # Create a mask for the nose bridge
    bridge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(bridge_mask, [nose_bridge_points], 255)
    
    # Apply a slight blur to the nose bridge region
    bridge_region = cv2.bitwise_and(result, result, mask=bridge_mask)
    blurred_bridge = cv2.GaussianBlur(bridge_region, (7, 7), 0)
    
    # Apply a slight brightness increase to the nose bridge region
    hsv = cv2.cvtColor(blurred_bridge, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + 0.1 * intensity)  # Increase brightness by 10% * intensity
    brightened_bridge = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose bridge region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(bridge_mask))
    result = cv2.add(result, brightened_bridge)
    
    # Apply a slight contour adjustment to simulate a more defined nose bridge
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(bridge_mask, kernel, iterations=1)
    contour_mask = cv2.bitwise_xor(dilated_mask, bridge_mask)
    
    # Apply a slight darkening to the contour to create definition
    contour_region = cv2.bitwise_and(result, result, mask=contour_mask)
    hsv_contour = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * (1 - 0.1 * intensity)  # Decrease brightness by 10% * intensity
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    # Apply a slight nose bridge reduction using warping
    h, w = image.shape[:2]
    
    # Create a grid of points
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xx, yy = np.meshgrid(x, y)
    
    # Create a displacement map for the nose bridge
    displacement = np.zeros((h, w, 2), dtype=np.float32)
    
    # Calculate the center of the nose bridge
    center_x = np.mean(nose_bridge_points[:, 0])
    center_y = np.mean(nose_bridge_points[:, 1])
    
    # Apply a vertical displacement to reduce the nose bridge height
    for i in range(h):
        for j in range(w):
            # Calculate the distance from the center of the nose bridge
            dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            
            # Apply a displacement based on the distance
            if dist < 50 * intensity:  # Adjust the radius based on intensity
                # Calculate the displacement factor
                factor = 1 - (dist / (50 * intensity))
                
                # Apply a vertical displacement to reduce the nose bridge height
                displacement[i, j, 1] = -10 * factor * intensity  # Adjust the magnitude based on intensity
    
    # Apply the displacement map
    result = cv2.remap(result, xx.astype(np.float32), yy.astype(np.float32) + displacement[:, :, 1], cv2.INTER_LINEAR)
    
    return result

def apply_tip_refinement(image, mask, nose_tip_points, intensity=0.6):
    # Create a copy of the image
    result = image.copy()
    
    # Create a mask for the nose tip
    tip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(tip_mask, [nose_tip_points], 255)
    
    # Apply a slight blur to the nose tip region
    tip_region = cv2.bitwise_and(result, result, mask=tip_mask)
    blurred_tip = cv2.GaussianBlur(tip_region, (5, 5), 0)
    
    # Apply a slight brightness increase to the nose tip region
    hsv = cv2.cvtColor(blurred_tip, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + 0.15 * intensity)  # Increase brightness by 15% * intensity
    brightened_tip = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose tip region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(tip_mask))
    result = cv2.add(result, brightened_tip)
    
    # Apply a slight contour adjustment to simulate a more defined nose tip
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(tip_mask, kernel, iterations=1)
    contour_mask = cv2.bitwise_xor(dilated_mask, tip_mask)
    
    # Apply a slight darkening to the contour to create definition
    contour_region = cv2.bitwise_and(result, result, mask=contour_mask)
    hsv_contour = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * (1 - 0.1 * intensity)  # Decrease brightness by 10% * intensity
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    # Apply a slight nose tip refinement using warping
    h, w = image.shape[:2]
    
    # Create a grid of points
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xx, yy = np.meshgrid(x, y)
    
    # Create a displacement map for the nose tip
    displacement = np.zeros((h, w, 2), dtype=np.float32)
    
    # Calculate the center of the nose tip
    center_x = np.mean(nose_tip_points[:, 0])
    center_y = np.mean(nose_tip_points[:, 1])
    
    # Apply a vertical displacement to refine the nose tip
    for i in range(h):
        for j in range(w):
            # Calculate the distance from the center of the nose tip
            dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            
            # Apply a displacement based on the distance
            if dist < 30 * intensity:  # Adjust the radius based on intensity
                # Calculate the displacement factor
                factor = 1 - (dist / (30 * intensity))
                
                # Apply a vertical displacement to refine the nose tip
                displacement[i, j, 1] = -5 * factor * intensity  # Adjust the magnitude based on intensity
    
    # Apply the displacement map
    result = cv2.remap(result, xx.astype(np.float32), yy.astype(np.float32) + displacement[:, :, 1], cv2.INTER_LINEAR)
    
    return result

def apply_nose_narrowing(image, mask, nose_bridge_points, nose_tip_points, nose_side_left_points, nose_side_right_points, intensity=0.8):
    # Create a copy of the image
    result = image.copy()
    
    # Create a mask for the nose
    nose_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(nose_mask, [nose_bridge_points, nose_tip_points], 255)
    
    # Apply a slight blur to the nose region
    nose_region = cv2.bitwise_and(result, result, mask=nose_mask)
    blurred_nose = cv2.GaussianBlur(nose_region, (7, 7), 0)
    
    # Apply a slight brightness increase to the nose region
    hsv = cv2.cvtColor(blurred_nose, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + 0.1 * intensity)  # Increase brightness by 10% * intensity
    brightened_nose = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(nose_mask))
    result = cv2.add(result, brightened_nose)
    
    # Apply a slight contour adjustment to simulate a narrower nose
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(nose_mask, kernel, iterations=1)
    contour_mask = cv2.bitwise_xor(nose_mask, eroded_mask)
    
    # Apply a slight darkening to the contour to create definition
    contour_region = cv2.bitwise_and(result, result, mask=contour_mask)
    hsv_contour = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * (1 - 0.1 * intensity)  # Decrease brightness by 10% * intensity
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    # Apply a slight nose narrowing using warping
    h, w = image.shape[:2]
    
    # Create a grid of points
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xx, yy = np.meshgrid(x, y)
    
    # Create a displacement map for the nose narrowing
    displacement = np.zeros((h, w, 2), dtype=np.float32)
    
    # Calculate the center of the nose
    center_x = np.mean(nose_bridge_points[:, 0])
    center_y = np.mean(nose_bridge_points[:, 1])
    
    # Apply a horizontal displacement to narrow the nose
    for i in range(h):
        for j in range(w):
            # Calculate the distance from the center of the nose
            dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            
            # Apply a displacement based on the distance
            if dist < 40 * intensity:  # Adjust the radius based on intensity
                # Calculate the displacement factor
                factor = 1 - (dist / (40 * intensity))
                
                # Apply a horizontal displacement to narrow the nose
                if j < center_x:
                    displacement[i, j, 0] = 5 * factor * intensity  # Adjust the magnitude based on intensity
                else:
                    displacement[i, j, 0] = -5 * factor * intensity  # Adjust the magnitude based on intensity
    
    # Apply the displacement map
    result = cv2.remap(result, xx.astype(np.float32) + displacement[:, :, 0], yy.astype(np.float32), cv2.INTER_LINEAR)
    
    return result

def apply_crooked_correction(image, mask, nose_bridge_points, intensity=0.7):
    # Create a copy of the image
    result = image.copy()
    
    # Create a mask for the nose bridge
    bridge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(bridge_mask, [nose_bridge_points], 255)
    
    # Apply a slight blur to the nose bridge region
    bridge_region = cv2.bitwise_and(result, result, mask=bridge_mask)
    blurred_bridge = cv2.GaussianBlur(bridge_region, (7, 7), 0)
    
    # Apply a slight brightness increase to the nose bridge region
    hsv = cv2.cvtColor(blurred_bridge, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + 0.1 * intensity)  # Increase brightness by 10% * intensity
    brightened_bridge = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose bridge region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(bridge_mask))
    result = cv2.add(result, brightened_bridge)
    
    # Apply a slight contour adjustment to simulate a straighter nose bridge
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(bridge_mask, kernel, iterations=1)
    contour_mask = cv2.bitwise_xor(dilated_mask, bridge_mask)
    
    # Apply a slight darkening to the contour to create definition
    contour_region = cv2.bitwise_and(result, result, mask=contour_mask)
    hsv_contour = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * (1 - 0.1 * intensity)  # Decrease brightness by 10% * intensity
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    # Apply a slight crooked nose correction using warping
    h, w = image.shape[:2]
    
    # Create a grid of points
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xx, yy = np.meshgrid(x, y)
    
    # Create a displacement map for the crooked nose correction
    displacement = np.zeros((h, w, 2), dtype=np.float32)
    
    # Calculate the center of the nose bridge
    center_x = np.mean(nose_bridge_points[:, 0])
    center_y = np.mean(nose_bridge_points[:, 1])
    
    # Calculate the angle of the nose bridge
    angle = np.arctan2(nose_bridge_points[-1, 1] - nose_bridge_points[0, 1], nose_bridge_points[-1, 0] - nose_bridge_points[0, 0])
    
    # Apply a displacement to straighten the nose bridge
    for i in range(h):
        for j in range(w):
            # Calculate the distance from the center of the nose bridge
            dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            
            # Apply a displacement based on the distance
            if dist < 50 * intensity:  # Adjust the radius based on intensity
                # Calculate the displacement factor
                factor = 1 - (dist / (50 * intensity))
                
                # Apply a displacement to straighten the nose bridge
                displacement[i, j, 0] = -5 * factor * intensity * np.sin(angle)  # Adjust the magnitude based on intensity
                displacement[i, j, 1] = 5 * factor * intensity * np.cos(angle)  # Adjust the magnitude based on intensity
    
    # Apply the displacement map
    result = cv2.remap(result, xx.astype(np.float32) + displacement[:, :, 0], yy.astype(np.float32) + displacement[:, :, 1], cv2.INTER_LINEAR)
    
    return result

def apply_combined_enhancement(image, mask, nose_bridge_points, nose_tip_points, nose_side_left_points, nose_side_right_points, intensity=0.6):
    # Apply multiple refinements
    result = apply_natural_refinement(image, mask, nose_bridge_points, nose_tip_points, intensity * 0.8)
    result = apply_bridge_reduction(result, mask, nose_bridge_points, intensity * 0.7)
    result = apply_tip_refinement(result, mask, nose_tip_points, intensity * 0.9)
    result = apply_nose_narrowing(result, mask, nose_bridge_points, nose_tip_points, nose_side_left_points, nose_side_right_points, intensity * 0.6)
    return result

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

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #2c3e50;'>AI Rhinoplasty Simulator</h1>
            <p style='color: #7f8c8d; font-size: 1.2em;'>
                Experience how you might look after rhinoplasty using advanced AI technology
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Check Banuba token status
    banuba_token_status = check_banuba_token()

    # Create tabs for different simulation methods
    tab1, tab2 = st.tabs(["AR Simulation", "Local Processing"])
    
    with tab1:
        st.markdown("""
            <div class='upload-section'>
                <h2 style='color: #2c3e50;'>AR Rhinoplasty Simulation</h2>
                <p style='color: #7f8c8d;'>Use your camera for real-time rhinoplasty simulation</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display the AR integration
        ar_html = get_ar_html()
        st.components.v1.html(ar_html, height=800, scrolling=True)
    
    with tab2:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
                <div class='upload-section'>
                    <h2 style='color: #2c3e50;'>Upload Your Photo</h2>
                    <p style='color: #7f8c8d;'>Choose a clear, front-facing photo for best results</p>
                </div>
            """, unsafe_allow_html=True)
            
            # File uploader with custom styling
            uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"], key="photo_upload", label_visibility="collapsed")

        with col2:
            st.markdown("""
                <div class='upload-section'>
                    <h2 style='color: #2c3e50;'>Choose Your Options</h2>
                    <p style='color: #7f8c8d;'>Select the rhinoplasty type</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Rhinoplasty options with custom styling
            option = st.selectbox(
                "Select Rhinoplasty Type",
                list(RHINOPLASTY_OPTIONS.keys()),
                format_func=lambda x: f"{RHINOPLASTY_OPTIONS[x]['icon']} {x}"
            )
            
            # Display option description
            st.markdown(f"""
                <div class='option-card'>
                    <p style='color: #2c3e50;'>{RHINOPLASTY_OPTIONS[option]['description']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Add intensity slider
            intensity = st.slider(
                "Adjust the intensity of the effect",
                min_value=0.0,
                max_value=1.0,
                value=RHINOPLASTY_OPTIONS[option]["intensity"],
                step=0.1,
                format="%.1f"
            )
        
        if uploaded_file is not None:
            try:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                # Create columns for before/after comparison
                before_col, after_col = st.columns(2)
                
                with before_col:
                    st.markdown("""
                        <h3 style='color: #2c3e50; text-align: center;'>Before</h3>
                    """, unsafe_allow_html=True)
                    st.image(image, channels="BGR", use_container_width=True)
                
                # Detect nose landmarks and create mask
                mask, nose_region, nose_landmarks = detect_nose_landmarks(image)
                
                if mask is not None:
                    if st.button("Generate Rhinoplasty Result", key="generate_btn"):
                        with st.spinner("Processing your image..."):
                            # Simulate the rhinoplasty
                            result_image = simulate_rhinoplasty(image, mask, nose_landmarks, option)
                            
                            with after_col:
                                st.markdown("""
                                    <h3 style='color: #2c3e50; text-align: center;'>After</h3>
                                """, unsafe_allow_html=True)
                                st.image(result_image, channels="BGR", use_container_width=True)
                            
                            # Add download button with custom styling
                            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                            result_pil = Image.fromarray(result_rgb)
                            buf = io.BytesIO()
                            result_pil.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            st.markdown("""
                                <div style='text-align: center; margin-top: 2rem;'>
                            """, unsafe_allow_html=True)
                            
                            st.download_button(
                                label="Download Result",
                                data=byte_im,
                                file_name=f"rhinoplasty_result_{option.lower().replace(' ', '_')}.png",
                                mime="image/png"
                            )
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("No face detected in the image. Please try another photo.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 