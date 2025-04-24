import os
import sys
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import src.app as app

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Set up Streamlit page config
st.set_page_config(
    page_title="AI Rhinoplasty Simulator",
    page_icon="👃",
    layout="wide"
)

# Rhinoplasty options
RHINOPLASTY_OPTIONS = {
    "Natural Refinement": {
        "description": "Subtle nose refinement for a more balanced profile",
        "prompt": "Apply subtle rhinoplasty to make the nose more refined and proportional."
    },
    "Bridge Reduction": {
        "description": "Reduce nose bridge height for a more delicate profile",
        "prompt": "Perform rhinoplasty to reduce the height of the nose bridge."
    },
    "Tip Refinement": {
        "description": "Refine nose tip for better definition",
        "prompt": "Refine the nose tip through rhinoplasty to create better definition."
    },
    "Nose Narrowing": {
        "description": "Reduce nose width for better proportion",
        "prompt": "Apply rhinoplasty to narrow the nose width."
    },
    "Crooked Correction": {
        "description": "Correct nose deviation for better symmetry",
        "prompt": "Correct any nose deviation through rhinoplasty."
    },
    "Combined Enhancement": {
        "description": "Comprehensive nose enhancement combining multiple effects",
        "prompt": "Apply comprehensive rhinoplasty combining multiple refinements."
    }
}

def detect_nose_landmarks(image):
    """Detect nose landmarks using MediaPipe Face Mesh."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    # Get nose landmarks
    nose_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        # Nose bridge points (27-31)
        nose_bridge = [face_landmarks.landmark[i] for i in range(27, 32)]
        # Nose tip point (4)
        nose_tip = face_landmarks.landmark[4]
        # Nostril points (129-134)
        nostrils = [face_landmarks.landmark[i] for i in range(129, 135)]
        
        nose_landmarks.extend(nose_bridge)
        nose_landmarks.append(nose_tip)
        nose_landmarks.extend(nostrils)
    
    return nose_landmarks

def create_nose_mask(image, landmarks):
    """Create a mask for the nose region."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if landmarks:
        points = []
        for landmark in landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)
    
    return mask

def apply_natural_refinement(img):
    """Apply natural refinement effect to the nose."""
    landmarks = detect_nose_landmarks(img)
    if landmarks is None:
        return img
    
    mask = create_nose_mask(img, landmarks)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance brightness and contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    img = cv2.bitwise_and(img, img, mask=~mask)
    result = cv2.add(img, result)
    
    return result

def apply_bridge_reduction(img):
    """Apply bridge reduction effect to the nose."""
    landmarks = detect_nose_landmarks(img)
    if landmarks is None:
        return img
    
    mask = create_nose_mask(img, landmarks)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Darken the bridge area
    l = cv2.addWeighted(l, 0.8, np.zeros_like(l), 0.2, 0)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    img = cv2.bitwise_and(img, img, mask=~mask)
    result = cv2.add(img, result)
    
    return result

def apply_tip_refinement(img):
    """Apply tip refinement effect to the nose."""
    landmarks = detect_nose_landmarks(img)
    if landmarks is None:
        return img
    
    # Create mask for just the tip
    tip_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    tip_point = landmarks[5]  # Nose tip landmark
    x = int(tip_point.x * img.shape[1])
    y = int(tip_point.y * img.shape[0])
    cv2.circle(tip_mask, (x, y), 20, 255, -1)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Brighten the tip area
    l = cv2.addWeighted(l, 1.2, np.ones_like(l) * 255, 0.1, 0)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply only to tip region
    result = cv2.bitwise_and(result, result, mask=tip_mask)
    img = cv2.bitwise_and(img, img, mask=~tip_mask)
    result = cv2.add(img, result)
    
    return result

def apply_nose_narrowing(img):
    """Apply nose narrowing effect."""
    landmarks = detect_nose_landmarks(img)
    if landmarks is None:
        return img
    
    mask = create_nose_mask(img, landmarks)
    height, width = img.shape[:2]
    
    # Apply warping to narrow the nose
    map_x = np.zeros((height, width), np.float32)
    map_y = np.zeros((height, width), np.float32)
    
    center_x = width // 2
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                # Calculate distance from center
                dx = x - center_x
                # Apply narrowing effect
                map_x[y, x] = x - dx * 0.2
                map_y[y, x] = y
            else:
                map_x[y, x] = x
                map_y[y, x] = y
    
    result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    img = cv2.bitwise_and(img, img, mask=~mask)
    result = cv2.add(img, result)
    
    return result

def apply_crooked_correction(img):
    """Apply crooked nose correction effect."""
    landmarks = detect_nose_landmarks(img)
    if landmarks is None:
        return img
    
    mask = create_nose_mask(img, landmarks)
    height, width = img.shape[:2]
    
    # Get nose bridge points
    bridge_points = landmarks[:5]
    
    # Calculate the angle of deviation
    start_point = bridge_points[0]
    end_point = bridge_points[-1]
    angle = np.arctan2(end_point.y - start_point.y, end_point.x - start_point.x)
    
    # Create a transformation matrix for rotation
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle * 180 / np.pi * 0.5, 1.0)
    
    # Apply rotation
    result = cv2.warpAffine(img, matrix, (width, height))
    
    # Apply only to nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    img = cv2.bitwise_and(img, img, mask=~mask)
    result = cv2.add(img, result)
    
    return result

def apply_combined_enhancement(img):
    """Apply all effects with reduced intensity."""
    result = img.copy()
    result = apply_natural_refinement(result)
    result = apply_bridge_reduction(result)
    result = apply_tip_refinement(result)
    result = apply_nose_narrowing(result)
    result = apply_crooked_correction(result)
    return result

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