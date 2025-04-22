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

# Rhinoplasty options
RHINOPLASTY_OPTIONS = {
    "Natural Refinement": {
        "description": "Subtle nose refinement for a more balanced profile",
        "intensity": 0.3
    },
    "Bridge Reduction": {
        "description": "Reduce nose bridge height for a more delicate profile",
        "intensity": 0.4
    },
    "Tip Refinement": {
        "description": "Refine nose tip for better definition",
        "intensity": 0.35
    },
    "Nose Narrowing": {
        "description": "Reduce nose width for better proportion",
        "intensity": 0.4
    },
    "Crooked Correction": {
        "description": "Correct nose deviation for better symmetry",
        "intensity": 0.45
    },
    "Combined Enhancement": {
        "description": "Comprehensive nose enhancement combining multiple effects",
        "intensity": 0.35
    }
}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

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

def apply_natural_refinement(image, intensity=0.3):
    """Apply natural nose refinement."""
    landmarks = detect_nose_landmarks(image)
    if not landmarks:
        return image
    
    mask = create_nose_mask(image, landmarks)
    
    # Create a copy of the image
    result = image.copy()
    
    # Apply subtle brightness and contrast adjustments
    alpha = 1.0 + (intensity * 0.2)  # Contrast
    beta = intensity * 10  # Brightness
    
    # Apply adjustments only to the nose region
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    result = cv2.bitwise_and(result, result, mask=mask)
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
    
    return result

def apply_bridge_reduction(image, intensity=0.4):
    """Apply nose bridge reduction."""
    landmarks = detect_nose_landmarks(image)
    if not landmarks:
        return image
    
    mask = create_nose_mask(image, landmarks)
    
    # Create a copy of the image
    result = image.copy()
    
    # Apply darkening effect to the bridge
    darkening = np.ones_like(image) * (1 - intensity * 0.5)
    result = cv2.multiply(result, darkening)
    
    # Apply only to the nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
    
    return result

def apply_tip_refinement(image, intensity=0.35):
    """Apply nose tip refinement."""
    landmarks = detect_nose_landmarks(image)
    if not landmarks:
        return image
    
    # Create a mask for just the tip
    tip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    tip_point = landmarks[5]  # Nose tip landmark
    x = int(tip_point.x * image.shape[1])
    y = int(tip_point.y * image.shape[0])
    cv2.circle(tip_mask, (x, y), 20, 255, -1)
    
    # Create a copy of the image
    result = image.copy()
    
    # Apply brightening effect to the tip
    brightening = np.ones_like(image) * (1 + intensity * 0.3)
    result = cv2.multiply(result, brightening)
    
    # Apply only to the tip region
    result = cv2.bitwise_and(result, result, mask=tip_mask)
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
    
    return result

def apply_nose_narrowing(image, intensity=0.4):
    """Apply nose narrowing effect."""
    landmarks = detect_nose_landmarks(image)
    if not landmarks:
        return image
    
    mask = create_nose_mask(image, landmarks)
    
    # Create a copy of the image
    result = image.copy()
    
    # Get nose width points
    left_point = landmarks[0]
    right_point = landmarks[-1]
    
    # Calculate center and width
    center_x = (left_point.x + right_point.x) / 2
    width = right_point.x - left_point.x
    
    # Create a transformation matrix for narrowing
    matrix = np.float32([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    # Apply narrowing transformation
    matrix[0, 0] = 1 - (intensity * 0.3)  # Scale factor
    matrix[0, 2] = center_x * (1 - matrix[0, 0])  # Offset
    
    # Apply transformation
    result = cv2.warpAffine(result, matrix[:2], (image.shape[1], image.shape[0]))
    
    # Apply only to the nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
    
    return result

def apply_crooked_correction(image, intensity=0.45):
    """Apply crooked nose correction."""
    landmarks = detect_nose_landmarks(image)
    if not landmarks:
        return image
    
    mask = create_nose_mask(image, landmarks)
    
    # Create a copy of the image
    result = image.copy()
    
    # Get nose bridge points
    bridge_points = landmarks[:5]
    
    # Calculate the angle of deviation
    start_point = bridge_points[0]
    end_point = bridge_points[-1]
    angle = np.arctan2(end_point.y - start_point.y, end_point.x - start_point.x)
    
    # Create a transformation matrix for rotation
    center = (image.shape[1] / 2, image.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, -angle * 180 / np.pi * intensity, 1.0)
    
    # Apply rotation
    result = cv2.warpAffine(result, matrix, (image.shape[1], image.shape[0]))
    
    # Apply only to the nose region
    result = cv2.bitwise_and(result, result, mask=mask)
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
    
    return result

def apply_combined_enhancement(image, intensity=0.35):
    """Apply combined nose enhancement."""
    # Apply all effects with reduced intensity
    result = apply_natural_refinement(image, intensity * 0.8)
    result = apply_bridge_reduction(result, intensity * 0.8)
    result = apply_tip_refinement(result, intensity * 0.8)
    result = apply_nose_narrowing(result, intensity * 0.8)
    result = apply_crooked_correction(result, intensity * 0.8)
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get image data from request
        image_data = request.json['image']
        effect = request.json['effect']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        
        # Apply selected effect
        if effect == "Natural Refinement":
            result = apply_natural_refinement(image, RHINOPLASTY_OPTIONS["Natural Refinement"]["intensity"])
        elif effect == "Bridge Reduction":
            result = apply_bridge_reduction(image, RHINOPLASTY_OPTIONS["Bridge Reduction"]["intensity"])
        elif effect == "Tip Refinement":
            result = apply_tip_refinement(image, RHINOPLASTY_OPTIONS["Tip Refinement"]["intensity"])
        elif effect == "Nose Narrowing":
            result = apply_nose_narrowing(image, RHINOPLASTY_OPTIONS["Nose Narrowing"]["intensity"])
        elif effect == "Crooked Correction":
            result = apply_crooked_correction(image, RHINOPLASTY_OPTIONS["Crooked Correction"]["intensity"])
        elif effect == "Combined Enhancement":
            result = apply_combined_enhancement(image, RHINOPLASTY_OPTIONS["Combined Enhancement"]["intensity"])
        else:
            return jsonify({"error": "Invalid effect selected"}), 400
        
        # Convert result to base64
        _, buffer = cv2.imencode('.jpg', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({"result": f"data:image/jpeg;base64,{result_base64}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    st.title("AI Rhinoplasty Simulator")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Effect selection
        selected_effect = st.selectbox(
            "Select Rhinoplasty Effect",
            list(RHINOPLASTY_OPTIONS.keys())
        )
        
        if st.button("Apply Effect"):
            with st.spinner("Processing..."):
                # Apply selected effect
                if selected_effect == "Natural Refinement":
                    result = apply_natural_refinement(image_cv, RHINOPLASTY_OPTIONS["Natural Refinement"]["intensity"])
                elif selected_effect == "Bridge Reduction":
                    result = apply_bridge_reduction(image_cv, RHINOPLASTY_OPTIONS["Bridge Reduction"]["intensity"])
                elif selected_effect == "Tip Refinement":
                    result = apply_tip_refinement(image_cv, RHINOPLASTY_OPTIONS["Tip Refinement"]["intensity"])
                elif selected_effect == "Nose Narrowing":
                    result = apply_nose_narrowing(image_cv, RHINOPLASTY_OPTIONS["Nose Narrowing"]["intensity"])
                elif selected_effect == "Crooked Correction":
                    result = apply_crooked_correction(image_cv, RHINOPLASTY_OPTIONS["Crooked Correction"]["intensity"])
                else:
                    result = apply_combined_enhancement(image_cv, RHINOPLASTY_OPTIONS["Combined Enhancement"]["intensity"])
                
                # Convert back to PIL Image
                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                
                # Display result
                st.image(result_image, caption=f"After {selected_effect}", use_column_width=True)
                
                # Add download button
                st.download_button(
                    label="Download Result",
                    data=result_image.tobytes(),
                    file_name="rhinoplasty_result.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()