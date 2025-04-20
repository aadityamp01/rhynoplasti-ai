import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import cv2
import mediapipe as mp

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
    </style>
""", unsafe_allow_html=True)

# Define rhinoplasty options with icons
RHINOPLASTY_OPTIONS = {
    "Natural Refinement": {
        "description": "Subtle changes to create a more balanced nose while maintaining natural appearance",
        "icon": "✨"
    },
    "Nose Bridge Reduction": {
        "description": "Reduce the height of the nose bridge for a more streamlined profile",
        "icon": "📉"
    },
    "Tip Refinement": {
        "description": "Refine the nose tip to be more defined and elegant",
        "icon": "🎯"
    },
    "Wide Nose Narrowing": {
        "description": "Narrow a wide nose for better facial harmony",
        "icon": "↔️"
    },
    "Crooked Nose Correction": {
        "description": "Straighten a crooked nose for better symmetry",
        "icon": "📏"
    },
    "Combined Enhancement": {
        "description": "Comprehensive nose reshaping with multiple refinements",
        "icon": "🌟"
    }
}

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
            
            return mask, nose_region, (nose_bridge_points, nose_tip_points)
    except Exception as e:
        st.error(f"Error in nose landmark detection: {str(e)}")
        return None, None, None

def simulate_rhinoplasty(image, mask, nose_landmarks, option):
    try:
        # Create a copy of the image
        result = image.copy()
        
        # Get nose bridge and tip points
        nose_bridge_points, nose_tip_points = nose_landmarks
        
        # Apply different rhinoplasty simulations based on the selected option
        if option == "Natural Refinement":
            # Subtle changes to create a more balanced nose
            result = apply_natural_refinement(result, mask, nose_bridge_points, nose_tip_points)
        elif option == "Nose Bridge Reduction":
            # Reduce the height of the nose bridge
            result = apply_bridge_reduction(result, mask, nose_bridge_points)
        elif option == "Tip Refinement":
            # Refine the nose tip
            result = apply_tip_refinement(result, mask, nose_tip_points)
        elif option == "Wide Nose Narrowing":
            # Narrow a wide nose
            result = apply_nose_narrowing(result, mask, nose_bridge_points, nose_tip_points)
        elif option == "Crooked Nose Correction":
            # Straighten a crooked nose
            result = apply_crooked_correction(result, mask, nose_bridge_points)
        elif option == "Combined Enhancement":
            # Apply multiple refinements
            result = apply_combined_enhancement(result, mask, nose_bridge_points, nose_tip_points)
        
        return result
    except Exception as e:
        st.error(f"Error in rhinoplasty simulation: {str(e)}")
        return image

def apply_natural_refinement(image, mask, nose_bridge_points, nose_tip_points):
    # Create a copy of the image
    result = image.copy()
    
    # Apply a slight blur to the nose region to simulate smoothing
    nose_region = cv2.bitwise_and(result, result, mask=mask)
    blurred_nose = cv2.GaussianBlur(nose_region, (5, 5), 0)
    
    # Apply a slight brightness increase to the nose region
    hsv = cv2.cvtColor(blurred_nose, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * 1.05  # Increase brightness by 5%
    brightened_nose = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(mask))
    result = cv2.add(result, brightened_nose)
    
    # Apply a slight contour adjustment to simulate a more defined nose
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    contour_mask = cv2.bitwise_xor(dilated_mask, mask)
    
    # Apply a slight darkening to the contour to create definition
    contour_region = cv2.bitwise_and(result, result, mask=contour_mask)
    hsv_contour = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * 0.95  # Decrease brightness by 5%
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    return result

def apply_bridge_reduction(image, mask, nose_bridge_points):
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
    hsv[:, :, 2] = hsv[:, :, 2] * 1.1  # Increase brightness by 10%
    brightened_bridge = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply the modified nose bridge region back to the image
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(bridge_mask))
    result = cv2.add(result, brightened_bridge)
    
    return result

def apply_tip_refinement(image, mask, nose_tip_points):
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
    hsv[:, :, 2] = hsv[:, :, 2] * 1.15  # Increase brightness by 15%
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
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * 0.9  # Decrease brightness by 10%
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    return result

def apply_nose_narrowing(image, mask, nose_bridge_points, nose_tip_points):
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
    hsv[:, :, 2] = hsv[:, :, 2] * 1.1  # Increase brightness by 10%
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
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * 0.9  # Decrease brightness by 10%
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    return result

def apply_crooked_correction(image, mask, nose_bridge_points):
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
    hsv[:, :, 2] = hsv[:, :, 2] * 1.1  # Increase brightness by 10%
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
    hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * 0.9  # Decrease brightness by 10%
    darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
    
    # Apply the contour adjustment
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
    result = cv2.add(result, darkened_contour)
    
    return result

def apply_combined_enhancement(image, mask, nose_bridge_points, nose_tip_points):
    # Apply multiple refinements
    result = apply_natural_refinement(image, mask, nose_bridge_points, nose_tip_points)
    result = apply_bridge_reduction(result, mask, nose_bridge_points)
    result = apply_tip_refinement(result, mask, nose_tip_points)
    return result

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
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="photo_upload")

    with col2:
        st.markdown("""
            <div class='upload-section'>
                <h2 style='color: #2c3e50;'>Choose Your Option</h2>
                <p style='color: #7f8c8d;'>Select the type of rhinoplasty you're interested in</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Rhinoplasty options with custom styling
        option = st.selectbox(
            "",
            list(RHINOPLASTY_OPTIONS.keys()),
            format_func=lambda x: f"{RHINOPLASTY_OPTIONS[x]['icon']} {x}"
        )
        
        # Display option description
        st.markdown(f"""
            <div class='option-card'>
                <p style='color: #2c3e50;'>{RHINOPLASTY_OPTIONS[option]['description']}</p>
            </div>
        """, unsafe_allow_html=True)

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
                st.image(image, channels="BGR", use_column_width=True)
            
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
                            st.image(result_image, channels="BGR", use_column_width=True)
                        
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
                            label="📥 Download Result",
                            data=byte_im,
                            file_name=f"rhinoplasty_result_{option.lower().replace(' ', '_')}.png",
                            mime="image/png"
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("No face detected in the image. Please try another photo.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 2rem; margin-top: 2rem;'>
            <p style='color: #7f8c8d;'>
                This is a simulation tool and should not be used as a substitute for professional medical advice.
                Please consult with a qualified healthcare provider for medical procedures.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 