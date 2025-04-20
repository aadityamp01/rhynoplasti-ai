import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import cv2
import mediapipe as mp

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
                return None, None
            
            # Get nose landmarks (indices 1-10 are typically nose-related)
            nose_landmarks = results.multi_face_landmarks[0].landmark[1:11]
            
            # Create a mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Convert landmarks to pixel coordinates
            h, w = image.shape[:2]
            nose_points = np.array([[int(l.x * w), int(l.y * h)] for l in nose_landmarks])
            
            # Draw the nose region on the mask
            cv2.fillConvexPoly(mask, nose_points, 255)
            
            # Get the nose region
            nose_region = cv2.bitwise_and(image, image, mask=mask)
            
            return mask, nose_region
    except Exception as e:
        st.error(f"Error in nose landmark detection: {str(e)}")
        return None, None

def simulate_rhinoplasty(image, mask, nose_region):
    try:
        # Create a copy of the image
        result = image.copy()
        
        # Apply a slight blur to the nose region to simulate smoothing
        blurred_nose = cv2.GaussianBlur(nose_region, (5, 5), 0)
        
        # Apply a slight brightness increase to the nose region
        hsv = cv2.cvtColor(blurred_nose, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * 1.1  # Increase brightness by 10%
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
        hsv_contour[:, :, 2] = hsv_contour[:, :, 2] * 0.9  # Decrease brightness by 10%
        darkened_contour = cv2.cvtColor(hsv_contour, cv2.COLOR_HSV2BGR)
        
        # Apply the contour adjustment
        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(contour_mask))
        result = cv2.add(result, darkened_contour)
        
        return result
    except Exception as e:
        st.error(f"Error in rhinoplasty simulation: {str(e)}")
        return image

def main():
    st.title("AI Rhinoplasty Simulator")
    st.write("Upload a photo to see how you might look after rhinoplasty")
    
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
            mask, nose_region = detect_nose_landmarks(image)
            
            if mask is not None:
                # Display mask
                st.image(mask, caption="Nose Region Mask")
                
                if st.button("Generate Rhinoplasty Result"):
                    # Simulate the rhinoplasty
                    result_image = simulate_rhinoplasty(image, mask, nose_region)
                    
                    # Display result
                    st.image(result_image, channels="BGR", caption="After Rhinoplasty")
            else:
                st.error("No face detected in the image. Please try another photo.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 