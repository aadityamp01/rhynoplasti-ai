import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Image as VertexImage
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def detect_nose_landmarks(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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

def generate_rhinoplasty_image(image, mask):
    # Initialize Vertex AI
    aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
    
    # Convert image to Vertex AI format
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    vertex_image = VertexImage.from_bytes(image_bytes)
    
    # Create the prompt
    prompt = """
    Perform a subtle rhinoplasty refinement on the masked area of the face.
    Maintain natural-looking results while improving the nose shape and profile.
    Keep the rest of the face unchanged.
    """
    
    # Generate the image
    model = GenerativeModel("imagegeneration@002")
    response = model.generate_content(
        [prompt, vertex_image],
        generation_config={
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 40
        }
    )
    
    # Convert response to PIL Image
    generated_image = Image.open(io.BytesIO(response.image))
    return generated_image

def main():
    st.title("AI Rhinoplasty Simulator")
    st.write("Upload a photo to see how you might look after rhinoplasty")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
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
                with st.spinner("Generating result..."):
                    # Generate the rhinoplasty image
                    result_image = generate_rhinoplasty_image(image, mask)
                    
                    # Display result
                    st.image(result_image, caption="After Rhinoplasty")
        else:
            st.error("No face detected in the image. Please try another photo.")

if __name__ == "__main__":
    main() 