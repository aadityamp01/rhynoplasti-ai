import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import json
import tempfile

try:
    import cv2
    import mediapipe as mp
    from google.cloud import aiplatform
    from vertexai.preview.generative_models import GenerativeModel, Image as VertexImage
    OPENCV_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing required libraries: {str(e)}")
    OPENCV_AVAILABLE = False

# Function to set up Google Cloud credentials from Streamlit secrets
def setup_google_credentials():
    try:
        # Check if credentials are in Streamlit secrets
        if 'google_credentials' in st.secrets:
            # Check if it's a dictionary (TOML format) or a string (JSON format)
            if isinstance(st.secrets['google_credentials'], dict):
                # TOML format - already a dictionary
                credentials = st.secrets['google_credentials']
            else:
                # JSON format - parse the string
                credentials = json.loads(st.secrets['google_credentials'])
            
            # Create a temporary file to store the credentials
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                json.dump(credentials, temp_file)
                temp_file_path = temp_file.name
            
            # Set the environment variable
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
            
            # Set project ID if available
            if 'google_cloud_project' in st.secrets and 'project_id' in st.secrets['google_cloud_project']:
                os.environ['GOOGLE_CLOUD_PROJECT'] = st.secrets['google_cloud_project']['project_id']
            elif 'project_id' in credentials:
                os.environ['GOOGLE_CLOUD_PROJECT'] = credentials['project_id']
                
            return True
        else:
            st.error("Google Cloud credentials not found in Streamlit secrets.")
            return False
    except Exception as e:
        st.error(f"Error setting up Google Cloud credentials: {str(e)}")
        return False

def detect_nose_landmarks(image):
    if not OPENCV_AVAILABLE:
        st.error("OpenCV is not available. Please check the installation.")
        return None
        
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
    except Exception as e:
        st.error(f"Error in nose landmark detection: {str(e)}")
        return None

def generate_rhinoplasty_image(image, mask):
    try:
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
    except Exception as e:
        st.error(f"Error in image generation: {str(e)}")
        return None

def main():
    st.title("AI Rhinoplasty Simulator")
    st.write("Upload a photo to see how you might look after rhinoplasty")
    
    if not OPENCV_AVAILABLE:
        st.error("Required libraries are not available. Please check the installation.")
        return
    
    # Set up Google Cloud credentials
    if not setup_google_credentials():
        st.error("Failed to set up Google Cloud credentials. Please check your Streamlit secrets.")
        return
    
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
            mask = detect_nose_landmarks(image)
            
            if mask is not None:
                # Display mask
                st.image(mask, caption="Nose Region Mask")
                
                if st.button("Generate Rhinoplasty Result"):
                    with st.spinner("Generating result..."):
                        # Generate the rhinoplasty image
                        result_image = generate_rhinoplasty_image(image, mask)
                        
                        if result_image:
                            # Display result
                            st.image(result_image, caption="After Rhinoplasty")
            else:
                st.error("No face detected in the image. Please try another photo.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 