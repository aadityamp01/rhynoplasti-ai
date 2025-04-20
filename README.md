# AI Rhinoplasty Simulator

This Streamlit application uses Hugging Face's Stable Diffusion and MediaPipe to simulate rhinoplasty results on uploaded facial images.

## Setup Instructions

1. **Local Environment Setup**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Running the App**
   ```bash
   streamlit run src/app.py
   ```

## Features

- Upload facial images
- Automatic nose region detection using MediaPipe
- AI-powered rhinoplasty simulation using Stable Diffusion
- Real-time preview of results

## Deployment

To deploy on Streamlit Community Cloud:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy!

## Requirements

- Python 3.8+
- All dependencies listed in requirements.txt

## How It Works

1. The app uses MediaPipe Face Mesh to detect facial landmarks
2. It creates a mask around the nose area
3. Stable Diffusion inpainting is used to generate a realistic rhinoplasty result
4. The result is displayed alongside the original image

## Notes

- The first run may take longer as it downloads the Stable Diffusion model
- For better performance, a GPU is recommended but not required 