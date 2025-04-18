# AI Rhinoplasty Simulator

This Streamlit application uses Google Cloud's Vertex AI and MediaPipe to simulate rhinoplasty results on uploaded facial images.

## Setup Instructions

1. **Google Cloud Setup**
   - Create a new Google Cloud project
   - Enable the Vertex AI API
   - Create a service account and download the JSON key file
   - Set the following environment variables:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
     GOOGLE_CLOUD_PROJECT=your-project-id
     ```

2. **Local Environment Setup**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Running the App**
   ```bash
   streamlit run app.py
   ```

## Features

- Upload facial images
- Automatic nose region detection
- AI-powered rhinoplasty simulation
- Real-time preview of results

## Deployment

To deploy on Streamlit Community Cloud:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Add your Google Cloud credentials as secrets in the Streamlit Cloud dashboard
4. Deploy!

## Requirements

- Python 3.8+
- Google Cloud account with Vertex AI enabled
- All dependencies listed in requirements.txt

## Security Note

Never commit your Google Cloud service account key to version control. Always use environment variables or secrets management for sensitive credentials. 