# AI Rhinoplasty Simulator

An advanced AI-powered application for simulating rhinoplasty results using computer vision and augmented reality technology.

## Features

- **Banuba AR Integration**: Real-time rhinoplasty simulation using your camera
- **Local Processing**: Upload photos for rhinoplasty simulation using computer vision
- **Multiple Rhinoplasty Options**: Choose from various rhinoplasty types
- **Adjustable Intensity**: Control the strength of the rhinoplasty effect
- **Download Results**: Save your simulated results for reference

## Rhinoplasty Options

- **Natural Refinement**: Subtle changes to create a more balanced nose
- **Nose Bridge Reduction**: Reduce the height of the nose bridge
- **Tip Refinement**: Refine the nose tip to be more defined
- **Wide Nose Narrowing**: Narrow a wide nose for better facial harmony
- **Crooked Nose Correction**: Straighten a crooked nose
- **Combined Enhancement**: Comprehensive nose reshaping

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Node.js and npm (for Banuba SDK)
- Web browser with WebGL 2.0 support

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rhinoplasty-ai.git
   cd rhinoplasty-ai
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Banuba SDK Setup

1. Obtain a Banuba client token:
   - Visit [Banuba's website](https://www.banuba.com/)
   - Sign up for a developer account
   - Create a new project and get your client token

2. Add your client token to Streamlit secrets:
   - For local development, create a `.streamlit/secrets.toml` file with:
     ```toml
     BANUBA_CLIENT_TOKEN = "your_banuba_client_token_here"
     ```
   - For Streamlit Cloud deployment, add the secret in the Streamlit Cloud dashboard

3. Create effect files for each rhinoplasty type:
   - Place the effect files in the `src/effects` directory
   - Name them according to the pattern in `banuba_integration.js`

## Running the Application

1. Start the Streamlit app:
   ```
   streamlit run src/app.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Choose between:
   - **Banuba AR Simulation**: Use your camera for real-time simulation
   - **Local Processing**: Upload a photo for simulation

## How It Works

### Banuba AR Integration

The Banuba SDK provides advanced facial tracking and augmented reality capabilities. The application uses this technology to:

1. Track facial landmarks in real-time
2. Apply 3D transformations to simulate rhinoplasty
3. Render the modified face with realistic lighting and shadows

### Local Processing

For uploaded photos, the application uses:

1. MediaPipe Face Mesh for facial landmark detection
2. Computer vision techniques for nose reshaping
3. Image processing algorithms for realistic results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and entertainment purposes only. It should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for medical procedures. 