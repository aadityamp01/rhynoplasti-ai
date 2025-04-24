import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the Streamlit app
import streamlit_app

if __name__ == "__main__":
    # This will be used when running locally with 'python main.py'
    streamlit_app.main() 