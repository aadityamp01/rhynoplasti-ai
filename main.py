import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Streamlit app
from streamlit_app import main

if __name__ == "__main__":
    # This will be used when running locally with 'python main.py'
    main() 