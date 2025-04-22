import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import streamlit as st
import src.app as app  # Changed to import the module directly

if __name__ == "__main__":
    app.main() 