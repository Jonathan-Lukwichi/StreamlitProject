"""
Redirect file for Streamlit Cloud compatibility.
This file exists to maintain compatibility with existing Streamlit Cloud deployments
that expect app.py as the entry point.

The actual application code is in Welcome.py.
"""

# Import and run the main Welcome.py application
import sys
import os

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Welcome module which contains the actual app
import Welcome
