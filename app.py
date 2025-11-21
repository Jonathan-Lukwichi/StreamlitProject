"""
Redirect file for Streamlit Cloud compatibility.
This file exists to maintain compatibility with existing Streamlit Cloud deployments
that expect app.py as the entry point.

The actual application code is in Welcome.py.
"""

# Execute Welcome.py content directly
with open('Welcome.py', 'r', encoding='utf-8') as f:
    exec(f.read())
