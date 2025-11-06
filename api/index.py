"""
Vercel serverless function entry point for Flask app
"""
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Vercel environment variable
os.environ['VERCEL'] = '1'

# Import and initialize the Flask app
from app import app, init_db

# Initialize database on first import
init_db()

# Export the Flask app for Vercel
# Vercel will automatically handle WSGI requests
__all__ = ['app']
