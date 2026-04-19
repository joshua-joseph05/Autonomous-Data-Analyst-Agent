"""
DataInsight AI - Main FastAPI application
"""
from fastapi import FastAPI
from app.main import app

__version__ = "1.0.0"
__all__ = ["app", "__version__"]
