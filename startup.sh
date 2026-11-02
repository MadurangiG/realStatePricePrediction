#!/bin/bash
# Azure App Service Linux Startup Script for Streamlit Dashboard
echo "Starting Melbourne Housing Valuation & XAI Streamlit Dashboard..."
python -m streamlit run src/app.py --server.port 8000 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false

