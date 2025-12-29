#!/bin/bash

# Load environment variables from .env file
set -a
source .env
set +a

# Start server
.venv/bin/python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
