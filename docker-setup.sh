#!/bin/bash

# Quick Docker Build and Test Script
# Builds the Docker image and runs a basic test

set -e  # Exit on error

echo "======================================"
echo "KV-Cache Inference Docker Setup"
echo "======================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running."
    echo "   Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "   Please create .env with: HF_TOKEN=your_token_here"
    exit 1
fi

echo "✅ .env file found"
echo ""

# Build the image
echo "🔨 Building Docker image..."
echo "   (This may take 5-10 minutes on first run)"
docker-compose build

echo ""
echo "✅ Image built successfully"
echo ""

# Start the container
echo "🚀 Starting container..."
docker-compose up -d

echo ""
echo "✅ Container started"
echo ""

# Wait for server to be ready
echo "⏳ Waiting for server to start (may take 1-2 minutes)..."
for i in {1..60}; do
    if curl -s http://localhost:8000/cache/stats > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        break
    fi
    sleep 2
    echo -n "."
done

echo ""
echo ""

# Run test
echo "🧪 Running tests..."
python3 test_docker.py

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "Server is running at: http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f     # View logs"
echo "  docker-compose down        # Stop server"
echo "  docker-compose restart     # Restart server"
echo ""
