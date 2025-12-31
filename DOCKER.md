# Docker Deployment Guide

This guide explains how to build and run the KV-Cache Inference server using Docker.

## 🚀 Quick Start

### Prerequisites
- Docker installed ([Download Docker Desktop](https://www.docker.com/products/docker-desktop))
- HuggingFace token (set in `.env` file)

### Build and Run

```bash
# 1. Make sure .env file exists with HF_TOKEN
echo "HF_TOKEN=your_token_here" > .env

# 2. Build and start the container
docker-compose up -d

# 3. Check logs
docker-compose logs -f

# 4. Test the server
python test_docker.py
```

## 📦 Building the Image

### Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Using Docker CLI
```bash
# Build image
docker build -t kv-cache-inference .

# Run container
docker run -d \
  --name kv-cache-server \
  -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  -v huggingface-cache:/root/.cache/huggingface \
  kv-cache-inference
```

## 🧪 Testing the Container

```bash
# Run test script
python test_docker.py

# Or manually test with curl
curl http://localhost:8000/cache/stats

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_new_tokens": 20}'
```

## 🛠️ Common Commands

```bash
# View running containers
docker ps

# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Restart container
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build

# Execute command inside container
docker exec -it kv-cache-server bash

# View resource usage
docker stats kv-cache-server
```

## 📊 Container Configuration

### Environment Variables
- `HF_TOKEN`: HuggingFace API token (required)
- `PYTHONUNBUFFERED`: Set to 1 for immediate log output

### Ports
- `8000`: FastAPI server port

### Volumes
- `huggingface-cache`: Persists downloaded models
- `./logs`: Optional log directory

### Resource Limits
- **CPU**: 4 cores (max)
- **Memory**: 8GB (max), 4GB (reserved)

## 🔧 Customization

### Change Model
Edit `api/server.py`:
```python
MODEL_NAME = "your-model-name"
```

### Change Port
Edit `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Host port 8080 → Container port 8000
```

### Adjust Resources
Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
```

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs for errors
docker-compose logs

# Verify .env file exists
cat .env

# Check if port 8000 is in use
lsof -i :8000
```

### Out of memory
```bash
# Increase memory limit in docker-compose.yml
# Or use a smaller model
```

### Model download fails
```bash
# Check HF_TOKEN is correct
docker exec kv-cache-server env | grep HF_TOKEN

# Manually download inside container
docker exec -it kv-cache-server bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
```

### Health check failing
```bash
# The server takes ~30-60 seconds to start on first run (model download)
# Wait for health check to pass or check logs
docker-compose logs -f
```

## 🚢 Deployment

### Push to Docker Hub
```bash
# Tag image
docker tag kv-cache-inference username/kv-cache-inference:latest

# Login to Docker Hub
docker login

# Push image
docker push username/kv-cache-inference:latest
```

### Deploy to Cloud

**AWS ECS / Azure Container Instances / Google Cloud Run:**
1. Push image to container registry
2. Create service using the image
3. Set environment variables (HF_TOKEN)
4. Configure port mapping (8000)
5. Set resource limits

## 📝 Notes

- **GPU Support**: Docker on Mac doesn't support MPS. Server will use CPU.
- **First Run**: Takes 5-10 minutes to download TinyLlama model (~2.2GB)
- **Model Cache**: Persisted in Docker volume `huggingface-cache`
- **Production**: Consider adding HTTPS, authentication, and monitoring

## 🔐 Security

- Never commit `.env` file to git
- Use secrets management in production (AWS Secrets Manager, etc.)
- Run container as non-root user in production
- Enable network security groups/firewalls

---

For more information, see the main [README.md](README.md).
