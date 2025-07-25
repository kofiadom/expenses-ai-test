# Docker Setup for Medical Processing Service

This document provides instructions for building and running the Medical Processing Service using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- External Redis instance (not included in Docker setup)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Application
NODE_ENV=production
PORT=3000

# Redis Configuration (External)
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0

# AI Service API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_KEY=your_anthropic_api_key_here
LLAMAINDEX_API_KEY=your_llamaindex_api_key_here

# File Upload Configuration
MAX_FILE_SIZE=50MB
UPLOAD_PATH=/app/uploads

# Job Queue Configuration
QUEUE_CONCURRENCY=10
MAX_RETRY_ATTEMPTS=3
JOB_TIMEOUT=300000

# Monitoring
ENABLE_SWAGGER=true
ENABLE_THROTTLING=true
THROTTLE_TTL=60
THROTTLE_LIMIT=100

# Webhook URLs (optional, comma-separated)
WEBHOOK_URLS=http://localhost:3001/webhooks/job-status,https://api.example.com/webhooks/medical-processing
```

## Building the Docker Image

### Build Production Image

```bash
docker build -t medical-processing-service:latest .
```

### Build with Custom Tag

```bash
docker build -t medical-processing-service:v1.0.0 .
```

## Running with Docker Compose

### Start the Service

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f medical-processing-service
```

### Stop the Service

```bash
docker-compose down
```

### Rebuild and Restart

```bash
docker-compose up -d --build
```

## Running with Docker Run

### Basic Run Command

```bash
docker run -d \
  --name medical-processing-app \
  -p 3000:3000 \
  --env-file .env \
  -v medical_uploads:/app/uploads \
  -v medical_results:/app/results \
  -v medical_logs:/app/logs \
  medical-processing-service:latest
```

### Run with Environment Variables

```bash
docker run -d \
  --name medical-processing-app \
  -p 3000:3000 \
  -e NODE_ENV=production \
  -e PORT=3000 \
  -e REDIS_HOST=your-redis-host \
  -e REDIS_PORT=6379 \
  -e OPENAI_API_KEY=your_key \
  -e ANTHROPIC_KEY=your_key \
  -e LLAMAINDEX_API_KEY=your_key \
  -v medical_uploads:/app/uploads \
  -v medical_results:/app/results \
  medical-processing-service:latest
```

## Health Checks

The container includes built-in health checks:

### Check Container Health

```bash
docker ps
# Look for "healthy" status in the STATUS column
```

### Manual Health Check

```bash
docker exec medical-processing-app node /app/healthcheck.js
```

### Health Check Endpoint

```bash
curl http://localhost:3000/health
```

## Volume Management

### List Volumes

```bash
docker volume ls | grep medical
```

### Inspect Volume

```bash
docker volume inspect medical-processing-service_uploads_data
```

### Backup Volumes

```bash
# Backup uploads
docker run --rm -v medical-processing-service_uploads_data:/data -v $(pwd):/backup alpine tar czf /backup/uploads-backup.tar.gz -C /data .

# Backup results
docker run --rm -v medical-processing-service_results_data:/data -v $(pwd):/backup alpine tar czf /backup/results-backup.tar.gz -C /data .
```

## Monitoring and Debugging

### View Container Logs

```bash
docker logs medical-processing-app -f
```

### Execute Commands in Container

```bash
docker exec -it medical-processing-app sh
```

### Monitor Resource Usage

```bash
docker stats medical-processing-app
```

## Production Deployment

### Environment-Specific Builds

```bash
# Staging
docker build -t medical-processing-service:staging .

# Production
docker build -t medical-processing-service:production .
```

### Container Registry

```bash
# Tag for registry
docker tag medical-processing-service:latest your-registry.com/medical-processing-service:latest

# Push to registry
docker push your-registry.com/medical-processing-service:latest
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Verify Redis host and credentials in environment variables
   - Ensure Redis is accessible from Docker container

2. **File Upload Issues**
   - Check volume mounts for uploads directory
   - Verify file size limits in environment variables

3. **API Key Issues**
   - Ensure all required API keys are set in environment variables
   - Check API key validity and quotas

4. **Memory Issues**
   - Monitor container memory usage with `docker stats`
   - Adjust Docker memory limits if needed

### Debug Mode

Run container with debug logging:

```bash
docker run -d \
  --name medical-processing-debug \
  -p 3000:3000 \
  --env-file .env \
  -e NODE_ENV=development \
  -v medical_uploads:/app/uploads \
  -v medical_results:/app/results \
  medical-processing-service:latest
```

### Container Shell Access

```bash
docker exec -it medical-processing-app sh
```

## API Documentation

Once the container is running, access the Swagger API documentation at:

```
http://localhost:3000/api
```

## Security Considerations

- Container runs as non-root user (nestjs:nodejs)
- Sensitive data should be passed via environment variables or Docker secrets
- Use proper network isolation in production
- Regularly update base images for security patches

## Performance Tuning

### Memory Limits

```bash
docker run -d \
  --name medical-processing-app \
  --memory=2g \
  --memory-swap=2g \
  -p 3000:3000 \
  --env-file .env \
  medical-processing-service:latest
```

### CPU Limits

```bash
docker run -d \
  --name medical-processing-app \
  --cpus=2 \
  -p 3000:3000 \
  --env-file .env \
  medical-processing-service:latest
