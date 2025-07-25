# OpenLongContext Deployment Guide

This guide covers deployment options for the OpenLongContext platform, from development to production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Development Deployment](#development-deployment)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Monitoring and Scaling](#monitoring-and-scaling)
- [Security Considerations](#security-considerations)

## Prerequisites

### System Requirements
- Python 3.9+
- 16GB+ RAM (32GB+ recommended for large models)
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- 50GB+ disk space
- Ubuntu 20.04+ or compatible Linux distribution

### Software Dependencies
- Docker and Docker Compose (for containerized deployment)
- NVIDIA Container Toolkit (for GPU support)
- PostgreSQL 13+ (for production database)
- Redis 6+ (for caching and queuing)
- Nginx (for reverse proxy)

## Development Deployment

### Quick Start
```bash
# Clone repository
git clone https://github.com/openlongcontext/openlongcontext.git
cd openlongcontext

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[all]"

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export OPENLONGCONTEXT_ENV="development"

# Run development server
uvicorn openlongcontext.api:app --reload --host 0.0.0.0 --port 8000
```

### Development Configuration
Create `.env` file:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=true

# Model Configuration
MODEL_CACHE_DIR=/tmp/model_cache
MAX_SEQUENCE_LENGTH=16384
BATCH_SIZE=8

# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4-turbo-preview

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=json
```

## Docker Deployment

### Single Container
```bash
# Build image
docker build -t openlongcontext:latest .

# Run container
docker run -d \
  --name openlongcontext \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/data:/app/data \
  --gpus all \
  openlongcontext:latest
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/openlongcontext
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - model_cache:/app/models
    depends_on:
      - db
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=openlongcontext
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
  model_cache:
```

### Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/certs/cert.pem;
        ssl_certificate_key /etc/nginx/certs/key.pem;

        client_max_body_size 100M;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;

        location / {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## Production Deployment

### Environment Setup
```bash
# Create production user
sudo useradd -m -s /bin/bash openlongcontext
sudo su - openlongcontext

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev \
    build-essential postgresql-client redis-tools nginx supervisor

# Setup application directory
mkdir -p /opt/openlongcontext
cd /opt/openlongcontext
```

### Gunicorn Configuration
```python
# gunicorn_config.py
import multiprocessing
import os

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
timeout = 120
graceful_timeout = 30
accesslog = "/var/log/openlongcontext/access.log"
errorlog = "/var/log/openlongcontext/error.log"
loglevel = "info"
preload_app = True

# SSL Configuration (if not using nginx)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
```

### Supervisor Configuration
```ini
# /etc/supervisor/conf.d/openlongcontext.conf
[program:openlongcontext]
command=/opt/openlongcontext/venv/bin/gunicorn openlongcontext.api:app -c /opt/openlongcontext/gunicorn_config.py
directory=/opt/openlongcontext
user=openlongcontext
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/openlongcontext/supervisor.log
environment=PATH="/opt/openlongcontext/venv/bin",OPENLONGCONTEXT_ENV="production"
```

### Database Setup
```sql
-- Create database and user
CREATE DATABASE openlongcontext;
CREATE USER openlongcontext WITH ENCRYPTED PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE openlongcontext TO openlongcontext;

-- Enable extensions
\c openlongcontext
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";
```

### Redis Configuration
```conf
# /etc/redis/redis.conf additions
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Cloud Deployments

### AWS EC2 Deployment
```bash
# Launch EC2 instance (p3.2xlarge or similar for GPU)
# AMI: Deep Learning AMI (Ubuntu 20.04)

# SSH into instance
ssh -i your-key.pem ubuntu@ec2-instance

# Setup application
git clone https://github.com/openlongcontext/openlongcontext.git
cd openlongcontext

# Use systemd service
sudo cp deployment/openlongcontext.service /etc/systemd/system/
sudo systemctl enable openlongcontext
sudo systemctl start openlongcontext
```

### Kubernetes Deployment
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openlongcontext
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openlongcontext
  template:
    metadata:
      labels:
        app: openlongcontext
    spec:
      containers:
      - name: api
        image: openlongcontext:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openlongcontext-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: openlongcontext
spec:
  selector:
    app: openlongcontext
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Google Cloud Run
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/openlongcontext

# Deploy to Cloud Run
gcloud run deploy openlongcontext \
  --image gcr.io/PROJECT_ID/openlongcontext \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --concurrency 100 \
  --max-instances 10 \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

## Monitoring and Scaling

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'openlongcontext'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard
Import dashboard JSON from `deployment/grafana-dashboard.json` for:
- Request rate and latency
- Model inference time
- Memory and GPU usage
- Error rates and logs

### Auto-scaling Configuration
```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: openlongcontext-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openlongcontext
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Security Considerations

### API Security
```python
# Enable in production settings
SECURITY_SETTINGS = {
    "enable_auth": True,
    "jwt_secret": os.environ["JWT_SECRET"],
    "jwt_algorithm": "HS256",
    "access_token_expire_minutes": 30,
    "refresh_token_expire_days": 7,
    "api_key_header": "X-API-Key",
    "rate_limit": "100/minute",
    "cors_origins": ["https://your-frontend.com"],
    "ssl_redirect": True,
    "content_security_policy": "default-src 'self'",
}
```

### Network Security
```bash
# Firewall rules
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw enable

# Fail2ban configuration
sudo apt-get install fail2ban
sudo systemctl enable fail2ban
```

### Secrets Management
```bash
# Use environment variables or secret management service
export DATABASE_URL=$(aws secretsmanager get-secret-value --secret-id prod/db/url --query SecretString --output text)
export OPENAI_API_KEY=$(aws secretsmanager get-secret-value --secret-id prod/openai/key --query SecretString --output text)
```

## Backup and Recovery

### Database Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U openlongcontext -d openlongcontext | gzip > "$BACKUP_DIR/backup_$TIMESTAMP.sql.gz"

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

### Model Cache Backup
```bash
# Sync model cache to S3
aws s3 sync /app/models s3://your-bucket/model-cache --delete
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size in configuration
   - Enable gradient checkpointing
   - Use model quantization

2. **Slow Inference**
   - Enable GPU acceleration
   - Use model caching
   - Implement request batching

3. **Connection Timeouts**
   - Increase nginx/gunicorn timeouts
   - Implement async processing for long tasks
   - Use background job queue

### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Database connection
psql -h localhost -U openlongcontext -c "SELECT 1"

# Redis connection
redis-cli ping

# GPU availability
nvidia-smi
```

## Performance Optimization

### Model Optimization
```python
# Enable optimizations
OPTIMIZATION_CONFIG = {
    "use_amp": True,  # Automatic Mixed Precision
    "compile_model": True,  # Torch compile
    "use_flash_attention": True,
    "gradient_checkpointing": True,
    "quantization": "int8",  # or "int4" for more compression
}
```

### Caching Strategy
```python
# Redis caching configuration
CACHE_CONFIG = {
    "embedding_cache_ttl": 3600,  # 1 hour
    "result_cache_ttl": 300,  # 5 minutes
    "max_cache_size": "10GB",
    "eviction_policy": "lru",
}
```

For additional deployment support, refer to the [documentation](https://openlongcontext.github.io/openlongcontext/) or open an issue on [GitHub](https://github.com/openlongcontext/openlongcontext/issues).