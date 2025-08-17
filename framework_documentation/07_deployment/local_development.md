# Local Development with Docker Compose

> 🐳 **Local Environment**: Complete local development setup with Docker Compose for testing before cloud deployment.

## Navigation
- **Previous**: [Environment Configuration](environment_configuration.md)
- **Next**: [Google Cloud Setup](google_cloud_setup.md)
- **Related**: [Project Structure](project_structure.md) → [Testing Frameworks](../04_specialized/testing_frameworks.md)

---

## Overview

Local development environment that mirrors production infrastructure using Docker Compose. This ensures consistent behavior between local testing and cloud deployment while enabling rapid development cycles.

## Docker Compose Architecture

```
Local Development Stack:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (Streamlit)   │───▶│   (FastAPI)     │───▶│  (PostgreSQL)   │
│   Port: 8501    │    │   Port: 8000    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │   (Caching)     │
                       │   Port: 6379    │
                       └─────────────────┘
```

## Complete Docker Compose Configuration

### Root docker-compose.yml
```yaml
# docker-compose.yml - Complete local development environment
version: '3.8'

services:
  # Backend API and Agent Orchestration
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: aiagents-backend
    ports:
      - "${PROJECTNAME_BACKEND_PORT:-8000}:8000"
    environment:
      - PROJECTNAME_ENV=development
      - PROJECTNAME_DATABASE_HOST=database
      - PROJECTNAME_DATABASE_PORT=5432
      - PROJECTNAME_DATABASE_NAME=${PROJECTNAME_DATABASE_NAME:-aiagents_dev}
      - PROJECTNAME_DATABASE_USER=${PROJECTNAME_DATABASE_USER:-aiagents_user}
      - PROJECTNAME_DATABASE_PASSWORD=${PROJECTNAME_DATABASE_PASSWORD:-dev_password}
      - PROJECTNAME_REDIS_URL=redis://redis:6379/0
      - PROJECTNAME_OPENAI_API_KEY=${PROJECTNAME_OPENAI_API_KEY}
      - PROJECTNAME_ANTHROPIC_API_KEY=${PROJECTNAME_ANTHROPIC_API_KEY}
      - PROJECTNAME_LOG_LEVEL=DEBUG
    env_file:
      - .env.local
    volumes:
      - ./backend:/app
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - database
      - redis
    networks:
      - aiagents-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Frontend Web Interface
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: aiagents-frontend
    ports:
      - "${PROJECTNAME_FRONTEND_PORT:-8501}:8501"
    environment:
      - PROJECTNAME_BACKEND_HOST=backend
      - PROJECTNAME_BACKEND_PORT=8000
      - PROJECTNAME_ENV=development
    env_file:
      - .env.local
    volumes:
      - ./frontend:/app
    depends_on:
      - backend
    networks:
      - aiagents-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Database
  database:
    image: postgres:15-alpine
    container_name: aiagents-database
    ports:
      - "${PROJECTNAME_DATABASE_PORT:-5432}:5432"
    environment:
      - POSTGRES_DB=${PROJECTNAME_DATABASE_NAME:-aiagents_dev}
      - POSTGRES_USER=${PROJECTNAME_DATABASE_USER:-aiagents_user}
      - POSTGRES_PASSWORD=${PROJECTNAME_DATABASE_PASSWORD:-dev_password}
      - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
      - ./database/schemas:/schemas
    networks:
      - aiagents-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${PROJECTNAME_DATABASE_USER:-aiagents_user} -d ${PROJECTNAME_DATABASE_NAME:-aiagents_dev}"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis for Caching and Session Management
  redis:
    image: redis:7-alpine
    container_name: aiagents-redis
    ports:
      - "${PROJECTNAME_REDIS_PORT:-6379}:6379"
    environment:
      - REDIS_PASSWORD=${PROJECTNAME_REDIS_PASSWORD:-}
    volumes:
      - redis_data:/data
      - ./infrastructure/redis.conf:/usr/local/etc/redis/redis.conf
    networks:
      - aiagents-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    command: redis-server /usr/local/etc/redis/redis.conf

  # Database Administration (Optional)
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: aiagents-pgadmin
    ports:
      - "8080:80"
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@aiagents.local
      - PGADMIN_DEFAULT_PASSWORD=admin
      - PGADMIN_CONFIG_SERVER_MODE=False
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    depends_on:
      - database
    networks:
      - aiagents-network
    restart: unless-stopped
    profiles:
      - admin  # Only start with --profile admin

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  pgadmin_data:
    driver: local

networks:
  aiagents-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Backend Dockerfile

### Backend Container Configuration
```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Backend Requirements
```txt
# backend/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.1
openai==1.3.7
anthropic==0.7.7
structlog==23.2.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
aiofiles==23.2.1
httpx==0.25.2
python-dotenv==1.0.0
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
```

## Frontend Dockerfile

### Frontend Container Configuration
```dockerfile
# frontend/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Frontend Requirements
```txt
# frontend/requirements.txt
streamlit==1.28.1
requests==2.31.0
pandas==2.1.4
plotly==5.17.0
altair==5.2.0
python-dotenv==1.0.0
httpx==0.25.2
streamlit-authenticator==0.2.3
streamlit-option-menu==0.3.6
```

## Development Scripts

### Quick Start Script
```bash
#!/bin/bash
# scripts/start-local.sh - Quick local development startup

set -e

echo "🚀 Starting AI Agents Local Development Environment"

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "📝 Creating .env.local from template..."
    if [ -f ".env.template" ]; then
        cp .env.template .env.local
        echo "⚠️  Please edit .env.local with your API keys and settings"
    else
        echo "❌ .env.template not found. Please create it first."
        exit 1
    fi
fi

# Check for required environment variables
source .env.local
if [ -z "$PROJECTNAME_OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: PROJECTNAME_OPENAI_API_KEY not set in .env.local"
fi

# Create necessary directories
mkdir -p data logs

# Start services
echo "🐳 Starting Docker Compose services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Run database migrations if needed
echo "📦 Running database setup..."
docker-compose exec backend python -c "
from database.setup import setup_database
setup_database()
print('Database setup completed')
"

echo "✅ Local development environment is ready!"
echo ""
echo "🌐 Services available at:"
echo "   Frontend (Streamlit): http://localhost:8501"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Database: localhost:5432"
echo "   PgAdmin: http://localhost:8080 (start with --profile admin)"
echo ""
echo "📊 To view logs: docker-compose logs -f [service_name]"
echo "🛑 To stop: docker-compose down"
```

### Testing Script
```bash
#!/bin/bash
# scripts/run-tests.sh - Comprehensive local testing

set -e

echo "🧪 Running AI Agents Test Suite"

# Start test database
echo "🐳 Starting test services..."
docker-compose -f docker-compose.test.yml up -d

# Wait for services
sleep 5

# Run tests in container
echo "🔬 Running unit tests..."
docker-compose exec backend python -m pytest tests/unit/ -v

echo "🔗 Running integration tests..."
docker-compose exec backend python -m pytest tests/integration/ -v

echo "🌐 Running end-to-end tests..."
docker-compose exec backend python -m pytest tests/e2e/ -v

# Run frontend tests if they exist
if [ -d "frontend/tests" ]; then
    echo "🎨 Running frontend tests..."
    docker-compose exec frontend python -m pytest tests/ -v
fi

# Clean up test services
docker-compose -f docker-compose.test.yml down

echo "✅ All tests completed!"
```

### Cleanup Script
```bash
#!/bin/bash
# scripts/cleanup.sh - Clean up development environment

set -e

echo "🧹 Cleaning up AI Agents Development Environment"

# Stop and remove containers
echo "🛑 Stopping containers..."
docker-compose down

# Remove volumes (optional)
read -p "🗑️  Remove persistent data volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing volumes..."
    docker-compose down -v
    docker volume prune -f
fi

# Remove unused images
read -p "🖼️  Remove unused Docker images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🖼️  Removing unused images..."
    docker image prune -f
fi

# Clean up logs
if [ -d "logs" ]; then
    echo "📝 Cleaning up logs..."
    rm -rf logs/*
fi

echo "✅ Cleanup completed!"
```

## Development Workflow

### Daily Development Process
1. **Start Environment**: `./scripts/start-local.sh`
2. **Develop**: Edit code with hot-reload enabled
3. **Test Changes**: `./scripts/run-tests.sh`
4. **View Logs**: `docker-compose logs -f backend`
5. **Stop Environment**: `docker-compose down`

### Database Management
```bash
# Connect to database
docker-compose exec database psql -U aiagents_user -d aiagents_dev

# Run migrations
docker-compose exec backend alembic upgrade head

# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "description"

# Reset database
docker-compose down
docker volume rm aiagents_postgres_data
docker-compose up -d database
```

### Debugging and Troubleshooting
```bash
# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend

# Execute commands in containers
docker-compose exec backend bash
docker-compose exec frontend bash

# Check container health
docker-compose ps

# Restart specific service
docker-compose restart backend

# Rebuild and restart
docker-compose up -d --build backend
```

## Environment Validation

### Health Check Endpoints
```python
# backend/api/v1/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.connection import get_db
from config.settings import settings
import redis

router = APIRouter()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check for local development"""
    
    health_status = {
        "status": "healthy",
        "environment": settings.env,
        "services": {}
    }
    
    # Check database connection
    try:
        db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Redis connection
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check OpenAI API key
    if settings.openai_api_key:
        health_status["services"]["openai"] = "configured"
    else:
        health_status["services"]["openai"] = "not_configured"
    
    return health_status
```

## Integration with Framework Levels

### Level 1: Simple Agent Development
```yaml
# docker-compose.simple.yml - Minimal setup for Level 1
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PROJECTNAME_ENV=development
      - PROJECTNAME_OPENAI_API_KEY=${PROJECTNAME_OPENAI_API_KEY}
    volumes:
      - ./backend:/app
```

### Level 2: Standard Multi-Agent Development
```yaml
# docker-compose.standard.yml - Standard setup for Level 2
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - database
  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
  database:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${PROJECTNAME_DATABASE_NAME}
      POSTGRES_USER: ${PROJECTNAME_DATABASE_USER}
      POSTGRES_PASSWORD: ${PROJECTNAME_DATABASE_PASSWORD}
```

## Performance Optimization

### Development Performance Tips
1. **Use .dockerignore** to exclude unnecessary files
2. **Mount volumes** for hot-reload during development
3. **Use Alpine images** where possible for smaller containers
4. **Enable BuildKit** for faster Docker builds
5. **Cache pip dependencies** in separate layer

### Resource Management
```yaml
# Resource limits for development
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
```

---

## Next Steps

- **Cloud Infrastructure**: [Google Cloud Setup](google_cloud_setup.md)
- **Production Deployment**: [CI/CD Pipeline](cicd_pipeline.md)
- **Monitoring**: [Observability Setup](observability.md)

---

*This local development environment provides a production-like experience that ensures code tested locally will work reliably when deployed to Google Cloud.*