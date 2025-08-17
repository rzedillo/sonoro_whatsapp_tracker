# Environment Configuration for Cloud Deployment

> ⚙️ **Standardized Configuration**: Project-prefixed environment variables for consistent local and cloud deployment.

## Navigation
- **Previous**: [Project Structure](project_structure.md)
- **Next**: [Local Development](local_development.md)
- **Related**: [Google Cloud Setup](google_cloud_setup.md) → [Security Guidelines](../06_reference/security_guidelines.md)

---

## Overview

Consistent environment configuration is crucial for seamless deployment across local development, staging, and production environments. This guide establishes standardized patterns for environment variables and configuration management.

## Environment Variable Naming Convention

### Project Prefix Pattern
All environment variables should start with your project name in uppercase:

```bash
# For project "MyAIProject"
MYAIPROJECT_ENV=development
MYAIPROJECT_API_KEY=your-openai-key
MYAIPROJECT_DATABASE_URL=postgresql://...

# For project "SonoroRevenue" 
SONOROREVENUE_ENV=production
SONOROREVENUE_OPENAI_API_KEY=sk-...
SONOROREVENUE_DATABASE_HOST=localhost
```

### Configuration Categories

**Core Application Settings**
```bash
PROJECTNAME_ENV=development                    # Environment: development, staging, production
PROJECTNAME_DEBUG=false                       # Debug mode toggle
PROJECTNAME_LOG_LEVEL=INFO                   # Logging level
PROJECTNAME_SECRET_KEY=your-secret-key-here  # Application secret key
```

**Database Configuration**
```bash
PROJECTNAME_DATABASE_URL=postgresql://user:pass@host:5432/dbname
PROJECTNAME_DATABASE_HOST=localhost
PROJECTNAME_DATABASE_PORT=5432
PROJECTNAME_DATABASE_NAME=your_project_db
PROJECTNAME_DATABASE_USER=your_user
PROJECTNAME_DATABASE_PASSWORD=secure_password
PROJECTNAME_DATABASE_POOL_SIZE=10
```

**LLM API Configuration**
```bash
PROJECTNAME_OPENAI_API_KEY=sk-...            # OpenAI API key
PROJECTNAME_OPENAI_MODEL=gpt-4o-mini         # Default model
PROJECTNAME_OPENAI_TIMEOUT=30                # Request timeout
PROJECTNAME_OPENAI_MAX_RETRIES=3             # Retry attempts
PROJECTNAME_ANTHROPIC_API_KEY=sk-ant-...     # Claude API key (optional)
PROJECTNAME_ANTHROPIC_MODEL=claude-3-sonnet  # Default Claude model
```

**Web Interface Configuration**
```bash
PROJECTNAME_FRONTEND_HOST=0.0.0.0           # Frontend host binding
PROJECTNAME_FRONTEND_PORT=8501               # Streamlit port
PROJECTNAME_BACKEND_HOST=0.0.0.0            # Backend API host
PROJECTNAME_BACKEND_PORT=8000                # FastAPI port
PROJECTNAME_CORS_ORIGINS=http://localhost:8501,https://yourdomain.com
```

**Google Cloud Configuration**
```bash
PROJECTNAME_GCP_PROJECT_ID=your-gcp-project-id
PROJECTNAME_GCP_REGION=us-central1
PROJECTNAME_GCP_ZONE=us-central1-a
PROJECTNAME_GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
PROJECTNAME_CLOUD_SQL_CONNECTION_NAME=project:region:instance
```

**Redis/Caching Configuration**
```bash
PROJECTNAME_REDIS_URL=redis://localhost:6379/0
PROJECTNAME_REDIS_HOST=localhost
PROJECTNAME_REDIS_PORT=6379
PROJECTNAME_REDIS_PASSWORD=redis_password
PROJECTNAME_CACHE_TTL=3600                   # Cache time-to-live in seconds
```

**Security and Authentication**
```bash
PROJECTNAME_JWT_SECRET=your-jwt-secret       # JWT signing secret
PROJECTNAME_JWT_ALGORITHM=HS256              # JWT algorithm
PROJECTNAME_JWT_EXPIRATION=86400             # JWT expiration (24 hours)
PROJECTNAME_ENCRYPTION_KEY=32-char-key       # Data encryption key
PROJECTNAME_RATE_LIMIT_REQUESTS=100          # Requests per window
PROJECTNAME_RATE_LIMIT_WINDOW=3600           # Rate limit window (1 hour)
```

---

## Environment Files Structure

### Template Files (.env.template)
```bash
# .env.template - Copy this to .env.local and fill in your values

# ============================================================================
# PROJECT_NAME Environment Configuration
# Copy this file to .env.local and update with your actual values
# NEVER commit .env files with real credentials to version control
# ============================================================================

# Environment Settings
PROJECTNAME_ENV=development
PROJECTNAME_DEBUG=true
PROJECTNAME_LOG_LEVEL=INFO
PROJECTNAME_SECRET_KEY=change-this-in-production

# Database Configuration
PROJECTNAME_DATABASE_URL=postgresql://localhost:5432/projectname_dev
PROJECTNAME_DATABASE_HOST=localhost
PROJECTNAME_DATABASE_PORT=5432
PROJECTNAME_DATABASE_NAME=projectname_dev
PROJECTNAME_DATABASE_USER=projectname_user
PROJECTNAME_DATABASE_PASSWORD=change-this-password

# LLM API Keys
PROJECTNAME_OPENAI_API_KEY=your-openai-api-key-here
PROJECTNAME_OPENAI_MODEL=gpt-4o-mini
PROJECTNAME_OPENAI_TIMEOUT=30
PROJECTNAME_ANTHROPIC_API_KEY=your-claude-api-key-here

# Web Interface
PROJECTNAME_FRONTEND_HOST=localhost
PROJECTNAME_FRONTEND_PORT=8501
PROJECTNAME_BACKEND_HOST=localhost
PROJECTNAME_BACKEND_PORT=8000
PROJECTNAME_CORS_ORIGINS=http://localhost:8501

# Google Cloud (for production deployment)
PROJECTNAME_GCP_PROJECT_ID=your-gcp-project-id
PROJECTNAME_GCP_REGION=us-central1
PROJECTNAME_GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json

# Redis/Caching
PROJECTNAME_REDIS_URL=redis://localhost:6379/0
PROJECTNAME_CACHE_TTL=3600

# Security
PROJECTNAME_JWT_SECRET=your-jwt-secret-here
PROJECTNAME_ENCRYPTION_KEY=your-32-character-encryption-key
PROJECTNAME_RATE_LIMIT_REQUESTS=100
PROJECTNAME_RATE_LIMIT_WINDOW=3600
```

### Environment-Specific Files

**Local Development (.env.local)**
```bash
PROJECTNAME_ENV=development
PROJECTNAME_DEBUG=true
PROJECTNAME_LOG_LEVEL=DEBUG
PROJECTNAME_DATABASE_HOST=localhost
PROJECTNAME_FRONTEND_HOST=localhost
PROJECTNAME_BACKEND_HOST=localhost
```

**Staging (.env.staging)**
```bash
PROJECTNAME_ENV=staging
PROJECTNAME_DEBUG=false
PROJECTNAME_LOG_LEVEL=INFO
PROJECTNAME_DATABASE_HOST=staging-db-host
PROJECTNAME_FRONTEND_HOST=0.0.0.0
PROJECTNAME_BACKEND_HOST=0.0.0.0
```

**Production (.env.production)**
```bash
PROJECTNAME_ENV=production
PROJECTNAME_DEBUG=false
PROJECTNAME_LOG_LEVEL=WARNING
PROJECTNAME_DATABASE_HOST=production-db-host
PROJECTNAME_FRONTEND_HOST=0.0.0.0
PROJECTNAME_BACKEND_HOST=0.0.0.0
PROJECTNAME_RATE_LIMIT_REQUESTS=50  # Stricter rate limiting
```

---

## Configuration Management in Code

### Python Configuration Class
```python
# backend/config/settings.py
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable loading"""
    
    # Environment
    env: str = os.getenv("PROJECTNAME_ENV", "development")
    debug: bool = os.getenv("PROJECTNAME_DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("PROJECTNAME_LOG_LEVEL", "INFO")
    secret_key: str = os.getenv("PROJECTNAME_SECRET_KEY", "dev-secret-key")
    
    # Database
    database_url: str = os.getenv("PROJECTNAME_DATABASE_URL", "postgresql://localhost:5432/projectname_dev")
    database_host: str = os.getenv("PROJECTNAME_DATABASE_HOST", "localhost")
    database_port: int = int(os.getenv("PROJECTNAME_DATABASE_PORT", "5432"))
    database_name: str = os.getenv("PROJECTNAME_DATABASE_NAME", "projectname_dev")
    database_user: str = os.getenv("PROJECTNAME_DATABASE_USER", "projectname_user")
    database_password: str = os.getenv("PROJECTNAME_DATABASE_PASSWORD", "")
    
    # LLM APIs
    openai_api_key: str = os.getenv("PROJECTNAME_OPENAI_API_KEY", "")
    openai_model: str = os.getenv("PROJECTNAME_OPENAI_MODEL", "gpt-4o-mini")
    openai_timeout: int = int(os.getenv("PROJECTNAME_OPENAI_TIMEOUT", "30"))
    anthropic_api_key: Optional[str] = os.getenv("PROJECTNAME_ANTHROPIC_API_KEY")
    
    # Web Interface
    frontend_host: str = os.getenv("PROJECTNAME_FRONTEND_HOST", "localhost")
    frontend_port: int = int(os.getenv("PROJECTNAME_FRONTEND_PORT", "8501"))
    backend_host: str = os.getenv("PROJECTNAME_BACKEND_HOST", "localhost")
    backend_port: int = int(os.getenv("PROJECTNAME_BACKEND_PORT", "8000"))
    cors_origins: list = os.getenv("PROJECTNAME_CORS_ORIGINS", "").split(",")
    
    # Google Cloud
    gcp_project_id: Optional[str] = os.getenv("PROJECTNAME_GCP_PROJECT_ID")
    gcp_region: str = os.getenv("PROJECTNAME_GCP_REGION", "us-central1")
    google_application_credentials: Optional[str] = os.getenv("PROJECTNAME_GOOGLE_APPLICATION_CREDENTIALS")
    
    # Redis
    redis_url: str = os.getenv("PROJECTNAME_REDIS_URL", "redis://localhost:6379/0")
    cache_ttl: int = int(os.getenv("PROJECTNAME_CACHE_TTL", "3600"))
    
    # Security
    jwt_secret: str = os.getenv("PROJECTNAME_JWT_SECRET", "dev-jwt-secret")
    jwt_algorithm: str = os.getenv("PROJECTNAME_JWT_ALGORITHM", "HS256")
    rate_limit_requests: int = int(os.getenv("PROJECTNAME_RATE_LIMIT_REQUESTS", "100"))
    
    @property
    def is_development(self) -> bool:
        return self.env == "development"
    
    @property
    def is_production(self) -> bool:
        return self.env == "production"
    
    def validate_required_settings(self):
        """Validate that required settings are present"""
        required_for_production = [
            "openai_api_key",
            "secret_key",
            "database_password",
            "jwt_secret"
        ]
        
        if self.is_production:
            missing = []
            for setting in required_for_production:
                if not getattr(self, setting):
                    missing.append(f"PROJECTNAME_{setting.upper()}")
            
            if missing:
                raise ValueError(f"Missing required production settings: {', '.join(missing)}")

# Global settings instance
settings = Settings()
```

### Usage in Application
```python
# backend/main.py
from config.settings import settings
from fastapi import FastAPI

# Validate settings on startup
settings.validate_required_settings()

app = FastAPI(
    title=f"Project Name - {settings.env.title()}",
    debug=settings.debug
)

# Use settings throughout the application
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.env,
        "database_host": settings.database_host
    }
```

---

## Environment Loading Strategies

### Docker Compose Environment
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    environment:
      - PROJECTNAME_ENV=development
      - PROJECTNAME_DATABASE_HOST=database
      - PROJECTNAME_REDIS_URL=redis://redis:6379/0
    env_file:
      - .env.local
    depends_on:
      - database
      - redis

  frontend:
    build: ./frontend
    environment:
      - PROJECTNAME_BACKEND_HOST=backend
      - PROJECTNAME_BACKEND_PORT=8000
    env_file:
      - .env.local
    depends_on:
      - backend

  database:
    image: postgres:15
    environment:
      POSTGRES_DB: ${PROJECTNAME_DATABASE_NAME}
      POSTGRES_USER: ${PROJECTNAME_DATABASE_USER}
      POSTGRES_PASSWORD: ${PROJECTNAME_DATABASE_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Google Cloud Secret Manager Integration
```python
# backend/config/secrets.py
from google.cloud import secretmanager
from config.settings import settings

class SecretManager:
    """Google Cloud Secret Manager integration"""
    
    def __init__(self):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = settings.gcp_project_id
    
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Google Cloud Secret Manager"""
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID not configured")
        
        name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    
    def load_secrets_to_env(self):
        """Load secrets from Cloud Secret Manager to environment"""
        secret_mappings = {
            "openai-api-key": "PROJECTNAME_OPENAI_API_KEY",
            "database-password": "PROJECTNAME_DATABASE_PASSWORD",
            "jwt-secret": "PROJECTNAME_JWT_SECRET",
        }
        
        for secret_name, env_var in secret_mappings.items():
            try:
                secret_value = self.get_secret(secret_name)
                os.environ[env_var] = secret_value
            except Exception as e:
                print(f"Warning: Could not load secret {secret_name}: {e}")

# Usage in production
if settings.is_production and settings.gcp_project_id:
    secret_manager = SecretManager()
    secret_manager.load_secrets_to_env()
```

---

## Security Best Practices

### Environment File Security
1. **Never commit real credentials** to version control
2. **Use .env.template** files with placeholder values
3. **Include .env files** in .gitignore
4. **Validate required variables** on application startup
5. **Use different secrets** for each environment

### Secret Management
1. **Local Development**: Use .env.local files (gitignored)
2. **Staging/Production**: Use Google Cloud Secret Manager
3. **Container Deployment**: Mount secrets as volumes or environment variables
4. **Rotation**: Regularly rotate API keys and passwords

### Access Control
1. **Principle of Least Privilege**: Grant minimal required permissions
2. **Environment Separation**: Use different credentials for dev/staging/prod
3. **Monitoring**: Log access to sensitive configuration
4. **Encryption**: Encrypt sensitive data at rest and in transit

---

## Next Steps

- **Local Development**: [Docker Compose Setup](local_development.md)
- **Cloud Deployment**: [Google Cloud Infrastructure](google_cloud_setup.md)
- **Security**: [Security Guidelines](../06_reference/security_guidelines.md)

---

*Consistent environment configuration ensures smooth transitions between development, staging, and production while maintaining security and flexibility.*