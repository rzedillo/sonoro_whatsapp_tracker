# WhatsApp Task Tracker

> 🤖 Multi-agent WhatsApp task management system with enhanced architecture V3.1

## Overview

WhatsApp Task Tracker is a sophisticated multi-agent system that monitors WhatsApp messages and manages tasks using advanced AI orchestration. Built with the enhanced framework V3.1, it provides intelligent task extraction, management, and notification capabilities.

## Features

- **WhatsApp Integration**: Seamless connection to WhatsApp Web
- **AI-Powered Task Management**: Intelligent task extraction from messages
- **Multi-Agent Architecture**: Specialized agents for different functionalities
- **Real-time Dashboard**: Streamlit web interface for monitoring and management
- **Local Development**: Docker Compose environment mirroring production
- **Cloud-Ready**: Google Cloud deployment with CI/CD pipeline

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for local testing)
- Anthropic API key (Claude AI)
- OpenAI API key (optional)

### Step-by-Step Setup

#### 1. **Environment Configuration**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your actual API keys
   # Required: ANTHROPIC_API_KEY=your_claude_api_key_here
   # Optional: OPENAI_API_KEY=your_openai_api_key_here
   # Configure: WHATSAPP_MONITORED_GROUPS=["your_group_1", "your_group_2"]
   ```

#### 2. **Start the System**
   ```bash
   # Make script executable (first time only)
   chmod +x scripts/start-local.sh
   
   # Start all services with Docker Compose
   ./scripts/start-local.sh
   
   # Alternative: Start manually
   docker-compose up -d
   ```

#### 3. **Verify System is Running**
   ```bash
   # Check all containers are running
   docker-compose ps
   
   # View logs to ensure no errors
   docker-compose logs -f
   ```

#### 4. **Access the Application**
   - **Main Dashboard**: http://localhost:8501 (Streamlit frontend)
   - **API Backend**: http://localhost:8000 (FastAPI)
   - **API Documentation**: http://localhost:8000/docs (Swagger UI)
   - **Database**: localhost:5432 (PostgreSQL)
   - **Cache**: localhost:6379 (Redis)

#### 5. **WhatsApp Setup**
   1. Open the dashboard at http://localhost:8501
   2. Navigate to WhatsApp section
   3. Scan the QR code with your WhatsApp mobile app
   4. Wait for connection confirmation
   5. Your WhatsApp groups will be monitored automatically

### Quick Test

```bash
# Run integration tests to verify everything works
python test_integration.py

# Test specific components
docker-compose exec backend python -m pytest tests/unit/ -v
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │───▶│     FastAPI     │───▶│   PostgreSQL    │
│   Frontend      │    │    Backend      │    │    Database     │
│   (Port 8501)   │    │   (Port 8000)   │    │   (Port 5432)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │   (Caching)     │
                       │   (Port 6379)   │
                       └─────────────────┘
```

## Development Workflow

1. **Test Locally First**: All changes must pass local tests
2. **Docker Environment**: Development environment mirrors production
3. **Multi-Agent Design**: Specialized agents for different functions
4. **CI/CD Pipeline**: Automated deployment after local validation

## Project Structure

```
whatsapp_tracker/
├── agents/              # Specialized agent implementations
├── core/               # Core framework and base classes
├── database/           # Database models and migrations
├── api/                # FastAPI endpoints and middleware
├── web/                # Streamlit frontend application
├── tests/              # Comprehensive test suite
├── scripts/            # Development and deployment scripts
├── infrastructure/     # Infrastructure configuration
└── docker-compose.yml  # Local development environment
```

## Testing

```bash
# Run comprehensive test suite
./scripts/run-tests.sh

# Run specific test types
docker-compose exec backend python -m pytest tests/unit/
docker-compose exec backend python -m pytest tests/integration/
docker-compose exec backend python -m pytest tests/e2e/
```

## Troubleshooting

### Common Issues

#### 1. **"Permission denied" when running scripts**
```bash
chmod +x scripts/start-local.sh
chmod +x scripts/run-tests.sh
```

#### 2. **"Port already in use" errors**
```bash
# Stop any existing containers
docker-compose down

# Check what's using the ports
lsof -i :8501 :8000 :5432 :6379

# Kill processes if needed, then restart
docker-compose up -d
```

#### 3. **"API key not found" errors**
- Ensure `.env` file exists and contains your API keys
- Check that ANTHROPIC_API_KEY is set correctly
- Restart containers after changing environment variables

#### 4. **WhatsApp connection issues**
```bash
# Check WhatsApp agent logs
docker-compose logs whatsapp_agent

# Ensure browser dependencies are installed
docker-compose exec backend pip install selenium undetected-chromedriver
```

#### 5. **Database connection errors**
```bash
# Reset database
docker-compose down -v
docker-compose up -d db
# Wait for database to initialize, then start other services
docker-compose up -d
```

### Getting Help

If you encounter issues:

1. **Check logs**: `docker-compose logs -f [service_name]`
2. **Run tests**: `python test_integration.py`
3. **Verify environment**: Ensure all required environment variables are set
4. **Check ports**: Make sure ports 8501, 8000, 5432, 6379 are available

## Deployment

The system follows the "test locally, deploy to cloud" philosophy:

1. **Local Testing**: Complete test suite must pass locally
2. **Cloud Pipeline**: Google Cloud Build automatically deploys tested code
3. **Multi-Environment**: Dev → Staging → Production progression

## Contributing

1. Ensure all local tests pass before committing
2. Follow the multi-agent architecture patterns
3. Update tests for new functionality
4. Maintain Docker environment compatibility

## License

MIT License - see LICENSE file for details