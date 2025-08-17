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

- Docker and Docker Compose
- Python 3.11+
- Anthropic API key

### Local Development

1. **Clone and Setup**
   ```bash
   cd whatsapp_tracker
   cp .env.template .env.local
   # Edit .env.local with your API keys
   ```

2. **Start Development Environment**
   ```bash
   ./scripts/start-local.sh
   ```

3. **Access Services**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

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