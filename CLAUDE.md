# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the WhatsApp Task Tracker codebase.

## 🚀 Quick Start Commands

- `./scripts/start-local.sh` - Start local development environment
- `./scripts/run-tests.sh` - Run comprehensive test suite  
- `docker-compose up -d` - Start all services
- `docker-compose down` - Stop all services

## 🏗️ Architecture Overview

This is a **multi-agent WhatsApp task management system** built with Enhanced Framework V3.1, featuring:

### Core Components
1. **Multi-Agent Orchestrator** (`core/orchestrator.py`) - Coordinates all agents
2. **Agent System** (`agents/`) - Specialized agents for different functions
3. **FastAPI Backend** (`main.py`, `api/`) - REST API with comprehensive endpoints
4. **Streamlit Frontend** (`web/main.py`) - Interactive dashboard
5. **PostgreSQL + Redis** - Database and caching layer

### Key Architecture Patterns

**Agent-Based Processing:**
1. WhatsApp message received → WhatsApp Agent → Message Analysis Agent
2. Task detected → Task Management Agent → Notification Agent
3. All coordinated through central Orchestrator with Redis caching

**Agent Classes:**
- `BaseAgent` - Core agent functionality with metrics, error handling, health checks
- `WhatsAppAgent` - Selenium-based WhatsApp Web integration
- `MessageAnalysisAgent` - Claude AI-powered task detection  
- `TaskManagementAgent` - Task CRUD, analytics, user patterns
- `NotificationAgent` - Multi-channel notifications and reminders

**Database Models:**
- `Task` - Main tasks with full lifecycle tracking
- `Conversation` - WhatsApp message history
- `TaskHistory` - Change tracking with audit trail
- `UserPattern` - Productivity analytics
- `WhatsAppSession` - Session management
- `AgentMetrics` - Performance monitoring

## 🔧 Development Workflow

### Local Development
1. **Environment Setup**: Copy `.env.template` to `.env.local` and configure
2. **Start Services**: `./scripts/start-local.sh`
3. **Access Points**:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - PgAdmin: http://localhost:8080 (with `--profile admin`)

### Testing (MANDATORY before deployment)
```bash
./scripts/run-tests.sh  # Must pass before any cloud deployment
```

**Test Types:**
- Unit tests (`tests/unit/`) - Individual component testing
- Integration tests (`tests/integration/`) - Service interaction testing  
- E2E tests (`tests/e2e/`) - Full API workflow testing
- Performance tests (`tests/performance/`) - Load and stress testing

### Cloud Deployment
**Framework Principle: Test Locally, Deploy to Cloud**

1. **Local Testing Required**: CI/CD pipeline blocks without local test evidence
2. **Environment Progression**: Dev → Staging → Production
3. **Automatic Rollback**: Health checks trigger rollback on failure
4. **Multi-Environment**: Different configurations per environment

## 📊 Key Features

### WhatsApp Integration
- QR code authentication with session persistence
- Real-time message monitoring from specified groups
- Selenium-based Web WhatsApp automation
- Message history tracking and context analysis

### AI-Powered Task Detection
- Claude AI integration for natural language task extraction
- Context-aware analysis using user patterns and recent tasks
- Confidence scoring and duplicate detection
- Support for priority detection and auto-assignment

### Advanced Task Management
- Complete CRUD operations with change history
- User productivity analytics and pattern recognition
- Due date reminders and escalation notifications
- Group-based task organization and filtering

### Multi-Channel Notifications
- Console, cache, webhook, and email notification support
- Smart reminder system based on due dates
- Real-time notification feed for web interface
- Configurable notification priorities and channels

## 🗄️ Database Schema

**Tasks Table (PostgreSQL):**
```sql
- id, descripcion, responsable, fecha_limite
- prioridad, estado, mensaje_original, autor_mensaje  
- timestamp, grupo_id, grupo_nombre, mensaje_id
- confidence_score, analysis_metadata, completion_date
```

**Key Relationships:**
- Tasks ↔ TaskHistory (1:many) - Change tracking
- Tasks ↔ UserPatterns (many:many) - Analytics
- Conversations → Groups (many:1) - Message organization

## ⚙️ Configuration

**Environment Variables:**
- `WHATSAPP_TRACKER_ENV` - Environment (development/test/staging/production)
- `WHATSAPP_TRACKER_DATABASE_URL` - PostgreSQL connection
- `WHATSAPP_TRACKER_REDIS_URL` - Redis connection
- `WHATSAPP_TRACKER_ANTHROPIC_API_KEY` - Claude AI integration

**Agent Configuration:**
- WhatsApp: Session path, QR timeout, headless mode
- Task Manager: Auto-assign, priority detection, duplicate threshold
- Message Analyzer: Confidence threshold, analysis timeout
- Notifier: Channels, reminder hours, queue size

## 🔍 Monitoring & Debugging

**Agent Health Checks:**
- `/api/v1/agents/` - All agent status
- `/api/v1/agents/{name}/health` - Individual agent health
- `/api/v1/agents/{name}/metrics` - Performance metrics

**Application Monitoring:**
- Structured logging with contextual information
- Redis-based real-time metrics and activity logs
- Database performance metrics and health checks
- Automatic error recovery and circuit breaker patterns

**Local Debugging:**
```bash
docker-compose logs -f backend    # Backend logs
docker-compose logs -f frontend   # Frontend logs  
docker-compose exec backend bash  # Backend shell
docker-compose exec frontend bash # Frontend shell
```

## 🚀 Production Deployment

**Google Cloud Architecture:**
- Cloud Run services for backend and frontend
- Cloud SQL (PostgreSQL) for production database  
- Cloud Memory Store (Redis) for caching
- Artifact Registry for container images
- Cloud Build for CI/CD pipeline

**Security & Best Practices:**
- Environment-specific configurations
- Secret management through Cloud Secret Manager
- Health checks and automatic scaling
- Comprehensive logging and monitoring
- Backup and disaster recovery procedures

The system follows the Enhanced Framework V3.1 principles with test-first deployment, multi-agent orchestration, and production-ready cloud infrastructure.