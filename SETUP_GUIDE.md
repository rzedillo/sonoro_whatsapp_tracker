# 🚀 WhatsApp Task Tracker - Setup Guide

## ✅ Quick Start

### 1. **Environment Setup**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Required Environment Variables:**
```bash
# Essential - Add your actual API key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# WhatsApp Configuration
WHATSAPP_MONITORED_GROUPS=["Group Name 1", "Group Name 2"]
WHATSAPP_SESSION_PATH=./data/whatsapp_session
WHATSAPP_HEADLESS=false  # Set to true for production

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/whatsapp_tracker

# Redis
REDIS_URL=redis://localhost:6379/0
```

### 2. **Dependencies Installation**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (for Chrome/Selenium)
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y chromium-browser chromium-chromedriver

# On macOS:
brew install --cask google-chrome
```

### 3. **Database Setup**
```bash
# Start services
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head
```

### 4. **System Validation**
```bash
# Run integration tests
python test_integration.py

# Expected output: "🎉 ALL TESTS PASSED! System is ready for deployment."
```

### 5. **Start the System**
```bash
# Option A: Full stack with Docker
docker-compose up

# Option B: Individual components
python main.py                    # Main orchestrator
python -m uvicorn api.main:app    # API server (separate terminal)
streamlit run web/app.py          # Web interface (separate terminal)
```

## 📱 WhatsApp Setup

### Authentication Process:
1. **First Run**: System will display QR code
2. **Scan QR Code**: Use WhatsApp on your phone
3. **Session Saved**: Subsequent runs will auto-connect

### Available Commands:
- `!help` - Show all available commands
- `!tasks` - List current tasks
- `!create [description]` - Create new task
- `!complete [id]` - Mark task as completed
- `!assign [id] [user]` - Assign task to user
- `!status` - System status
- `!stats` - Task statistics

## 🔧 Production Deployment

### Docker Production:
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d
```

### Google Cloud Deployment:
```bash
# Deploy using included CI/CD pipeline
gcloud builds submit --config cloudbuild.yaml
```

## 🧪 Testing

### Component Testing:
```bash
# Run integration tests
python test_integration.py

# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=.
```

### Manual Testing:
1. **WhatsApp Commands**: Send `!help` in monitored group
2. **Web Interface**: Visit `http://localhost:8501`
3. **API**: Visit `http://localhost:8000/docs`

## 🔍 Troubleshooting

### Common Issues:

**❌ "No module named 'undetected_chromedriver'"**
```bash
pip install undetected-chromedriver==3.5.4
```

**❌ WhatsApp Authentication Failed**
- Ensure Chrome/Chromium is installed
- Check `WHATSAPP_HEADLESS=false` for first setup
- Verify session path permissions

**❌ Database Connection Error**
```bash
# Check if PostgreSQL is running
docker-compose up -d postgres

# Verify connection
psql $DATABASE_URL -c "SELECT 1;"
```

**❌ Redis Connection Error**
```bash
# Check if Redis is running
docker-compose up -d redis

# Test connection
redis-cli -u $REDIS_URL ping
```

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WhatsApp Web  │────│   Orchestrator  │────│   Notification  │
│     Agent       │    │                 │    │     Agent       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Command        │    │  Message        │    │  Task           │
│  Handler        │    │  Analysis       │    │  Management     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │     Redis       │
                    └─────────────────┘
```

## 🎯 Feature Comparison

| Feature | V1 (Node.js) | V2 (Python) | Status |
|---------|--------------|-------------|--------|
| WhatsApp Commands | ✅ | ✅ | **IMPROVED** |
| Message Sending | ✅ | ✅ | **IMPROVED** |
| Task Management | ✅ | ✅ | **ENHANCED** |
| Multi-channel Notifications | ❌ | ✅ | **NEW** |
| Web Dashboard | ❌ | ✅ | **NEW** |
| REST API | ❌ | ✅ | **NEW** |
| Docker Deployment | ❌ | ✅ | **NEW** |
| Cloud Pipeline | ❌ | ✅ | **NEW** |
| Testing Framework | ❌ | ✅ | **NEW** |

## 🔐 Security Notes

- Never commit `.env` file to version control
- Use strong database passwords in production
- Configure firewall rules for production deployment
- Regularly update dependencies for security patches

## 📞 Support

For issues or questions:
1. Check this setup guide
2. Review the troubleshooting section
3. Run integration tests to identify specific issues
4. Check logs in `./logs/` directory

---

🎉 **System successfully refactored from V1 with 100% feature parity and significant enhancements!**