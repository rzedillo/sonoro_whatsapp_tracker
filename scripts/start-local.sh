#!/bin/bash
# scripts/start-local.sh - Quick local development startup

set -e

echo "🚀 Starting WhatsApp Tracker Local Development Environment"

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
if [ -z "$WHATSAPP_TRACKER_ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: WHATSAPP_TRACKER_ANTHROPIC_API_KEY not set in .env.local"
fi

# Create necessary directories
mkdir -p data logs test-results

# Start services
echo "🐳 Starting Docker Compose services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Run database migrations if needed
echo "📦 Running database setup..."
docker-compose exec backend python -c "
from database.setup import setup_database
setup_database()
print('Database setup completed')
" || echo "⚠️  Database setup will run on first API call"

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