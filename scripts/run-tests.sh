#!/bin/bash
# scripts/run-tests.sh - Comprehensive local testing

set -e

echo "🧪 Running WhatsApp Task Tracker Test Suite"

# Environment setup
export WHATSAPP_TRACKER_ENV=test
export WHATSAPP_TRACKER_LOG_LEVEL=DEBUG

# Create test results directory
mkdir -p test-results

# Start test environment
echo "🐳 Starting test environment..."
docker-compose -f docker-compose.test.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check test services
echo "🔍 Health checking test services..."
docker-compose -f docker-compose.test.yml exec -T backend curl -f http://localhost:8000/health || {
    echo "❌ Backend health check failed"
    docker-compose -f docker-compose.test.yml logs backend
    exit 1
}

# Run unit tests
echo "🔬 Running unit tests..."
docker-compose -f docker-compose.test.yml exec -T backend python -m pytest tests/unit/ \
    -v \
    --junitxml=/app/test-results/unit-tests.xml \
    --cov=core \
    --cov=agents \
    --cov=database \
    --cov-report=xml:/app/test-results/coverage.xml

# Run integration tests
echo "🔗 Running integration tests..."
docker-compose -f docker-compose.test.yml exec -T backend python -m pytest tests/integration/ \
    -v \
    --junitxml=/app/test-results/integration-tests.xml

# Run end-to-end tests
echo "🌐 Running end-to-end tests..."
docker-compose -f docker-compose.test.yml exec -T backend python -m pytest tests/e2e/ \
    -v \
    --junitxml=/app/test-results/e2e-tests.xml

# Run performance tests
echo "⚡ Running performance tests..."
if [ -d "tests/performance" ]; then
    docker-compose -f docker-compose.test.yml exec -T backend python -m pytest tests/performance/ \
        -v \
        --junitxml=/app/test-results/performance-tests.xml
fi

# Cleanup test environment
echo "🧹 Cleaning up test environment..."
docker-compose -f docker-compose.test.yml down

# Generate test summary
echo "📊 Generating test summary..."
python scripts/generate-test-summary.py test-results/ || echo "Test summary generation skipped"

# Create cloud deployment evidence
echo "✅ Creating cloud deployment evidence..."
cat > test-results.json << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "tests_run": true,
    "local_validation": "passed",
    "test_types": ["unit", "integration", "e2e"],
    "cloud_ready": true,
    "commit_hash": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
}
EOF

echo "✅ All tests completed successfully!"
echo "📤 Ready for cloud deployment."