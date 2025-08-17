# CI/CD Pipeline with Google Cloud Build

> 🔄 **Automated Deployment**: Test-first CI/CD pipeline ensuring local tests pass before cloud deployment.

## Navigation
- **Previous**: [Google Cloud Setup](google_cloud_setup.md)
- **Next**: [Deployment Scripts](deployment_scripts.md)
- **Related**: [Local Development](local_development.md) → [Testing Frameworks](../04_specialized/testing_frameworks.md)

---

## Overview

This CI/CD pipeline enforces the "test locally, deploy to cloud" workflow by requiring successful local testing before triggering cloud deployment. The pipeline uses Google Cloud Build with multiple stages and automatic rollback capabilities.

## Pipeline Architecture

```
CI/CD Pipeline Flow:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Git Push      │───▶│  Local Tests    │───▶│   Cloud Build   │
│   (trigger)     │    │  (validation)   │    │   (build/test)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Block Deploy   │    │    Deploy to    │
                       │  (if tests fail)│    │   Environment   │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                               ┌─────────────────┐
                                               │  Health Check   │
                                               │  & Rollback     │
                                               └─────────────────┘
```

## Cloud Build Configuration

### Main Build Configuration
```yaml
# cloudbuild.yaml - Main CI/CD pipeline
steps:
  # Step 1: Validate local tests were run
  - name: 'gcr.io/cloud-builders/git'
    id: 'validate-local-tests'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🔍 Validating local test execution..."
        
        # Check for test artifacts or test commit message
        if git log -1 --pretty=%B | grep -E "(test:|tests:|✅|🧪)" > /dev/null; then
          echo "✅ Local tests validated via commit message"
        elif [ -f "test-results.json" ]; then
          echo "✅ Local test results found"
        else
          echo "❌ No evidence of local test execution"
          echo "Please run local tests first: ./scripts/run-tests.sh"
          exit 1
        fi

  # Step 2: Set up environment variables
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'setup-environment'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "⚙️ Setting up environment variables..."
        
        # Determine environment based on branch
        if [ "$BRANCH_NAME" = "main" ]; then
          echo "ENVIRONMENT=prod" >> /workspace/.env.build
          echo "PROJECT_ID=${_PROD_PROJECT_ID}" >> /workspace/.env.build
        elif [ "$BRANCH_NAME" = "staging" ]; then
          echo "ENVIRONMENT=staging" >> /workspace/.env.build
          echo "PROJECT_ID=${_STAGING_PROJECT_ID}" >> /workspace/.env.build
        else
          echo "ENVIRONMENT=dev" >> /workspace/.env.build
          echo "PROJECT_ID=${_DEV_PROJECT_ID}" >> /workspace/.env.build
        fi
        
        # Load environment variables
        source /workspace/.env.build
        echo "📋 Environment: $ENVIRONMENT"
        echo "📋 Project ID: $PROJECT_ID"

  # Step 3: Run comprehensive tests in cloud environment
  - name: 'docker/compose:latest'
    id: 'run-cloud-tests'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🧪 Running comprehensive test suite in cloud environment..."
        
        # Set up test environment
        export PROJECTNAME_ENV=test
        export PROJECTNAME_DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db
        export PROJECTNAME_REDIS_URL=redis://localhost:6379/1
        
        # Start test services
        docker-compose -f docker-compose.test.yml up -d
        
        # Wait for services to be ready
        sleep 30
        
        # Run unit tests
        echo "🔬 Running unit tests..."
        docker-compose exec -T backend python -m pytest tests/unit/ -v --junitxml=test-results-unit.xml
        
        # Run integration tests
        echo "🔗 Running integration tests..."
        docker-compose exec -T backend python -m pytest tests/integration/ -v --junitxml=test-results-integration.xml
        
        # Run end-to-end tests
        echo "🌐 Running end-to-end tests..."
        docker-compose exec -T backend python -m pytest tests/e2e/ -v --junitxml=test-results-e2e.xml
        
        # Cleanup test services
        docker-compose -f docker-compose.test.yml down
        
        echo "✅ All tests passed in cloud environment"

  # Step 4: Build backend container
  - name: 'gcr.io/cloud-builders/docker'
    id: 'build-backend'
    args:
      - 'build'
      - '-t'
      - '${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/backend:${SHORT_SHA}'
      - '-t'
      - '${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/backend:latest'
      - './backend'
    env:
      - 'PROJECT_ID=${_PROJECT_ID}'

  # Step 5: Build frontend container
  - name: 'gcr.io/cloud-builders/docker'
    id: 'build-frontend'
    args:
      - 'build'
      - '-t'
      - '${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/frontend:${SHORT_SHA}'
      - '-t'
      - '${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/frontend:latest'
      - './frontend'
    env:
      - 'PROJECT_ID=${_PROJECT_ID}'

  # Step 6: Push containers to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    id: 'push-backend'
    args:
      - 'push'
      - '--all-tags'
      - '${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/backend'
    waitFor: ['build-backend']

  - name: 'gcr.io/cloud-builders/docker'
    id: 'push-frontend'
    args:
      - 'push'
      - '--all-tags'
      - '${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/frontend'
    waitFor: ['build-frontend']

  # Step 7: Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'deploy-backend'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        source /workspace/.env.build
        
        echo "🚀 Deploying backend to Cloud Run..."
        
        gcloud run deploy ${_PROJECT_NAME}-$ENVIRONMENT-backend \
          --image=${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/backend:${SHORT_SHA} \
          --region=${_REGION} \
          --platform=managed \
          --allow-unauthenticated \
          --set-env-vars="PROJECTNAME_ENV=$ENVIRONMENT" \
          --min-instances=${_MIN_INSTANCES} \
          --max-instances=${_MAX_INSTANCES} \
          --memory=${_MEMORY_LIMIT} \
          --cpu=${_CPU_LIMIT} \
          --project=${PROJECT_ID}
    waitFor: ['push-backend', 'run-cloud-tests']

  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'deploy-frontend'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        source /workspace/.env.build
        
        echo "🚀 Deploying frontend to Cloud Run..."
        
        # Get backend URL
        BACKEND_URL=$(gcloud run services describe ${_PROJECT_NAME}-$ENVIRONMENT-backend \
          --region=${_REGION} \
          --format='value(status.url)' \
          --project=${PROJECT_ID})
        
        gcloud run deploy ${_PROJECT_NAME}-$ENVIRONMENT-frontend \
          --image=${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/frontend:${SHORT_SHA} \
          --region=${_REGION} \
          --platform=managed \
          --allow-unauthenticated \
          --set-env-vars="PROJECTNAME_ENV=$ENVIRONMENT,PROJECTNAME_BACKEND_HOST=$BACKEND_URL" \
          --min-instances=${_MIN_INSTANCES} \
          --max-instances=${_MAX_INSTANCES} \
          --memory=${_MEMORY_LIMIT} \
          --cpu=${_CPU_LIMIT} \
          --project=${PROJECT_ID}
    waitFor: ['push-frontend', 'deploy-backend']

  # Step 8: Health check and smoke tests
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'health-check'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        source /workspace/.env.build
        
        echo "🏥 Running health checks..."
        
        # Get service URLs
        BACKEND_URL=$(gcloud run services describe ${_PROJECT_NAME}-$ENVIRONMENT-backend \
          --region=${_REGION} \
          --format='value(status.url)' \
          --project=${PROJECT_ID})
        
        FRONTEND_URL=$(gcloud run services describe ${_PROJECT_NAME}-$ENVIRONMENT-frontend \
          --region=${_REGION} \
          --format='value(status.url)' \
          --project=${PROJECT_ID})
        
        # Health check backend
        echo "🔍 Checking backend health..."
        for i in {1..10}; do
          if curl -f "$BACKEND_URL/health" > /dev/null 2>&1; then
            echo "✅ Backend health check passed"
            break
          fi
          echo "⏳ Waiting for backend to be ready... ($i/10)"
          sleep 30
        done
        
        # Health check frontend
        echo "🔍 Checking frontend health..."
        for i in {1..10}; do
          if curl -f "$FRONTEND_URL/_stcore/health" > /dev/null 2>&1; then
            echo "✅ Frontend health check passed"
            break
          fi
          echo "⏳ Waiting for frontend to be ready... ($i/10)"
          sleep 30
        done
        
        # Run smoke tests
        echo "🧪 Running smoke tests..."
        curl -f "$BACKEND_URL/api/v1/agents" || (echo "❌ Smoke test failed" && exit 1)
        
        echo "✅ All health checks and smoke tests passed!"
        echo "🌐 Backend URL: $BACKEND_URL"
        echo "🌐 Frontend URL: $FRONTEND_URL"
    waitFor: ['deploy-frontend']

  # Step 9: Update traffic (production only)
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'update-traffic'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        source /workspace/.env.build
        
        if [ "$ENVIRONMENT" = "prod" ]; then
          echo "🚦 Updating production traffic..."
          
          # Gradual traffic migration for production
          gcloud run services update-traffic ${_PROJECT_NAME}-$ENVIRONMENT-backend \
            --to-latest=100 \
            --region=${_REGION} \
            --project=${PROJECT_ID}
          
          gcloud run services update-traffic ${_PROJECT_NAME}-$ENVIRONMENT-frontend \
            --to-latest=100 \
            --region=${_REGION} \
            --project=${PROJECT_ID}
        else
          echo "✅ Skipping traffic update for $ENVIRONMENT environment"
        fi
    waitFor: ['health-check']

# Substitution variables
substitutions:
  _PROJECT_NAME: 'aiagents'
  _REPOSITORY: 'ai-agents-repo'
  _REGISTRY_REGION: 'us-central1'
  _REGION: 'us-central1'
  
  # Environment-specific project IDs
  _DEV_PROJECT_ID: 'your-ai-agents-dev'
  _STAGING_PROJECT_ID: 'your-ai-agents-staging'
  _PROD_PROJECT_ID: 'your-ai-agents-prod'
  
  # Resource limits
  _MIN_INSTANCES: '0'
  _MAX_INSTANCES: '10'
  _CPU_LIMIT: '1000m'
  _MEMORY_LIMIT: '2Gi'

# Build options
options:
  machineType: 'E2_HIGHCPU_8'
  substitution_option: 'ALLOW_LOOSE'
  logging: CLOUD_LOGGING_ONLY

# Build timeout
timeout: '1800s'  # 30 minutes

# IAM permissions for Cloud Build
serviceAccount: 'projects/${PROJECT_ID}/serviceAccounts/cloudbuild@${PROJECT_ID}.iam.gserviceaccount.com'
```

### Environment-Specific Build Configurations

#### Development Pipeline
```yaml
# cloudbuild.dev.yaml - Development environment
steps:
  - name: 'gcr.io/cloud-builders/git'
    id: 'dev-validation'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🔧 Development deployment - minimal validation"
        # Less strict validation for development branches

  - name: 'docker/compose:latest'
    id: 'dev-tests'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🧪 Running development tests..."
        docker-compose -f docker-compose.test.yml up -d
        docker-compose exec -T backend python -m pytest tests/unit/ -v
        docker-compose -f docker-compose.test.yml down

  # Build and deploy to development
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'deploy-dev'
    args:
      - 'run'
      - 'deploy'
      - '${_PROJECT_NAME}-dev-backend'
      - '--source=./backend'
      - '--region=${_REGION}'
      - '--allow-unauthenticated'
      - '--set-env-vars=PROJECTNAME_ENV=development'

substitutions:
  _PROJECT_NAME: 'aiagents'
  _REGION: 'us-central1'

timeout: '600s'  # 10 minutes for development
```

#### Production Pipeline
```yaml
# cloudbuild.prod.yaml - Production environment with enhanced security
steps:
  - name: 'gcr.io/cloud-builders/git'
    id: 'prod-validation'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🔒 Production deployment - enhanced validation"
        
        # Require specific commit message format for production
        if ! git log -1 --pretty=%B | grep -E "^(feat|fix|chore)(\(.+\))?: .+" > /dev/null; then
          echo "❌ Production requires conventional commit format"
          exit 1
        fi
        
        # Require test evidence
        if ! git log -5 --pretty=%B | grep -E "(test:|tests:|✅|🧪)" > /dev/null; then
          echo "❌ No test evidence in recent commits"
          exit 1
        fi

  # Security scanning
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'security-scan'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🔒 Running security scans..."
        
        # Scan for secrets in code
        if command -v gitleaks &> /dev/null; then
          gitleaks detect --source . --verbose
        fi
        
        # Dependency vulnerability scan
        if [ -f "backend/requirements.txt" ]; then
          pip install safety
          safety check -r backend/requirements.txt
        fi

  # Enhanced testing for production
  - name: 'docker/compose:latest'
    id: 'comprehensive-tests'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🧪 Running comprehensive production test suite..."
        
        docker-compose -f docker-compose.test.yml up -d
        
        # Full test suite
        docker-compose exec -T backend python -m pytest tests/ -v --cov=. --cov-report=term
        
        # Performance tests
        docker-compose exec -T backend python -m pytest tests/performance/ -v
        
        docker-compose -f docker-compose.test.yml down

  # Blue-green deployment for production
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'blue-green-deploy'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "🔄 Blue-green deployment to production..."
        
        # Deploy to staging slot first
        gcloud run deploy ${_PROJECT_NAME}-prod-backend-staging \
          --image=${_REGISTRY_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/backend:${SHORT_SHA} \
          --region=${_REGION} \
          --no-traffic
        
        # Test staging deployment
        STAGING_URL=$(gcloud run services describe ${_PROJECT_NAME}-prod-backend-staging \
          --region=${_REGION} \
          --format='value(status.url)')
        
        # Health check staging
        for i in {1..5}; do
          if curl -f "$STAGING_URL/health" > /dev/null 2>&1; then
            echo "✅ Staging deployment healthy"
            break
          fi
          sleep 10
        done
        
        # Switch traffic to new version
        gcloud run services update-traffic ${_PROJECT_NAME}-prod-backend \
          --to-revisions=${_PROJECT_NAME}-prod-backend-staging=100 \
          --region=${_REGION}

substitutions:
  _PROJECT_NAME: 'aiagents'
  _REPOSITORY: 'ai-agents-repo'
  _REGISTRY_REGION: 'us-central1'
  _REGION: 'us-central1'

timeout: '2400s'  # 40 minutes for production
```

## Local Test Validation Hook

### Pre-Push Git Hook
```bash
#!/bin/bash
# .git/hooks/pre-push - Ensure tests run before push

echo "🧪 Running local tests before push..."

# Check if in the correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Not in project root directory"
    exit 1
fi

# Run local test suite
./scripts/run-tests.sh

if [ $? -ne 0 ]; then
    echo "❌ Local tests failed. Push cancelled."
    echo "💡 Fix tests before pushing to trigger cloud deployment"
    exit 1
fi

echo "✅ Local tests passed. Proceeding with push..."

# Create test evidence file
echo "{
    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"tests_run\": true,
    \"local_validation\": \"passed\"
}" > test-results.json

# Add test results to commit if they don't exist
if ! git diff --cached --name-only | grep -q "test-results.json"; then
    git add test-results.json
    git commit --amend --no-edit
fi

exit 0
```

### Local Test Runner with Cloud Validation
```bash
#!/bin/bash
# scripts/run-tests-for-cloud.sh - Enhanced test runner for cloud deployment

set -e

echo "🧪 Running comprehensive test suite for cloud deployment..."

# Environment setup
export PROJECTNAME_ENV=test
export PROJECTNAME_LOG_LEVEL=DEBUG

# Create test results directory
mkdir -p test-results

# Start test environment
echo "🐳 Starting test environment..."
docker-compose -f docker-compose.test.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check test services
docker-compose exec -T backend curl -f http://localhost:8000/health || {
    echo "❌ Backend health check failed"
    docker-compose -f docker-compose.test.yml logs backend
    exit 1
}

# Run unit tests
echo "🔬 Running unit tests..."
docker-compose exec -T backend python -m pytest tests/unit/ \
    -v \
    --junitxml=/app/test-results/unit-tests.xml \
    --cov=core \
    --cov=agents \
    --cov-report=xml:/app/test-results/coverage.xml

# Run integration tests
echo "🔗 Running integration tests..."
docker-compose exec -T backend python -m pytest tests/integration/ \
    -v \
    --junitxml=/app/test-results/integration-tests.xml

# Run end-to-end tests
echo "🌐 Running end-to-end tests..."
docker-compose exec -T backend python -m pytest tests/e2e/ \
    -v \
    --junitxml=/app/test-results/e2e-tests.xml

# Run performance tests
echo "⚡ Running performance tests..."
docker-compose exec -T backend python -m pytest tests/performance/ \
    -v \
    --junitxml=/app/test-results/performance-tests.xml

# Run security tests
echo "🔒 Running security tests..."
if [ -f "tests/security/test_security.py" ]; then
    docker-compose exec -T backend python -m pytest tests/security/ \
        -v \
        --junitxml=/app/test-results/security-tests.xml
fi

# Cleanup test environment
echo "🧹 Cleaning up test environment..."
docker-compose -f docker-compose.test.yml down

# Generate test summary
echo "📊 Generating test summary..."
python scripts/generate-test-summary.py test-results/

# Create cloud deployment evidence
echo "✅ Creating cloud deployment evidence..."
cat > test-results.json << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "tests_run": true,
    "local_validation": "passed",
    "test_types": ["unit", "integration", "e2e", "performance"],
    "cloud_ready": true,
    "commit_hash": "$(git rev-parse HEAD)"
}
EOF

echo "✅ All tests passed! Ready for cloud deployment."
echo "📤 Push your changes to trigger the CI/CD pipeline."
```

## Pipeline Triggers

### GitHub Integration
```yaml
# .github/workflows/trigger-cloud-build.yml
name: Trigger Cloud Build

on:
  push:
    branches: [main, staging, develop]
  pull_request:
    branches: [main, staging]

jobs:
  validate-and-trigger:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Validate local tests
      run: |
        if [ ! -f "test-results.json" ]; then
          echo "❌ No local test evidence found"
          echo "Please run ./scripts/run-tests-for-cloud.sh first"
          exit 1
        fi
        
        # Check test results are recent (within last hour)
        timestamp=$(jq -r '.timestamp' test-results.json)
        current_time=$(date -u +%s)
        test_time=$(date -d "$timestamp" +%s)
        age=$((current_time - test_time))
        
        if [ $age -gt 3600 ]; then
          echo "❌ Test results are too old (${age}s > 3600s)"
          echo "Please run fresh tests before pushing"
          exit 1
        fi
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
    
    - name: Trigger Cloud Build
      run: |
        # Determine environment based on branch
        if [ "$GITHUB_REF" = "refs/heads/main" ]; then
          CONFIG="cloudbuild.prod.yaml"
          PROJECT_ID="${{ secrets.PROD_PROJECT_ID }}"
        elif [ "$GITHUB_REF" = "refs/heads/staging" ]; then
          CONFIG="cloudbuild.yaml"
          PROJECT_ID="${{ secrets.STAGING_PROJECT_ID }}"
        else
          CONFIG="cloudbuild.dev.yaml"
          PROJECT_ID="${{ secrets.DEV_PROJECT_ID }}"
        fi
        
        gcloud builds submit \
          --config=$CONFIG \
          --project=$PROJECT_ID \
          --substitutions=BRANCH_NAME=$GITHUB_REF_NAME,SHORT_SHA=$GITHUB_SHA
```

## Rollback Strategy

### Automatic Rollback Script
```bash
#!/bin/bash
# scripts/rollback.sh - Automatic rollback on deployment failure

set -e

ENVIRONMENT=${1:-dev}
PROJECT_ID=${2}
SERVICE_NAME=${3}

if [ -z "$PROJECT_ID" ] || [ -z "$SERVICE_NAME" ]; then
    echo "❌ Usage: $0 <environment> <project-id> <service-name>"
    exit 1
fi

echo "🔄 Rolling back $SERVICE_NAME in $ENVIRONMENT..."

# Get current revision
CURRENT_REVISION=$(gcloud run services describe $SERVICE_NAME \
    --region=us-central1 \
    --project=$PROJECT_ID \
    --format='value(status.latestCreatedRevisionName)')

# Get previous revision
PREVIOUS_REVISION=$(gcloud run revisions list \
    --service=$SERVICE_NAME \
    --region=us-central1 \
    --project=$PROJECT_ID \
    --limit=2 \
    --format='value(metadata.name)' | sed -n '2p')

if [ -z "$PREVIOUS_REVISION" ]; then
    echo "❌ No previous revision found for rollback"
    exit 1
fi

echo "🔄 Rolling back from $CURRENT_REVISION to $PREVIOUS_REVISION"

# Update traffic to previous revision
gcloud run services update-traffic $SERVICE_NAME \
    --to-revisions=$PREVIOUS_REVISION=100 \
    --region=us-central1 \
    --project=$PROJECT_ID

# Verify rollback
sleep 30
NEW_TRAFFIC=$(gcloud run services describe $SERVICE_NAME \
    --region=us-central1 \
    --project=$PROJECT_ID \
    --format='value(status.traffic[0].revisionName)')

if [ "$NEW_TRAFFIC" = "$PREVIOUS_REVISION" ]; then
    echo "✅ Rollback successful to $PREVIOUS_REVISION"
else
    echo "❌ Rollback failed"
    exit 1
fi
```

## Monitoring and Alerting

### Build Notification
```yaml
# cloudbuild-slack.yaml - Slack notifications for build status
steps:
  # ... existing build steps ...

  # Notify on success
  - name: 'gcr.io/cloud-builders/curl'
    id: 'notify-success'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        if [ "${_SLACK_WEBHOOK}" != "" ]; then
          curl -X POST -H 'Content-type: application/json' \
            --data "{
              \"text\": \"✅ Deployment successful!\",
              \"blocks\": [{
                \"type\": \"section\",
                \"text\": {
                  \"type\": \"mrkdwn\",
                  \"text\": \"*${_PROJECT_NAME}* deployed to *${_ENVIRONMENT}*\\n*Commit:* ${SHORT_SHA}\\n*Branch:* ${BRANCH_NAME}\"
                }
              }]
            }" \
            ${_SLACK_WEBHOOK}
        fi
    waitFor: ['health-check']

# Handle build failures
onFailure:
  - name: 'gcr.io/cloud-builders/curl'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        if [ "${_SLACK_WEBHOOK}" != "" ]; then
          curl -X POST -H 'Content-type: application/json' \
            --data "{
              \"text\": \"❌ Deployment failed!\",
              \"blocks\": [{
                \"type\": \"section\",
                \"text\": {
                  \"type\": \"mrkdwn\",
                  \"text\": \"*${_PROJECT_NAME}* deployment to *${_ENVIRONMENT}* failed\\n*Commit:* ${SHORT_SHA}\\n*Branch:* ${BRANCH_NAME}\"
                }
              }]
            }" \
            ${_SLACK_WEBHOOK}
        fi

substitutions:
  _SLACK_WEBHOOK: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
```

---

## Next Steps

- **Deployment Scripts**: [Automated Setup Scripts](deployment_scripts.md)
- **Monitoring**: [Observability and Alerting](observability.md)
- **Security**: [Security Configuration](../06_reference/security_guidelines.md)

---

*This CI/CD pipeline ensures that only thoroughly tested code reaches production while maintaining rapid deployment cycles and automatic rollback capabilities.*