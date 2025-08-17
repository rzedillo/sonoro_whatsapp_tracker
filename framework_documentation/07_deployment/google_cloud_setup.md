# Google Cloud Infrastructure Setup

> ☁️ **Cloud Deployment**: Complete Google Cloud infrastructure using Terraform for scalable, production-ready deployment.

## Navigation
- **Previous**: [Local Development](local_development.md)
- **Next**: [CI/CD Pipeline](cicd_pipeline.md)
- **Related**: [Environment Configuration](environment_configuration.md) → [Security Guidelines](../06_reference/security_guidelines.md)

---

## Overview

This guide provides Infrastructure as Code (IaC) templates using Terraform to deploy the AI agents framework to Google Cloud Platform. The infrastructure supports all framework levels from simple agents to enterprise production systems.

## Infrastructure Architecture

```
Google Cloud Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Project                         │
├─────────────────────────────────────────────────────────────────┤
│  Load Balancer → Cloud Run (Backend) → Cloud SQL (PostgreSQL)  │
│                     ↓                      ↓                    │
│                Cloud Run (Frontend)   Memory Store (Redis)      │
│                     ↓                      ↓                    │
│              Cloud Storage (Files)   Secret Manager (Keys)      │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required Tools
```bash
# Install required tools
# Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Verify installations
gcloud --version
terraform --version
```

### Google Cloud Setup
```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Set project (replace with your project ID)
export PROJECT_ID="your-ai-agents-project"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable sql-component.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable redis.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

## Terraform Infrastructure

### Main Infrastructure Configuration
```hcl
# infrastructure/terraform/main.tf
terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "terraform/state"
  }
}

# Configure providers
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Local values for consistent naming
locals {
  project_name = var.project_name
  environment  = var.environment
  
  # Resource naming convention
  name_prefix = "${local.project_name}-${local.environment}"
  
  # Common labels
  common_labels = {
    project     = local.project_name
    environment = local.environment
    managed_by  = "terraform"
  }
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460
}

# Subnet for Cloud Run and other services
resource "google_compute_subnetwork" "subnet" {
  name          = "${local.name_prefix}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
  
  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "192.168.1.0/24"
  }
  
  secondary_ip_range {
    range_name    = "pod-ranges"
    ip_cidr_range = "192.168.64.0/22"
  }
}

# Cloud Router for NAT
resource "google_compute_router" "router" {
  name    = "${local.name_prefix}-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

# NAT Gateway for outbound internet access
resource "google_compute_router_nat" "nat" {
  name   = "${local.name_prefix}-nat"
  router = google_compute_router.router.name
  region = var.region

  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
```

### Variables Configuration
```hcl
# infrastructure/terraform/variables.tf
variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "aiagents"
}

variable "region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Google Cloud zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "database_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

variable "redis_memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "cpu_limit" {
  description = "CPU limit for Cloud Run services"
  type        = string
  default     = "1000m"
}

variable "memory_limit" {
  description = "Memory limit for Cloud Run services"
  type        = string
  default     = "2Gi"
}

variable "domain_name" {
  description = "Custom domain name (optional)"
  type        = string
  default     = ""
}
```

### Database Module
```hcl
# infrastructure/terraform/modules/database/main.tf
resource "google_sql_database_instance" "main" {
  name             = "${var.name_prefix}-db"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier                        = var.database_tier
    availability_type           = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    disk_type                   = "PD_SSD"
    disk_size                   = var.environment == "prod" ? 100 : 20
    disk_autoresize            = true
    disk_autoresize_limit      = var.environment == "prod" ? 500 : 100
    
    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      location                       = var.region
      point_in_time_recovery_enabled = var.environment == "prod"
      
      backup_retention_settings {
        retained_backups = var.environment == "prod" ? 30 : 7
        retention_unit   = "COUNT"
      }
    }
    
    maintenance_window {
      hour = 3
      day  = 1
    }
    
    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
    
    database_flags {
      name  = "log_connections"
      value = "on"
    }
    
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
    
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = var.vpc_network
      enable_private_path_for_google_cloud_services = true
    }
  }
  
  deletion_protection = var.environment == "prod"
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Private VPC connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.name_prefix}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = var.vpc_network
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = var.vpc_network
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Create database
resource "google_sql_database" "database" {
  name     = var.database_name
  instance = google_sql_database_instance.main.name
}

# Database user
resource "google_sql_user" "user" {
  name     = var.database_user
  instance = google_sql_database_instance.main.name
  password = var.database_password
}

# Store database credentials in Secret Manager
resource "google_secret_manager_secret" "database_url" {
  secret_id = "${var.name_prefix}-database-url"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = "postgresql://${var.database_user}:${var.database_password}@${google_sql_database_instance.main.private_ip_address}:5432/${var.database_name}"
}
```

### Redis Module
```hcl
# infrastructure/terraform/modules/redis/main.tf
resource "google_redis_instance" "cache" {
  name           = "${var.name_prefix}-redis"
  tier           = var.environment == "prod" ? "STANDARD_HA" : "BASIC"
  memory_size_gb = var.memory_size_gb
  region         = var.region
  
  location_id             = var.zone
  alternative_location_id = var.environment == "prod" ? "${substr(var.region, 0, length(var.region)-1)}b" : null
  
  authorized_network = var.vpc_network
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  redis_version     = "REDIS_7_0"
  display_name      = "${var.name_prefix} Redis Cache"
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    notify-keyspace-events = "Ex"
  }
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
}

# Store Redis connection string in Secret Manager
resource "google_secret_manager_secret" "redis_url" {
  secret_id = "${var.name_prefix}-redis-url"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "redis_url" {
  secret      = google_secret_manager_secret.redis_url.id
  secret_data = "redis://${google_redis_instance.cache.host}:${google_redis_instance.cache.port}"
}
```

### Cloud Run Backend Service
```hcl
# infrastructure/terraform/modules/backend/main.tf
resource "google_cloud_run_v2_service" "backend" {
  name     = "${var.name_prefix}-backend"
  location = var.region
  
  template {
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    containers {
      image = var.backend_image
      
      ports {
        container_port = 8000
      }
      
      env {
        name  = "${upper(var.project_name)}_ENV"
        value = var.environment
      }
      
      env {
        name = "${upper(var.project_name)}_DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = var.database_url_secret
            version = "latest"
          }
        }
      }
      
      env {
        name = "${upper(var.project_name)}_REDIS_URL"
        value_source {
          secret_key_ref {
            secret  = var.redis_url_secret
            version = "latest"
          }
        }
      }
      
      env {
        name = "${upper(var.project_name)}_OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = var.openai_api_key_secret
            version = "latest"
          }
        }
      }
      
      env {
        name = "${upper(var.project_name)}_ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = var.anthropic_api_key_secret
            version = "latest"
          }
        }
      }
      
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
        cpu_idle = var.environment == "prod" ? false : true
      }
      
      startup_probe {
        initial_delay_seconds = 30
        timeout_seconds      = 10
        period_seconds       = 5
        failure_threshold    = 3
        
        http_get {
          path = "/health"
          port = 8000
        }
      }
      
      liveness_probe {
        initial_delay_seconds = 30
        timeout_seconds      = 10
        period_seconds       = 30
        failure_threshold    = 3
        
        http_get {
          path = "/health"
          port = 8000
        }
      }
    }
    
    vpc_access {
      connector = var.vpc_connector
      egress    = "PRIVATE_RANGES_ONLY"
    }
    
    service_account = var.service_account_email
  }
  
  traffic {
    percent = 100
  }
}

# IAM binding for Cloud Run service
resource "google_cloud_run_service_iam_binding" "backend_public" {
  location = google_cloud_run_v2_service.backend.location
  service  = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}
```

### Cloud Run Frontend Service
```hcl
# infrastructure/terraform/modules/frontend/main.tf
resource "google_cloud_run_v2_service" "frontend" {
  name     = "${var.name_prefix}-frontend"
  location = var.region
  
  template {
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    containers {
      image = var.frontend_image
      
      ports {
        container_port = 8501
      }
      
      env {
        name  = "${upper(var.project_name)}_BACKEND_HOST"
        value = var.backend_url
      }
      
      env {
        name  = "${upper(var.project_name)}_BACKEND_PORT"
        value = "443"
      }
      
      env {
        name  = "${upper(var.project_name)}_ENV"
        value = var.environment
      }
      
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
        cpu_idle = var.environment == "prod" ? false : true
      }
      
      startup_probe {
        initial_delay_seconds = 30
        timeout_seconds      = 10
        period_seconds       = 5
        failure_threshold    = 3
        
        http_get {
          path = "/_stcore/health"
          port = 8501
        }
      }
    }
    
    service_account = var.service_account_email
  }
  
  traffic {
    percent = 100
  }
}

# IAM binding for Cloud Run service
resource "google_cloud_run_service_iam_binding" "frontend_public" {
  location = google_cloud_run_v2_service.frontend.location
  service  = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}
```

### Secret Manager for API Keys
```hcl
# infrastructure/terraform/modules/secrets/main.tf
# OpenAI API Key Secret
resource "google_secret_manager_secret" "openai_api_key" {
  secret_id = "${var.name_prefix}-openai-api-key"
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

# Anthropic API Key Secret
resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = "${var.name_prefix}-anthropic-api-key"
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

# JWT Secret
resource "google_secret_manager_secret" "jwt_secret" {
  secret_id = "${var.name_prefix}-jwt-secret"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "jwt_secret" {
  secret      = google_secret_manager_secret.jwt_secret.id
  secret_data = var.jwt_secret
}

# Application Secret Key
resource "google_secret_manager_secret" "app_secret_key" {
  secret_id = "${var.name_prefix}-app-secret-key"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "app_secret_key" {
  secret      = google_secret_manager_secret.app_secret_key.id
  secret_data = var.app_secret_key
}
```

## Environment-Specific Configurations

### Development Environment
```hcl
# infrastructure/terraform/environments/dev/terraform.tfvars
project_id   = "your-ai-agents-dev"
project_name = "aiagents"
environment  = "dev"
region       = "us-central1"
zone         = "us-central1-a"

# Small instances for development
database_tier        = "db-f1-micro"
redis_memory_size_gb = 1
min_instances        = 0
max_instances        = 3
cpu_limit           = "1000m"
memory_limit        = "1Gi"
```

### Staging Environment
```hcl
# infrastructure/terraform/environments/staging/terraform.tfvars
project_id   = "your-ai-agents-staging"
project_name = "aiagents"
environment  = "staging"
region       = "us-central1"
zone         = "us-central1-a"

# Medium instances for staging
database_tier        = "db-g1-small"
redis_memory_size_gb = 2
min_instances        = 1
max_instances        = 5
cpu_limit           = "1000m"
memory_limit        = "2Gi"
```

### Production Environment
```hcl
# infrastructure/terraform/environments/prod/terraform.tfvars
project_id   = "your-ai-agents-prod"
project_name = "aiagents"
environment  = "prod"
region       = "us-central1"
zone         = "us-central1-a"

# Production-grade instances
database_tier        = "db-standard-2"
redis_memory_size_gb = 4
min_instances        = 2
max_instances        = 20
cpu_limit           = "2000m"
memory_limit        = "4Gi"

domain_name = "api.yourdomain.com"
```

## Deployment Scripts

### Terraform Deployment Script
```bash
#!/bin/bash
# infrastructure/scripts/deploy.sh

set -e

ENVIRONMENT=${1:-dev}
PROJECT_NAME=${2:-aiagents}

echo "🚀 Deploying $PROJECT_NAME to $ENVIRONMENT environment"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo "❌ Invalid environment. Use: dev, staging, or prod"
    exit 1
fi

# Set working directory
cd "$(dirname "$0")/../terraform"

# Initialize Terraform
echo "🔧 Initializing Terraform..."
terraform init -backend-config="bucket=${PROJECT_NAME}-terraform-state-${ENVIRONMENT}"

# Select workspace
terraform workspace select $ENVIRONMENT || terraform workspace new $ENVIRONMENT

# Plan deployment
echo "📋 Planning deployment..."
terraform plan -var-file="environments/${ENVIRONMENT}/terraform.tfvars" -out=tfplan

# Apply deployment
read -p "🚀 Deploy to $ENVIRONMENT? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Deploying infrastructure..."
    terraform apply tfplan
    
    echo "✅ Deployment completed!"
    
    # Output important information
    echo "📊 Infrastructure details:"
    terraform output
else
    echo "❌ Deployment cancelled"
    rm -f tfplan
fi
```

### Secret Management Script
```bash
#!/bin/bash
# infrastructure/scripts/setup-secrets.sh

set -e

PROJECT_ID=${1}
ENVIRONMENT=${2:-dev}
PROJECT_NAME=${3:-aiagents}

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Usage: $0 <project-id> [environment] [project-name]"
    exit 1
fi

echo "🔐 Setting up secrets for $PROJECT_NAME in $ENVIRONMENT"

# Set project
gcloud config set project $PROJECT_ID

# Create OpenAI API Key secret (user will need to add the actual value)
read -p "🤖 Enter OpenAI API Key: " -s OPENAI_KEY
echo
echo $OPENAI_KEY | gcloud secrets create ${PROJECT_NAME}-${ENVIRONMENT}-openai-api-key --data-file=-

# Create Anthropic API Key secret (optional)
read -p "🤖 Enter Anthropic API Key (optional): " -s ANTHROPIC_KEY
echo
if [ ! -z "$ANTHROPIC_KEY" ]; then
    echo $ANTHROPIC_KEY | gcloud secrets create ${PROJECT_NAME}-${ENVIRONMENT}-anthropic-api-key --data-file=-
fi

# Generate and store JWT secret
JWT_SECRET=$(openssl rand -base64 32)
echo $JWT_SECRET | gcloud secrets create ${PROJECT_NAME}-${ENVIRONMENT}-jwt-secret --data-file=-

# Generate and store app secret key
APP_SECRET=$(openssl rand -base64 32)
echo $APP_SECRET | gcloud secrets create ${PROJECT_NAME}-${ENVIRONMENT}-app-secret-key --data-file=-

echo "✅ Secrets created successfully!"
echo "🔑 You can update secrets later using:"
echo "   gcloud secrets versions add <secret-name> --data-file=-"
```

## Monitoring and Observability

### Cloud Monitoring Setup
```hcl
# infrastructure/terraform/modules/monitoring/main.tf
# Custom dashboard for the application
resource "google_monitoring_dashboard" "main" {
  dashboard_json = jsonencode({
    displayName = "${var.name_prefix} Dashboard"
    
    gridLayout = {
      widgets = [
        {
          title = "Cloud Run CPU Utilization"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"${var.name_prefix}-.*\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
              plotType = "LINE"
            }]
          }
        },
        {
          title = "Cloud Run Memory Utilization"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"${var.name_prefix}-.*\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
              plotType = "LINE"
            }]
          }
        }
      ]
    }
  })
}

# Alerting policies
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "${var.name_prefix} High Error Rate"
  combiner     = "OR"
  
  conditions {
    display_name = "Cloud Run Error Rate"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"${var.name_prefix}-.*\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.1
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  alert_strategy {
    auto_close = "1800s"
  }
}
```

---

## Next Steps

- **CI/CD Pipeline**: [Automated Deployment](cicd_pipeline.md)
- **Monitoring**: [Observability Setup](observability.md)
- **Security**: [Security Configuration](../06_reference/security_guidelines.md)

---

*This Terraform configuration provides a complete, scalable Google Cloud infrastructure that can grow from development to enterprise production deployment.*