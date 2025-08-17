# Framework Integration with Cloud Deployment

> 🔗 **Seamless Integration**: How to integrate cloud deployment patterns with all framework implementation levels.

## Navigation
- **Previous**: [Deployment Scripts](deployment_scripts.md)
- **Next**: [Production Checklist](production_checklist.md)
- **Related**: [Level 1 Simple](../05_implementation_levels/level_1_simple.md) → [Level 4 Production](../05_implementation_levels/level_4_production.md)

---

## Overview

This guide shows how to integrate the streamlined cloud deployment approach with each framework implementation level, maintaining the framework's progressive complexity while adding cloud-native capabilities.

## Integration by Framework Level

### Level 1: Simple Agent Systems + Cloud

**Enhanced for Cloud Deployment**

```python
# Level 1 with Cloud Integration
# backend/core/agents/simple_cloud_agent.py

import os
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import openai
import structlog
from google.cloud import secretmanager

@dataclass
class CloudAgentConfig:
    """Cloud-enhanced configuration for simple agents"""
    name: str
    role: str
    expertise: list
    project_name: str  # For environment variable prefixes
    
    # Cloud-specific settings
    use_secret_manager: bool = True
    gcp_project_id: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout: int = 30

class SimpleCloudAgent:
    """Level 1 agent enhanced for cloud deployment"""
    
    def __init__(self, config: CloudAgentConfig):
        self.config = config
        self.logger = structlog.get_logger(agent=config.name)
        self.prefix = config.project_name.upper()
        
        # Initialize OpenAI client with cloud-aware configuration
        self.client = self._initialize_openai_client()
        
        # Metrics
        self.execution_count = 0
        self.success_count = 0
    
    def _initialize_openai_client(self) -> openai.AsyncOpenAI:
        """Initialize OpenAI client with cloud secret management"""
        api_key = self._get_openai_api_key()
        
        if not api_key:
            raise ValueError(f"OpenAI API key not found. Set {self.prefix}_OPENAI_API_KEY")
        
        return openai.AsyncOpenAI(api_key=api_key)
    
    def _get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment or Secret Manager"""
        # Try environment variable first
        api_key = os.getenv(f"{self.prefix}_OPENAI_API_KEY")
        
        if api_key:
            return api_key
        
        # Try Google Secret Manager in cloud environment
        if self.config.use_secret_manager and self.config.gcp_project_id:
            try:
                client = secretmanager.SecretManagerServiceClient()
                secret_name = f"projects/{self.config.gcp_project_id}/secrets/{self.config.project_name}-openai-api-key/versions/latest"
                response = client.access_secret_version(request={"name": secret_name})
                return response.payload.data.decode("UTF-8")
            except Exception as e:
                self.logger.warning("Could not access Secret Manager", error=str(e))
        
        return ""
    
    async def execute(self, task: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute task with cloud-enhanced logging and monitoring"""
        execution_id = f"{self.config.name}_{int(asyncio.get_event_loop().time() * 1000)}"
        self.execution_count += 1
        
        self.logger.info("Starting task execution", 
                        execution_id=execution_id, 
                        task=task,
                        data_size=len(str(data)) if data else 0)
        
        try:
            # Validate input
            if not task or not task.strip():
                raise ValueError("Task cannot be empty")
            
            # Build cloud-aware system prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(task, data)
            
            # Execute with retry logic
            response = await self._execute_with_retry(system_prompt, user_prompt)
            
            # Process response
            try:
                import json
                result = json.loads(response)
            except json.JSONDecodeError:
                result = {"content": response}
            
            self.success_count += 1
            
            final_result = {
                "status": "success",
                "agent": self.config.name,
                "task": task,
                "result": result,
                "execution_id": execution_id,
                "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown")
            }
            
            self.logger.info("Task execution completed successfully", 
                           execution_id=execution_id,
                           success_rate=self.success_count / self.execution_count)
            
            return final_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "agent": self.config.name,
                "task": task,
                "error": str(e),
                "execution_id": execution_id,
                "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown")
            }
            
            self.logger.error("Task execution failed", 
                            execution_id=execution_id,
                            error=str(e),
                            error_rate=(self.execution_count - self.success_count) / self.execution_count)
            
            return error_result
    
    async def _execute_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Execute with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    timeout=self.config.timeout
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = (2 ** attempt) + (asyncio.get_event_loop().time() % 1)
                    self.logger.warning("Retrying after error", 
                                      attempt=attempt + 1, 
                                      wait_time=wait_time,
                                      error=str(e))
                    await asyncio.sleep(wait_time)
        
        raise last_exception
    
    def _build_system_prompt(self) -> str:
        """Build cloud-aware system prompt"""
        environment = os.getenv(f"{self.prefix}_ENV", "development")
        
        return f"""You are {self.config.name}, a {self.config.role}.

Your expertise areas: {', '.join(self.config.expertise)}

Environment: {environment}
Cloud Platform: Google Cloud Platform

Guidelines:
- Focus on your area of expertise
- Provide clear, actionable results
- Return JSON responses when possible
- Include confidence levels for assessments
- Be precise and concise
- Consider cloud deployment context

Response format:
{{
    "result": "your main result",
    "confidence": 0.95,
    "explanation": "brief explanation",
    "recommendations": ["list of recommendations if applicable"],
    "cloud_compatible": true
}}"""
    
    def _build_user_prompt(self, task: str, data: Dict[str, Any]) -> str:
        """Build user prompt with cloud context"""
        prompt = f"Task: {task}\n\n"
        
        if data:
            import json
            prompt += f"Data to process:\n{json.dumps(data, indent=2)}\n\n"
        
        # Add cloud context
        environment = os.getenv(f"{self.prefix}_ENV", "development")
        prompt += f"Environment: {environment}\n"
        prompt += "Please complete this task according to your expertise and cloud deployment context."
        
        return prompt
    
    def get_cloud_metrics(self) -> Dict[str, Any]:
        """Get cloud-enhanced metrics"""
        return {
            "agent_name": self.config.name,
            "total_executions": self.execution_count,
            "successful_executions": self.success_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0,
            "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown"),
            "cloud_platform": "Google Cloud Platform",
            "secret_management": "enabled" if self.config.use_secret_manager else "disabled"
        }

# Cloud deployment configuration
def create_level1_cloud_config(project_name: str) -> CloudAgentConfig:
    """Create Level 1 cloud configuration"""
    return CloudAgentConfig(
        name="SimpleCloudAgent",
        role="Cloud-native simple agent",
        expertise=["data_processing", "cloud_operations"],
        project_name=project_name,
        use_secret_manager=os.getenv(f"{project_name.upper()}_ENV") == "production",
        gcp_project_id=os.getenv(f"{project_name.upper()}_GCP_PROJECT_ID")
    )
```

**Level 1 Cloud Deployment Structure:**
```
level1-cloud-project/
├── backend/
│   ├── core/
│   │   └── agents/
│   │       └── simple_cloud_agent.py
│   ├── main.py                    # FastAPI with single agent
│   └── requirements.txt
├── frontend/                      # Optional Streamlit interface
│   ├── main.py
│   └── requirements.txt
├── infrastructure/
│   └── terraform/
│       └── level1/               # Minimal cloud resources
│           ├── main.tf          # Cloud Run + secrets only
│           └── variables.tf
├── .env.template                 # Project-prefixed variables
└── docker-compose.yml           # Local development
```

### Level 2: Standard Multi-Agent Systems + Cloud

**Enhanced Orchestration with Cloud Features**

```python
# Level 2 with Cloud Integration
# backend/core/orchestrator/cloud_orchestrator.py

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import structlog
from google.cloud import monitoring_v3
from google.cloud import logging

@dataclass
class CloudWorkflowStage:
    """Cloud-enhanced workflow stage"""
    name: str
    agent_name: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    required: bool = True
    timeout: int = 300
    cloud_monitoring: bool = True
    retry_policy: str = "exponential_backoff"

@dataclass
class CloudWorkflowConfig:
    """Cloud-enhanced workflow configuration"""
    name: str
    stages: List[CloudWorkflowStage]
    project_name: str
    context_sharing: bool = True
    progress_tracking: bool = True
    cloud_monitoring: bool = True
    auto_scaling: bool = True

class CloudContextManager:
    """Cloud-native context manager with persistence"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.prefix = project_name.upper()
        self.contexts = {}
        self.logger = structlog.get_logger(component="context_manager")
        
        # Cloud storage for context persistence (Redis in cloud)
        self.redis_client = self._init_redis_client()
    
    def _init_redis_client(self):
        """Initialize Redis client for cloud context storage"""
        redis_url = os.getenv(f"{self.prefix}_REDIS_URL")
        if redis_url:
            import redis
            return redis.from_url(redis_url)
        return None
    
    async def create_context(self, workflow_id: str, initial_data: Dict) -> str:
        """Create persistent cloud context"""
        context = {
            "workflow_id": workflow_id,
            "data": initial_data.copy(),
            "stage_results": {},
            "created_at": datetime.utcnow().isoformat(),
            "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown"),
            "project_name": self.project_name
        }
        
        # Store in memory
        self.contexts[workflow_id] = context
        
        # Persist to Redis if available
        if self.redis_client:
            try:
                import json
                self.redis_client.setex(
                    f"context:{workflow_id}",
                    3600,  # 1 hour TTL
                    json.dumps(context, default=str)
                )
            except Exception as e:
                self.logger.warning("Could not persist context to Redis", error=str(e))
        
        return workflow_id
    
    async def update_context(self, workflow_id: str, stage_name: str, result: Dict):
        """Update context with cloud persistence"""
        if workflow_id in self.contexts:
            self.contexts[workflow_id]["stage_results"][stage_name] = result
            self.contexts[workflow_id]["data"].update(result.get("context_data", {}))
            self.contexts[workflow_id]["updated_at"] = datetime.utcnow().isoformat()
            
            # Update persistent storage
            if self.redis_client:
                try:
                    import json
                    self.redis_client.setex(
                        f"context:{workflow_id}",
                        3600,
                        json.dumps(self.contexts[workflow_id], default=str)
                    )
                except Exception as e:
                    self.logger.warning("Could not update context in Redis", error=str(e))

class Level2CloudOrchestrator:
    """Level 2 orchestrator enhanced for cloud deployment"""
    
    def __init__(self, agents: Dict[str, Any], project_name: str):
        self.agents = agents
        self.project_name = project_name
        self.prefix = project_name.upper()
        self.context_manager = CloudContextManager(project_name)
        self.execution_history = []
        
        # Cloud monitoring setup
        self.logger = structlog.get_logger(component="orchestrator", project=project_name)
        self.monitoring_client = self._init_monitoring_client()
        
        # Performance metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0
        }
    
    def _init_monitoring_client(self):
        """Initialize Google Cloud Monitoring client"""
        gcp_project_id = os.getenv(f"{self.prefix}_GCP_PROJECT_ID")
        if gcp_project_id and os.getenv(f"{self.prefix}_ENV") in ["staging", "production"]:
            try:
                return monitoring_v3.MetricServiceClient()
            except Exception as e:
                self.logger.warning("Could not initialize monitoring client", error=str(e))
        return None
    
    async def execute_workflow(self, config: CloudWorkflowConfig, 
                             input_data: Dict[str, Any],
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute workflow with cloud monitoring and scaling"""
        
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.metrics["total_workflows"] += 1
        
        self.logger.info("Starting cloud workflow execution", 
                        workflow_id=workflow_id,
                        workflow_name=config.name,
                        input_size=len(str(input_data)))
        
        # Send metrics to Cloud Monitoring
        if config.cloud_monitoring:
            self._send_workflow_metric("workflow_started", 1)
        
        try:
            # Create cloud context
            context_id = None
            if config.context_sharing:
                context_id = await self.context_manager.create_context(workflow_id, input_data)
            
            # Execute stages with cloud monitoring
            stage_results = {}
            
            for stage in config.stages:
                if not self._dependencies_met(stage, stage_results):
                    raise ValueError(f"Dependencies not met for stage {stage.name}")
                
                # Progress tracking
                if progress_callback:
                    progress_callback(stage.name, "started", 0)
                
                # Execute stage with cloud monitoring
                stage_start = time.time()
                result = await self._execute_cloud_stage(stage, context_id, workflow_id)
                stage_duration = time.time() - stage_start
                
                stage_results[stage.name] = result
                
                # Update context
                if context_id:
                    await self.context_manager.update_context(context_id, stage.name, result)
                
                # Send stage metrics
                if config.cloud_monitoring:
                    self._send_stage_metric(stage.name, stage_duration, result["status"] == "success")
                
                if progress_callback:
                    progress_callback(stage.name, "completed", 100)
            
            # Build final result
            execution_time = time.time() - start_time
            
            final_result = {
                "workflow_id": workflow_id,
                "workflow_name": config.name,
                "status": "success",
                "execution_time": execution_time,
                "stage_results": stage_results,
                "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown"),
                "cloud_monitoring": config.cloud_monitoring,
                "input_data": input_data
            }
            
            # Update metrics
            self.metrics["successful_workflows"] += 1
            self._update_average_execution_time(execution_time)
            
            # Send success metrics
            if config.cloud_monitoring:
                self._send_workflow_metric("workflow_completed", 1)
                self._send_workflow_metric("workflow_duration", execution_time)
            
            self.logger.info("Cloud workflow execution completed", 
                           workflow_id=workflow_id,
                           execution_time=execution_time,
                           success_rate=self.metrics["successful_workflows"] / self.metrics["total_workflows"])
            
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = {
                "workflow_id": workflow_id,
                "workflow_name": config.name,
                "status": "error",
                "error": str(e),
                "execution_time": execution_time,
                "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown")
            }
            
            # Update metrics
            self.metrics["failed_workflows"] += 1
            
            # Send error metrics
            if config.cloud_monitoring:
                self._send_workflow_metric("workflow_failed", 1)
            
            self.logger.error("Cloud workflow execution failed", 
                            workflow_id=workflow_id,
                            error=str(e),
                            execution_time=execution_time)
            
            return error_result
    
    async def _execute_cloud_stage(self, stage: CloudWorkflowStage, context_id: str, workflow_id: str) -> Dict[str, Any]:
        """Execute stage with cloud-specific enhancements"""
        
        if stage.agent_name not in self.agents:
            raise ValueError(f"Agent {stage.agent_name} not found")
        
        agent = self.agents[stage.agent_name]
        
        # Prepare stage input with cloud context
        stage_input = {
            "task": stage.task,
            "workflow_id": workflow_id,
            "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown")
        }
        
        if context_id:
            context_data = await self.context_manager.get_context_for_stage(context_id, stage.name)
            stage_input.update(context_data)
        
        # Execute with timeout and retry
        try:
            result = await asyncio.wait_for(
                agent.execute(stage.task, stage_input),
                timeout=stage.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"Stage {stage.name} timed out after {stage.timeout} seconds")
    
    def _send_workflow_metric(self, metric_name: str, value: float):
        """Send custom metric to Google Cloud Monitoring"""
        if not self.monitoring_client:
            return
        
        try:
            gcp_project_id = os.getenv(f"{self.prefix}_GCP_PROJECT_ID")
            project_name = f"projects/{gcp_project_id}"
            
            series = monitoring_v3.TimeSeries()
            series.metric.type = f"custom.googleapis.com/{self.project_name}/{metric_name}"
            series.resource.type = "global"
            
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10 ** 9)
            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )
            
            point = monitoring_v3.Point(
                {"interval": interval, "value": {"double_value": value}}
            )
            series.points = [point]
            
            self.monitoring_client.create_time_series(
                name=project_name, time_series=[series]
            )
            
        except Exception as e:
            self.logger.warning("Could not send metric to Cloud Monitoring", 
                              metric=metric_name, error=str(e))
    
    def _send_stage_metric(self, stage_name: str, duration: float, success: bool):
        """Send stage-specific metrics"""
        self._send_workflow_metric(f"stage_{stage_name}_duration", duration)
        self._send_workflow_metric(f"stage_{stage_name}_success", 1.0 if success else 0.0)
    
    def _dependencies_met(self, stage: CloudWorkflowStage, completed_stages: Dict) -> bool:
        """Check if stage dependencies are met"""
        for dep in stage.depends_on:
            if dep not in completed_stages:
                return False
            if completed_stages[dep].get("status") != "success":
                return False
        return True
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        total_workflows = self.metrics["total_workflows"]
        current_avg = self.metrics["average_execution_time"]
        
        new_avg = ((current_avg * (total_workflows - 1)) + execution_time) / total_workflows
        self.metrics["average_execution_time"] = new_avg
    
    def get_cloud_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cloud metrics"""
        return {
            **self.metrics,
            "project_name": self.project_name,
            "cloud_environment": os.getenv(f"{self.prefix}_ENV", "unknown"),
            "monitoring_enabled": self.monitoring_client is not None,
            "context_persistence": self.context_manager.redis_client is not None,
            "active_contexts": len(self.context_manager.contexts)
        }
```

### Level 3-4: Complex/Production Systems + Cloud

**Enterprise-Grade Cloud Integration**

For Level 3-4 systems, the cloud integration includes:

1. **Advanced Monitoring**: Full observability with distributed tracing
2. **Auto-scaling**: Kubernetes or advanced Cloud Run configurations
3. **Multi-region Deployment**: Global distribution and disaster recovery
4. **Advanced Security**: IAM, VPC, encryption at rest and in transit
5. **Performance Optimization**: Caching, CDN, database optimization

```python
# Level 3-4 Enterprise Cloud Features
# backend/core/enterprise/cloud_enterprise_orchestrator.py

class EnterpriseCloudOrchestrator(Level2CloudOrchestrator):
    """Enterprise-grade cloud orchestrator for Level 3-4 systems"""
    
    def __init__(self, agents: Dict[str, Any], project_name: str):
        super().__init__(agents, project_name)
        
        # Enterprise features
        self.circuit_breakers = {name: CircuitBreaker() for name in agents.keys()}
        self.distributed_tracing = self._init_tracing()
        self.performance_profiler = self._init_profiler()
        self.security_manager = SecurityManager(project_name)
        
        # Multi-region support
        self.region_manager = RegionManager(project_name)
        self.disaster_recovery = DisasterRecoveryManager(project_name)
    
    async def execute_workflow(self, config: EnterpriseWorkflowConfig, 
                             input_data: Dict[str, Any],
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute workflow with enterprise cloud features"""
        
        # Security validation
        await self.security_manager.validate_request(input_data)
        
        # Distributed tracing
        trace_id = self._start_trace(config.name)
        
        # Performance profiling
        with self.performance_profiler.profile(f"workflow_{config.name}"):
            
            # Select optimal region
            optimal_region = await self.region_manager.select_optimal_region(input_data)
            
            # Execute with enterprise features
            result = await super().execute_workflow(config, input_data, progress_callback)
            
            # Add enterprise metadata
            result.update({
                "trace_id": trace_id,
                "execution_region": optimal_region,
                "security_level": "enterprise",
                "performance_profile": self.performance_profiler.get_profile(f"workflow_{config.name}")
            })
            
            return result
```

## Cloud Deployment Patterns by Level

### Level 1: Minimal Cloud Resources
```terraform
# infrastructure/terraform/level1/main.tf
# Minimal cloud deployment for Level 1

resource "google_cloud_run_service" "simple_agent" {
  name     = "${var.project_name}-agent"
  location = var.region

  template {
    spec {
      containers {
        image = var.agent_image
        env {
          name = "${upper(var.project_name)}_ENV"
          value = var.environment
        }
        resources {
          limits = {
            cpu    = "1000m"
            memory = "1Gi"
          }
        }
      }
    }
  }
}

# Minimal secret management
resource "google_secret_manager_secret" "openai_key" {
  secret_id = "${var.project_name}-openai-key"
  replication {
    automatic = true
  }
}
```

### Level 2: Standard Cloud Architecture
```terraform
# infrastructure/terraform/level2/main.tf
# Standard cloud deployment for Level 2

module "database" {
  source = "../modules/database"
  # Cloud SQL for workflow state
}

module "redis" {
  source = "../modules/redis"
  # Redis for context sharing
}

module "backend" {
  source = "../modules/cloud_run"
  # Multi-agent orchestrator
}

module "frontend" {
  source = "../modules/cloud_run"
  # Streamlit interface
}

module "monitoring" {
  source = "../modules/monitoring"
  # Basic monitoring and alerting
}
```

### Level 3-4: Enterprise Cloud Architecture
```terraform
# infrastructure/terraform/level3-4/main.tf
# Enterprise cloud deployment

module "vpc" {
  source = "../modules/vpc"
  # Private networking
}

module "gke_cluster" {
  source = "../modules/gke"
  # Kubernetes for complex workflows
}

module "databases" {
  source = "../modules/databases"
  # Multiple databases with replication
}

module "security" {
  source = "../modules/security"
  # Advanced security controls
}

module "monitoring" {
  source = "../modules/enterprise_monitoring"
  # Full observability stack
}

module "disaster_recovery" {
  source = "../modules/disaster_recovery"
  # Multi-region backup and recovery
}
```

## Environment Configuration Integration

### Framework-Level Environment Variables
```bash
# Level 1 Simple Systems
PROJECTNAME_ENV=development
PROJECTNAME_OPENAI_API_KEY=sk-...
PROJECTNAME_LOG_LEVEL=INFO

# Level 2 Standard Systems (adds coordination)
PROJECTNAME_DATABASE_URL=postgresql://...
PROJECTNAME_REDIS_URL=redis://...
PROJECTNAME_WORKFLOW_TIMEOUT=300

# Level 3 Complex Systems (adds enterprise features)
PROJECTNAME_CIRCUIT_BREAKER_ENABLED=true
PROJECTNAME_DISTRIBUTED_TRACING=true
PROJECTNAME_SECURITY_LEVEL=high

# Level 4 Production Systems (adds full enterprise)
PROJECTNAME_MULTI_REGION_ENABLED=true
PROJECTNAME_DISASTER_RECOVERY_ENABLED=true
PROJECTNAME_PERFORMANCE_MONITORING=advanced
PROJECTNAME_COMPLIANCE_MODE=enterprise
```

## Migration Path Between Levels

### Level 1 → Level 2 Cloud Migration
```bash
#!/bin/bash
# scripts/migrate-level1-to-level2.sh

echo "🔄 Migrating from Level 1 to Level 2 cloud deployment..."

# Add database and Redis to infrastructure
cp infrastructure/terraform/level2/database.tf infrastructure/terraform/
cp infrastructure/terraform/level2/redis.tf infrastructure/terraform/

# Update application configuration
sed -i 's/simple_agent/multi_agent_orchestrator/' backend/main.py

# Deploy additional resources
terraform plan -var-file="environments/${ENV}/terraform.tfvars"
terraform apply

echo "✅ Migration to Level 2 completed"
```

### Level 2 → Level 3 Cloud Migration
```bash
#!/bin/bash
# scripts/migrate-level2-to-level3.sh

echo "🔄 Migrating from Level 2 to Level 3 cloud deployment..."

# Add enterprise features
mkdir -p backend/core/enterprise
cp templates/level3/enterprise_orchestrator.py backend/core/enterprise/

# Add monitoring and security
cp infrastructure/terraform/level3/monitoring.tf infrastructure/terraform/
cp infrastructure/terraform/level3/security.tf infrastructure/terraform/

# Update environment variables
echo "PROJECTNAME_CIRCUIT_BREAKER_ENABLED=true" >> .env.template
echo "PROJECTNAME_DISTRIBUTED_TRACING=true" >> .env.template

terraform apply -var-file="environments/${ENV}/terraform.tfvars"

echo "✅ Migration to Level 3 completed"
```

## Testing Integration Across Levels

### Cloud-Native Testing Pipeline
```python
# tests/cloud/test_level_integration.py

class TestCloudLevelIntegration:
    """Test cloud integration across framework levels"""
    
    @pytest.mark.parametrize("level", ["1", "2", "3", "4"])
    async def test_level_cloud_deployment(self, level):
        """Test each level deploys correctly to cloud"""
        
        # Load level-specific configuration
        config = load_level_config(level)
        
        # Deploy to test environment
        deployment = await deploy_to_test_cloud(config)
        
        # Verify deployment
        assert deployment.status == "healthy"
        assert deployment.endpoints_responding()
        
        # Level-specific assertions
        if level == "1":
            assert deployment.has_simple_agent()
        elif level == "2":
            assert deployment.has_orchestrator()
            assert deployment.has_database()
        elif level in ["3", "4"]:
            assert deployment.has_enterprise_features()
            assert deployment.has_monitoring()
        
        # Cleanup
        await cleanup_test_deployment(deployment)
```

---

## Next Steps

- **Production Checklist**: [Production Deployment Guide](production_checklist.md)
- **Monitoring**: [Observability and Alerting](observability.md)
- **Security**: [Security Configuration](../06_reference/security_guidelines.md)

---

*This integration approach ensures that cloud deployment capabilities scale naturally with framework complexity while maintaining the simplicity and progressive enhancement that makes the framework accessible to all skill levels.*