# Level 4: Production Agent Systems

> 🏭 **Mission-Critical**: Deploy enterprise-grade agent systems with full observability, security, and scalability for production environments.

## Navigation
- **Previous**: [Level 3: Complex](level_3_complex.md)
- **Next**: [Reference Templates](../06_reference/templates.md)
- **Specialized**: [Financial Precision](../04_specialized/financial_precision.md) → [Testing Frameworks](../04_specialized/testing_frameworks.md)
- **Reference**: [Security Guidelines](../06_reference/security_guidelines.md) → [Troubleshooting](../06_reference/troubleshooting.md)

---

## Overview

Level 4 systems represent production-ready, mission-critical agent deployments with enterprise security, compliance, observability, and scalability. These implementations handle high-volume workloads with strict SLAs, comprehensive monitoring, and automated operations.

## Level 4 Characteristics

| Aspect | Level 4 Specification |
|--------|----------------------|
| **Agents** | 5+ specialized agents with auto-scaling |
| **Patterns** | All patterns + Production hardening |
| **Complexity** | Expert + Production operations |
| **Deployment** | Container orchestration, CI/CD, multi-environment |
| **Context** | Distributed state with persistence and replication |
| **Time to Production** | 2-4 weeks |

---

## Production Architecture

### Container-Based Deployment

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  orchestrator:
    build:
      context: .
      dockerfile: docker/Dockerfile.orchestrator
    image: agent-system/orchestrator:${VERSION}
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
      - TRACING_ENABLED=true
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
      - monitoring
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    networks:
      - agent-network
    volumes:
      - agent-logs:/app/logs
      - agent-data:/app/data

  web-interface:
    build:
      context: .
      dockerfile: docker/Dockerfile.web
    image: agent-system/web:${VERSION}
    environment:
      - ORCHESTRATOR_URL=http://orchestrator:8000
      - AUTH_ENABLED=true
      - SESSION_SECRET=${SESSION_SECRET}
    ports:
      - "8501:8501"
    depends_on:
      - orchestrator
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    networks:
      - agent-network

  redis:
    image: redis:7-alpine
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
    networks:
      - agent-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=${DATABASE_NAME}
      - POSTGRES_USER=${DATABASE_USER}
      - POSTGRES_PASSWORD=${DATABASE_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init:/docker-entrypoint-initdb.d
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    networks:
      - agent-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DATABASE_USER} -d ${DATABASE_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - agent-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - agent-network
    depends_on:
      - monitoring

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - agent-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - orchestrator
      - web-interface
    networks:
      - agent-network

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:
  agent-logs:
  agent-data:

networks:
  agent-network:
    driver: bridge
```

### Production Orchestrator

```python
# production_orchestrator.py
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import uuid
from datetime import datetime
import structlog
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from prometheus_client import Counter, Histogram, Gauge
import aioredis
import asyncpg

# Metrics
WORKFLOW_COUNTER = Counter('workflows_total', 'Total workflows executed', ['status', 'workflow_name'])
WORKFLOW_DURATION = Histogram('workflow_duration_seconds', 'Workflow execution duration')
ACTIVE_WORKFLOWS = Gauge('active_workflows', 'Number of active workflows')
AGENT_COUNTER = Counter('agent_executions_total', 'Total agent executions', ['agent_name', 'status'])

@dataclass
class ProductionConfig:
    """Production configuration with security and compliance"""
    environment: str = "production"
    log_level: str = "INFO"
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    security_enabled: bool = True
    audit_enabled: bool = True
    encryption_enabled: bool = True
    backup_enabled: bool = True
    monitoring_interval: int = 30
    health_check_interval: int = 10
    max_concurrent_workflows: int = 100
    workflow_timeout: int = 3600
    database_pool_size: int = 20
    redis_pool_size: int = 50

class ProductionOrchestrator:
    """Production-grade orchestrator with full observability and security"""
    
    def __init__(self, agents: Dict[str, Any], config: ProductionConfig):
        self.agents = agents
        self.config = config
        self.logger = self._setup_structured_logging()
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Production components
        self.security_manager = SecurityManager(config)
        self.audit_logger = AuditLogger(config)
        self.health_checker = HealthChecker(self.agents)
        self.performance_monitor = PerformanceMonitor()
        self.backup_manager = BackupManager(config)
        
        # Infrastructure
        self.redis_pool = None
        self.db_pool = None
        self.message_queue = None
        
        # Runtime state
        self.active_workflows = {}
        self.system_status = "initializing"
        
        # Initialize instrumentation
        if config.tracing_enabled:
            AsyncioInstrumentor().instrument()
    
    async def initialize(self):
        """Initialize production infrastructure"""
        with self.tracer.start_as_current_span("orchestrator_initialization"):
            try:
                self.logger.info("Initializing production orchestrator", 
                               environment=self.config.environment)
                
                # Initialize infrastructure connections
                await self._initialize_infrastructure()
                
                # Initialize security
                await self.security_manager.initialize()
                
                # Start health monitoring
                asyncio.create_task(self._health_monitoring_loop())
                
                # Start performance monitoring
                asyncio.create_task(self._performance_monitoring_loop())
                
                # Start backup scheduling
                if self.config.backup_enabled:
                    asyncio.create_task(self._backup_scheduling_loop())
                
                self.system_status = "healthy"
                self.logger.info("Production orchestrator initialized successfully")
                
            except Exception as e:
                self.system_status = "failed"
                self.logger.error("Failed to initialize orchestrator", error=str(e))
                raise
    
    async def execute_workflow(self, workflow_config: 'ProductionWorkflowConfig',
                             input_data: Dict[str, Any],
                             user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute workflow with full production features"""
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Increment active workflows metric
        ACTIVE_WORKFLOWS.inc()
        
        with self.tracer.start_as_current_span("workflow_execution",
                                             attributes={
                                                 "workflow.id": execution_id,
                                                 "workflow.name": workflow_config.workflow_name
                                             }) as span:
            
            try:
                # Security validation
                if self.config.security_enabled:
                    await self.security_manager.validate_workflow_request(
                        workflow_config, input_data, user_context
                    )
                
                # Audit log workflow start
                await self.audit_logger.log_workflow_start(
                    execution_id, workflow_config, user_context
                )
                
                # Rate limiting check
                if not await self._check_rate_limits(user_context):
                    raise Exception("Rate limit exceeded")
                
                # Resource availability check
                if not await self._check_resource_availability(workflow_config):
                    raise Exception("Insufficient resources available")
                
                # Initialize workflow context
                workflow_context = await self._create_secure_workflow_context(
                    execution_id, workflow_config, input_data
                )
                
                # Track active workflow
                self.active_workflows[execution_id] = {
                    "config": workflow_config,
                    "start_time": start_time,
                    "status": "running",
                    "user_context": user_context
                }
                
                # Execute workflow with monitoring
                result = await self._execute_monitored_workflow(
                    execution_id, workflow_config, workflow_context
                )
                
                # Update metrics
                execution_time = time.time() - start_time
                WORKFLOW_DURATION.observe(execution_time)
                WORKFLOW_COUNTER.labels(
                    status=result["status"],
                    workflow_name=workflow_config.workflow_name
                ).inc()
                
                # Audit log completion
                await self.audit_logger.log_workflow_completion(
                    execution_id, result, execution_time
                )
                
                # Add production metadata
                result.update({
                    "execution_environment": self.config.environment,
                    "security_validated": self.config.security_enabled,
                    "audit_logged": self.config.audit_enabled,
                    "compliance_data": await self._generate_compliance_data(execution_id)
                })
                
                span.set_attribute("workflow.status", result["status"])
                span.set_attribute("workflow.duration", execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                WORKFLOW_COUNTER.labels(
                    status="error",
                    workflow_name=workflow_config.workflow_name
                ).inc()
                
                # Audit log error
                await self.audit_logger.log_workflow_error(
                    execution_id, str(e), execution_time
                )
                
                # Security incident check
                if self.config.security_enabled:
                    await self.security_manager.handle_workflow_error(
                        execution_id, e, user_context
                    )
                
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                raise
                
            finally:
                # Cleanup
                ACTIVE_WORKFLOWS.dec()
                if execution_id in self.active_workflows:
                    del self.active_workflows[execution_id]
    
    async def _execute_monitored_workflow(self, execution_id: str,
                                        config: 'ProductionWorkflowConfig',
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with comprehensive monitoring"""
        
        stage_results = {}
        
        for stage in config.stages:
            stage_start_time = time.time()
            
            with self.tracer.start_as_current_span(f"stage_{stage.name}",
                                                 attributes={
                                                     "stage.name": stage.name,
                                                     "stage.agent": stage.agent_name
                                                 }) as stage_span:
                
                try:
                    # Monitor stage execution
                    result = await self._execute_monitored_stage(
                        stage, stage_results, context, execution_id
                    )
                    
                    stage_results[stage.name] = result
                    
                    # Record stage metrics
                    stage_duration = time.time() - stage_start_time
                    AGENT_COUNTER.labels(
                        agent_name=stage.agent_name,
                        status=result.get("status", "unknown")
                    ).inc()
                    
                    stage_span.set_attribute("stage.status", result.get("status", "unknown"))
                    stage_span.set_attribute("stage.duration", stage_duration)
                    
                except Exception as e:
                    # Record stage error
                    AGENT_COUNTER.labels(
                        agent_name=stage.agent_name,
                        status="error"
                    ).inc()
                    
                    stage_span.record_exception(e)
                    raise
        
        return {
            "execution_id": execution_id,
            "status": "success",
            "stage_results": stage_results,
            "execution_time": time.time() - context["start_time"]
        }
    
    async def _initialize_infrastructure(self):
        """Initialize production infrastructure connections"""
        
        # Redis connection pool
        self.redis_pool = aioredis.ConnectionPool.from_url(
            "redis://redis:6379",
            max_connections=self.config.redis_pool_size,
            health_check_interval=30
        )
        
        # PostgreSQL connection pool
        self.db_pool = await asyncpg.create_pool(
            "postgresql://user:password@postgres:5432/agentdb",
            min_size=5,
            max_size=self.config.database_pool_size,
            command_timeout=30
        )
        
        # Message queue for async processing
        self.message_queue = ProductionMessageQueue(self.redis_pool)
        
        self.logger.info("Infrastructure connections initialized")
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                health_status = await self.health_checker.check_system_health()
                
                if health_status["status"] != "healthy":
                    self.logger.warning("System health issues detected",
                                      health_status=health_status)
                    
                    # Trigger alerts if needed
                    await self._handle_health_issues(health_status)
                
            except Exception as e:
                self.logger.error("Health monitoring error", error=str(e))
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring"""
        while True:
            try:
                await asyncio.sleep(self.config.monitoring_interval)
                
                # Collect performance metrics
                performance_data = await self.performance_monitor.collect_metrics()
                
                # Check for performance issues
                issues = await self.performance_monitor.analyze_performance(performance_data)
                
                if issues:
                    self.logger.warning("Performance issues detected",
                                      issues=issues)
                    
                    # Auto-scaling decisions
                    await self._handle_performance_issues(issues)
                
            except Exception as e:
                self.logger.error("Performance monitoring error", error=str(e))
    
    def _setup_structured_logging(self) -> structlog.BoundLogger:
        """Setup structured logging for production"""
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.config.log_level.upper())
        )
        
        return structlog.get_logger()

class SecurityManager:
    """Production security management"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.jwt_secret = self._load_jwt_secret()
        self.encryption_key = self._load_encryption_key()
        self.rate_limiter = RateLimiter()
        self.intrusion_detector = IntrusionDetector()
    
    async def validate_workflow_request(self, workflow_config: Any,
                                      input_data: Dict[str, Any],
                                      user_context: Dict[str, Any]) -> bool:
        """Validate workflow request for security compliance"""
        
        # JWT token validation
        if user_context:
            if not await self._validate_jwt_token(user_context.get("token")):
                raise SecurityException("Invalid authentication token")
        
        # Input sanitization
        sanitized_input = await self._sanitize_input_data(input_data)
        
        # Authorization check
        if not await self._check_workflow_authorization(workflow_config, user_context):
            raise SecurityException("Insufficient permissions")
        
        # Rate limiting
        if not await self.rate_limiter.check_rate_limit(user_context):
            raise SecurityException("Rate limit exceeded")
        
        # Intrusion detection
        if await self.intrusion_detector.detect_suspicious_activity(user_context, input_data):
            raise SecurityException("Suspicious activity detected")
        
        return True
    
    async def _sanitize_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data to prevent injection attacks"""
        
        def sanitize_value(value):
            if isinstance(value, str):
                # Remove potentially dangerous characters
                import re
                return re.sub(r'[<>"\']', '', value)
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            return value
        
        return sanitize_value(input_data)

class AuditLogger:
    """Comprehensive audit logging for compliance"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.audit_db = None
        self.encryption = AuditEncryption(config)
    
    async def log_workflow_start(self, execution_id: str, 
                               workflow_config: Any,
                               user_context: Dict[str, Any]):
        """Log workflow initiation for audit trail"""
        
        audit_entry = {
            "event_type": "workflow_start",
            "execution_id": execution_id,
            "workflow_name": workflow_config.workflow_name,
            "user_id": user_context.get("user_id") if user_context else None,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": user_context.get("ip_address") if user_context else None,
            "user_agent": user_context.get("user_agent") if user_context else None,
            "workflow_config_hash": self._hash_config(workflow_config)
        }
        
        # Encrypt sensitive data
        encrypted_entry = await self.encryption.encrypt_audit_entry(audit_entry)
        
        # Store in audit database
        await self._store_audit_entry(encrypted_entry)
    
    async def log_workflow_completion(self, execution_id: str,
                                    result: Dict[str, Any],
                                    execution_time: float):
        """Log workflow completion"""
        
        audit_entry = {
            "event_type": "workflow_completion",
            "execution_id": execution_id,
            "status": result["status"],
            "execution_time": execution_time,
            "timestamp": datetime.utcnow().isoformat(),
            "result_hash": self._hash_result(result)
        }
        
        encrypted_entry = await self.encryption.encrypt_audit_entry(audit_entry)
        await self._store_audit_entry(encrypted_entry)

class HealthChecker:
    """Comprehensive system health monitoring"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.health_history = []
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        
        health_results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check agent health
        for agent_name, agent in self.agents.items():
            agent_health = await self._check_agent_health(agent_name, agent)
            health_results["components"][f"agent_{agent_name}"] = agent_health
            
            if agent_health["status"] != "healthy":
                health_results["status"] = "degraded"
        
        # Check infrastructure health
        infra_checks = [
            ("database", self._check_database_health),
            ("redis", self._check_redis_health),
            ("message_queue", self._check_message_queue_health)
        ]
        
        for component, check_func in infra_checks:
            component_health = await check_func()
            health_results["components"][component] = component_health
            
            if component_health["status"] != "healthy":
                health_results["status"] = "degraded"
        
        # Store health history
        self.health_history.append(health_results)
        if len(self.health_history) > 1000:  # Keep last 1000 entries
            self.health_history = self.health_history[-1000:]
        
        return health_results
    
    async def _check_agent_health(self, agent_name: str, agent: Any) -> Dict[str, Any]:
        """Check individual agent health"""
        
        try:
            # Test agent with health check task
            start_time = time.time()
            
            health_task = "Perform health check - respond with status"
            health_context = {"health_check": True, "timestamp": datetime.utcnow().isoformat()}
            
            result = await asyncio.wait_for(
                agent.execute(health_task, health_context),
                timeout=10.0
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if result.get("status") == "success" else "unhealthy",
                "response_time": response_time,
                "last_check": datetime.utcnow().isoformat(),
                "details": result
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "error": "Health check timeout",
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }

class PerformanceMonitor:
    """Advanced performance monitoring and optimization"""
    
    def __init__(self):
        self.performance_history = []
        self.baseline_metrics = None
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics"""
        
        import psutil
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict(),
                "process_count": len(psutil.pids())
            },
            "application": {
                "active_workflows": len(ACTIVE_WORKFLOWS._value._value),
                "total_workflows": WORKFLOW_COUNTER._value.sum(),
                "average_response_time": self._calculate_average_response_time(),
                "error_rate": self._calculate_error_rate(),
                "throughput": self._calculate_throughput()
            }
        }
        
        # Store metrics history
        self.performance_history.append(metrics)
        if len(self.performance_history) > 1440:  # Keep 24 hours of data (1 min intervals)
            self.performance_history = self.performance_history[-1440:]
        
        return metrics
    
    async def analyze_performance(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance and identify issues"""
        
        issues = []
        
        # CPU usage analysis
        if current_metrics["system"]["cpu_percent"] > 80:
            issues.append({
                "type": "high_cpu_usage",
                "severity": "warning" if current_metrics["system"]["cpu_percent"] < 90 else "critical",
                "value": current_metrics["system"]["cpu_percent"],
                "recommendation": "Consider horizontal scaling or workload optimization"
            })
        
        # Memory usage analysis
        if current_metrics["system"]["memory_percent"] > 85:
            issues.append({
                "type": "high_memory_usage",
                "severity": "warning" if current_metrics["system"]["memory_percent"] < 95 else "critical",
                "value": current_metrics["system"]["memory_percent"],
                "recommendation": "Review memory leaks or increase memory allocation"
            })
        
        # Error rate analysis
        error_rate = current_metrics["application"]["error_rate"]
        if error_rate > 0.05:  # 5% error rate threshold
            issues.append({
                "type": "high_error_rate",
                "severity": "warning" if error_rate < 0.1 else "critical",
                "value": error_rate,
                "recommendation": "Investigate error patterns and improve error handling"
            })
        
        # Response time analysis
        avg_response_time = current_metrics["application"]["average_response_time"]
        if avg_response_time > 30:  # 30 second threshold
            issues.append({
                "type": "slow_response_time",
                "severity": "warning" if avg_response_time < 60 else "critical",
                "value": avg_response_time,
                "recommendation": "Optimize workflow stages or increase compute resources"
            })
        
        return issues

class BackupManager:
    """Automated backup and disaster recovery"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.backup_schedule = BackupSchedule()
        self.storage_backend = S3BackupStorage()
    
    async def create_system_backup(self) -> Dict[str, Any]:
        """Create comprehensive system backup"""
        
        backup_id = f"backup_{int(time.time())}"
        backup_start = time.time()
        
        try:
            # Database backup
            db_backup_path = await self._backup_database()
            
            # Configuration backup
            config_backup_path = await self._backup_configurations()
            
            # Logs backup
            logs_backup_path = await self._backup_logs()
            
            # Upload to storage
            backup_manifest = {
                "backup_id": backup_id,
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "database": db_backup_path,
                    "configurations": config_backup_path,
                    "logs": logs_backup_path
                },
                "backup_size": await self._calculate_backup_size([
                    db_backup_path, config_backup_path, logs_backup_path
                ]),
                "duration": time.time() - backup_start
            }
            
            await self.storage_backend.upload_backup(backup_manifest)
            
            return {
                "status": "success",
                "backup_id": backup_id,
                "manifest": backup_manifest
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "backup_id": backup_id,
                "error": str(e),
                "duration": time.time() - backup_start
            }

# Kubernetes Deployment Configuration
kubernetes_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  labels:
    app: agent-orchestrator
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    metadata:
      labels:
        app: agent-orchestrator
        version: v1.0.0
    spec:
      containers:
      - name: orchestrator
        image: agent-system/orchestrator:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: agent-config
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: agent-orchestrator-service
spec:
  selector:
    app: agent-orchestrator
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-orchestrator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-orchestrator
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
```

### CI/CD Pipeline

```yaml
# .github/workflows/production-deploy.yml
name: Production Deployment

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
        pip install -r requirements-production.txt
    
    - name: Run security scan
      run: |
        bandit -r . -f json -o security-report.json
        safety check --json --output safety-report.json
    
    - name: Run linting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Run type checking
      run: mypy .
    
    - name: Run unit tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
        ENVIRONMENT: test
      run: |
        pytest tests/unit/ -v --cov=. --cov-report=xml
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
        ENVIRONMENT: test
      run: |
        pytest tests/integration/ -v
    
    - name: Run end-to-end tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
        ENVIRONMENT: test
      run: |
        pytest tests/e2e/ -v
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          security-report.json
          safety-report.json

  build:
    needs: test
    runs-on: ubuntu-latest
    
    outputs:
      image: ${{ steps.image.outputs.image }}
      digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix=sha-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./docker/Dockerfile.production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Output image
      id: image
      run: |
        echo "image=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}" >> $GITHUB_OUTPUT

  security-scan:
    needs: build
    runs-on: ubuntu-latest
    
    steps:
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ needs.build.outputs.image }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  deploy-staging:
    needs: [test, build, security-scan]
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # kubectl apply -f k8s/staging/
        # helm upgrade --install agent-system ./helm-chart --namespace staging

  deploy-production:
    needs: [test, build, security-scan, deploy-staging]
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # kubectl apply -f k8s/production/
        # helm upgrade --install agent-system ./helm-chart --namespace production
    
    - name: Run smoke tests
      run: |
        echo "Running production smoke tests"
        # curl -f https://api.agentsystem.com/health
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Monitoring and Alerting

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'agent-orchestrator'
    static_configs:
      - targets: ['orchestrator:8000']
    metrics_path: /metrics
    scrape_interval: 5s
    
  - job_name: 'agent-web'
    static_configs:
      - targets: ['web-interface:8501']
    metrics_path: /metrics
    scrape_interval: 15s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

```yaml
# monitoring/alerts.yml
groups:
- name: agent-system-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(workflows_total{status="error"}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
      description: "Error rate is {{ $value }} errors per second"
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(workflow_duration_seconds_bucket[5m])) > 60
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High response time detected
      description: "95th percentile response time is {{ $value }} seconds"
  
  - alert: SystemResourcesHigh
    expr: (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) < 0.1
    for: 3m
    labels:
      severity: critical
    annotations:
      summary: Low memory available
      description: "Only {{ $value | humanizePercentage }} memory available"
  
  - alert: DatabaseConnectionsHigh
    expr: pg_stat_database_numbackends > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High database connections
      description: "Database has {{ $value }} active connections"
  
  - alert: AgentHealthCheckFailed
    expr: up{job="agent-orchestrator"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Agent orchestrator is down
      description: "Agent orchestrator health check failed"
```

---

## Best Practices for Level 4

### Security
1. **Defense in Depth**: Multiple layers of security controls
2. **Zero Trust Architecture**: Verify everything, trust nothing
3. **Encryption Everywhere**: Data at rest, in transit, and in processing
4. **Regular Security Audits**: Automated and manual security assessments
5. **Incident Response Plan**: Documented procedures for security incidents

### Compliance
1. **Data Governance**: Clear data handling and retention policies
2. **Audit Trails**: Comprehensive logging for compliance requirements
3. **Access Controls**: Role-based access with principle of least privilege
4. **Data Privacy**: GDPR, CCPA, and other privacy regulation compliance
5. **Regular Compliance Reviews**: Automated compliance checking

### Operations
1. **Infrastructure as Code**: All infrastructure defined in version control
2. **Automated Deployments**: Zero-downtime deployments with rollback capability
3. **Comprehensive Monitoring**: Full observability stack with alerting
4. **Disaster Recovery**: Automated backup and recovery procedures
5. **Capacity Planning**: Proactive scaling based on usage patterns

### Performance
1. **Auto-scaling**: Horizontal and vertical scaling based on metrics
2. **Performance Testing**: Regular load testing and performance optimization
3. **Caching Strategy**: Multi-level caching for optimal performance
4. **Database Optimization**: Query optimization and connection pooling
5. **CDN Integration**: Global content delivery for web interfaces

---

## Migration to Level 4

### From Level 3 to Level 4
1. **Security Hardening**: Implement comprehensive security controls
2. **Monitoring Infrastructure**: Deploy full observability stack
3. **Container Orchestration**: Move to Kubernetes or similar platform
4. **CI/CD Pipeline**: Implement automated testing and deployment
5. **Compliance Framework**: Add audit logging and compliance controls
6. **Disaster Recovery**: Implement backup and recovery systems
7. **Performance Optimization**: Add auto-scaling and optimization
8. **Documentation**: Complete operational runbooks and procedures

### Production Readiness Checklist
- [ ] Security audit completed and issues resolved
- [ ] Comprehensive test suite with >90% coverage
- [ ] Performance testing and optimization completed
- [ ] Monitoring and alerting configured
- [ ] Backup and disaster recovery tested
- [ ] Documentation complete and up-to-date
- [ ] Incident response procedures defined
- [ ] Compliance requirements verified
- [ ] Production deployment pipeline tested
- [ ] Team training completed

---

## Next Steps

- **Reference Materials**: [Implementation Templates](../06_reference/templates.md) for production deployment
- **Security Guidelines**: [Security Best Practices](../06_reference/security_guidelines.md)
- **Operational Procedures**: [Operations Manual](../06_reference/troubleshooting.md)

---

*Level 4 systems represent the pinnacle of agent system implementation, providing enterprise-grade reliability, security, and scalability for mission-critical production environments.*