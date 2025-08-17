# Dual Interface Architecture Design

> 🌐 **Multi-Modal Access**: Design systems that serve both technical users (CLI/API) and business users (Web) from a single codebase.

## Navigation
- **Previous**: [Architecture Patterns](../02_architecture_patterns.md)
- **Next**: [Web Integration](web_integration.md) → [Progress Tracking](progress_tracking.md)
- **Implementation**: [Level 2: Standard](../05_implementation_levels/level_2_standard.md) → [Level 3: Complex](../05_implementation_levels/level_3_complex.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Decision Matrices](../06_reference/decision_matrices.md)

---

## Overview

Modern agent systems require multiple interaction modes: programmatic (CLI/API) for automation and human-friendly (Web) for business users. This section provides proven patterns for building systems that seamlessly support both paradigms from a single codebase.

## Core Design Principles

1. **Shared Orchestration Layer**: Single agent orchestrator serves all interfaces
2. **Interface-Specific Wrappers**: Thin adapters handle interface-specific concerns
3. **Common Data Models**: Unified data structures across all interfaces
4. **Progressive Enhancement**: CLI-first design enhanced with web capabilities
5. **Async Compatibility**: Web interfaces handle long-running operations gracefully

---

## Architecture Pattern

```
User Request
    ↓
Interface Router (CLI/Web/API)
    ↓
Common Orchestrator
    ↓
Agent Execution Engine
    ↓
Interface-Specific Formatter
    ↓
Response (CLI/Web/API)
```

## Implementation Pattern

### 1. Core Orchestrator (Interface-Agnostic)

```python
class UniversalOrchestrator:
    """Core orchestrator that serves all interfaces"""
    
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.context_manager = SharedContextManager()
        self.progress_manager = ProgressManager()
    
    async def execute_workflow(self, request: WorkflowRequest, 
                             progress_callback: Optional[Callable] = None):
        """Execute workflow with optional progress tracking"""
        
        # Validate request
        validated_request = self.validate_request(request)
        
        # Create execution context
        context_id = f"exec_{int(time.time() * 1000)}"
        execution_context = ExecutionContext(context_id, validated_request)
        
        # Initialize progress tracking
        if progress_callback:
            progress_tracker = self.progress_manager.create_tracker(
                context_id, len(validated_request.stages)
            )
            progress_tracker.add_callback(progress_callback)
        else:
            progress_tracker = None
        
        # Execute workflow stages
        results = []
        for i, stage in enumerate(validated_request.stages):
            if progress_tracker:
                progress_tracker.start_stage(stage.name)
            
            # Get appropriate agent
            agent = self.agents[stage.agent_name]
            
            # Execute with context
            stage_result = await agent.execute(
                stage.task, 
                execution_context.get_context_for_stage(stage),
                progress_callback=progress_tracker.stage_callback if progress_tracker else None
            )
            
            results.append(stage_result)
            execution_context.add_stage_result(stage.name, stage_result)
            
            if progress_tracker:
                progress_tracker.complete_stage(stage.name)
        
        # Synthesize final result
        final_result = await self.synthesize_results(results, execution_context)
        
        if progress_tracker:
            progress_tracker.complete_operation("Workflow completed successfully")
        
        return final_result
```

### 2. Interface-Specific Wrappers

```python
class WebOrchestrator:
    """Streamlit-compatible wrapper"""
    
    def __init__(self, core_orchestrator: UniversalOrchestrator):
        self.core = core_orchestrator
        self.session_manager = WebSessionManager()
    
    def generate_report(self, params: dict, progress_bar=None, status_container=None):
        """Web-specific method with Streamlit progress"""
        
        # Create web-specific progress callback
        callback = StreamlitProgressCallback(progress_bar, status_container)
        
        # Convert web params to workflow request
        request = self.convert_web_params_to_request(params)
        
        # Execute with core orchestrator
        try:
            result = asyncio.run(self.core.execute_workflow(request, callback.update))
            
            # Format for web display
            formatted_result = self.format_for_web_display(result)
            
            # Store in session
            self.session_manager.store_result(params.get('session_id'), formatted_result)
            
            return formatted_result
            
        except Exception as e:
            callback.error(str(e))
            raise WebOrchestrationError(f"Web workflow failed: {str(e)}")

class CLIOrchestrator:
    """CLI-compatible wrapper"""
    
    def __init__(self, core_orchestrator: UniversalOrchestrator):
        self.core = core_orchestrator
        self.output_formatter = CLIOutputFormatter()
    
    async def run_workflow(self, params: dict, verbose: bool = True):
        """CLI-specific method with text progress"""
        
        # Create CLI-specific progress callback
        callback = CLIProgressCallback(verbose)
        
        # Convert CLI params to workflow request
        request = self.convert_cli_params_to_request(params)
        
        # Execute with core orchestrator
        try:
            result = await self.core.execute_workflow(request, callback.update)
            
            # Format for CLI output
            formatted_result = self.output_formatter.format_cli_output(result)
            
            return formatted_result
            
        except Exception as e:
            callback.error(str(e))
            raise CLIOrchestrationError(f"CLI workflow failed: {str(e)}")

class APIOrchestrator:
    """FastAPI-compatible wrapper"""
    
    def __init__(self, core_orchestrator: UniversalOrchestrator):
        self.core = core_orchestrator
        self.webhook_manager = WebhookManager()
    
    async def process_api_request(self, request: APIRequest):
        """API-specific method with webhook progress"""
        
        # Create API-specific progress callback
        callback = APIProgressCallback(
            webhook_url=request.progress_webhook_url,
            operation_id=request.operation_id
        )
        
        # Convert API request to workflow request
        workflow_request = self.convert_api_to_workflow_request(request)
        
        # Execute with core orchestrator
        try:
            result = await self.core.execute_workflow(workflow_request, callback.update)
            
            # Format for API response
            api_response = APIResponse(
                operation_id=request.operation_id,
                status="completed",
                result=result,
                execution_time=result.get('execution_time', 0)
            )
            
            return api_response
            
        except Exception as e:
            callback.error(str(e))
            return APIResponse(
                operation_id=request.operation_id,
                status="failed",
                error=str(e)
            )
```

### 3. Interface Coordination Patterns

**Pattern 1: Shared State Management**

```python
class StateManager:
    """Manages state across interfaces"""
    
    def __init__(self, storage_backend="redis"):
        self.storage = self._init_storage(storage_backend)
        self.active_operations = {}
        self.results_cache = {}
    
    def track_operation(self, operation_id: str, interface_type: str, metadata: dict):
        """Track operation across interfaces"""
        operation_data = {
            "interface": interface_type,
            "start_time": time.time(),
            "status": "running",
            "metadata": metadata
        }
        
        self.active_operations[operation_id] = operation_data
        self.storage.store_operation_state(operation_id, operation_data)
    
    def get_operation_status(self, operation_id: str) -> Optional[dict]:
        """Get status from any interface"""
        if operation_id in self.active_operations:
            return self.active_operations[operation_id]
        
        return self.storage.get_operation_state(operation_id)
    
    def complete_operation(self, operation_id: str, result: dict):
        """Mark operation as completed"""
        if operation_id in self.active_operations:
            self.active_operations[operation_id]["status"] = "completed"
            self.active_operations[operation_id]["end_time"] = time.time()
            self.results_cache[operation_id] = result
            
            # Persist to storage
            self.storage.store_operation_result(operation_id, result)
```

**Pattern 2: Progress Callback Adaptation**

```python
class ProgressCallbackAdapter:
    """Adapts progress callbacks for different interfaces"""
    
    @staticmethod
    def create_callback(interface_type: str, **kwargs):
        if interface_type == "streamlit":
            return StreamlitProgressCallback(
                kwargs.get("progress_bar"),
                kwargs.get("status_container")
            )
        elif interface_type == "cli":
            return CLIProgressCallback(kwargs.get("verbose", True))
        elif interface_type == "api":
            return APIProgressCallback(
                kwargs.get("webhook_url"),
                kwargs.get("operation_id")
            )
        else:
            return NoOpProgressCallback()

class StreamlitProgressCallback:
    def __init__(self, progress_bar=None, status_container=None):
        self.progress_bar = progress_bar
        self.status_container = status_container
        self.start_time = time.time()
    
    def update(self, stage: str, progress: int, message: str = ""):
        if self.progress_bar:
            self.progress_bar.progress(progress / 100.0)
        
        if self.status_container:
            elapsed = time.time() - self.start_time
            status_text = f"""
            **{stage}** - {message}
            
            ⏱️ Elapsed: {elapsed:.1f}s | 📊 Progress: {progress}%
            """
            self.status_container.markdown(status_text)
    
    def error(self, error_message: str):
        if self.status_container:
            self.status_container.error(f"❌ Error: {error_message}")

class CLIProgressCallback:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = time.time()
    
    def update(self, stage: str, progress: int, message: str = ""):
        elapsed = time.time() - self.start_time
        
        if self.verbose:
            progress_bar = "█" * (progress // 10) + "░" * (10 - (progress // 10))
            print(f"[{progress_bar}] {progress:3d}% | {stage}: {message} | Elapsed: {elapsed:.1f}s")
        else:
            print(f"[{progress:3d}%] {stage}")
    
    def error(self, error_message: str):
        print(f"❌ Error: {error_message}")

class APIProgressCallback:
    def __init__(self, webhook_url: Optional[str] = None, operation_id: str = None):
        self.webhook_url = webhook_url
        self.operation_id = operation_id
        self.progress_log = []
    
    async def update(self, stage: str, progress: int, message: str = ""):
        update_data = {
            "operation_id": self.operation_id,
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": time.time()
        }
        
        self.progress_log.append(update_data)
        
        if self.webhook_url:
            await self._send_webhook(update_data)
    
    async def _send_webhook(self, update_data: dict):
        """Send progress update via webhook"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    self.webhook_url,
                    json=update_data,
                    timeout=aiohttp.ClientTimeout(total=5)
                )
        except Exception as e:
            print(f"Webhook delivery failed: {e}")
```

### 4. Configuration Management

```python
class InterfaceConfig:
    """Interface-specific configuration"""
    
    CLI_CONFIG = {
        "output_format": "text",
        "progress_style": "simple",
        "error_format": "structured",
        "color_output": True,
        "verbose_by_default": False
    }
    
    WEB_CONFIG = {
        "output_format": "rich",
        "progress_style": "visual",
        "error_format": "user_friendly",
        "session_timeout": 3600,
        "auto_refresh": True,
        "theme": "light"
    }
    
    API_CONFIG = {
        "output_format": "json",
        "progress_style": "webhook",
        "error_format": "structured",
        "rate_limit": 100,
        "timeout": 300,
        "async_by_default": True
    }
    
    @classmethod
    def get_config(cls, interface_type: str) -> dict:
        config_map = {
            "cli": cls.CLI_CONFIG,
            "web": cls.WEB_CONFIG,
            "api": cls.API_CONFIG
        }
        return config_map.get(interface_type, cls.CLI_CONFIG)
```

---

## File Structure for Dual Interface

```
dual_interface_system/
├── main.py                    # Entry point router
├── core/
│   ├── orchestrator.py       # Universal orchestrator
│   ├── agents/              # Interface-agnostic agents
│   └── models.py            # Shared data models
├── interfaces/
│   ├── cli/
│   │   ├── cli_wrapper.py   # CLI-specific wrapper
│   │   ├── commands.py      # CLI commands
│   │   └── formatters.py    # CLI output formatting
│   ├── web/
│   │   ├── web_wrapper.py   # Streamlit wrapper
│   │   ├── pages/           # Streamlit pages
│   │   ├── components/      # Reusable web components
│   │   └── session.py       # Session management
│   └── api/
│       ├── api_wrapper.py   # FastAPI wrapper
│       ├── endpoints.py     # REST endpoints
│       ├── models.py        # API request/response models
│       └── webhooks.py      # Webhook handling
├── shared/
│   ├── state_manager.py     # Cross-interface state
│   ├── progress_callbacks.py # Progress adaptation
│   └── config.py           # Interface configurations
└── config/
    └── interface_config.py  # Interface configurations
```

---

## Implementation Examples

### Entry Point Router

```python
# main.py
import asyncio
import sys
from core.orchestrator import UniversalOrchestrator
from interfaces.cli.cli_wrapper import CLIOrchestrator
from interfaces.web.web_wrapper import WebOrchestrator
from interfaces.api.api_wrapper import APIOrchestrator

def main():
    # Initialize core orchestrator
    core_orchestrator = UniversalOrchestrator(agents=load_agents())
    
    # Determine interface type
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Launch web interface
        web_orchestrator = WebOrchestrator(core_orchestrator)
        web_orchestrator.launch()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        # Launch API interface
        api_orchestrator = APIOrchestrator(core_orchestrator)
        api_orchestrator.run_server()
    
    else:
        # Default to CLI interface
        cli_orchestrator = CLIOrchestrator(core_orchestrator)
        asyncio.run(cli_orchestrator.run_cli())

if __name__ == "__main__":
    main()
```

### Usage Examples

**CLI Usage:**
```bash
# Standard CLI usage
python main.py --task "process_data" --input data.csv

# Web interface
python main.py web

# API server
python main.py api --port 8000
```

**Web Usage:**
```python
# streamlit run main.py web
# Accessible at http://localhost:8501
```

**API Usage:**
```bash
# Start API server
python main.py api

# Make API request
curl -X POST http://localhost:8000/api/workflow \
  -H "Content-Type: application/json" \
  -d '{"task": "process_data", "params": {...}}'
```

---

## Practical API Integration Examples

### Production API Implementation

**Complete FastAPI Integration with Authentication and Monitoring:**

```python
# production_api.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import logging
import time
import uuid
from datetime import datetime

# Enhanced Request/Response Models
class EnhancedAgentRequest(BaseModel):
    task: str = Field(..., description="Task description for the agent")
    data: Dict[str, Any] = Field(default_factory=dict, description="Input data for processing")
    agent_type: str = Field("general", description="Type of agent to use")
    priority: str = Field("normal", description="Priority level: low, normal, high")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DetailedAgentResponse(BaseModel):
    operation_id: str
    status: str  # "queued", "processing", "completed", "failed", "timeout"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = Field(0, description="Progress percentage (0-100)")
    processing_time: Optional[float] = None
    queue_position: Optional[int] = None
    estimated_completion: Optional[str] = None
    agent_metadata: Dict[str, Any] = Field(default_factory=dict)

class ProductionAPISystem:
    """Production-ready API system with comprehensive features"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Agent System API v2.0",
            description="Production Agent Orchestration API with comprehensive monitoring and management",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_tags=[
                {"name": "agents", "description": "Agent execution operations"},
                {"name": "monitoring", "description": "System monitoring and health"},
                {"name": "management", "description": "System management operations"}
            ]
        )
        
        # Core components
        self.orchestrator = UniversalOrchestrator()
        self.operation_queue = asyncio.Queue(maxsize=100)
        self.active_operations = {}
        self.operation_history = {}
        
        # Monitoring components
        self.metrics_collector = MetricsCollector()
        self.rate_limiter = RateLimiter()
        
        self.setup_middleware()
        self.setup_routes()
        self.start_background_workers()
    
    def setup_middleware(self):
        """Setup comprehensive middleware stack"""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Log request
            logging.info(f"API Request: {request.method} {request.url}")
            
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logging.info(f"API Response: {response.status_code} in {process_time:.3f}s")
            
            return response
    
    def setup_routes(self):
        """Setup comprehensive API routes"""
        
        # Agent execution endpoints
        @self.app.post("/v2/agents/execute", 
                      response_model=DetailedAgentResponse,
                      tags=["agents"],
                      summary="Execute agent task",
                      description="Submit a task for agent processing with optional callbacks")
        async def execute_agent_v2(
            request: EnhancedAgentRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
            client_request: Request = None
        ):
            # Rate limiting
            user_id = await self.validate_and_get_user(credentials.credentials)
            if not await self.rate_limiter.check_rate_limit(user_id, client_request.client.host):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            operation_id = str(uuid.uuid4())
            
            # Validate request
            await self._validate_agent_request(request, user_id)
            
            # Queue or execute immediately based on priority and system load
            current_load = len(self.active_operations)
            
            if request.priority == "high" or current_load < 5:
                # Execute immediately
                background_tasks.add_task(
                    self._execute_operation,
                    operation_id, request, user_id
                )
                
                return DetailedAgentResponse(
                    operation_id=operation_id,
                    status="processing",
                    progress=0,
                    estimated_completion=self._estimate_completion_time(request),
                    agent_metadata={"execution_mode": "immediate"}
                )
            else:
                # Queue for later execution
                await self.operation_queue.put({
                    "operation_id": operation_id,
                    "request": request,
                    "user_id": user_id,
                    "queued_at": datetime.utcnow()
                })
                
                return DetailedAgentResponse(
                    operation_id=operation_id,
                    status="queued",
                    queue_position=self.operation_queue.qsize(),
                    estimated_completion=self._estimate_queue_completion_time(),
                    agent_metadata={"execution_mode": "queued"}
                )
        
        @self.app.get("/v2/operations/{operation_id}/status", 
                     response_model=DetailedAgentResponse,
                     tags=["agents"],
                     summary="Get operation status")
        async def get_operation_status_v2(
            operation_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            user_id = await self.validate_and_get_user(credentials.credentials)
            
            # Check active operations
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]
                if operation["user_id"] != user_id:
                    raise HTTPException(status_code=403, detail="Access denied")
                
                return DetailedAgentResponse(**operation)
            
            # Check operation history
            if operation_id in self.operation_history:
                operation = self.operation_history[operation_id]
                if operation["user_id"] != user_id:
                    raise HTTPException(status_code=403, detail="Access denied")
                
                return DetailedAgentResponse(**operation)
            
            raise HTTPException(status_code=404, detail="Operation not found")
        
        @self.app.delete("/v2/operations/{operation_id}",
                        tags=["agents"],
                        summary="Cancel operation")
        async def cancel_operation(
            operation_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            user_id = await self.validate_and_get_user(credentials.credentials)
            
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]
                if operation["user_id"] != user_id:
                    raise HTTPException(status_code=403, detail="Access denied")
                
                # Mark for cancellation
                operation["status"] = "cancelled"
                operation["cancel_requested"] = True
                
                return {"message": "Operation cancellation requested"}
            
            raise HTTPException(status_code=404, detail="Operation not found or already completed")
        
        # System monitoring endpoints
        @self.app.get("/v2/health/detailed",
                     tags=["monitoring"],
                     summary="Detailed health check")
        async def detailed_health_check():
            """Comprehensive health check with system metrics"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": {
                    "active_operations": len(self.active_operations),
                    "queued_operations": self.operation_queue.qsize(),
                    "total_operations_today": self.metrics_collector.get_daily_count(),
                    "average_response_time": self.metrics_collector.get_avg_response_time(),
                    "error_rate": self.metrics_collector.get_error_rate(),
                    "uptime_seconds": self.metrics_collector.get_uptime()
                },
                "agent_status": {
                    "available_agents": list(self.orchestrator.agents.keys()),
                    "agent_health": await self._check_agent_health()
                },
                "api_metrics": {
                    "requests_per_minute": self.metrics_collector.get_rpm(),
                    "rate_limit_hits": self.rate_limiter.get_hits_count(),
                    "active_connections": self._get_active_connections()
                }
            }
        
        @self.app.get("/v2/metrics/prometheus",
                     tags=["monitoring"],
                     summary="Prometheus metrics endpoint")
        async def prometheus_metrics():
            """Prometheus-compatible metrics endpoint"""
            return self.metrics_collector.generate_prometheus_metrics()
        
        # Management endpoints
        @self.app.get("/v2/admin/operations",
                     tags=["management"],
                     summary="List all operations (admin only)")
        async def list_operations(
            status: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 50,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            admin_user = await self.validate_admin_user(credentials.credentials)
            
            operations = []
            
            # Get active operations
            for op_id, op_data in self.active_operations.items():
                if status is None or op_data["status"] == status:
                    if user_id is None or op_data["user_id"] == user_id:
                        operations.append({
                            "operation_id": op_id,
                            **op_data
                        })
            
            return {
                "operations": operations[:limit],
                "total_count": len(operations),
                "filters_applied": {"status": status, "user_id": user_id}
            }
    
    async def _execute_operation(self, operation_id: str, 
                               request: EnhancedAgentRequest, user_id: str):
        """Execute operation with comprehensive monitoring"""
        start_time = time.time()
        
        try:
            # Update operation status
            self.active_operations[operation_id] = {
                "operation_id": operation_id,
                "status": "processing",
                "user_id": user_id,
                "progress": 0,
                "started_at": datetime.utcnow().isoformat(),
                "agent_metadata": {"request": request.dict()}
            }
            
            # Create progress callback
            def progress_callback(progress: int, message: str = ""):
                if operation_id in self.active_operations:
                    self.active_operations[operation_id]["progress"] = progress
                    self.active_operations[operation_id]["status_message"] = message
            
            # Execute with timeout
            timeout = request.timeout or 300  # 5 minutes default
            
            result = await asyncio.wait_for(
                self.orchestrator.execute_workflow(
                    request.task,
                    request.data,
                    interface_mode="api",
                    progress_callback=progress_callback
                ),
                timeout=timeout
            )
            
            processing_time = time.time() - start_time
            
            # Update with completion
            completion_data = {
                "status": "completed",
                "result": result,
                "progress": 100,
                "processing_time": processing_time,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            self.active_operations[operation_id].update(completion_data)
            
            # Send webhook if requested
            if request.callback_url:
                await self._send_webhook(request.callback_url, operation_id, "completed", result)
            
            # Move to history after delay
            await self._schedule_history_move(operation_id)
            
        except asyncio.TimeoutError:
            self.active_operations[operation_id].update({
                "status": "timeout",
                "error": f"Operation timed out after {timeout} seconds",
                "processing_time": time.time() - start_time
            })
            
        except Exception as e:
            self.active_operations[operation_id].update({
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            })
            
            if request.callback_url:
                await self._send_webhook(request.callback_url, operation_id, "failed", {"error": str(e)})
    
    async def validate_and_get_user(self, api_key: str) -> str:
        """Enhanced API key validation with user information"""
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        # Validate format
        if not api_key.startswith(("sk-", "ak-")):
            raise HTTPException(status_code=401, detail="Invalid API key format")
        
        # Mock validation - implement proper database/cache validation
        if api_key.startswith("sk-test-"):
            return f"user_{api_key[-8:]}"
        
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    async def validate_admin_user(self, api_key: str) -> str:
        """Validate admin API key"""
        user_id = await self.validate_and_get_user(api_key)
        
        # Check admin privileges
        if not api_key.startswith("sk-admin-"):
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        return user_id

# Usage with proper startup
def create_production_api():
    """Create production API with all features enabled"""
    api_system = ProductionAPISystem()
    
    # Add agents
    api_system.orchestrator.add_agents([
        DataValidatorAgent(),
        AnalyticsAgent(),
        ReportGeneratorAgent()
    ])
    
    return api_system.app

# Run with production settings
if __name__ == "__main__":
    import uvicorn
    
    app = create_production_api()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        access_log=True,
        log_level="info"
    )
```

### Client SDK Example

**Python Client SDK for easy integration:**

```python
# agent_api_client.py
import aiohttp
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class AgentAPIClient:
    """Python client for Agent System API"""
    
    base_url: str
    api_key: str
    timeout: int = 300
    
    def __post_init__(self):
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def execute_task(self, task: str, data: Dict[str, Any] = None,
                          agent_type: str = "general", priority: str = "normal",
                          callback_url: str = None, wait_for_completion: bool = True):
        """Execute a task and optionally wait for completion"""
        
        request_data = {
            "task": task,
            "data": data or {},
            "agent_type": agent_type,
            "priority": priority,
            "callback_url": callback_url
        }
        
        # Submit task
        async with self.session.post(
            f"{self.base_url}/v2/agents/execute",
            json=request_data
        ) as response:
            
            if response.status != 200:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data}")
            
            result_data = await response.json()
            operation_id = result_data["operation_id"]
        
        if not wait_for_completion:
            return operation_id
        
        # Poll for completion
        return await self.wait_for_completion(operation_id)
    
    async def wait_for_completion(self, operation_id: str, 
                                poll_interval: int = 2) -> Dict[str, Any]:
        """Wait for operation to complete with polling"""
        
        start_time = time.time()
        
        while True:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Operation {operation_id} timed out")
            
            # Get status
            async with self.session.get(
                f"{self.base_url}/v2/operations/{operation_id}/status"
            ) as response:
                
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"Status check failed: {error_data}")
                
                status_data = await response.json()
                
                if status_data["status"] == "completed":
                    return status_data["result"]
                elif status_data["status"] == "failed":
                    raise Exception(f"Operation failed: {status_data.get('error')}")
                elif status_data["status"] in ["timeout", "cancelled"]:
                    raise Exception(f"Operation {status_data['status']}")
            
            await asyncio.sleep(poll_interval)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get detailed system health information"""
        async with self.session.get(
            f"{self.base_url}/v2/health/detailed"
        ) as response:
            return await response.json()

# Usage examples
async def example_usage():
    """Example usage of the API client"""
    
    async with AgentAPIClient(
        base_url="http://localhost:8000",
        api_key="sk-test-12345678"
    ) as client:
        
        # Simple task execution
        result = await client.execute_task(
            task="Validate this financial data",
            data={"revenue": "1000.50", "currency": "USD"},
            agent_type="financial_validator"
        )
        print("Task result:", result)
        
        # High priority task
        urgent_result = await client.execute_task(
            task="Process urgent transaction",
            data={"transaction_id": "tx_12345"},
            priority="high"
        )
        print("Urgent task result:", urgent_result)
        
        # Async task with manual polling
        operation_id = await client.execute_task(
            task="Generate comprehensive report",
            data={"report_type": "monthly"},
            wait_for_completion=False
        )
        
        print(f"Started operation: {operation_id}")
        result = await client.wait_for_completion(operation_id)
        print("Report generated:", result)

if __name__ == "__main__":
    asyncio.run(example_usage())
```

---

## Best Practices

1. **CLI-First Design**: Start with CLI interface, then enhance for web
2. **Async Everywhere**: Use async/await throughout for consistency
3. **Progress Transparency**: Always provide progress feedback across interfaces
4. **Error Consistency**: Maintain consistent error handling across interfaces
5. **State Isolation**: Keep interface-specific state separate from core logic
6. **Configuration Flexibility**: Allow interface-specific customization
7. **Testing Strategy**: Test each interface independently and together

## Interface Selection Guide

| Interface | Best For | Use Cases |
|-----------|----------|-----------|
| **CLI** | Automation, scripting, power users | CI/CD, batch processing, technical users |
| **Web** | Business users, visualization, dashboards | Business analytics, reporting, collaboration |
| **API** | Integration, services, automation | Microservices, third-party integration, mobile apps |

---

## Next Steps

- **Web Integration**: [Detailed Web Patterns](web_integration.md)
- **Progress Tracking**: [Real-time Feedback Systems](progress_tracking.md)
- **Implementation**: [Level 2 Standard Systems](../05_implementation_levels/level_2_standard.md)

---

*This dual interface architecture enables serving diverse user needs from a single, maintainable codebase while preserving the strengths of each interface paradigm.*