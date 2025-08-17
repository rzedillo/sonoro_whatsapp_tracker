# Architecture Patterns - Core Design Templates

> 🔧 **Design Templates**: Proven architectural patterns for agent systems, from simple workflows to enterprise orchestration.

## Navigation
- **Previous**: [Quick Start Guide](01_quick_start.md)
- **Next**: [Dual Interface Design](03_interfaces/dual_interface_design.md) → [Web Integration](03_interfaces/web_integration.md)
- **Patterns**: [Financial Precision](04_specialized/financial_precision.md) → [Context Management](04_specialized/context_management.md)
- **Reference**: [Decision Matrices](06_reference/decision_matrices.md) → [Templates](06_reference/templates.md)

---

## Pattern Selection Guide

| Pattern | Use Case | Complexity | Agents | Best For |
|---------|----------|------------|---------|-----------|
| **Agent Specialization** | Single task, clear expertise | Simple | 1 | Data validation, formatting |
| **Multi-Agent Workflow** | Sequential processing | Standard | 2-3 | Data pipelines, analysis |
| **Fallback Resilience** | High reliability needed | Standard | 1+ | Critical operations |
| **Context Sharing** | Collaborative tasks | Complex | 2+ | Knowledge sharing |
| **Dual Interface** | Multiple user types | Standard | Any | Business + technical users |
| **Progress Tracking** | Long-running operations | Standard | Any | User feedback required |
| **Financial Data** | Precision critical | Complex | 1+ | Financial, scientific data |
| **Advanced Orchestration** | Enterprise workflows | Expert | 3+ | Production systems |

---

## Core Architecture Patterns

### 1. Agent Specialization Pattern

**When to Use**: Single-purpose tasks requiring specific expertise.

```
Problem → Agent Assignment → Execution → Result
    ↓           ↓              ↓         ↓
Analyze    Match Expertise   Execute   Validate
Complexity  & Capability     with      & Format
           Tier Selection    Context    Output
```

**Implementation Template**:
```python
class SpecializedAgent:
    def __init__(self, personality: AgentPersonality):
        self.personality = personality
        self.expertise_validator = ExpertiseValidator(personality.expertise)
    
    async def execute(self, task: str, context: dict):
        # 1. Validate task matches expertise
        if not self.expertise_validator.can_handle(task):
            raise ValueError(f"Task outside expertise: {self.personality.expertise}")
        
        # 2. Select appropriate capability tier
        complexity = self.analyze_task_complexity(task)
        model = self.select_model_for_capability(complexity)
        
        # 3. Execute with specialized context
        result = await self.process_with_expertise(task, context, model)
        
        # 4. Validate and format output
        return self.format_specialized_output(result)
```

**Best Practices**:
- Match agent expertise to task domain
- Use capability tiers to optimize cost/performance
- Validate inputs and outputs for consistency
- Implement fallback strategies for edge cases

---

### 2. Multi-Agent Workflow Pattern

**When to Use**: Sequential processing where each step requires different expertise.

```
Input → Agent A → Agent B → Agent C → Final Output
  ↓       ↓         ↓         ↓          ↓
Parse   Extract   Validate  Enhance    Present
Data    Info      Quality   Analysis   Results
```

**Implementation Template**:
```python
class WorkflowOrchestrator:
    def __init__(self, workflow_stages: List[WorkflowStage]):
        self.stages = workflow_stages
        self.context_manager = SharedContextManager()
    
    async def execute_workflow(self, input_data: dict, context_id: str):
        workflow_context = {"context_id": context_id, "results": {}}
        
        for stage in self.stages:
            # Pass context from previous stages
            stage_input = self.prepare_stage_input(stage, workflow_context)
            
            # Execute stage with appropriate agent
            agent = self.get_agent_for_stage(stage)
            result = await agent.execute(stage_input, workflow_context)
            
            # Store result for next stages
            workflow_context["results"][stage.name] = result
            
            # Update shared context
            await self.context_manager.update_context(
                context_id, stage.agent_name, result
            )
        
        return self.synthesize_final_result(workflow_context)
```

**Best Practices**:
- Design clear handoffs between stages
- Maintain context continuity across agents
- Implement stage validation and error recovery
- Plan for parallel execution where possible

---

### 3. Fallback Resilience Pattern

**When to Use**: Critical operations requiring high reliability and graceful degradation.

```
Primary Strategy → Success? → Output
       ↓              ↓
    Failure       Secondary Strategy → Success? → Output
       ↓              ↓                  ↓
    Continue      Failure          Minimal Strategy → Output
```

**Implementation Template**:
```python
class ResilientAgent:
    def __init__(self, personality: AgentPersonality):
        self.personality = personality
        self.fallback_chain = self.build_fallback_chain()
        self.retry_config = RetryConfig()
    
    async def execute_with_resilience(self, task: str, context: dict):
        for strategy_level, strategy in enumerate(self.fallback_chain):
            try:
                result = await self.execute_with_strategy(
                    task, context, strategy
                )
                
                # Add resilience metadata
                result["resilience_info"] = {
                    "strategy_used": strategy.name,
                    "fallback_level": strategy_level,
                    "confidence": self.calculate_confidence(result, strategy_level)
                }
                
                return result
                
            except Exception as e:
                if strategy_level == len(self.fallback_chain) - 1:
                    # Last strategy failed
                    return self.create_minimal_response(task, str(e))
                
                # Log and continue to next strategy
                self.log_fallback_attempt(strategy, e)
                continue
    
    def build_fallback_chain(self):
        return [
            PrimaryStrategy(self.personality),
            SecondaryStrategy(self.personality.fallback_tier),
            MinimalStrategy("simple")
        ]
```

**Best Practices**:
- Define clear fallback hierarchy
- Maintain confidence scoring across strategies
- Log all fallback attempts for analysis
- Design minimal viable responses for worst-case scenarios

---

### 4. Context Sharing Pattern

**When to Use**: Collaborative tasks requiring information exchange between agents.

```
Agent A Context → Shared Context Store → Agent B Context
       ↓               ↓                      ↓
   Task Result    State Management      Enhanced Input
   + Metadata     + Persistence         + History
```

**Implementation Template**:
```python
class SharedContextManager:
    def __init__(self, storage_backend="memory"):
        self.storage = self.init_storage(storage_backend)
        self.context_filters = AgentContextFilters()
    
    async def get_context_for_agent(self, context_id: str, agent_name: str):
        # Get full context
        full_context = await self.storage.get_context(context_id)
        
        # Filter relevant information for this agent
        filtered_context = self.context_filters.filter_for_agent(
            full_context, agent_name
        )
        
        return filtered_context
    
    async def update_context(self, context_id: str, agent_name: str, 
                           new_data: dict, merge_strategy="smart_merge"):
        current_context = await self.storage.get_context(context_id)
        
        # Intelligent merging based on strategy
        merged_context = await self.merge_contexts(
            current_context, new_data, merge_strategy
        )
        
        # Persist updated context
        await self.storage.save_context(context_id, merged_context)
        
        # Notify other agents of update
        await self.notify_context_update(context_id, agent_name, new_data)
```

**Best Practices**:
- Filter context relevance per agent type
- Use intelligent merging strategies
- Implement change notifications
- Maintain context version history

---

### 5. V3.1 🔥 Dual Interface Pattern

**When to Use**: Systems serving both technical users (CLI/API) and business users (Web).

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

**Implementation Template**:
```python
class DualInterfaceSystem:
    def __init__(self):
        self.core_orchestrator = UniversalOrchestrator()
        self.interface_router = InterfaceRouter()
        self.formatters = {
            "cli": CLIFormatter(),
            "web": WebFormatter(), 
            "api": APIFormatter()
        }
    
    async def process_request(self, request: Request, interface_type: str):
        # Route to appropriate interface handler
        interface_handler = self.interface_router.get_handler(interface_type)
        
        # Execute with core orchestrator (interface-agnostic)
        core_result = await self.core_orchestrator.execute_workflow(
            request, progress_callback=interface_handler.create_progress_callback()
        )
        
        # Format for specific interface
        formatter = self.formatters[interface_type]
        formatted_result = await formatter.format_result(core_result)
        
        return formatted_result

class UniversalOrchestrator:
    """Core orchestrator that serves all interfaces"""
    
    async def execute_workflow(self, request, progress_callback=None):
        # This method is called by all interfaces
        return await self.process_with_agents(request, progress_callback)
```

**Best Practices**:
- Shared orchestration layer for all interfaces
- Interface-specific wrappers for UI concerns
- Common data models across interfaces
- Progressive enhancement (CLI-first, then Web)

---

### 6. V3.1 🔥 Progress Tracking Pattern

**When to Use**: Long-running operations requiring real-time user feedback.

```
Operation Start → Progress Callbacks → Real-time Updates
       ↓                ↓                     ↓
  Initialize        Stage Updates        User Interface
   Tracker           + Metadata          Updates (CLI/Web)
       ↓                ↓                     ↓
   Execution      Context Updates       Completion
   Monitoring     + Error Handling      Notification
```

**Implementation Template**:
```python
class ProgressTracker:
    def __init__(self, operation_id: str, total_stages: int = 1):
        self.operation_id = operation_id
        self.total_stages = total_stages
        self.current_stage = 0
        self.callbacks = []
        self.status = ProgressStatus.PENDING
    
    def add_callback(self, callback: Callable[[ProgressUpdate], None]):
        self.callbacks.append(callback)
    
    def update_stage(self, stage_name: str, progress: int, message: str = ""):
        update = ProgressUpdate(
            operation_id=self.operation_id,
            stage=stage_name,
            progress_percentage=progress,
            message=message,
            status=self.status,
            timestamp=time.time()
        )
        
        self._notify_callbacks(update)
    
    def complete_stage(self, stage_name: str, message: str = "Stage completed"):
        self.current_stage += 1
        overall_progress = int((self.current_stage / self.total_stages) * 100)
        
        self.update_stage(stage_name, 100, message)
        
        if self.current_stage >= self.total_stages:
            self.complete_operation("All stages completed successfully")
```

**Best Practices**:
- Provide granular, meaningful updates
- Handle callback failures gracefully
- Persist progress for long-running operations
- Maintain consistent progress semantics across interfaces

---

### 7. V3.1 🔥 Financial Data Pattern

**When to Use**: Systems handling financial data, scientific measurements, or precision-critical calculations.

```
Raw Data → Validation → Decimal Conversion → Calculation → Audit Trail
    ↓          ↓             ↓                 ↓             ↓
  Source    Range Check   Precision         Business      Compliance
  Format    + Type Val.   Handling          Rules         Logging
```

**Implementation Template**:
```python
class FinancialDataProcessor:
    def __init__(self, precision_config: PrecisionConfig):
        self.config = precision_config
        self.context = Context(prec=28, rounding=precision_config.rounding_mode)
        self.audit_trail = FinancialAuditTrail()
    
    def validate_input(self, value: Union[str, float, int, Decimal]) -> Decimal:
        try:
            # Convert avoiding float precision issues
            if isinstance(value, float):
                decimal_value = Decimal(str(value))
            else:
                decimal_value = Decimal(str(value))
            
            # Apply precision configuration
            decimal_value = decimal_value.quantize(
                Decimal('0.' + '0' * self.config.decimal_places),
                rounding=self.config.rounding_mode
            )
            
            # Validate range
            self._validate_range(decimal_value)
            
            # Record audit trail
            if self.config.audit_trail_enabled:
                self.audit_trail.record_validation(str(value), decimal_value)
            
            return decimal_value
            
        except Exception as e:
            raise ValueError(f"Invalid financial value '{value}': {str(e)}")
    
    def calculate_revenue_share(self, total_revenue: Decimal, 
                              share_percentage: Decimal) -> Dict[str, Decimal]:
        with self.context:
            creator_amount = self.calculate_percentage(total_revenue, share_percentage)
            platform_amount = total_revenue - creator_amount
            
            # Ensure exact total (handle rounding)
            actual_total = creator_amount + platform_amount
            if actual_total != total_revenue:
                platform_amount = total_revenue - creator_amount
            
            return {
                "total": total_revenue,
                "creator_share": creator_amount,
                "platform_share": platform_amount
            }
```

**Best Practices**:
- Always use Decimal for financial calculations
- Validate all inputs before processing
- Maintain comprehensive audit trails
- Handle rounding consistently
- Track currency throughout operations

---

### 8. V3.1 🔥 Advanced Orchestration Pattern

**When to Use**: Enterprise workflows with complex dependencies and parallel execution requirements.

```
Workflow Definition → Dependency Analysis → Parallel Execution → Result Synthesis
        ↓                    ↓                     ↓                    ↓
   Stage Config         Execution Plan         Stage Monitoring      Final Output
   + Dependencies       + Resource Mgmt        + Error Recovery      + Metadata
```

**Implementation Template**:
```python
class AdvancedOrchestrator:
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.dependency_analyzer = DependencyAnalyzer()
        self.resource_manager = ResourceManager()
        self.error_recovery = ErrorRecoveryManager()
    
    async def execute_workflow(self, workflow_definition: WorkflowDefinition):
        # Analyze dependencies and create execution plan
        execution_plan = self.dependency_analyzer.analyze(workflow_definition)
        
        # Execute stages according to plan
        execution_context = ExecutionContext(workflow_definition.name)
        
        for execution_stage in execution_plan.stages:
            if execution_stage.can_run_parallel:
                # Execute parallel stages
                stage_results = await self._execute_parallel_stages(
                    execution_stage.parallel_stages, execution_context
                )
            else:
                # Execute sequential stage
                stage_result = await self._execute_single_stage(
                    execution_stage.stage, execution_context
                )
                stage_results = [stage_result]
            
            # Update execution context with results
            for result in stage_results:
                execution_context.add_stage_result(result)
            
            # Check for errors and apply recovery if needed
            await self._handle_stage_errors(stage_results, execution_context)
        
        # Synthesize final results
        return await self._synthesize_workflow_results(execution_context)
    
    async def _execute_parallel_stages(self, stages: List[WorkflowStage], 
                                     context: ExecutionContext):
        # Resource allocation for parallel execution
        resources = self.resource_manager.allocate_for_parallel(len(stages))
        
        # Create parallel tasks
        tasks = []
        for stage in stages:
            task = self._create_stage_task(stage, context, resources)
            tasks.append(task)
        
        # Execute with monitoring
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        return self._process_parallel_results(results, stages)
```

**Best Practices**:
- Analyze dependencies before execution
- Implement parallel execution where safe
- Monitor resource usage and allocation
- Design comprehensive error recovery strategies
- Maintain detailed execution metadata

---

### 9. V3.1 🔥 API-First Architecture Pattern

**When to Use**: Modern systems requiring programmatic access, integration with external services, and scalable multi-interface support.

```
API Gateway → Authentication → Agent Orchestrator → Response Formatter
     ↓              ↓               ↓                    ↓
REST Endpoints  JWT/API Keys    Core Agent Logic    JSON/XML Output
+ OpenAPI      + Rate Limiting  + Context Mgmt     + Status Codes
+ Versioning   + Audit Logs     + Error Handling   + Metadata
```

**Implementation Template**:
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import uuid

# Request/Response Models
class AgentRequest(BaseModel):
    task: str
    data: Dict[str, Any] = {}
    agent_type: str = "general"
    priority: str = "normal"
    callback_url: Optional[str] = None

class AgentResponse(BaseModel):
    operation_id: str
    status: str  # "processing", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    agent_metadata: Dict[str, Any] = {}

class APIFirstAgentSystem:
    """API-First agent system with FastAPI integration"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Agent System API",
            description="Production-ready agent orchestration API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        self.agent_orchestrator = UniversalOrchestrator()
        self.active_operations = {}
        self.setup_routes()
        self.setup_middleware()
    
    def setup_routes(self):
        """Setup API routes with proper error handling"""
        
        @self.app.post("/v1/agents/execute", response_model=AgentResponse)
        async def execute_agent(
            request: AgentRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            """Execute agent task asynchronously"""
            
            # Validate authentication
            user_id = await self.validate_api_key(credentials.credentials)
            
            # Create operation ID for tracking
            operation_id = str(uuid.uuid4())
            
            # For quick tasks, execute synchronously
            if request.priority == "high" or request.agent_type == "simple":
                try:
                    result = await self.agent_orchestrator.execute_workflow(
                        request.task, 
                        request.data,
                        interface_mode="api",
                        user_id=user_id
                    )
                    
                    return AgentResponse(
                        operation_id=operation_id,
                        status="completed",
                        result=result,
                        processing_time=result.get("processing_time", 0),
                        agent_metadata=result.get("metadata", {})
                    )
                    
                except Exception as e:
                    return AgentResponse(
                        operation_id=operation_id,
                        status="failed",
                        error=str(e)
                    )
            
            # For complex tasks, execute in background
            else:
                self.active_operations[operation_id] = {
                    "status": "processing",
                    "user_id": user_id,
                    "started_at": time.time()
                }
                
                background_tasks.add_task(
                    self.execute_background_task,
                    operation_id, request, user_id
                )
                
                return AgentResponse(
                    operation_id=operation_id,
                    status="processing"
                )
        
        @self.app.get("/v1/operations/{operation_id}", response_model=AgentResponse)
        async def get_operation_status(
            operation_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            """Get status of async operation"""
            
            user_id = await self.validate_api_key(credentials.credentials)
            
            if operation_id not in self.active_operations:
                raise HTTPException(status_code=404, detail="Operation not found")
            
            operation = self.active_operations[operation_id]
            
            # Check user access
            if operation["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return AgentResponse(
                operation_id=operation_id,
                status=operation["status"],
                result=operation.get("result"),
                error=operation.get("error"),
                processing_time=operation.get("processing_time")
            )
        
        @self.app.get("/v1/health")
        async def health_check():
            """Health check endpoint for monitoring"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "active_operations": len(self.active_operations),
                "system_info": {
                    "agents_available": len(self.agent_orchestrator.agents),
                    "memory_usage": "normal"  # Could add actual monitoring
                }
            }
        
        @self.app.get("/v1/agents/capabilities")
        async def get_agent_capabilities(
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            """Get available agent types and capabilities"""
            
            await self.validate_api_key(credentials.credentials)
            
            return {
                "available_agents": list(self.agent_orchestrator.agents.keys()),
                "capabilities": {
                    agent_name: agent.personality.expertise
                    for agent_name, agent in self.agent_orchestrator.agents.items()
                },
                "supported_priorities": ["low", "normal", "high"],
                "max_concurrent_operations": 10
            }
    
    def setup_middleware(self):
        """Setup middleware for logging, CORS, rate limiting"""
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        
        # CORS configuration
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security middleware
        self.app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=["*"]  # Configure for production
        )
    
    async def validate_api_key(self, api_key: str) -> str:
        """Validate API key and return user ID"""
        # In production, validate against database/cache
        if not api_key or not api_key.startswith("sk-"):
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key format"
            )
        
        # Mock validation - implement proper validation
        if api_key == "sk-test-key-12345":
            return "user_123"
        
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    async def execute_background_task(self, operation_id: str, 
                                    request: AgentRequest, user_id: str):
        """Execute long-running task in background"""
        try:
            start_time = time.time()
            
            result = await self.agent_orchestrator.execute_workflow(
                request.task,
                request.data,
                interface_mode="api",
                user_id=user_id
            )
            
            processing_time = time.time() - start_time
            
            # Update operation status
            self.active_operations[operation_id].update({
                "status": "completed",
                "result": result,
                "processing_time": processing_time
            })
            
            # Send webhook notification if requested
            if request.callback_url:
                await self.send_webhook_notification(
                    request.callback_url, operation_id, "completed", result
                )
            
        except Exception as e:
            self.active_operations[operation_id].update({
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            })
            
            if request.callback_url:
                await self.send_webhook_notification(
                    request.callback_url, operation_id, "failed", {"error": str(e)}
                )
    
    async def send_webhook_notification(self, webhook_url: str, 
                                      operation_id: str, status: str, data: Dict):
        """Send webhook notification for operation completion"""
        import aiohttp
        
        payload = {
            "operation_id": operation_id,
            "status": status,
            "timestamp": time.time(),
            "data": data
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        print(f"✅ Webhook notification sent for {operation_id}")
                    else:
                        print(f"⚠️ Webhook notification failed: {response.status}")
        except Exception as e:
            print(f"❌ Webhook notification error: {e}")

# Usage Example
def create_api_first_system():
    """Create and configure API-first agent system"""
    api_system = APIFirstAgentSystem()
    
    # Add agents to orchestrator
    api_system.agent_orchestrator.add_agents([
        DataValidatorAgent(),
        AnalyticsAgent(),
        ReportGeneratorAgent()
    ])
    
    return api_system.app

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
app = create_api_first_system()
```

**Best Practices**:
- Use proper HTTP status codes and error handling
- Implement authentication and rate limiting
- Provide comprehensive API documentation (OpenAPI/Swagger)
- Support both synchronous and asynchronous operations
- Include health checks and monitoring endpoints
- Version your APIs for backward compatibility
- Use webhook notifications for long-running tasks

---

## Pattern Combination Strategies

### Simple System (1-2 Patterns)
- **Agent Specialization** + **Fallback Resilience**
- Best for: Single-purpose applications with reliability requirements

### Standard System (2-3 Patterns)  
- **Multi-Agent Workflow** + **Context Sharing** + **Progress Tracking**
- Best for: Business applications with moderate complexity

### Complex System (3-5 Patterns)
- **Dual Interface** + **Financial Data** + **Advanced Orchestration** + **Context Sharing** + **Progress Tracking**
- Best for: Enterprise applications with multiple user types

### Production System (All Patterns)
- All patterns integrated with comprehensive monitoring and testing
- Best for: Mission-critical enterprise systems

---

## Next Steps

Choose your implementation level based on complexity needs:

- **Level 1**: [Simple Systems](05_implementation_levels/level_1_simple.md) - 1-2 patterns
- **Level 2**: [Standard Systems](05_implementation_levels/level_2_standard.md) - 2-3 patterns  
- **Level 3**: [Complex Systems](05_implementation_levels/level_3_complex.md) - 3-5 patterns
- **Level 4**: [Production Systems](05_implementation_levels/level_4_production.md) - All patterns

Or explore specific capabilities:
- [Dual Interface Design](03_interfaces/dual_interface_design.md)
- [Financial Data Handling](04_specialized/financial_precision.md)
- [Testing Frameworks](04_specialized/testing_frameworks.md)

---

*These patterns provide the architectural foundation for scalable agent systems. Combine them based on your specific requirements and complexity needs.*