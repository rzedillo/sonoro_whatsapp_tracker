# Level 3: Complex Agent Systems

> ⚡ **Enterprise Workflows**: Orchestrate 3-5 specialized agents with advanced features for enterprise-grade applications.

## Navigation
- **Previous**: [Level 2: Standard](level_2_standard.md)
- **Next**: [Level 4: Production](level_4_production.md)
- **Specialized**: [Financial Precision](../04_specialized/financial_precision.md) → [Context Management](../04_specialized/context_management.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Security Guidelines](../06_reference/security_guidelines.md)

---

## Overview

Level 3 systems manage complex enterprise workflows with multiple specialized agents, advanced error recovery, parallel execution, and sophisticated monitoring. These implementations handle real-world business complexity while maintaining reliability and performance.

## Level 3 Characteristics

| Aspect | Level 3 Specification |
|--------|----------------------|
| **Agents** | 3-5 specialized agents |
| **Patterns** | Advanced Orchestration + Dual Interface + Financial Data + Context Management |
| **Complexity** | Complex to Expert tasks |
| **Deployment** | Full dual interface with enterprise features |
| **Context** | Advanced state management with persistence |
| **Time to MVP** | 1-2 weeks |

---

## Use Cases and Examples

### Perfect for Level 3
- **Enterprise Data Pipelines**: Multi-source → Processing → Validation → Distribution
- **Financial Systems**: Revenue collection → Calculation → Compliance → Reporting
- **Content Management**: Creation → Review → Enhancement → Publication → Analytics
- **Quality Assurance**: Multi-stage validation → Testing → Approval → Deployment
- **Business Intelligence**: Data collection → Analysis → Prediction → Reporting
- **Compliance Workflows**: Data gathering → Validation → Audit → Certification

### Level 3 Requirements
- Multiple data sources or processing stages
- Enterprise-grade error handling and recovery
- Real-time monitoring and alerting needed
- Parallel processing for performance
- Advanced user interfaces required
- Integration with external enterprise systems

---

## Advanced Orchestration Architecture

### Enterprise Orchestrator

```python
# level3_orchestrator.py
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor

class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"

class RetryPolicy(Enum):
    NONE = "none"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    CUSTOM = "custom"

@dataclass
class AdvancedWorkflowStage:
    """Advanced workflow stage with enterprise features"""
    name: str
    agent_name: str
    task_template: str
    depends_on: List[str] = field(default_factory=list)
    parallel_group: str = None
    required: bool = True
    timeout: int = 300
    retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class WorkflowExecutionPlan:
    """Detailed execution plan for complex workflows"""
    execution_groups: List[List[AdvancedWorkflowStage]]
    total_stages: int
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    critical_path: List[str]
    parallel_groups: Dict[str, List[str]]

class Level3Orchestrator:
    """Advanced orchestrator for Level 3 complex workflows"""
    
    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any] = None):
        self.agents = agents
        self.config = config or {}
        self.context_manager = AdvancedContextManager()
        self.progress_manager = EnterpriseProgressManager()
        self.resource_manager = ResourceManager()
        self.circuit_breakers = CircuitBreakerManager()
        self.monitoring = WorkflowMonitoring()
        self.execution_history = []
        self.active_executions = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 10))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def execute_workflow(self, config: 'AdvancedWorkflowConfig',
                             input_data: Dict[str, Any],
                             execution_strategy: ExecutionStrategy = ExecutionStrategy.HYBRID,
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute complex workflow with advanced features"""
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create execution plan
        execution_plan = self._create_execution_plan(config, execution_strategy)
        
        # Initialize monitoring
        self.monitoring.start_workflow_monitoring(execution_id, config.workflow_name)
        
        # Initialize progress tracking
        progress_tracker = None
        if config.progress_tracking_enabled:
            progress_tracker = self.progress_manager.create_advanced_tracker(
                execution_id, execution_plan.total_stages
            )
            if progress_callback:
                progress_tracker.add_callback(progress_callback)
        
        # Initialize context
        context_id = None
        if config.context_sharing_enabled:
            context_id = await self.context_manager.create_advanced_workflow_context(
                execution_id, input_data, config
            )
        
        # Track active execution
        self.active_executions[execution_id] = {
            "config": config,
            "start_time": start_time,
            "status": "running",
            "progress_tracker": progress_tracker
        }
        
        try:
            # Pre-execution validation
            validation_result = await self._validate_workflow_preconditions(
                config, input_data, execution_plan
            )
            
            if not validation_result["valid"]:
                raise ValueError(f"Workflow validation failed: {validation_result['errors']}")
            
            # Reserve resources
            resource_reservation = await self.resource_manager.reserve_resources(
                execution_plan.resource_requirements
            )
            
            try:
                # Execute workflow stages
                stage_results = {}
                
                for group_index, execution_group in enumerate(execution_plan.execution_groups):
                    group_start_time = time.time()
                    
                    if len(execution_group) == 1:
                        # Sequential execution
                        stage = execution_group[0]
                        result = await self._execute_stage_with_advanced_features(
                            stage, stage_results, context_id, progress_tracker, execution_id
                        )
                        stage_results[stage.name] = result
                        
                    else:
                        # Parallel execution
                        parallel_results = await self._execute_parallel_group_advanced(
                            execution_group, stage_results, context_id, 
                            progress_tracker, execution_id
                        )
                        stage_results.update(parallel_results)
                    
                    # Monitor group completion
                    group_duration = time.time() - group_start_time
                    self.monitoring.record_group_completion(
                        execution_id, group_index, group_duration
                    )
                    
                    # Check for early termination conditions
                    if await self._should_terminate_early(stage_results, config):
                        break
                
                # Post-execution validation
                final_validation = await self._validate_workflow_results(
                    stage_results, config
                )
                
                execution_time = time.time() - start_time
                
                final_result = {
                    "execution_id": execution_id,
                    "workflow_name": config.workflow_name,
                    "status": "success",
                    "execution_time": execution_time,
                    "execution_plan": {
                        "total_stages": execution_plan.total_stages,
                        "parallel_groups": len([g for g in execution_plan.execution_groups if len(g) > 1]),
                        "critical_path": execution_plan.critical_path
                    },
                    "stage_results": stage_results,
                    "validation": final_validation,
                    "performance_metrics": self.monitoring.get_workflow_metrics(execution_id),
                    "resource_usage": resource_reservation.get_usage_stats(),
                    "input_data": input_data
                }
                
                if progress_tracker:
                    progress_tracker.complete_operation("Workflow completed successfully")
                
                self.monitoring.complete_workflow_monitoring(execution_id, "success")
                
                return final_result
                
            finally:
                # Release resources
                await self.resource_manager.release_reservation(resource_reservation)
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = {
                "execution_id": execution_id,
                "workflow_name": config.workflow_name,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time,
                "completed_stages": [name for name, result in stage_results.items() 
                                   if result.get("status") == "success"],
                "failed_stage": self._identify_failed_stage(stage_results),
                "recovery_suggestions": self._generate_recovery_suggestions(e, stage_results),
                "input_data": input_data
            }
            
            if progress_tracker:
                progress_tracker.error_operation(f"Workflow failed: {str(e)}")
            
            self.monitoring.complete_workflow_monitoring(execution_id, "error", str(e))
            
            # Store error for analysis
            self.logger.error(f"Workflow {execution_id} failed: {str(e)}", 
                            extra={"execution_id": execution_id, "workflow": config.workflow_name})
            
            return error_result
            
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            # Store execution history
            self.execution_history.append({
                "execution_id": execution_id,
                "config": config,
                "result": final_result if 'final_result' in locals() else error_result,
                "timestamp": datetime.now()
            })
    
    def _create_execution_plan(self, config: 'AdvancedWorkflowConfig',
                             strategy: ExecutionStrategy) -> WorkflowExecutionPlan:
        """Create detailed execution plan with optimization"""
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            return self._create_sequential_plan(config)
        elif strategy == ExecutionStrategy.PARALLEL:
            return self._create_parallel_plan(config)
        else:  # HYBRID
            return self._create_hybrid_plan(config)
    
    def _create_hybrid_plan(self, config: 'AdvancedWorkflowConfig') -> WorkflowExecutionPlan:
        """Create optimized hybrid execution plan"""
        
        stages_by_name = {stage.name: stage for stage in config.stages}
        execution_groups = []
        executed_stages = set()
        parallel_groups = {}
        
        # Calculate stage priorities and dependencies
        stage_priorities = self._calculate_stage_priorities(config.stages)
        critical_path = self._identify_critical_path(config.stages)
        
        while len(executed_stages) < len(config.stages):
            # Find stages ready for execution
            ready_stages = []
            
            for stage in config.stages:
                if stage.name in executed_stages:
                    continue
                
                # Check dependencies
                dependencies_met = all(dep in executed_stages for dep in stage.depends_on)
                
                # Check circuit breaker status
                if self.circuit_breakers.is_open(stage.agent_name):
                    continue
                
                if dependencies_met:
                    ready_stages.append(stage)
            
            if not ready_stages:
                raise ValueError("Circular dependency or circuit breaker blocking execution")
            
            # Optimize grouping for parallel execution
            optimized_groups = self._optimize_parallel_grouping(ready_stages, stage_priorities)
            
            for group in optimized_groups:
                execution_groups.append(group)
                for stage in group:
                    executed_stages.add(stage.name)
                    
                # Track parallel groups
                if len(group) > 1:
                    group_name = f"parallel_group_{len(parallel_groups)}"
                    parallel_groups[group_name] = [s.name for s in group]
        
        # Calculate resource requirements
        total_resources = self._calculate_resource_requirements(config.stages)
        
        # Estimate duration
        estimated_duration = self._estimate_workflow_duration(execution_groups, critical_path)
        
        return WorkflowExecutionPlan(
            execution_groups=execution_groups,
            total_stages=len(config.stages),
            estimated_duration=estimated_duration,
            resource_requirements=total_resources,
            critical_path=critical_path,
            parallel_groups=parallel_groups
        )
    
    async def _execute_stage_with_advanced_features(self, stage: AdvancedWorkflowStage,
                                                  previous_results: Dict[str, Any],
                                                  context_id: Optional[str],
                                                  progress_tracker: Optional[Any],
                                                  execution_id: str) -> Dict[str, Any]:
        """Execute stage with advanced error handling and monitoring"""
        
        stage_start_time = time.time()
        
        # Check circuit breaker
        if self.circuit_breakers.is_open(stage.agent_name):
            raise Exception(f"Circuit breaker open for agent {stage.agent_name}")
        
        # Update progress
        if progress_tracker:
            progress_tracker.start_stage(stage.name, {
                "agent": stage.agent_name,
                "timeout": stage.timeout,
                "retry_policy": stage.retry_policy.value
            })
        
        # Start monitoring
        self.monitoring.start_stage_monitoring(execution_id, stage.name)
        
        retry_count = 0
        last_exception = None
        
        while retry_count <= stage.max_retries:
            try:
                # Get agent
                if stage.agent_name not in self.agents:
                    raise ValueError(f"Agent '{stage.agent_name}' not found")
                
                agent = self.agents[stage.agent_name]
                
                # Prepare stage input
                stage_input = await self._prepare_advanced_stage_input(
                    stage, previous_results, context_id
                )
                
                # Pre-execution validation
                if stage.validation_rules:
                    validation_result = await self._validate_stage_input(
                        stage_input, stage.validation_rules
                    )
                    if not validation_result["valid"]:
                        raise ValueError(f"Stage input validation failed: {validation_result['errors']}")
                
                # Execute with timeout and monitoring
                result = await asyncio.wait_for(
                    self._execute_agent_with_monitoring(
                        agent, stage.task_template, stage_input, stage.name, execution_id
                    ),
                    timeout=stage.timeout
                )
                
                # Post-execution validation
                if stage.success_criteria:
                    success_validation = await self._validate_stage_success(
                        result, stage.success_criteria
                    )
                    if not success_validation["valid"]:
                        raise ValueError(f"Stage success criteria not met: {success_validation['errors']}")
                
                # Update context
                if context_id and result.get("status") == "success":
                    await self.context_manager.update_advanced_context(
                        context_id, stage.agent_name, {
                            f"{stage.name}_result": result,
                            f"{stage.name}_metadata": {
                                "execution_time": time.time() - stage_start_time,
                                "retry_count": retry_count,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                    )
                
                # Record success
                self.circuit_breakers.record_success(stage.agent_name)
                self.monitoring.complete_stage_monitoring(execution_id, stage.name, "success")
                
                if progress_tracker:
                    progress_tracker.complete_stage(stage.name, "Stage completed successfully")
                
                # Add execution metadata
                result["execution_metadata"] = {
                    "stage_name": stage.name,
                    "agent_name": stage.agent_name,
                    "execution_time": time.time() - stage_start_time,
                    "retry_count": retry_count,
                    "timestamp": datetime.now().isoformat()
                }
                
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                error_msg = f"Stage '{stage.name}' timed out after {stage.timeout} seconds"
                
            except Exception as e:
                last_exception = e
                error_msg = f"Stage '{stage.name}' failed: {str(e)}"
            
            # Record failure
            self.circuit_breakers.record_failure(stage.agent_name)
            retry_count += 1
            
            # Check if we should retry
            if retry_count <= stage.max_retries and stage.retry_policy != RetryPolicy.NONE:
                retry_delay = self._calculate_retry_delay(stage, retry_count)
                
                self.logger.warning(f"Stage {stage.name} failed (attempt {retry_count}), retrying in {retry_delay}s: {error_msg}")
                
                if progress_tracker:
                    progress_tracker.update_stage_progress(
                        stage.name, 
                        progress=50,  # Partial progress during retry
                        message=f"Retrying... (attempt {retry_count + 1})"
                    )
                
                await asyncio.sleep(retry_delay)
            else:
                break
        
        # All retries exhausted
        self.monitoring.complete_stage_monitoring(execution_id, stage.name, "error", str(last_exception))
        
        if progress_tracker:
            progress_tracker.error_stage(stage.name, str(last_exception))
        
        if stage.required:
            raise last_exception
        else:
            return {
                "status": "skipped",
                "reason": str(last_exception),
                "execution_metadata": {
                    "stage_name": stage.name,
                    "agent_name": stage.agent_name,
                    "execution_time": time.time() - stage_start_time,
                    "retry_count": retry_count,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    async def _execute_parallel_group_advanced(self, stages: List[AdvancedWorkflowStage],
                                             previous_results: Dict[str, Any],
                                             context_id: Optional[str],
                                             progress_tracker: Optional[Any],
                                             execution_id: str) -> Dict[str, Any]:
        """Execute parallel group with advanced resource management"""
        
        # Reserve resources for parallel execution
        group_resources = {}
        for stage in stages:
            if stage.resource_requirements:
                group_resources[stage.name] = stage.resource_requirements
        
        if group_resources:
            resource_reservation = await self.resource_manager.reserve_group_resources(group_resources)
        else:
            resource_reservation = None
        
        try:
            # Create semaphore for concurrency control
            max_concurrent = self.config.get('max_concurrent_stages', len(stages))
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def execute_with_semaphore(stage):
                async with semaphore:
                    return await self._execute_stage_with_advanced_features(
                        stage, previous_results, context_id, progress_tracker, execution_id
                    )
            
            # Execute stages in parallel
            tasks = {stage.name: execute_with_semaphore(stage) for stage in stages}
            
            # Wait for completion with progress tracking
            results = {}
            completed = 0
            total = len(tasks)
            
            async for stage_name, result in self._gather_with_progress(tasks):
                results[stage_name] = result
                completed += 1
                
                if progress_tracker:
                    group_progress = int((completed / total) * 100)
                    progress_tracker.update_group_progress(
                        f"parallel_group_{len(results)}", 
                        group_progress,
                        f"Completed {completed}/{total} parallel stages"
                    )
            
            return results
            
        finally:
            if resource_reservation:
                await self.resource_manager.release_reservation(resource_reservation)
    
    async def _gather_with_progress(self, tasks: Dict[str, asyncio.Task]):
        """Gather tasks with progress reporting"""
        pending_tasks = {name: asyncio.create_task(task) for name, task in tasks.items()}
        
        while pending_tasks:
            done, pending = await asyncio.wait(
                pending_tasks.values(), 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for completed_task in done:
                # Find task name
                task_name = None
                for name, task in pending_tasks.items():
                    if task == completed_task:
                        task_name = name
                        break
                
                if task_name:
                    try:
                        result = await completed_task
                        yield task_name, result
                    except Exception as e:
                        yield task_name, {"status": "error", "error": str(e)}
                    
                    del pending_tasks[task_name]
    
    def _calculate_retry_delay(self, stage: AdvancedWorkflowStage, retry_count: int) -> float:
        """Calculate retry delay based on retry policy"""
        
        if stage.retry_policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            return stage.retry_delay * (2 ** (retry_count - 1))
        elif stage.retry_policy == RetryPolicy.FIXED_INTERVAL:
            return stage.retry_delay
        else:
            return stage.retry_delay
    
    def _optimize_parallel_grouping(self, stages: List[AdvancedWorkflowStage],
                                  priorities: Dict[str, int]) -> List[List[AdvancedWorkflowStage]]:
        """Optimize stage grouping for parallel execution"""
        
        # Group by parallel_group if specified
        groups = {}
        ungrouped = []
        
        for stage in stages:
            if stage.parallel_group:
                if stage.parallel_group not in groups:
                    groups[stage.parallel_group] = []
                groups[stage.parallel_group].append(stage)
            else:
                ungrouped.append(stage)
        
        # Create execution groups
        execution_groups = list(groups.values())
        
        # Handle ungrouped stages
        if ungrouped:
            # Sort by priority and create small parallel groups
            ungrouped.sort(key=lambda s: priorities.get(s.name, 0), reverse=True)
            
            # Group high-priority stages together
            max_group_size = self.config.get('max_parallel_group_size', 3)
            
            for i in range(0, len(ungrouped), max_group_size):
                group = ungrouped[i:i + max_group_size]
                execution_groups.append(group)
        
        return execution_groups
    
    def _calculate_stage_priorities(self, stages: List[AdvancedWorkflowStage]) -> Dict[str, int]:
        """Calculate stage execution priorities"""
        priorities = {}
        
        for stage in stages:
            priority = 0
            
            # Higher priority for stages with more dependents
            dependents = sum(1 for s in stages if stage.name in s.depends_on)
            priority += dependents * 10
            
            # Higher priority for required stages
            if stage.required:
                priority += 50
            
            # Lower priority for stages with higher timeout (likely slower)
            priority -= stage.timeout // 60
            
            priorities[stage.name] = priority
        
        return priorities
    
    def _identify_critical_path(self, stages: List[AdvancedWorkflowStage]) -> List[str]:
        """Identify critical path through workflow"""
        
        # Build dependency graph
        graph = {stage.name: stage.depends_on for stage in stages}
        stage_durations = {stage.name: stage.timeout for stage in stages}
        
        # Find longest path (critical path)
        def calculate_longest_path(node, visited=None):
            if visited is None:
                visited = set()
            
            if node in visited:
                return 0, []  # Circular dependency
            
            visited.add(node)
            
            if not graph.get(node):  # No dependencies
                return stage_durations[node], [node]
            
            max_duration = 0
            max_path = []
            
            for dep in graph[node]:
                dep_duration, dep_path = calculate_longest_path(dep, visited.copy())
                if dep_duration > max_duration:
                    max_duration = dep_duration
                    max_path = dep_path
            
            return max_duration + stage_durations[node], max_path + [node]
        
        # Find critical path
        critical_duration = 0
        critical_path = []
        
        for stage in stages:
            if not any(stage.name in s.depends_on for s in stages):  # End nodes
                duration, path = calculate_longest_path(stage.name)
                if duration > critical_duration:
                    critical_duration = duration
                    critical_path = path
        
        return critical_path
    
    async def cancel_execution(self, execution_id: str, reason: str = "User requested cancellation") -> bool:
        """Cancel running workflow execution"""
        
        if execution_id not in self.active_executions:
            return False
        
        execution_info = self.active_executions[execution_id]
        
        # Update status
        execution_info["status"] = "cancelled"
        execution_info["cancellation_reason"] = reason
        
        # Cancel progress tracker
        if execution_info["progress_tracker"]:
            execution_info["progress_tracker"].cancel_operation(reason)
        
        # Record cancellation
        self.monitoring.cancel_workflow_monitoring(execution_id, reason)
        
        self.logger.info(f"Workflow {execution_id} cancelled: {reason}")
        
        return True
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently running executions"""
        return {
            execution_id: {
                "workflow_name": info["config"].workflow_name,
                "start_time": info["start_time"],
                "elapsed_time": time.time() - info["start_time"],
                "status": info["status"]
            }
            for execution_id, info in self.active_executions.items()
        }
```

### Resource Management System

```python
# resource_manager.py
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import psutil
import time

@dataclass
class ResourceLimits:
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_concurrent_operations: int = 10
    max_network_connections: int = 100

@dataclass
class ResourceReservation:
    reservation_id: str
    resources: Dict[str, Any]
    start_time: float
    
    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "duration": time.time() - self.start_time,
            "reserved_resources": self.resources
        }

class ResourceManager:
    """Manage system resources for agent execution"""
    
    def __init__(self, limits: ResourceLimits = None):
        self.limits = limits or ResourceLimits()
        self.active_reservations = {}
        self.resource_usage_history = []
        self._lock = asyncio.Lock()
    
    async def reserve_resources(self, requirements: Dict[str, Any]) -> ResourceReservation:
        """Reserve resources for workflow execution"""
        
        async with self._lock:
            # Check if resources are available
            current_usage = self._get_current_usage()
            
            if not self._can_satisfy_requirements(requirements, current_usage):
                raise Exception("Insufficient resources available")
            
            # Create reservation
            reservation_id = f"res_{int(time.time() * 1000)}"
            reservation = ResourceReservation(
                reservation_id=reservation_id,
                resources=requirements,
                start_time=time.time()
            )
            
            self.active_reservations[reservation_id] = reservation
            
            return reservation
    
    def _get_current_usage(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            "memory_mb": psutil.virtual_memory().used // 1024 // 1024,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "concurrent_operations": len(self.active_reservations),
            "network_connections": len(psutil.net_connections())
        }
    
    def _can_satisfy_requirements(self, requirements: Dict[str, Any], 
                                current_usage: Dict[str, Any]) -> bool:
        """Check if requirements can be satisfied"""
        
        required_memory = requirements.get("memory_mb", 0)
        if current_usage["memory_mb"] + required_memory > self.limits.max_memory_mb:
            return False
        
        if current_usage["concurrent_operations"] >= self.limits.max_concurrent_operations:
            return False
        
        return True

class CircuitBreakerManager:
    """Manage circuit breakers for agent reliability"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_counts = {}
        self.last_failure_times = {}
        self.circuit_states = {}  # open, closed, half_open
    
    def is_open(self, agent_name: str) -> bool:
        """Check if circuit breaker is open for agent"""
        
        state = self.circuit_states.get(agent_name, "closed")
        
        if state == "open":
            # Check if reset timeout has passed
            last_failure = self.last_failure_times.get(agent_name, 0)
            if time.time() - last_failure > self.reset_timeout:
                self.circuit_states[agent_name] = "half_open"
                return False
            return True
        
        return False
    
    def record_success(self, agent_name: str):
        """Record successful execution"""
        self.failure_counts[agent_name] = 0
        self.circuit_states[agent_name] = "closed"
    
    def record_failure(self, agent_name: str):
        """Record failed execution"""
        self.failure_counts[agent_name] = self.failure_counts.get(agent_name, 0) + 1
        self.last_failure_times[agent_name] = time.time()
        
        if self.failure_counts[agent_name] >= self.failure_threshold:
            self.circuit_states[agent_name] = "open"

class WorkflowMonitoring:
    """Monitor workflow execution performance"""
    
    def __init__(self):
        self.workflow_metrics = {}
        self.stage_metrics = {}
        self.alerts = []
    
    def start_workflow_monitoring(self, execution_id: str, workflow_name: str):
        """Start monitoring workflow execution"""
        self.workflow_metrics[execution_id] = {
            "workflow_name": workflow_name,
            "start_time": time.time(),
            "stages": {},
            "alerts": []
        }
    
    def start_stage_monitoring(self, execution_id: str, stage_name: str):
        """Start monitoring stage execution"""
        if execution_id in self.workflow_metrics:
            self.workflow_metrics[execution_id]["stages"][stage_name] = {
                "start_time": time.time(),
                "status": "running"
            }
    
    def complete_stage_monitoring(self, execution_id: str, stage_name: str, 
                                status: str, error: str = None):
        """Complete stage monitoring"""
        if execution_id in self.workflow_metrics:
            stage_info = self.workflow_metrics[execution_id]["stages"].get(stage_name, {})
            stage_info.update({
                "end_time": time.time(),
                "status": status,
                "duration": time.time() - stage_info.get("start_time", time.time())
            })
            
            if error:
                stage_info["error"] = error
    
    def get_workflow_metrics(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        return self.workflow_metrics.get(execution_id, {})
```

### Enterprise Web Interface

```python
# level3_web_interface.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

class Level3WebInterface:
    """Enterprise web interface for Level 3 systems"""
    
    def __init__(self, orchestrator: Level3Orchestrator):
        self.orchestrator = orchestrator
        self.setup_enterprise_page()
    
    def setup_enterprise_page(self):
        """Setup enterprise-grade page configuration"""
        st.set_page_config(
            page_title="Enterprise Agent System",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for enterprise look
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(45deg, #f0f2f6, #ffffff);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e1e5e9;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_enterprise_dashboard(self):
        """Render enterprise dashboard"""
        st.title("⚡ Enterprise Agent System Dashboard")
        
        # Navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Overview", "🚀 Execute", "📊 Analytics", "🔧 Operations", "⚙️ Administration"
        ])
        
        with tab1:
            self.render_overview_dashboard()
        
        with tab2:
            self.render_workflow_execution()
        
        with tab3:
            self.render_analytics_dashboard()
        
        with tab4:
            self.render_operations_dashboard()
        
        with tab5:
            self.render_administration_panel()
    
    def render_overview_dashboard(self):
        """Render overview dashboard with KPIs"""
        
        # KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            self.render_kpi_card("Active Executions", 
                                len(self.orchestrator.get_active_executions()),
                                "🔄", "success")
        
        with col2:
            success_rate = self._calculate_success_rate()
            self.render_kpi_card("Success Rate", f"{success_rate:.1%}", 
                               "✅", "success" if success_rate > 0.95 else "warning")
        
        with col3:
            avg_time = self._calculate_average_execution_time()
            self.render_kpi_card("Avg Execution", f"{avg_time:.1f}s", 
                               "⏱️", "success" if avg_time < 60 else "warning")
        
        with col4:
            error_rate = self._calculate_error_rate()
            self.render_kpi_card("Error Rate", f"{error_rate:.1%}", 
                               "❌", "error" if error_rate > 0.05 else "success")
        
        with col5:
            agent_health = self._calculate_agent_health()
            self.render_kpi_card("Agent Health", f"{agent_health:.0%}", 
                               "🤖", "success" if agent_health > 0.9 else "warning")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_execution_timeline()
        
        with col2:
            self.render_agent_performance_chart()
        
        # Recent Activity
        st.subheader("Recent Workflow Executions")
        self.render_recent_executions_table()
    
    def render_kpi_card(self, title: str, value: str, icon: str, status: str):
        """Render KPI card with status indicator"""
        status_class = f"status-{status}"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span class="status-indicator {status_class}"></span>
                <strong>{title}</strong>
            </div>
            <div style="font-size: 2em; font-weight: bold; color: #1f2937;">
                {icon} {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_execution_timeline(self):
        """Render execution timeline chart"""
        st.subheader("Execution Timeline (Last 24 Hours)")
        
        # Sample data - in real implementation, get from orchestrator
        if self.orchestrator.execution_history:
            df = self._prepare_timeline_data()
            
            fig = px.timeline(df, x_start="start", x_end="end", y="workflow_name",
                            color="status", title="Workflow Executions")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No execution data available")
    
    def render_agent_performance_chart(self):
        """Render agent performance metrics"""
        st.subheader("Agent Performance Metrics")
        
        # Sample performance data
        agent_data = []
        for agent_name in self.orchestrator.agents.keys():
            # In real implementation, get actual metrics
            agent_data.append({
                "Agent": agent_name,
                "Success Rate": 0.95 + (hash(agent_name) % 10) / 200,  # Sample data
                "Avg Response Time": 2.5 + (hash(agent_name) % 20) / 10,
                "Error Count": hash(agent_name) % 5
            })
        
        df = pd.DataFrame(agent_data)
        
        fig = go.Figure()
        
        # Success rate bars
        fig.add_trace(go.Bar(
            name='Success Rate',
            x=df['Agent'],
            y=df['Success Rate'],
            yaxis='y',
            offsetgroup=1
        ))
        
        # Response time line
        fig.add_trace(go.Scatter(
            name='Avg Response Time (s)',
            x=df['Agent'],
            y=df['Avg Response Time'],
            yaxis='y2',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            xaxis=dict(title='Agents'),
            yaxis=dict(title='Success Rate', side='left'),
            yaxis2=dict(title='Response Time (s)', side='right', overlaying='y'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_workflow_execution(self):
        """Render advanced workflow execution interface"""
        st.header("Execute Enterprise Workflow")
        
        with st.form("enterprise_workflow"):
            col1, col2 = st.columns(2)
            
            with col1:
                workflow_name = st.text_input("Workflow Name", "Enterprise Data Pipeline")
                execution_strategy = st.selectbox(
                    "Execution Strategy", 
                    ["Hybrid (Recommended)", "Sequential", "Parallel"]
                )
                priority = st.selectbox("Priority", ["Normal", "High", "Critical"])
            
            with col2:
                max_retries = st.slider("Max Retries per Stage", 0, 5, 3)
                timeout = st.slider("Stage Timeout (seconds)", 30, 600, 300)
                enable_monitoring = st.checkbox("Enable Advanced Monitoring", True)
            
            st.subheader("Agent Configuration")
            
            # Dynamic agent selection
            selected_agents = st.multiselect(
                "Select Agents for Workflow",
                list(self.orchestrator.agents.keys()),
                default=list(self.orchestrator.agents.keys())[:3]
            )
            
            # Stage configuration
            stages_config = []
            for i, agent in enumerate(selected_agents):
                with st.expander(f"Stage {i+1}: {agent}", expanded=i == 0):
                    task = st.text_area(f"Task for {agent}", 
                                      f"Process data using {agent} capabilities")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        required = st.checkbox(f"Required", True, key=f"req_{i}")
                    with col2:
                        stage_timeout = st.number_input(f"Timeout (s)", 60, 1800, timeout, key=f"timeout_{i}")
                    with col3:
                        parallel_group = st.text_input(f"Parallel Group", "", key=f"group_{i}")
                    
                    stages_config.append({
                        "agent": agent,
                        "task": task,
                        "required": required,
                        "timeout": stage_timeout,
                        "parallel_group": parallel_group
                    })
            
            st.subheader("Input Data")
            input_method = st.radio("Input Method", ["JSON", "File Upload", "Database Query"])
            
            if input_method == "JSON":
                input_data = st.text_area("Input Data (JSON)", 
                    '{"data_source": "enterprise_db", "processing_date": "2024-11-15"}',
                    height=150)
            elif input_method == "File Upload":
                uploaded_file = st.file_uploader("Upload Input File", type=["json", "csv", "xlsx"])
                input_data = '{"file_uploaded": true}' if uploaded_file else '{}'
            else:
                db_query = st.text_area("Database Query", "SELECT * FROM revenue_data WHERE date >= '2024-11-01'")
                input_data = f'{{"database_query": "{db_query}"}}'
            
            submitted = st.form_submit_button("🚀 Execute Enterprise Workflow", type="primary")
        
        if submitted:
            self._execute_enterprise_workflow(
                workflow_name, execution_strategy, stages_config, input_data, 
                max_retries, enable_monitoring
            )
    
    def render_analytics_dashboard(self):
        """Render analytics and insights dashboard"""
        st.header("Analytics & Insights")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Analytics tabs
        tab1, tab2, tab3 = st.tabs(["Performance", "Resource Usage", "Trends"])
        
        with tab1:
            self.render_performance_analytics()
        
        with tab2:
            self.render_resource_analytics()
        
        with tab3:
            self.render_trend_analytics()
    
    def render_operations_dashboard(self):
        """Render operations monitoring dashboard"""
        st.header("Operations Dashboard")
        
        # Real-time status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Active Executions")
            active_executions = self.orchestrator.get_active_executions()
            
            if active_executions:
                for exec_id, info in active_executions.items():
                    with st.expander(f"🔄 {info['workflow_name']} ({exec_id[:8]}...)"):
                        st.text(f"Status: {info['status']}")
                        st.text(f"Elapsed: {info['elapsed_time']:.1f}s")
                        
                        if st.button(f"Cancel", key=f"cancel_{exec_id}"):
                            if self.orchestrator.cancel_execution(exec_id):
                                st.success("Execution cancelled")
                                st.rerun()
            else:
                st.info("No active executions")
        
        with col2:
            st.subheader("System Health")
            self.render_system_health_indicators()
        
        # Circuit breaker status
        st.subheader("Circuit Breaker Status")
        self.render_circuit_breaker_status()
        
        # Resource monitoring
        st.subheader("Resource Monitoring")
        self.render_resource_monitoring()
    
    def _execute_enterprise_workflow(self, workflow_name: str, strategy: str,
                                   stages_config: List[Dict], input_data: str,
                                   max_retries: int, enable_monitoring: bool):
        """Execute enterprise workflow with progress tracking"""
        
        # Progress containers
        progress_container = st.container()
        
        with progress_container:
            st.info(f"🚀 Executing {workflow_name}...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.empty()
            
            # Real-time metrics placeholder
            with metrics_container.container():
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                
                with met_col1:
                    stages_completed = st.metric("Stages Completed", "0")
                with met_col2:
                    current_stage = st.metric("Current Stage", "Initializing")
                with met_col3:
                    elapsed_time = st.metric("Elapsed Time", "0s")
                with met_col4:
                    estimated_remaining = st.metric("ETA", "Calculating...")
            
            # Execute workflow (in real implementation, this would be async)
            try:
                # Simulate execution with progress updates
                total_stages = len(stages_config)
                
                for i, stage_config in enumerate(stages_config):
                    # Update progress
                    progress = int((i / total_stages) * 100)
                    progress_bar.progress(progress / 100)
                    
                    status_text.markdown(f"**Stage {i+1}/{total_stages}**: Processing with {stage_config['agent']}")
                    
                    # Simulate stage execution
                    import time
                    time.sleep(2)  # Simulate processing
                
                # Completion
                progress_bar.progress(1.0)
                status_text.markdown("**✅ Workflow completed successfully!**")
                
                st.success("Enterprise workflow executed successfully!")
                
                # Show results summary
                with st.expander("Execution Summary", expanded=True):
                    st.json({
                        "workflow_name": workflow_name,
                        "status": "success",
                        "stages_completed": total_stages,
                        "execution_time": "8.5s",
                        "strategy": strategy
                    })
                
            except Exception as e:
                st.error(f"Workflow execution failed: {str(e)}")
    
    def _calculate_success_rate(self) -> float:
        """Calculate system success rate"""
        if not self.orchestrator.execution_history:
            return 1.0
        
        successful = sum(1 for exec in self.orchestrator.execution_history
                        if exec['result']['status'] == 'success')
        return successful / len(self.orchestrator.execution_history)
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time"""
        if not self.orchestrator.execution_history:
            return 0.0
        
        total_time = sum(exec['result']['execution_time'] 
                        for exec in self.orchestrator.execution_history)
        return total_time / len(self.orchestrator.execution_history)
    
    def _calculate_error_rate(self) -> float:
        """Calculate system error rate"""
        if not self.orchestrator.execution_history:
            return 0.0
        
        errors = sum(1 for exec in self.orchestrator.execution_history
                    if exec['result']['status'] == 'error')
        return errors / len(self.orchestrator.execution_history)
    
    def _calculate_agent_health(self) -> float:
        """Calculate overall agent health score"""
        # In real implementation, check actual agent health
        return 0.95  # Sample value

# Launch enterprise interface
def launch_level3_web():
    """Launch Level 3 enterprise web interface"""
    
    # Initialize complex orchestrator
    agents = {
        "data_extractor": "DataExtractionAgent",
        "data_processor": "DataProcessingAgent", 
        "financial_calculator": "FinancialCalculationAgent",
        "report_generator": "ReportGenerationAgent"
    }
    
    orchestrator = Level3Orchestrator(agents)
    
    # Create and render enterprise interface
    enterprise_interface = Level3WebInterface(orchestrator)
    enterprise_interface.render_enterprise_dashboard()

if __name__ == "__main__":
    launch_level3_web()
```

---

## Best Practices for Level 3

### Architecture Design
1. **Modular Agent Design**: Each agent should be independently deployable and testable
2. **Circuit Breaker Pattern**: Implement circuit breakers for external dependencies
3. **Resource Management**: Monitor and limit resource usage per workflow
4. **Graceful Degradation**: System should continue operating with reduced functionality
5. **Comprehensive Monitoring**: Track all aspects of system performance

### Error Handling Strategy
1. **Multi-Level Recovery**: Stage-level, workflow-level, and system-level recovery
2. **Intelligent Retry Logic**: Different retry strategies based on error types
3. **Circuit Breakers**: Prevent cascade failures across agents
4. **Fallback Mechanisms**: Alternative execution paths for critical workflows
5. **Error Categorization**: Classify errors for appropriate response strategies

### Performance Optimization
1. **Parallel Execution**: Maximize parallel processing where dependencies allow
2. **Resource Pooling**: Efficient resource allocation and reuse
3. **Caching Strategies**: Cache expensive operations and external data
4. **Load Balancing**: Distribute work across available resources
5. **Performance Monitoring**: Continuous monitoring with alerting

---

## Migration Paths

### From Level 2 to Level 3
1. **Add Advanced Orchestrator**: Upgrade to Level 3 orchestrator with advanced features
2. **Implement Circuit Breakers**: Add reliability patterns for external dependencies
3. **Add Resource Management**: Implement resource monitoring and allocation
4. **Enhance Error Handling**: Add sophisticated retry and recovery mechanisms
5. **Upgrade Web Interface**: Implement enterprise-grade dashboard and monitoring

### To Level 4 Production
- Add comprehensive logging and monitoring infrastructure
- Implement security and compliance features
- Add deployment automation and CI/CD pipelines
- Implement high availability and disaster recovery
- Add performance optimization and auto-scaling

---

## Next Steps

- **Production Ready**: [Level 4 Production Systems](level_4_production.md) for mission-critical deployments
- **Reference Materials**: [Templates and Examples](../06_reference/) for implementation guidance
- **Specialized Features**: [Testing Frameworks](../04_specialized/testing_frameworks.md) for quality assurance

---

*Level 3 systems provide enterprise-grade agent orchestration with advanced reliability, monitoring, and performance features suitable for production business applications.*