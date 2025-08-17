# Real-Time Progress Tracking Systems

> 📊 **Transparent Operations**: Implement real-time progress feedback for long-running agent workflows across all interfaces.

## Navigation
- **Previous**: [Web Integration](web_integration.md)
- **Next**: [Financial Data Handling](../04_specialized/financial_precision.md)
- **Specialized**: [Context Management](../04_specialized/context_management.md) → [Testing Frameworks](../04_specialized/testing_frameworks.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Implementation Levels](../05_implementation_levels/)

---

## Overview

Long-running agent workflows require transparent progress feedback for user experience and operational monitoring. This section provides proven patterns for implementing real-time progress tracking across different interfaces and workflow complexities.

---

## Progress Tracking Pattern

```
Operation Start → Progress Callbacks → Real-time Updates
       ↓                ↓                     ↓
  Initialize        Stage Updates        User Interface
   Tracker           + Metadata          Updates (CLI/Web)
       ↓                ↓                     ↓
   Execution      Context Updates       Completion
   Monitoring     + Error Handling      Notification
```

---

## Core Progress Tracking Components

### Progress Status and Data Structures

```python
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List
import time
from enum import Enum
import uuid

class ProgressStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ProgressUpdate:
    """Standard progress update structure"""
    operation_id: str
    stage: str
    progress_percentage: int
    message: str
    status: ProgressStatus
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation_id": self.operation_id,
            "stage": self.stage,
            "progress_percentage": self.progress_percentage,
            "message": self.message,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }

@dataclass
class StageInfo:
    """Information about workflow stage"""
    name: str
    description: str
    estimated_duration: Optional[float] = None
    dependencies: List[str] = None
    critical: bool = True
```

### Universal Progress Tracker

```python
class ProgressTracker:
    """Universal progress tracker for agent operations"""
    
    def __init__(self, operation_id: str, total_stages: int = 1):
        self.operation_id = operation_id
        self.total_stages = total_stages
        self.current_stage = 0
        self.start_time = time.time()
        self.stage_history = []
        self.callbacks = []
        self.status = ProgressStatus.PENDING
        self.estimated_completion = None
        
    def add_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Add progress callback function"""
        self.callbacks.append(callback)
    
    def start_operation(self, message: str = "Starting operation..."):
        """Mark operation as started"""
        self.status = ProgressStatus.IN_PROGRESS
        self.start_time = time.time()
        self._notify_callbacks(0, "Initialization", message)
    
    def update_stage(self, stage_name: str, progress: int, message: str = ""):
        """Update current stage progress"""
        # Calculate estimated completion time
        if progress > 0:
            elapsed = time.time() - self.start_time
            total_estimated = (elapsed / progress) * 100
            self.estimated_completion = self.start_time + total_estimated
        
        update = ProgressUpdate(
            operation_id=self.operation_id,
            stage=stage_name,
            progress_percentage=progress,
            message=message,
            status=self.status,
            timestamp=time.time(),
            metadata={
                "elapsed_time": time.time() - self.start_time,
                "current_stage": self.current_stage,
                "total_stages": self.total_stages,
                "estimated_completion": self.estimated_completion,
                "stage_history": len(self.stage_history)
            }
        )
        
        self._notify_callbacks_update(update)
    
    def complete_stage(self, stage_name: str, message: str = "Stage completed"):
        """Mark current stage as complete and advance"""
        stage_completion_time = time.time()
        
        self.stage_history.append({
            "name": stage_name,
            "started_at": getattr(self, '_current_stage_start', self.start_time),
            "completed_at": stage_completion_time,
            "duration": stage_completion_time - getattr(self, '_current_stage_start', self.start_time),
            "message": message
        })
        
        self.current_stage += 1
        overall_progress = int((self.current_stage / self.total_stages) * 100)
        
        self.update_stage(stage_name, 100, message)
        
        if self.current_stage >= self.total_stages:
            self.complete_operation("All stages completed successfully")
        else:
            # Prepare for next stage
            self._current_stage_start = time.time()
    
    def complete_operation(self, message: str = "Operation completed"):
        """Mark entire operation as complete"""
        self.status = ProgressStatus.COMPLETED
        completion_time = time.time()
        
        # Calculate final statistics
        total_duration = completion_time - self.start_time
        
        final_update = ProgressUpdate(
            operation_id=self.operation_id,
            stage="Completed",
            progress_percentage=100,
            message=message,
            status=self.status,
            timestamp=completion_time,
            metadata={
                "total_duration": total_duration,
                "stages_completed": len(self.stage_history),
                "average_stage_duration": total_duration / max(len(self.stage_history), 1),
                "completion_time": completion_time
            }
        )
        
        self._notify_callbacks_update(final_update)
    
    def error_operation(self, error_message: str):
        """Mark operation as failed"""
        self.status = ProgressStatus.ERROR
        error_time = time.time()
        
        error_update = ProgressUpdate(
            operation_id=self.operation_id,
            stage="Error",
            progress_percentage=int((self.current_stage / self.total_stages) * 100),
            message=error_message,
            status=self.status,
            timestamp=error_time,
            metadata={
                "error_time": error_time,
                "elapsed_time": error_time - self.start_time,
                "stages_completed": len(self.stage_history),
                "failed_stage": self.current_stage
            }
        )
        
        self._notify_callbacks_update(error_update)
    
    def cancel_operation(self, reason: str = "Operation cancelled"):
        """Cancel the operation"""
        self.status = ProgressStatus.CANCELLED
        self._notify_callbacks(
            int((self.current_stage / self.total_stages) * 100),
            "Cancelled",
            reason
        )
    
    def _notify_callbacks(self, progress: int, stage: str, message: str):
        """Notify all registered callbacks"""
        update = ProgressUpdate(
            operation_id=self.operation_id,
            stage=stage,
            progress_percentage=progress,
            message=message,
            status=self.status,
            timestamp=time.time()
        )
        self._notify_callbacks_update(update)
    
    def _notify_callbacks_update(self, update: ProgressUpdate):
        """Send update to all callbacks"""
        for callback in self.callbacks:
            try:
                callback(update)
            except Exception as e:
                # Log callback errors but don't fail the operation
                print(f"Progress callback error: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get operation summary"""
        return {
            "operation_id": self.operation_id,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "total_stages": self.total_stages,
            "elapsed_time": time.time() - self.start_time,
            "stages_completed": len(self.stage_history),
            "estimated_completion": self.estimated_completion
        }
```

---

## Interface-Specific Progress Handlers

### Streamlit Progress Handler

```python
import streamlit as st

class StreamlitProgressHandler:
    """Streamlit-specific progress visualization"""
    
    def __init__(self, progress_bar=None, status_container=None, 
                 metrics_container=None):
        self.progress_bar = progress_bar
        self.status_container = status_container
        self.metrics_container = metrics_container
        self.start_time = time.time()
        self.last_update = None
    
    def handle_progress(self, update: ProgressUpdate):
        """Handle progress update for Streamlit interface"""
        self.last_update = update
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.progress(update.progress_percentage / 100.0)
        
        # Update status display
        if self.status_container:
            self._update_status_display(update)
        
        # Update metrics
        if self.metrics_container:
            self._update_metrics_display(update)
    
    def _update_status_display(self, update: ProgressUpdate):
        """Update status text with rich formatting"""
        elapsed = time.time() - self.start_time
        
        # Status icon based on current status
        status_icons = {
            ProgressStatus.PENDING: "⏳",
            ProgressStatus.IN_PROGRESS: "🔄",
            ProgressStatus.COMPLETED: "✅",
            ProgressStatus.ERROR: "❌",
            ProgressStatus.CANCELLED: "🚫"
        }
        
        status_icon = status_icons.get(update.status, "🔄")
        
        # Estimate remaining time
        eta_text = ""
        if update.metadata and update.metadata.get('estimated_completion'):
            remaining = update.metadata['estimated_completion'] - time.time()
            if remaining > 0:
                eta_text = f" | ⏰ ETA: {remaining:.0f}s"
        
        status_text = f"""
        {status_icon} **{update.stage}** - {update.message}
        
        ⏱️ Elapsed: {elapsed:.1f}s | 📊 Progress: {update.progress_percentage}%{eta_text}
        🔄 Status: {update.status.value.replace('_', ' ').title()}
        """
        
        self.status_container.markdown(status_text)
    
    def _update_metrics_display(self, update: ProgressUpdate):
        """Update metrics in dedicated container"""
        if not update.metadata:
            return
        
        col1, col2, col3 = self.metrics_container.columns(3)
        
        with col1:
            st.metric(
                "Current Stage", 
                f"{update.metadata.get('current_stage', 0) + 1}/{update.metadata.get('total_stages', 1)}"
            )
        
        with col2:
            elapsed = update.metadata.get('elapsed_time', 0)
            st.metric("Elapsed Time", f"{elapsed:.1f}s")
        
        with col3:
            if update.metadata.get('estimated_completion'):
                remaining = update.metadata['estimated_completion'] - time.time()
                st.metric("ETA", f"{max(0, remaining):.0f}s")
    
    def create_stage_visualization(self, stages: List[StageInfo], current_stage: int):
        """Create visual stage progress indicator"""
        if not self.status_container:
            return
        
        stage_display = []
        for i, stage in enumerate(stages):
            if i < current_stage:
                icon = "✅"
                status = "completed"
            elif i == current_stage:
                icon = "🔄"
                status = "in-progress"
            else:
                icon = "⏳"
                status = "pending"
            
            stage_display.append(f"{icon} {stage.name}")
        
        self.status_container.markdown("**Workflow Stages:**\n" + "\n".join(stage_display))

class CLIProgressHandler:
    """CLI-specific progress display with rich formatting"""
    
    def __init__(self, verbose: bool = True, show_eta: bool = True):
        self.verbose = verbose
        self.show_eta = show_eta
        self.start_time = time.time()
        self.last_progress = 0
    
    def handle_progress(self, update: ProgressUpdate):
        """Handle progress update for CLI interface"""
        elapsed = time.time() - self.start_time
        
        if self.verbose:
            self._display_verbose_progress(update, elapsed)
        else:
            self._display_simple_progress(update)
        
        self.last_progress = update.progress_percentage
    
    def _display_verbose_progress(self, update: ProgressUpdate, elapsed: float):
        """Display detailed progress information"""
        # Create progress bar
        bar_width = 20
        filled = int((update.progress_percentage / 100) * bar_width)
        progress_bar = "█" * filled + "░" * (bar_width - filled)
        
        # Calculate ETA if available
        eta_text = ""
        if self.show_eta and update.metadata and update.metadata.get('estimated_completion'):
            remaining = update.metadata['estimated_completion'] - time.time()
            if remaining > 0:
                eta_text = f" | ETA: {remaining:.0f}s"
        
        # Stage information
        stage_info = ""
        if update.metadata:
            current = update.metadata.get('current_stage', 0) + 1
            total = update.metadata.get('total_stages', 1)
            stage_info = f" | Stage: {current}/{total}"
        
        print(f"\r[{progress_bar}] {update.progress_percentage:3d}% | "
              f"{update.stage}: {update.message} | "
              f"Elapsed: {elapsed:.1f}s{eta_text}{stage_info}", end="", flush=True)
        
        # New line on completion or error
        if update.status in [ProgressStatus.COMPLETED, ProgressStatus.ERROR, ProgressStatus.CANCELLED]:
            print()  # New line
    
    def _display_simple_progress(self, update: ProgressUpdate):
        """Display minimal progress information"""
        print(f"[{update.progress_percentage:3d}%] {update.stage}")
    
    def display_summary(self, tracker: ProgressTracker):
        """Display operation summary"""
        summary = tracker.get_summary()
        
        print(f"\n📋 Operation Summary:")
        print(f"   Status: {summary['status'].title()}")
        print(f"   Duration: {summary['elapsed_time']:.1f}s")
        print(f"   Stages: {summary['stages_completed']}/{summary['total_stages']}")
        
        if tracker.stage_history:
            print(f"\n🕐 Stage Timings:")
            for stage in tracker.stage_history:
                print(f"   {stage['name']}: {stage['duration']:.1f}s")

class APIProgressHandler:
    """API-specific progress handling with webhooks and persistence"""
    
    def __init__(self, webhook_url: Optional[str] = None, 
                 operation_id: str = None,
                 persistence_backend: str = "memory"):
        self.webhook_url = webhook_url
        self.operation_id = operation_id
        self.progress_log = []
        self.persistence = self._init_persistence(persistence_backend)
    
    async def handle_progress(self, update: ProgressUpdate):
        """Handle progress update for API interface"""
        # Store in progress log
        self.progress_log.append(update)
        
        # Persist update
        if self.persistence:
            await self.persistence.store_progress_update(update)
        
        # Send webhook if configured
        if self.webhook_url:
            await self._send_webhook(update)
        
        # Update operation status in storage
        await self._update_operation_status(update)
    
    async def _send_webhook(self, update: ProgressUpdate):
        """Send progress update via webhook"""
        import aiohttp
        
        payload = update.to_dict()
        
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status != 200:
                        print(f"Webhook delivery warning: HTTP {response.status}")
        except Exception as e:
            print(f"Webhook delivery failed: {e}")
    
    async def _update_operation_status(self, update: ProgressUpdate):
        """Update operation status in persistent storage"""
        if self.persistence:
            await self.persistence.update_operation_status(
                update.operation_id, 
                update.status,
                update.to_dict()
            )
    
    def _init_persistence(self, backend: str):
        """Initialize persistence backend"""
        if backend == "redis":
            return RedisProgressPersistence()
        elif backend == "database":
            return DatabaseProgressPersistence()
        else:
            return MemoryProgressPersistence()
    
    async def get_progress_history(self, operation_id: str) -> List[ProgressUpdate]:
        """Retrieve progress history for an operation"""
        if self.persistence:
            return await self.persistence.get_progress_history(operation_id)
        return []
```

---

## Workflow Integration Patterns

### Progress-Aware Orchestrator

```python
class ProgressAwareOrchestrator:
    """Orchestrator with integrated progress tracking"""
    
    def __init__(self, agents: Dict[str, 'Agent']):
        self.agents = agents
        self.active_trackers = {}
    
    async def execute_workflow_with_progress(self, workflow_config: 'WorkflowConfig', 
                                           progress_callback: Optional[Callable] = None):
        """Execute workflow with comprehensive progress tracking"""
        
        operation_id = f"workflow_{int(time.time() * 1000)}"
        
        # Initialize progress tracking
        total_stages = len(workflow_config.stages)
        progress_tracker = ProgressTracker(operation_id, total_stages)
        
        if progress_callback:
            progress_tracker.add_callback(progress_callback)
        
        self.active_trackers[operation_id] = progress_tracker
        progress_tracker.start_operation("Initializing workflow...")
        
        try:
            results = []
            
            for i, stage in enumerate(workflow_config.stages):
                stage_name = stage.get("name", f"Stage {i+1}")
                
                # Start stage
                progress_tracker.update_stage(
                    stage_name, 
                    0, 
                    f"Starting {stage_name.lower()}..."
                )
                
                # Execute stage with sub-progress tracking
                stage_result = await self._execute_stage_with_progress(
                    stage, progress_tracker, stage_name, operation_id
                )
                
                results.append(stage_result)
                
                # Complete stage
                progress_tracker.complete_stage(
                    stage_name,
                    f"{stage_name} completed successfully"
                )
            
            progress_tracker.complete_operation("Workflow completed successfully")
            
            return {
                "operation_id": operation_id,
                "status": "completed",
                "results": results,
                "summary": progress_tracker.get_summary()
            }
            
        except Exception as e:
            progress_tracker.error_operation(f"Workflow failed: {str(e)}")
            raise
        finally:
            # Cleanup
            if operation_id in self.active_trackers:
                del self.active_trackers[operation_id]
    
    async def _execute_stage_with_progress(self, stage: Dict, progress_tracker: ProgressTracker, 
                                         stage_name: str, operation_id: str):
        """Execute individual stage with progress updates"""
        agent_name = stage["agent_name"]
        
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        
        # Create stage-specific progress callback
        def stage_progress_callback(sub_progress: int, message: str = ""):
            # Map sub-progress to overall stage progress (reserve 10% for completion)
            stage_progress = min(90, sub_progress)
            progress_tracker.update_stage(stage_name, stage_progress, message)
        
        # Execute agent with progress callback
        result = await agent.execute(
            stage["task"], 
            stage.get("context", {}),
            progress_callback=stage_progress_callback
        )
        
        return result
    
    def get_operation_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for an operation"""
        if operation_id in self.active_trackers:
            return self.active_trackers[operation_id].get_summary()
        return None
    
    def cancel_operation(self, operation_id: str, reason: str = "User requested cancellation"):
        """Cancel a running operation"""
        if operation_id in self.active_trackers:
            self.active_trackers[operation_id].cancel_operation(reason)
            return True
        return False
```

### Agent-Level Progress Integration

```python
class ProgressAwareAgent:
    """Agent with built-in progress reporting"""
    
    def __init__(self, personality: 'AgentPersonality'):
        self.personality = personality
        self.current_progress_callback = None
    
    async def execute(self, task: str, context: Dict = None, 
                     progress_callback: Optional[Callable] = None):
        """Execute with progress reporting"""
        self.current_progress_callback = progress_callback
        
        try:
            # Initialization
            self._report_progress(5, "Initializing agent...")
            
            # Task analysis
            self._report_progress(15, "Analyzing task requirements...")
            task_analysis = await self._analyze_task(task, context)
            
            # Processing
            self._report_progress(30, "Processing with LLM...")
            llm_result = await self._process_with_llm(task, context, task_analysis)
            
            # Validation
            self._report_progress(70, "Validating results...")
            validated_result = await self._validate_result(llm_result)
            
            # Formatting
            self._report_progress(90, "Formatting output...")
            formatted_result = await self._format_output(validated_result)
            
            # Completion
            self._report_progress(100, "Task completed successfully")
            
            return {
                "status": "success",
                "agent": self.personality.name,
                "content": formatted_result,
                "confidence": validated_result.get("confidence", 0.9)
            }
            
        except Exception as e:
            self._report_progress(0, f"Error: {str(e)}")
            raise
    
    def _report_progress(self, progress: int, message: str):
        """Report progress if callback is available"""
        if self.current_progress_callback:
            try:
                self.current_progress_callback(progress, message)
            except Exception as e:
                # Don't fail the main operation due to progress reporting errors
                print(f"Progress reporting error: {e}")
    
    async def _analyze_task(self, task: str, context: Dict):
        """Analyze task (with progress reporting)"""
        # Simulate analysis work
        await asyncio.sleep(0.5)
        return {"complexity": "standard", "estimated_time": 10}
    
    async def _process_with_llm(self, task: str, context: Dict, analysis: Dict):
        """Process with LLM (with progress reporting)"""
        # Simulate LLM processing with intermediate progress
        for i in range(3):
            await asyncio.sleep(1)
            progress = 30 + (i + 1) * 10
            self._report_progress(progress, f"LLM processing step {i + 1}/3...")
        
        return {"content": f"Processed: {task}", "confidence": 0.92}
    
    async def _validate_result(self, result: Dict):
        """Validate result (with progress reporting)"""
        await asyncio.sleep(0.3)
        return result
    
    async def _format_output(self, result: Dict):
        """Format output (with progress reporting)"""
        await asyncio.sleep(0.2)
        return result["content"]
```

---

## Progress Persistence and Recovery

### Progress Persistence

```python
class ProgressPersistence:
    """Base class for progress persistence"""
    
    async def store_progress_update(self, update: ProgressUpdate):
        """Store progress update"""
        raise NotImplementedError
    
    async def get_progress_history(self, operation_id: str) -> List[ProgressUpdate]:
        """Get progress history for operation"""
        raise NotImplementedError
    
    async def update_operation_status(self, operation_id: str, status: ProgressStatus, metadata: Dict):
        """Update operation status"""
        raise NotImplementedError

class RedisProgressPersistence(ProgressPersistence):
    """Redis-based progress persistence"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)
    
    async def store_progress_update(self, update: ProgressUpdate):
        """Store progress update in Redis"""
        key = f"progress:{update.operation_id}"
        await self.redis.lpush(key, json.dumps(update.to_dict()))
        await self.redis.expire(key, 86400)  # 24 hour expiry
    
    async def get_progress_history(self, operation_id: str) -> List[ProgressUpdate]:
        """Get progress history from Redis"""
        key = f"progress:{operation_id}"
        history = await self.redis.lrange(key, 0, -1)
        
        updates = []
        for item in reversed(history):
            data = json.loads(item)
            update = ProgressUpdate(
                operation_id=data["operation_id"],
                stage=data["stage"],
                progress_percentage=data["progress_percentage"],
                message=data["message"],
                status=ProgressStatus(data["status"]),
                timestamp=data["timestamp"],
                metadata=data.get("metadata")
            )
            updates.append(update)
        
        return updates

class ProgressRecovery:
    """Recover and resume operations after system restart"""
    
    def __init__(self, persistence: ProgressPersistence):
        self.persistence = persistence
    
    async def recover_operations(self) -> List[str]:
        """Recover interrupted operations"""
        # Implementation depends on persistence backend
        # Could scan for operations with IN_PROGRESS status
        pass
    
    async def resume_operation(self, operation_id: str) -> bool:
        """Attempt to resume specific operation"""
        history = await self.persistence.get_progress_history(operation_id)
        
        if not history:
            return False
        
        last_update = history[-1]
        if last_update.status == ProgressStatus.IN_PROGRESS:
            # Determine if operation can be resumed or should be marked as failed
            # This is highly dependent on the specific workflow
            return await self._attempt_resume(operation_id, last_update)
        
        return False
    
    async def _attempt_resume(self, operation_id: str, last_update: ProgressUpdate) -> bool:
        """Attempt to resume operation from last known state"""
        # Implementation depends on specific workflow requirements
        # Could involve restarting from last checkpoint or marking as failed
        return False
```

---

## Best Practices

1. **Granular Updates**: Provide frequent, meaningful progress updates
2. **Error Transparency**: Clear error messages with recovery suggestions  
3. **Performance Impact**: Minimize overhead of progress tracking
4. **Callback Safety**: Handle callback failures gracefully
5. **Persistence Strategy**: Save progress for long-running operations
6. **User Feedback**: Provide ETA and remaining time estimates
7. **Cross-Interface Consistency**: Maintain progress semantics across interfaces

## Progress Design Patterns

### Pattern 1: Stage-Based Progress
- Break workflows into logical stages
- Report completion percentage per stage
- Provide stage-specific status messages

### Pattern 2: Nested Progress
- Parent operation tracks overall progress
- Child operations report sub-progress
- Automatically aggregate progress levels

### Pattern 3: Estimated Time Remaining
- Calculate ETA based on historical performance
- Update estimates as operation progresses
- Handle variable-duration stages gracefully

---

## Next Steps

- **Financial Data**: [Precision Data Handling](../04_specialized/financial_precision.md)
- **Implementation**: [Level 2 Standard Systems](../05_implementation_levels/level_2_standard.md)
- **Web Integration**: [Complete Web Interface Patterns](web_integration.md)

---

*Real-time progress tracking transforms long-running agent operations from black boxes into transparent, user-friendly experiences across all interface types.*