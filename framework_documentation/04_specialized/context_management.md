# Context Management and Shared State

> 🧠 **Intelligent Coordination**: Advanced patterns for managing shared state and context across multi-agent workflows.

## Navigation
- **Previous**: [Testing Frameworks](testing_frameworks.md)
- **Next**: [Level 1: Simple](../05_implementation_levels/level_1_simple.md)
- **Implementation**: [Level 2: Standard](../05_implementation_levels/level_2_standard.md) → [Level 3: Complex](../05_implementation_levels/level_3_complex.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Architecture Patterns](../02_architecture_patterns.md)

---

## Overview

Context management is critical for multi-agent systems where agents need to share information, maintain workflow state, and coordinate decisions. This section provides proven patterns for intelligent context sharing, state persistence, and agent coordination.

---

## Context Management Architecture

```
Agent A → Context Update → Shared Context Store → Context Filter → Agent B
   ↓           ↓                    ↓                    ↓           ↓
Task Data   Metadata         State Management      Relevant Info   Enhanced Input
+ Results   + Timestamps     + Versioning         + History       + Dependencies
```

### Core Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Context Store** | Central state repository | Memory, Redis, Database |
| **Context Filters** | Relevance-based filtering | Agent-specific, role-based |
| **State Manager** | Lifecycle and persistence | Versioning, cleanup, recovery |
| **Coordination Engine** | Agent synchronization | Dependencies, notifications |

---

## Context Data Structures

### Base Context Models

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid
import json

class ContextScope(Enum):
    """Define context visibility and access levels"""
    GLOBAL = "global"           # Visible to all agents
    WORKFLOW = "workflow"       # Visible within workflow
    AGENT_GROUP = "agent_group" # Visible to specific agent types
    PRIVATE = "private"         # Agent-specific only

class ContextPriority(Enum):
    """Context data priority levels"""
    CRITICAL = "critical"       # Must be preserved
    HIGH = "high"              # Important for workflow
    MEDIUM = "medium"          # Useful context
    LOW = "low"                # Optional information

@dataclass
class ContextEntry:
    """Individual context data entry"""
    key: str
    value: Any
    agent_id: str
    timestamp: datetime
    scope: ContextScope
    priority: ContextPriority
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    expires_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage"""
        return {
            "key": self.key,
            "value": self._serialize_value(self.value),
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "scope": self.scope.value,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "version": self.version,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "dependencies": self.dependencies
        }
    
    def _serialize_value(self, value: Any) -> Any:
        """Handle complex value serialization"""
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

@dataclass
class WorkflowContext:
    """Complete context for a workflow execution"""
    workflow_id: str
    context_entries: Dict[str, ContextEntry] = field(default_factory=dict)
    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_entry(self, entry: ContextEntry):
        """Add or update context entry"""
        self.context_entries[entry.key] = entry
        self.updated_at = datetime.now()
    
    def get_entry(self, key: str) -> Optional[ContextEntry]:
        """Retrieve context entry"""
        return self.context_entries.get(key)
    
    def get_entries_for_agent(self, agent_id: str, scope_filter: List[ContextScope] = None) -> List[ContextEntry]:
        """Get relevant entries for specific agent"""
        if scope_filter is None:
            scope_filter = [ContextScope.GLOBAL, ContextScope.WORKFLOW]
        
        relevant_entries = []
        for entry in self.context_entries.values():
            # Include if in scope or if agent owns the entry
            if entry.scope in scope_filter or entry.agent_id == agent_id:
                # Check if not expired
                if entry.expires_at is None or entry.expires_at > datetime.now():
                    relevant_entries.append(entry)
        
        return sorted(relevant_entries, key=lambda x: (x.priority.value, x.timestamp), reverse=True)
```

---

## Context Manager Implementation

### Core Context Manager

```python
from abc import ABC, abstractmethod
import asyncio
from typing import Protocol

class ContextStorage(Protocol):
    """Protocol for context storage backends"""
    
    async def store_context(self, workflow_id: str, context: WorkflowContext) -> bool:
        """Store complete workflow context"""
        ...
    
    async def get_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Retrieve workflow context"""
        ...
    
    async def update_entry(self, workflow_id: str, entry: ContextEntry) -> bool:
        """Update single context entry"""
        ...
    
    async def delete_context(self, workflow_id: str) -> bool:
        """Delete workflow context"""
        ...

class SharedContextManager:
    """Advanced context manager with intelligent filtering and coordination"""
    
    def __init__(self, storage: ContextStorage, config: Dict[str, Any] = None):
        self.storage = storage
        self.config = config or {}
        self.context_filters = ContextFilterRegistry()
        self.notification_manager = ContextNotificationManager()
        self.dependency_tracker = ContextDependencyTracker()
        self._active_contexts = {}  # In-memory cache
        self._lock = asyncio.Lock()
    
    async def create_workflow_context(self, workflow_id: str, initial_data: Dict[str, Any] = None) -> WorkflowContext:
        """Create new workflow context"""
        async with self._lock:
            context = WorkflowContext(
                workflow_id=workflow_id,
                execution_metadata=initial_data or {}
            )
            
            # Store in backend and cache
            await self.storage.store_context(workflow_id, context)
            self._active_contexts[workflow_id] = context
            
            return context
    
    async def get_context_for_agent(self, workflow_id: str, agent_id: str, 
                                  agent_role: str = None) -> Dict[str, Any]:
        """Get filtered context for specific agent"""
        context = await self._get_workflow_context(workflow_id)
        if not context:
            return {}
        
        # Get relevant entries for agent
        relevant_entries = context.get_entries_for_agent(agent_id)
        
        # Apply agent-specific filters
        filtered_entries = await self.context_filters.filter_for_agent(
            relevant_entries, agent_id, agent_role
        )
        
        # Build context dictionary
        agent_context = {
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "context_data": {},
            "metadata": context.execution_metadata,
            "dependencies": []
        }
        
        for entry in filtered_entries:
            agent_context["context_data"][entry.key] = entry.value
            agent_context["dependencies"].extend(entry.dependencies)
        
        # Add agent-specific state
        if agent_id in context.agent_states:
            agent_context["agent_state"] = context.agent_states[agent_id]
        
        return agent_context
    
    async def update_context(self, workflow_id: str, agent_id: str, 
                           updates: Dict[str, Any], merge_strategy: str = "smart_merge"):
        """Update context with new data from agent"""
        context = await self._get_workflow_context(workflow_id)
        if not context:
            raise ValueError(f"Workflow context not found: {workflow_id}")
        
        async with self._lock:
            # Process each update
            for key, value in updates.items():
                await self._process_context_update(
                    context, agent_id, key, value, merge_strategy
                )
            
            # Update workflow context
            context.updated_at = datetime.now()
            await self.storage.store_context(workflow_id, context)
            
            # Notify other agents of updates
            await self.notification_manager.notify_context_update(
                workflow_id, agent_id, list(updates.keys())
            )
    
    async def _process_context_update(self, context: WorkflowContext, agent_id: str,
                                    key: str, value: Any, merge_strategy: str):
        """Process individual context update"""
        existing_entry = context.get_entry(key)
        
        if existing_entry and merge_strategy == "smart_merge":
            # Intelligent merging based on data types and priorities
            merged_value = await self._smart_merge_values(
                existing_entry.value, value, existing_entry.priority
            )
            
            # Update existing entry
            existing_entry.value = merged_value
            existing_entry.version += 1
            existing_entry.timestamp = datetime.now()
            existing_entry.metadata["last_updated_by"] = agent_id
            
        else:
            # Create new entry or replace existing
            new_entry = ContextEntry(
                key=key,
                value=value,
                agent_id=agent_id,
                timestamp=datetime.now(),
                scope=self._determine_scope(key, value),
                priority=self._determine_priority(key, value),
                metadata={"created_by": agent_id}
            )
            
            context.add_entry(new_entry)
        
        # Update dependency tracking
        await self.dependency_tracker.update_dependencies(context, key, agent_id)
    
    async def _smart_merge_values(self, existing: Any, new: Any, priority: ContextPriority) -> Any:
        """Intelligently merge context values"""
        # Handle different data types intelligently
        if isinstance(existing, dict) and isinstance(new, dict):
            # Deep merge dictionaries
            merged = existing.copy()
            for k, v in new.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = await self._smart_merge_values(merged[k], v, priority)
                else:
                    merged[k] = v
            return merged
        
        elif isinstance(existing, list) and isinstance(new, list):
            # Merge lists based on priority
            if priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]:
                return existing + [item for item in new if item not in existing]
            else:
                return new  # Replace with new list
        
        elif hasattr(existing, 'merge') and hasattr(new, 'merge'):
            # Custom merge method
            return existing.merge(new)
        
        else:
            # Default: replace with new value
            return new
    
    def _determine_scope(self, key: str, value: Any) -> ContextScope:
        """Determine appropriate scope for context entry"""
        # Business logic for scope determination
        if key.startswith("global_"):
            return ContextScope.GLOBAL
        elif key.startswith("private_"):
            return ContextScope.PRIVATE
        elif isinstance(value, dict) and value.get("shared", False):
            return ContextScope.WORKFLOW
        else:
            return ContextScope.WORKFLOW
    
    def _determine_priority(self, key: str, value: Any) -> ContextPriority:
        """Determine priority level for context entry"""
        critical_keys = ["error_state", "workflow_status", "security_context"]
        high_keys = ["user_preferences", "calculation_results", "validation_results"]
        
        if key in critical_keys:
            return ContextPriority.CRITICAL
        elif key in high_keys:
            return ContextPriority.HIGH
        elif isinstance(value, dict) and value.get("priority"):
            return ContextPriority(value["priority"])
        else:
            return ContextPriority.MEDIUM
    
    async def _get_workflow_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get workflow context from cache or storage"""
        # Check cache first
        if workflow_id in self._active_contexts:
            return self._active_contexts[workflow_id]
        
        # Load from storage
        context = await self.storage.get_context(workflow_id)
        if context:
            self._active_contexts[workflow_id] = context
        
        return context
    
    async def cleanup_expired_contexts(self):
        """Clean up expired context entries"""
        for workflow_id, context in list(self._active_contexts.items()):
            expired_keys = []
            
            for key, entry in context.context_entries.items():
                if entry.expires_at and entry.expires_at <= datetime.now():
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del context.context_entries[key]
            
            # Update storage if changes were made
            if expired_keys:
                await self.storage.store_context(workflow_id, context)
```

---

## Context Filtering and Access Control

### Intelligent Context Filtering

```python
class ContextFilterRegistry:
    """Registry of context filters for different agent types and roles"""
    
    def __init__(self):
        self.filters = {}
        self.role_filters = {}
        self._register_default_filters()
    
    def register_agent_filter(self, agent_type: str, filter_func: callable):
        """Register filter function for specific agent type"""
        self.filters[agent_type] = filter_func
    
    def register_role_filter(self, role: str, filter_func: callable):
        """Register filter function for specific role"""
        self.role_filters[role] = filter_func
    
    async def filter_for_agent(self, entries: List[ContextEntry], 
                             agent_id: str, agent_role: str = None) -> List[ContextEntry]:
        """Apply appropriate filters for agent"""
        filtered_entries = entries.copy()
        
        # Apply agent-specific filters
        agent_type = self._extract_agent_type(agent_id)
        if agent_type in self.filters:
            filtered_entries = await self.filters[agent_type](filtered_entries, agent_id)
        
        # Apply role-based filters
        if agent_role and agent_role in self.role_filters:
            filtered_entries = await self.role_filters[agent_role](filtered_entries, agent_id)
        
        # Apply security filters
        filtered_entries = await self._apply_security_filters(filtered_entries, agent_id)
        
        return filtered_entries
    
    def _register_default_filters(self):
        """Register default filters for common agent types"""
        
        async def web_extractor_filter(entries: List[ContextEntry], agent_id: str) -> List[ContextEntry]:
            """Filter for web extraction agents"""
            relevant_entries = []
            for entry in entries:
                # Include data extraction results, validation rules, error states
                if any(keyword in entry.key.lower() for keyword in 
                      ["extraction", "validation", "error", "credential", "website"]):
                    relevant_entries.append(entry)
                # Include high-priority global entries
                elif entry.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]:
                    relevant_entries.append(entry)
            return relevant_entries
        
        async def financial_calculator_filter(entries: List[ContextEntry], agent_id: str) -> List[ContextEntry]:
            """Filter for financial calculation agents"""
            relevant_entries = []
            for entry in entries:
                # Include financial data, exchange rates, calculation results
                if any(keyword in entry.key.lower() for keyword in 
                      ["financial", "revenue", "calculation", "exchange", "currency", "precision"]):
                    relevant_entries.append(entry)
                # Include critical workflow state
                elif entry.priority == ContextPriority.CRITICAL:
                    relevant_entries.append(entry)
            return relevant_entries
        
        async def report_generator_filter(entries: List[ContextEntry], agent_id: str) -> List[ContextEntry]:
            """Filter for report generation agents"""
            relevant_entries = []
            for entry in entries:
                # Include processed data, templates, formatting preferences
                if any(keyword in entry.key.lower() for keyword in 
                      ["processed", "template", "format", "output", "report", "chart"]):
                    relevant_entries.append(entry)
                # Include all high-priority entries for comprehensive reporting
                elif entry.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]:
                    relevant_entries.append(entry)
            return relevant_entries
        
        # Register filters
        self.filters["web_extractor"] = web_extractor_filter
        self.filters["financial_calculator"] = financial_calculator_filter
        self.filters["report_generator"] = report_generator_filter
    
    def _extract_agent_type(self, agent_id: str) -> str:
        """Extract agent type from agent ID"""
        # Assume agent_id format like "web_extractor_001"
        parts = agent_id.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:-1])  # Remove numeric suffix
        return agent_id
    
    async def _apply_security_filters(self, entries: List[ContextEntry], agent_id: str) -> List[ContextEntry]:
        """Apply security-based filtering"""
        filtered_entries = []
        
        for entry in entries:
            # Check if entry contains sensitive information
            if self._is_sensitive_data(entry):
                # Only include if agent has appropriate access
                if await self._check_agent_access(agent_id, entry):
                    filtered_entries.append(entry)
            else:
                filtered_entries.append(entry)
        
        return filtered_entries
    
    def _is_sensitive_data(self, entry: ContextEntry) -> bool:
        """Check if context entry contains sensitive data"""
        sensitive_keywords = ["password", "key", "token", "credential", "secret"]
        
        # Check key name
        if any(keyword in entry.key.lower() for keyword in sensitive_keywords):
            return True
        
        # Check metadata
        if entry.metadata.get("sensitive", False):
            return True
        
        # Check value content (basic check)
        if isinstance(entry.value, str):
            if any(keyword in entry.value.lower() for keyword in sensitive_keywords):
                return True
        
        return False
    
    async def _check_agent_access(self, agent_id: str, entry: ContextEntry) -> bool:
        """Check if agent has access to sensitive entry"""
        # Simple access control - in production, use proper ACL
        agent_type = self._extract_agent_type(agent_id)
        
        # Web extractors need access to credentials
        if agent_type == "web_extractor" and "credential" in entry.key.lower():
            return True
        
        # Financial agents need access to API keys for exchange rates
        if agent_type == "financial_calculator" and "api_key" in entry.key.lower():
            return True
        
        # Default: no access to sensitive data
        return False
```

---

## Dependency Tracking and Coordination

### Context Dependency Management

```python
class ContextDependencyTracker:
    """Track and manage dependencies between context entries"""
    
    def __init__(self):
        self.dependency_graph = {}
        self.reverse_dependencies = {}
        self.dependency_notifications = {}
    
    async def update_dependencies(self, context: WorkflowContext, 
                                key: str, agent_id: str):
        """Update dependency tracking for context entry"""
        entry = context.get_entry(key)
        if not entry:
            return
        
        # Build dependency relationships
        for dep_key in entry.dependencies:
            if dep_key not in self.dependency_graph:
                self.dependency_graph[dep_key] = set()
            self.dependency_graph[dep_key].add(key)
            
            if key not in self.reverse_dependencies:
                self.reverse_dependencies[key] = set()
            self.reverse_dependencies[key].add(dep_key)
        
        # Check for dependency resolution
        await self._check_dependency_resolution(context, key)
    
    async def _check_dependency_resolution(self, context: WorkflowContext, key: str):
        """Check if dependencies are resolved and notify waiting agents"""
        entry = context.get_entry(key)
        if not entry:
            return
        
        # Check if this entry resolves dependencies for other entries
        if key in self.dependency_graph:
            dependent_keys = self.dependency_graph[key]
            
            for dependent_key in dependent_keys:
                dependent_entry = context.get_entry(dependent_key)
                if dependent_entry and self._are_dependencies_resolved(context, dependent_entry):
                    # Notify agents waiting for this dependency
                    await self._notify_dependency_resolved(
                        context.workflow_id, dependent_key, key
                    )
    
    def _are_dependencies_resolved(self, context: WorkflowContext, entry: ContextEntry) -> bool:
        """Check if all dependencies for an entry are resolved"""
        for dep_key in entry.dependencies:
            dep_entry = context.get_entry(dep_key)
            if not dep_entry:
                return False
            
            # Check if dependency is valid/fresh
            if dep_entry.expires_at and dep_entry.expires_at <= datetime.now():
                return False
        
        return True
    
    async def _notify_dependency_resolved(self, workflow_id: str, entry_key: str, resolved_key: str):
        """Notify agents that a dependency has been resolved"""
        notification_key = f"{workflow_id}:{entry_key}:{resolved_key}"
        
        if notification_key in self.dependency_notifications:
            callbacks = self.dependency_notifications[notification_key]
            for callback in callbacks:
                try:
                    await callback(workflow_id, entry_key, resolved_key)
                except Exception as e:
                    print(f"Dependency notification error: {e}")
    
    def register_dependency_callback(self, workflow_id: str, entry_key: str, 
                                   dependency_key: str, callback: callable):
        """Register callback for when specific dependency is resolved"""
        notification_key = f"{workflow_id}:{entry_key}:{dependency_key}"
        
        if notification_key not in self.dependency_notifications:
            self.dependency_notifications[notification_key] = []
        
        self.dependency_notifications[notification_key].append(callback)

class ContextNotificationManager:
    """Manage notifications about context changes"""
    
    def __init__(self):
        self.subscribers = {}
        self.notification_history = {}
    
    def subscribe_to_context_changes(self, workflow_id: str, agent_id: str, 
                                   keys: List[str], callback: callable):
        """Subscribe agent to context changes for specific keys"""
        subscription_key = f"{workflow_id}:{agent_id}"
        
        if subscription_key not in self.subscribers:
            self.subscribers[subscription_key] = {}
        
        for key in keys:
            if key not in self.subscribers[subscription_key]:
                self.subscribers[subscription_key][key] = []
            self.subscribers[subscription_key][key].append(callback)
    
    async def notify_context_update(self, workflow_id: str, updating_agent_id: str, 
                                  updated_keys: List[str]):
        """Notify subscribers about context updates"""
        notification_time = datetime.now()
        
        for subscription_key, key_callbacks in self.subscribers.items():
            stored_workflow_id, subscriber_agent_id = subscription_key.split(":", 1)
            
            # Only notify if it's the same workflow and different agent
            if stored_workflow_id == workflow_id and subscriber_agent_id != updating_agent_id:
                
                for updated_key in updated_keys:
                    if updated_key in key_callbacks:
                        callbacks = key_callbacks[updated_key]
                        
                        for callback in callbacks:
                            try:
                                await callback(workflow_id, updating_agent_id, updated_key)
                                
                                # Record notification
                                self._record_notification(
                                    workflow_id, subscriber_agent_id, updated_key, notification_time
                                )
                                
                            except Exception as e:
                                print(f"Context notification error: {e}")
    
    def _record_notification(self, workflow_id: str, agent_id: str, 
                           key: str, timestamp: datetime):
        """Record notification for audit and debugging"""
        record_key = f"{workflow_id}:{agent_id}"
        
        if record_key not in self.notification_history:
            self.notification_history[record_key] = []
        
        self.notification_history[record_key].append({
            "key": key,
            "timestamp": timestamp,
            "type": "context_update"
        })
        
        # Keep only recent history (last 100 notifications)
        if len(self.notification_history[record_key]) > 100:
            self.notification_history[record_key] = self.notification_history[record_key][-100:]
```

---

## Storage Backend Implementations

### Memory Storage Backend

```python
class MemoryContextStorage:
    """In-memory context storage for development and testing"""
    
    def __init__(self):
        self.contexts = {}
        self._lock = asyncio.Lock()
    
    async def store_context(self, workflow_id: str, context: WorkflowContext) -> bool:
        """Store context in memory"""
        async with self._lock:
            self.contexts[workflow_id] = context
            return True
    
    async def get_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Retrieve context from memory"""
        return self.contexts.get(workflow_id)
    
    async def update_entry(self, workflow_id: str, entry: ContextEntry) -> bool:
        """Update single context entry"""
        context = self.contexts.get(workflow_id)
        if context:
            context.add_entry(entry)
            return True
        return False
    
    async def delete_context(self, workflow_id: str) -> bool:
        """Delete context from memory"""
        if workflow_id in self.contexts:
            del self.contexts[workflow_id]
            return True
        return False

class RedisContextStorage:
    """Redis-based context storage for production"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)
        self.key_prefix = "context:"
        self.expiration_time = 86400  # 24 hours
    
    async def store_context(self, workflow_id: str, context: WorkflowContext) -> bool:
        """Store context in Redis"""
        try:
            key = f"{self.key_prefix}{workflow_id}"
            serialized_context = self._serialize_context(context)
            
            await self.redis.setex(key, self.expiration_time, serialized_context)
            return True
            
        except Exception as e:
            print(f"Redis storage error: {e}")
            return False
    
    async def get_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Retrieve context from Redis"""
        try:
            key = f"{self.key_prefix}{workflow_id}"
            serialized_context = await self.redis.get(key)
            
            if serialized_context:
                return self._deserialize_context(serialized_context)
            
            return None
            
        except Exception as e:
            print(f"Redis retrieval error: {e}")
            return None
    
    async def update_entry(self, workflow_id: str, entry: ContextEntry) -> bool:
        """Update single entry in Redis context"""
        context = await self.get_context(workflow_id)
        if context:
            context.add_entry(entry)
            return await self.store_context(workflow_id, context)
        return False
    
    async def delete_context(self, workflow_id: str) -> bool:
        """Delete context from Redis"""
        try:
            key = f"{self.key_prefix}{workflow_id}"
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            print(f"Redis deletion error: {e}")
            return False
    
    def _serialize_context(self, context: WorkflowContext) -> str:
        """Serialize context for Redis storage"""
        context_dict = {
            "workflow_id": context.workflow_id,
            "context_entries": {k: v.to_dict() for k, v in context.context_entries.items()},
            "agent_states": context.agent_states,
            "execution_metadata": context.execution_metadata,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat()
        }
        return json.dumps(context_dict)
    
    def _deserialize_context(self, serialized_data: str) -> WorkflowContext:
        """Deserialize context from Redis storage"""
        data = json.loads(serialized_data)
        
        context = WorkflowContext(
            workflow_id=data["workflow_id"],
            agent_states=data["agent_states"],
            execution_metadata=data["execution_metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        # Reconstruct context entries
        for key, entry_data in data["context_entries"].items():
            entry = ContextEntry(
                key=entry_data["key"],
                value=entry_data["value"],
                agent_id=entry_data["agent_id"],
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                scope=ContextScope(entry_data["scope"]),
                priority=ContextPriority(entry_data["priority"]),
                metadata=entry_data["metadata"],
                version=entry_data["version"],
                expires_at=datetime.fromisoformat(entry_data["expires_at"]) if entry_data["expires_at"] else None,
                dependencies=entry_data["dependencies"]
            )
            context.add_entry(entry)
        
        return context
```

---

## Agent Integration Patterns

### Context-Aware Agent Base Class

```python
class ContextAwareAgent:
    """Base class for agents with advanced context management"""
    
    def __init__(self, personality: AgentPersonality, context_manager: SharedContextManager):
        self.personality = personality
        self.context_manager = context_manager
        self.context_subscriptions = []
        self.dependency_callbacks = {}
    
    async def execute_with_context(self, task: str, workflow_id: str, 
                                 initial_context: Dict[str, Any] = None):
        """Execute task with full context management"""
        
        # Get current context for this agent
        agent_context = await self.context_manager.get_context_for_agent(
            workflow_id, self.personality.name, self.personality.role
        )
        
        # Merge with initial context
        if initial_context:
            agent_context["context_data"].update(initial_context)
        
        # Execute the actual task
        result = await self.execute_task_with_context(task, agent_context)
        
        # Update context with results
        if "context_updates" in result:
            await self.context_manager.update_context(
                workflow_id, self.personality.name, result["context_updates"]
            )
        
        return result
    
    async def execute_task_with_context(self, task: str, context: Dict[str, Any]):
        """Override this method in specific agents"""
        raise NotImplementedError
    
    def subscribe_to_context_changes(self, workflow_id: str, keys: List[str]):
        """Subscribe to specific context changes"""
        async def context_change_callback(wf_id: str, updating_agent: str, key: str):
            await self.handle_context_change(wf_id, updating_agent, key)
        
        self.context_manager.notification_manager.subscribe_to_context_changes(
            workflow_id, self.personality.name, keys, context_change_callback
        )
        
        self.context_subscriptions.append((workflow_id, keys))
    
    async def handle_context_change(self, workflow_id: str, updating_agent: str, key: str):
        """Handle context changes from other agents"""
        # Default implementation - override in specific agents
        print(f"Agent {self.personality.name} notified of context change: {key} by {updating_agent}")
    
    def wait_for_dependency(self, workflow_id: str, dependency_key: str, timeout: float = 30.0):
        """Wait for specific context dependency to be available"""
        async def dependency_callback(wf_id: str, entry_key: str, resolved_key: str):
            if resolved_key == dependency_key:
                # Dependency resolved, continue execution
                self.dependency_callbacks[dependency_key].set()
        
        # Create event for this dependency
        self.dependency_callbacks[dependency_key] = asyncio.Event()
        
        # Register callback
        self.context_manager.dependency_tracker.register_dependency_callback(
            workflow_id, f"{self.personality.name}_waiting", dependency_key, dependency_callback
        )
        
        return self.dependency_callbacks[dependency_key].wait()

# Example: Context-aware Financial Calculator Agent
class ContextAwareFinancialAgent(ContextAwareAgent):
    """Financial agent with intelligent context usage"""
    
    async def execute_task_with_context(self, task: str, context: Dict[str, Any]):
        """Execute financial calculations with context awareness"""
        
        # Extract relevant financial data from context
        revenue_data = context["context_data"].get("revenue_data", [])
        exchange_rates = context["context_data"].get("exchange_rates", {})
        calculation_preferences = context["context_data"].get("calculation_preferences", {})
        
        # Check for required dependencies
        if not revenue_data:
            # Wait for revenue data from data consolidation agent
            await self.wait_for_dependency(context["workflow_id"], "consolidated_revenue_data")
            
            # Re-fetch context after dependency is resolved
            updated_context = await self.context_manager.get_context_for_agent(
                context["workflow_id"], self.personality.name
            )
            revenue_data = updated_context["context_data"].get("revenue_data", [])
        
        # Perform calculations
        calculation_results = await self._perform_financial_calculations(
            revenue_data, exchange_rates, calculation_preferences
        )
        
        # Prepare context updates
        context_updates = {
            "calculation_results": calculation_results,
            "exchange_rates_used": exchange_rates,
            "calculation_timestamp": datetime.now().isoformat(),
            "calculation_confidence": 0.95
        }
        
        return {
            "status": "success",
            "results": calculation_results,
            "context_updates": context_updates
        }
    
    async def handle_context_change(self, workflow_id: str, updating_agent: str, key: str):
        """Handle context changes relevant to financial calculations"""
        if key == "exchange_rates" and updating_agent != self.personality.name:
            # Exchange rates updated - may need to recalculate
            print(f"Exchange rates updated by {updating_agent}, considering recalculation...")
            
            # Get updated rates
            context = await self.context_manager.get_context_for_agent(
                workflow_id, self.personality.name
            )
            
            new_rates = context["context_data"].get("exchange_rates", {})
            
            # Check if rates significantly changed
            if self._rates_significantly_changed(new_rates):
                # Trigger recalculation
                await self._trigger_recalculation(workflow_id)
    
    def _rates_significantly_changed(self, new_rates: Dict[str, float]) -> bool:
        """Check if exchange rates changed significantly"""
        # Implementation depends on business requirements
        # For example, check if any rate changed by more than 1%
        return True  # Simplified for example
    
    async def _trigger_recalculation(self, workflow_id: str):
        """Trigger recalculation due to context changes"""
        # Update context to indicate recalculation is needed
        await self.context_manager.update_context(
            workflow_id, self.personality.name, {
                "recalculation_requested": True,
                "recalculation_reason": "exchange_rates_updated",
                "recalculation_timestamp": datetime.now().isoformat()
            }
        )
```

---

## Best Practices and Patterns

### Context Management Guidelines

1. **Granular Context Scoping**: Use appropriate scope levels for different data types
2. **Intelligent Filtering**: Filter context based on agent capabilities and roles
3. **Dependency Management**: Track and resolve dependencies automatically
4. **Performance Optimization**: Cache frequently accessed context data
5. **Security**: Apply proper access controls to sensitive context information
6. **Lifecycle Management**: Clean up expired and unused context data
7. **Notification Efficiency**: Use targeted notifications to avoid spam

### Common Anti-Patterns to Avoid

1. **Context Pollution**: Adding irrelevant data to shared context
2. **Circular Dependencies**: Creating dependency loops between context entries
3. **Over-Subscribing**: Subscribing to too many context changes
4. **Context Leakage**: Exposing sensitive data to unauthorized agents
5. **State Inconsistency**: Not maintaining consistency across context updates
6. **Memory Leaks**: Not cleaning up expired context and subscriptions

---

## Memory Types in Agent Systems

### Understanding Agent Memory Architecture

**Memory Types Integration**: Advanced context management incorporates different memory types for enhanced agent intelligence and learning capabilities.

```
Input → Short-term Memory → Long-term Memory → Knowledge Retrieval
  ↓           ↓                   ↓               ↓
Working     Episodic          Semantic         Vector
Context     Events            Knowledge        Database
           + Recent          + Facts          + Embeddings
           + Temporary       + Procedures     + Similarity
```

#### Memory Type Classifications

**Short-term Memory (Working Context)**:
- Current workflow state and active data
- Temporary information for immediate processing
- Session-specific context and user interactions
- Limited capacity, automatically cleaned up

**Long-term Memory (Persistent Storage)**:
- Learned patterns and historical insights
- Agent knowledge base and experience
- Cross-session learning and adaptation
- Permanent storage with retrieval mechanisms

**Specialized Memory Types**:

1. **Semantic Memory**: Facts, knowledge, and structured information
2. **Episodic Memory**: Specific events, experiences, and workflow histories  
3. **Procedural Memory**: Skills, processes, and learned workflows

### Vector Database Integration

**Enhanced Context with Semantic Search:**

```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class MemoryEntry:
    """Enhanced memory entry with vector embeddings"""
    id: str
    content: str
    embedding: np.ndarray
    memory_type: str  # "semantic", "episodic", "procedural"
    metadata: Dict[str, Any]
    timestamp: datetime
    access_count: int = 0
    relevance_score: float = 0.0

class VectorMemoryManager:
    """Vector-based memory management for intelligent context retrieval"""
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        self.embedding_model = embedding_model
        self.memory_store: Dict[str, MemoryEntry] = {}
        self.embedding_cache = {}
        self.similarity_threshold = 0.7
        
        # Memory type configurations
        self.memory_configs = {
            "semantic": {"retention_days": 365, "max_entries": 10000},
            "episodic": {"retention_days": 90, "max_entries": 5000},
            "procedural": {"retention_days": 180, "max_entries": 2000}
        }
    
    async def store_memory(self, content: str, memory_type: str, 
                          metadata: Dict[str, Any] = None) -> str:
        """Store content in vector memory with embedding"""
        
        # Generate embedding for content
        embedding = await self._generate_embedding(content)
        
        # Create memory entry
        memory_id = f"{memory_type}_{datetime.utcnow().timestamp()}_{hash(content)}"
        
        memory_entry = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            metadata=metadata or {},
            timestamp=datetime.utcnow()
        )
        
        # Store in memory
        self.memory_store[memory_id] = memory_entry
        
        # Cleanup old memories if needed
        await self._cleanup_memory_type(memory_type)
        
        return memory_id
    
    async def retrieve_relevant_memories(self, query: str, memory_types: List[str] = None,
                                       max_results: int = 5) -> List[MemoryEntry]:
        """Retrieve memories most relevant to query using semantic similarity"""
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Filter by memory types if specified
        candidate_memories = []
        for memory in self.memory_store.values():
            if memory_types is None or memory.memory_type in memory_types:
                candidate_memories.append(memory)
        
        # Calculate similarities
        memory_similarities = []
        for memory in candidate_memories:
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            
            if similarity >= self.similarity_threshold:
                memory.access_count += 1
                memory.relevance_score = similarity
                memory_similarities.append((similarity, memory))
        
        # Sort by similarity and return top results
        memory_similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for _, memory in memory_similarities[:max_results]]
    
    async def store_workflow_experience(self, workflow_id: str, 
                                      agents_used: List[str], success: bool,
                                      execution_time: float, insights: str):
        """Store episodic memory of workflow execution"""
        
        episodic_content = {
            "workflow_id": workflow_id,
            "agents_used": agents_used,
            "success": success,
            "execution_time": execution_time,
            "insights": insights,
            "pattern_type": "workflow_execution"
        }
        
        content_text = f"""
        Workflow Execution Experience:
        - Workflow: {workflow_id}
        - Agents: {', '.join(agents_used)}
        - Success: {success}
        - Duration: {execution_time:.2f}s
        - Key Insights: {insights}
        """
        
        await self.store_memory(
            content=content_text.strip(),
            memory_type="episodic",
            metadata=episodic_content
        )
    
    async def store_semantic_knowledge(self, concept: str, definition: str, 
                                     related_concepts: List[str] = None):
        """Store semantic knowledge and relationships"""
        
        semantic_content = {
            "concept": concept,
            "definition": definition,
            "related_concepts": related_concepts or [],
            "knowledge_type": "concept_definition"
        }
        
        content_text = f"""
        Concept: {concept}
        Definition: {definition}
        Related: {', '.join(related_concepts or [])}
        """
        
        await self.store_memory(
            content=content_text.strip(),
            memory_type="semantic", 
            metadata=semantic_content
        )
    
    async def store_procedural_skill(self, skill_name: str, steps: List[str], 
                                   success_rate: float, contexts: List[str]):
        """Store procedural memory of learned skills/processes"""
        
        procedural_content = {
            "skill_name": skill_name,
            "steps": steps,
            "success_rate": success_rate,
            "applicable_contexts": contexts,
            "skill_type": "learned_procedure"
        }
        
        content_text = f"""
        Skill: {skill_name}
        Success Rate: {success_rate:.2%}
        Steps: {' → '.join(steps)}
        Contexts: {', '.join(contexts)}
        """
        
        await self.store_memory(
            content=content_text.strip(),
            memory_type="procedural",
            metadata=procedural_content
        )
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (placeholder - integrate with actual embedding service)"""
        
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # In production, call actual embedding service (OpenAI, etc.)
        # For now, return a mock embedding
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        mock_embedding = np.random.rand(1536)  # OpenAI embedding dimension
        
        # Cache the embedding
        self.embedding_cache[text] = mock_embedding
        
        return mock_embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def _cleanup_memory_type(self, memory_type: str):
        """Clean up old memories based on retention policies"""
        
        config = self.memory_configs.get(memory_type, {})
        max_entries = config.get("max_entries", 1000)
        retention_days = config.get("retention_days", 30)
        
        # Get memories of this type
        type_memories = [
            (memory_id, memory) for memory_id, memory in self.memory_store.items()
            if memory.memory_type == memory_type
        ]
        
        # Remove old memories
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        for memory_id, memory in type_memories:
            if memory.timestamp < cutoff_date:
                del self.memory_store[memory_id]
        
        # Limit total entries (keep most recently accessed)
        remaining_memories = [
            (memory_id, memory) for memory_id, memory in self.memory_store.items()
            if memory.memory_type == memory_type
        ]
        
        if len(remaining_memories) > max_entries:
            # Sort by access count and recency
            remaining_memories.sort(
                key=lambda x: (x[1].access_count, x[1].timestamp.timestamp()),
                reverse=True
            )
            
            # Remove least accessed/oldest memories
            for memory_id, _ in remaining_memories[max_entries:]:
                del self.memory_store[memory_id]

class MemoryEnhancedContextManager(SharedContextManager):
    """Context manager enhanced with intelligent memory capabilities"""
    
    def __init__(self, storage: ContextStorage, memory_manager: VectorMemoryManager):
        super().__init__(storage)
        self.memory_manager = memory_manager
    
    async def get_context_with_memory(self, workflow_id: str, agent_id: str,
                                    current_task: str, agent_role: str = None) -> Dict[str, Any]:
        """Get context enhanced with relevant memories"""
        
        # Get standard context
        base_context = await self.get_context_for_agent(workflow_id, agent_id, agent_role)
        
        # Retrieve relevant memories based on current task
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(
            query=current_task,
            memory_types=["semantic", "episodic", "procedural"],
            max_results=3
        )
        
        # Add memory insights to context
        if relevant_memories:
            memory_insights = []
            
            for memory in relevant_memories:
                insight = {
                    "type": memory.memory_type,
                    "relevance": memory.relevance_score,
                    "content": memory.content,
                    "metadata": memory.metadata
                }
                memory_insights.append(insight)
            
            base_context["memory_insights"] = memory_insights
            base_context["has_relevant_experience"] = True
        else:
            base_context["has_relevant_experience"] = False
        
        return base_context
    
    async def learn_from_execution(self, workflow_id: str, agent_id: str,
                                 task: str, result: Dict[str, Any], 
                                 execution_time: float):
        """Learn from agent execution and store insights"""
        
        success = result.get("status") == "success"
        
        # Store episodic memory of this execution
        insights = self._extract_insights_from_result(result)
        
        await self.memory_manager.store_workflow_experience(
            workflow_id=workflow_id,
            agents_used=[agent_id],
            success=success,
            execution_time=execution_time,
            insights=insights
        )
        
        # Store procedural knowledge if successful
        if success and "learned_pattern" in result:
            pattern = result["learned_pattern"]
            
            await self.memory_manager.store_procedural_skill(
                skill_name=f"{agent_id}_{pattern.get('skill_name', 'task_execution')}",
                steps=pattern.get("steps", []),
                success_rate=1.0,  # Will be updated over time
                contexts=[task]
            )
    
    def _extract_insights_from_result(self, result: Dict[str, Any]) -> str:
        """Extract meaningful insights from execution result"""
        
        insights = []
        
        if result.get("status") == "success":
            insights.append("Execution completed successfully")
            
            if "processing_time" in result:
                insights.append(f"Processing took {result['processing_time']:.2f}s")
            
            if "confidence" in result:
                insights.append(f"Result confidence: {result['confidence']:.2%}")
        
        else:
            insights.append(f"Execution failed: {result.get('error', 'Unknown error')}")
        
        return "; ".join(insights)

# Usage Example
async def example_memory_enhanced_workflow():
    """Example of using memory-enhanced context management"""
    
    # Initialize components
    memory_manager = VectorMemoryManager()
    storage = MemoryContextStorage()
    context_manager = MemoryEnhancedContextManager(storage, memory_manager)
    
    # Store some initial knowledge
    await memory_manager.store_semantic_knowledge(
        concept="Financial Data Validation",
        definition="Process of verifying financial data accuracy, format, and completeness",
        related_concepts=["Data Quality", "Compliance", "Accuracy Checking"]
    )
    
    # Simulate workflow execution with memory
    workflow_id = "demo_workflow_001"
    agent_id = "financial_validator"
    task = "Validate quarterly revenue data for compliance"
    
    # Get context with memory insights
    enhanced_context = await context_manager.get_context_with_memory(
        workflow_id, agent_id, task, "financial_analyst"
    )
    
    print("Enhanced context with memory:")
    print(f"Has relevant experience: {enhanced_context['has_relevant_experience']}")
    
    if enhanced_context.get("memory_insights"):
        print("Relevant memories found:")
        for insight in enhanced_context["memory_insights"]:
            print(f"- {insight['type']}: {insight['content'][:100]}... (relevance: {insight['relevance']:.2f})")
    
    # Simulate task execution result
    execution_result = {
        "status": "success",
        "processing_time": 2.5,
        "confidence": 0.95,
        "learned_pattern": {
            "skill_name": "quarterly_revenue_validation",
            "steps": ["Load data", "Check formats", "Validate ranges", "Generate report"]
        }
    }
    
    # Learn from this execution
    await context_manager.learn_from_execution(
        workflow_id, agent_id, task, execution_result, 2.5
    )
    
    print("Learning completed - knowledge stored for future use")

if __name__ == "__main__":
    asyncio.run(example_memory_enhanced_workflow())
```

### Integration with Existing Context Management

**Memory-Context Bridge**:

```python
class UnifiedContextMemorySystem:
    """Unified system combining context management with memory capabilities"""
    
    def __init__(self):
        self.context_manager = SharedContextManager(MemoryContextStorage())
        self.memory_manager = VectorMemoryManager()
        self.learning_enabled = True
    
    async def process_with_memory_learning(self, workflow_id: str, agent_id: str,
                                         task: str, input_data: Dict[str, Any]):
        """Process task with memory-enhanced context and learning"""
        
        # 1. Get memory-enhanced context
        context = await self.context_manager.get_context_with_memory(
            workflow_id, agent_id, task
        )
        
        # 2. Execute task (would call actual agent)
        result = await self._execute_agent_task(agent_id, task, input_data, context)
        
        # 3. Learn from execution if enabled
        if self.learning_enabled:
            await self.context_manager.learn_from_execution(
                workflow_id, agent_id, task, result, result.get("execution_time", 0)
            )
        
        # 4. Update context with results
        await self.context_manager.update_context(
            workflow_id, agent_id, {"latest_result": result}
        )
        
        return result
    
    async def _execute_agent_task(self, agent_id: str, task: str, 
                                input_data: Dict[str, Any], context: Dict[str, Any]):
        """Placeholder for actual agent execution"""
        # In real implementation, this would call the actual agent
        return {
            "status": "success",
            "result": "Task completed with memory insights",
            "execution_time": 1.5,
            "used_memory_insights": len(context.get("memory_insights", []))
        }
```

**Memory-Enhanced Best Practices**:

1. **Selective Memory Storage**: Store only meaningful experiences and insights
2. **Relevance Filtering**: Use similarity thresholds to avoid information overload
3. **Memory Cleanup**: Implement retention policies to manage memory growth
4. **Learning Integration**: Automatically learn from successful workflows
5. **Context Balance**: Balance memory insights with current context relevance

---

## Next Steps

- **Implementation**: [Level 1 Simple Systems](../05_implementation_levels/level_1_simple.md)
- **Advanced Patterns**: [Level 3 Complex Systems](../05_implementation_levels/level_3_complex.md)
- **Production Deployment**: [Level 4 Production Systems](../05_implementation_levels/level_4_production.md)

---

*Advanced context management enables sophisticated agent coordination while maintaining performance, security, and reliability across complex multi-agent workflows.*