"""
Base Agent Class for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import structlog
import asyncio
from datetime import datetime
from enum import Enum
import uuid

from core.settings import get_settings
from core.redis_client import cache_manager
from database.connection import get_db_context
from database.models import AgentMetrics


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    STOPPED = "stopped"


class BaseAgent(ABC):
    """
    Base agent class implementing the Enhanced Framework V3.1 patterns
    
    Features:
    - Structured logging with context
    - Performance metrics tracking
    - Error recovery mechanisms
    - State management
    - Async/await support
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.settings = get_settings()
        
        # Set up structured logger with agent context
        self.logger = structlog.get_logger(self.name).bind(
            agent_id=self.agent_id,
            agent_name=self.name
        )
        
        # Performance tracking
        self.metrics = {
            "tasks_processed": 0,
            "errors_count": 0,
            "total_processing_time": 0.0,
            "last_activity": None,
            "start_time": datetime.utcnow(),
        }
        
        # State management
        self.state = {}
        self.is_running = False
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        self.logger.info("Agent initialized", config=self.config)
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.logger.info("Initializing agent")
            
            # Agent-specific initialization
            await self._initialize_agent()
            
            self.status = AgentStatus.IDLE
            self.is_running = True
            
            await self._log_metric("agent_initialized", 1)
            self.logger.info("Agent initialization completed")
            return True
            
        except Exception as e:
            self.logger.error("Agent initialization failed", error=str(e))
            self.status = AgentStatus.ERROR
            await self._log_metric("initialization_errors", 1)
            return False
    
    async def shutdown(self):
        """Shutdown the agent gracefully"""
        try:
            self.logger.info("Shutting down agent")
            
            # Agent-specific cleanup
            await self._cleanup_agent()
            
            self.is_running = False
            self.status = AgentStatus.STOPPED
            
            # Save final metrics
            await self._save_metrics()
            
            self.logger.info("Agent shutdown completed")
            
        except Exception as e:
            self.logger.error("Agent shutdown error", error=str(e))
    
    async def process(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing method with error handling and metrics
        
        Args:
            data: Input data to process
            context: Additional context information
            
        Returns:
            Processing result dictionary
        """
        start_time = datetime.utcnow()
        context = context or {}
        
        try:
            self.status = AgentStatus.PROCESSING
            self.logger.info("Processing started", data_keys=list(data.keys()))
            
            # Add agent context
            context.update({
                "agent_name": self.name,
                "agent_id": self.agent_id,
                "processing_id": str(uuid.uuid4()),
                "start_time": start_time.isoformat(),
            })
            
            # Execute agent-specific processing
            result = await self._process_with_retry(data, context)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_processing_metrics(processing_time, success=True)
            
            self.status = AgentStatus.IDLE
            
            self.logger.info(
                "Processing completed",
                processing_time=processing_time,
                result_keys=list(result.keys())
            )
            
            return result
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_processing_metrics(processing_time, success=False)
            
            self.status = AgentStatus.ERROR
            self.logger.error(
                "Processing failed",
                error=str(e),
                processing_time=processing_time,
                data=data
            )
            
            return {
                "success": False,
                "error": str(e),
                "agent_name": self.name,
                "processing_time": processing_time,
            }
    
    async def _process_with_retry(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info("Retrying processing", attempt=attempt)
                    await asyncio.sleep(self.retry_delay * attempt)
                
                result = await self._process_data(data, context)
                
                if attempt > 0:
                    await self._log_metric("retry_success", 1)
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Processing attempt failed",
                    attempt=attempt,
                    error=str(e)
                )
                
                if attempt < self.max_retries:
                    await self._log_metric("retry_attempts", 1)
        
        # All retries failed
        await self._log_metric("retry_failures", 1)
        raise last_error
    
    async def _update_processing_metrics(self, processing_time: float, success: bool):
        """Update processing metrics"""
        self.metrics["last_activity"] = datetime.utcnow().isoformat()
        self.metrics["total_processing_time"] += processing_time
        
        if success:
            self.metrics["tasks_processed"] += 1
            await self._log_metric("tasks_processed", 1)
            await self._log_metric("processing_time", processing_time)
        else:
            self.metrics["errors_count"] += 1
            await self._log_metric("processing_errors", 1)
        
        # Log agent activity to Redis
        await cache_manager.log_agent_activity(
            self.name,
            {
                "action": "process",
                "success": success,
                "processing_time": processing_time,
                "status": self.status.value,
            }
        )
    
    async def _log_metric(self, metric_name: str, value: Union[int, float]):
        """Log metric to cache and database"""
        try:
            # Increment in Redis for real-time metrics
            await cache_manager.increment_metric(f"{self.name}:{metric_name}")
            
            # Store in database for historical analysis
            if not self.settings.is_testing:
                with get_db_context() as db:
                    metric = AgentMetrics(
                        agent_name=self.name,
                        metric_type=metric_name,
                        metric_value=float(value),
                        metric_data=None,
                    )
                    db.add(metric)
                    
        except Exception as e:
            self.logger.error("Metric logging failed", metric=metric_name, error=str(e))
    
    async def _save_metrics(self):
        """Save current metrics state"""
        try:
            await cache_manager.cache_user_patterns(
                f"agent_metrics_{self.name}",
                self.metrics,
                ttl=3600
            )
        except Exception as e:
            self.logger.error("Metrics saving failed", error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "is_running": self.is_running,
            "metrics": self.metrics.copy(),
            "config": self.config.copy(),
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check"""
        try:
            # Basic health check
            health_status = {
                "healthy": True,
                "status": self.status.value,
                "is_running": self.is_running,
                "uptime_seconds": (datetime.utcnow() - self.metrics["start_time"]).total_seconds(),
            }
            
            # Agent-specific health check
            agent_health = await self._health_check()
            health_status.update(agent_health)
            
            return health_status
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e),
                "status": self.status.value,
            }
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _initialize_agent(self):
        """Agent-specific initialization logic"""
        pass
    
    @abstractmethod
    async def _cleanup_agent(self):
        """Agent-specific cleanup logic"""
        pass
    
    @abstractmethod
    async def _process_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-specific processing logic"""
        pass
    
    @abstractmethod
    async def _health_check(self) -> Dict[str, Any]:
        """Agent-specific health check logic"""
        pass