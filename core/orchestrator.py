"""
Agent Orchestrator for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

import asyncio
from typing import Dict, List, Any, Optional, Type
import structlog
from datetime import datetime
from enum import Enum

from core.base_agent import BaseAgent, AgentStatus
from core.settings import get_settings
from core.redis_client import cache_manager, initialize_redis, shutdown_redis


class OrchestratorStatus(str, Enum):
    """Orchestrator status enumeration"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentOrchestrator:
    """
    Multi-agent orchestrator implementing Enhanced Framework V3.1 patterns
    
    Features:
    - Agent lifecycle management
    - Message routing and processing
    - Error recovery and circuit breaker patterns
    - Performance monitoring
    - Graceful shutdown
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = structlog.get_logger("orchestrator")
        
        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        
        # State management
        self.status = OrchestratorStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        self.is_running = False
        
        # Message processing
        self.message_queue = asyncio.Queue()
        self.processing_tasks = set()
        
        # Performance metrics
        self.metrics = {
            "messages_processed": 0,
            "agents_started": 0,
            "agents_failed": 0,
            "total_uptime": 0,
            "last_activity": None,
        }
        
        self.logger.info("Orchestrator initialized")
    
    async def initialize(self):
        """Initialize the orchestrator and all agents"""
        try:
            self.logger.info("Initializing orchestrator")
            
            # Initialize Redis connection
            await initialize_redis()
            
            # Register and initialize agents
            await self._register_agents()
            await self._initialize_agents()
            
            # Start message processing
            await self._start_message_processing()
            
            self.status = OrchestratorStatus.RUNNING
            self.is_running = True
            
            self.logger.info(
                "Orchestrator initialization completed",
                agents_count=len(self.agents)
            )
            
        except Exception as e:
            self.logger.error("Orchestrator initialization failed", error=str(e))
            self.status = OrchestratorStatus.ERROR
            raise
    
    async def shutdown(self):
        """Shutdown orchestrator and all agents gracefully"""
        try:
            self.logger.info("Shutting down orchestrator")
            self.status = OrchestratorStatus.STOPPING
            
            # Stop message processing
            self.is_running = False
            
            # Wait for current processing to complete
            if self.processing_tasks:
                self.logger.info("Waiting for processing tasks to complete")
                await asyncio.wait_for(
                    asyncio.gather(*self.processing_tasks, return_exceptions=True),
                    timeout=30.0
                )
            
            # Shutdown all agents
            await self._shutdown_agents()
            
            # Close Redis connection
            await shutdown_redis()
            
            self.status = OrchestratorStatus.STOPPED
            
            self.logger.info("Orchestrator shutdown completed")
            
        except Exception as e:
            self.logger.error("Orchestrator shutdown error", error=str(e))
    
    async def _register_agents(self):
        """Register all agent types"""
        try:
            # Import agent classes
            from agents.whatsapp_agent import WhatsAppAgent
            from agents.task_management_agent import TaskManagementAgent
            from agents.message_analysis_agent import MessageAnalysisAgent
            from agents.notification_agent import NotificationAgent
            
            # Define agent configurations
            self.agent_configs = {
                "whatsapp": {
                    "class": WhatsAppAgent,
                    "config": {
                        "session_path": self.settings.whatsapp_session_path,
                        "qr_timeout": self.settings.whatsapp_qr_timeout,
                        "headless": self.settings.whatsapp_headless,
                        "monitored_groups": self.settings.whatsapp_monitored_groups,
                        "rate_limit_delay": self.settings.whatsapp_rate_limit_delay,
                    }
                },
                "task_manager": {
                    "class": TaskManagementAgent,
                    "config": {
                        "auto_assign": True,
                        "priority_detection": True,
                    }
                },
                "message_analyzer": {
                    "class": MessageAnalysisAgent,
                    "config": {
                        "anthropic_api_key": self.settings.anthropic_api_key,
                        "analysis_timeout": 30,
                    }
                },
                "notifier": {
                    "class": NotificationAgent,
                    "config": {
                        "enabled": self.settings.enable_notifications,
                        "channels": self.settings.notification_channels,
                    }
                },
            }
            
            self.logger.info("Agent configurations registered", agents=list(self.agent_configs.keys()))
            
        except ImportError as e:
            self.logger.error("Failed to import agent classes", error=str(e))
            raise
    
    async def _initialize_agents(self):
        """Initialize all registered agents"""
        initialization_tasks = []
        
        for agent_name, config in self.agent_configs.items():
            task = asyncio.create_task(
                self._initialize_single_agent(agent_name, config)
            )
            initialization_tasks.append(task)
        
        # Wait for all agents to initialize
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Check results
        successful_agents = 0
        for i, result in enumerate(results):
            agent_name = list(self.agent_configs.keys())[i]
            if isinstance(result, Exception):
                self.logger.error(
                    "Agent initialization failed",
                    agent=agent_name,
                    error=str(result)
                )
                self.metrics["agents_failed"] += 1
            else:
                successful_agents += 1
                self.metrics["agents_started"] += 1
        
        if successful_agents == 0:
            raise Exception("No agents were successfully initialized")
        
        self.logger.info(
            "Agent initialization completed",
            successful=successful_agents,
            failed=self.metrics["agents_failed"]
        )
        
        # Setup inter-agent connections
        await self._setup_agent_connections()
    
    async def _setup_agent_connections(self):
        """Setup connections between agents"""
        try:
            self.logger.info("Setting up inter-agent connections")
            
            # Connect WhatsApp agent with notification agent
            whatsapp_agent = self.agents.get("whatsapp")
            notification_agent = self.agents.get("notifier")
            
            if whatsapp_agent and notification_agent:
                # Set WhatsApp agent reference in notification agent
                notification_agent.set_whatsapp_agent(whatsapp_agent)
                
                # Set orchestrator reference in WhatsApp agent for command handling
                await whatsapp_agent.process({
                    "command": "set_orchestrator",
                    "orchestrator": self
                })
                
                self.logger.info("WhatsApp and notification agents connected")
            else:
                self.logger.warning(
                    "Could not connect WhatsApp and notification agents",
                    whatsapp_available=whatsapp_agent is not None,
                    notification_available=notification_agent is not None
                )
            
            self.logger.info("Inter-agent connections setup completed")
            
        except Exception as e:
            self.logger.error("Agent connections setup failed", error=str(e))
    
    async def _initialize_single_agent(self, agent_name: str, config: Dict[str, Any]):
        """Initialize a single agent"""
        try:
            agent_class = config["class"]
            agent_config = config["config"]
            
            # Create agent instance
            agent = agent_class(agent_name, agent_config)
            
            # Initialize agent
            success = await agent.initialize()
            
            if success:
                self.agents[agent_name] = agent
                self.logger.info("Agent initialized successfully", agent=agent_name)
            else:
                raise Exception(f"Agent {agent_name} initialization returned False")
                
        except Exception as e:
            self.logger.error(
                "Single agent initialization failed",
                agent=agent_name,
                error=str(e)
            )
            raise
    
    async def _shutdown_agents(self):
        """Shutdown all agents"""
        shutdown_tasks = []
        
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(
                self._shutdown_single_agent(agent_name, agent)
            )
            shutdown_tasks.append(task)
        
        # Wait for all agents to shutdown
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    async def _shutdown_single_agent(self, agent_name: str, agent: BaseAgent):
        """Shutdown a single agent"""
        try:
            await agent.shutdown()
            self.logger.info("Agent shutdown completed", agent=agent_name)
        except Exception as e:
            self.logger.error(
                "Agent shutdown failed",
                agent=agent_name,
                error=str(e)
            )
    
    async def _start_message_processing(self):
        """Start background message processing"""
        self.logger.info("Starting message processing")
        
        # Create processing task
        processing_task = asyncio.create_task(self._process_messages())
        self.processing_tasks.add(processing_task)
        
        # Clean up completed tasks
        processing_task.add_done_callback(self.processing_tasks.discard)
    
    async def _process_messages(self):
        """Main message processing loop"""
        while self.is_running:
            try:
                # Wait for message with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                await self._handle_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                self.logger.error("Message processing error", error=str(e))
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle individual message"""
        try:
            message_type = message.get("type", "unknown")
            self.logger.info("Processing message", type=message_type)
            
            # Route message based on type
            if message_type == "whatsapp_message":
                await self._handle_whatsapp_message(message)
            elif message_type == "task_update":
                await self._handle_task_update(message)
            elif message_type == "user_command":
                await self._handle_user_command(message)
            else:
                self.logger.warning("Unknown message type", type=message_type)
            
            # Update metrics
            self.metrics["messages_processed"] += 1
            self.metrics["last_activity"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            self.logger.error("Message handling failed", error=str(e), message=message)
    
    async def _handle_whatsapp_message(self, message: Dict[str, Any]):
        """Handle WhatsApp message processing workflow"""
        try:
            self.logger.info("Starting WhatsApp message workflow", 
                           author=message.get("author"), 
                           chat=message.get("chat_name"))
            
            # Step 1: Analyze message for task detection
            message_analyzer = self.agents.get("message_analyzer")
            if not message_analyzer:
                self.logger.error("Message analyzer agent not available")
                return
            
            analysis_result = await message_analyzer.process(message)
            
            if not analysis_result.get("success"):
                self.logger.warning("Message analysis failed", error=analysis_result.get("error"))
                return
            
            # Step 2: If task detected, create task
            if analysis_result.get("task_detected"):
                await self._handle_task_creation(analysis_result)
            else:
                self.logger.info("No task detected in message", 
                               confidence=analysis_result.get("confidence", 0))
            
        except Exception as e:
            self.logger.error("WhatsApp message handling failed", error=str(e))
    
    async def _handle_task_creation(self, analysis_result: Dict[str, Any]):
        """Handle task creation from analysis result"""
        try:
            self.logger.info("Creating task from analysis result")
            
            # Step 1: Create task via task management agent
            task_manager = self.agents.get("task_manager")
            if not task_manager:
                self.logger.error("Task manager agent not available")
                return
            
            task_result = await task_manager.process(analysis_result)
            
            if not task_result.get("success"):
                self.logger.error("Task creation failed", error=task_result.get("error"))
                return
            
            # Step 2: Send notification
            notification_data = {
                **analysis_result,
                "task_created": True,
                "task_id": task_result.get("task_id"),
                "task_data": task_result.get("task_data", {})
            }
            
            await self._send_notification(notification_data)
            
            self.logger.info("Task creation workflow completed", 
                           task_id=task_result.get("task_id"))
            
        except Exception as e:
            self.logger.error("Task creation handling failed", error=str(e))
    
    async def _handle_task_update(self, message: Dict[str, Any]):
        """Handle task update notifications"""
        try:
            # Send notification about task update
            await self._send_notification(message)
            
        except Exception as e:
            self.logger.error("Task update handling failed", error=str(e))
    
    async def _handle_user_command(self, message: Dict[str, Any]):
        """Handle user commands (if needed)"""
        try:
            # User commands are handled directly by WhatsApp agent
            # This is for any orchestrator-level command handling
            self.logger.info("User command received", command=message.get("command"))
            
        except Exception as e:
            self.logger.error("User command handling failed", error=str(e))
    
    async def _send_notification(self, data: Dict[str, Any]):
        """Send notification via notification agent"""
        try:
            notifier = self.agents.get("notifier")
            if not notifier:
                self.logger.warning("Notification agent not available")
                return
            
            notification_result = await notifier.process(data)
            
            if notification_result.get("success"):
                self.logger.info("Notification sent successfully")
            else:
                self.logger.warning("Notification failed", 
                                  error=notification_result.get("error"))
                
        except Exception as e:
            self.logger.error("Notification sending failed", error=str(e))
    
    # Public interface methods
    async def send_message(self, message_data: Dict[str, Any]):
        """Public method to send message for processing"""
        try:
            # Add message type if not specified
            if "type" not in message_data:
                message_data["type"] = "whatsapp_message"
            
            # Add to processing queue
            await self.message_queue.put(message_data)
            
            self.logger.debug("Message queued for processing", 
                            type=message_data.get("type"))
            
        except Exception as e:
            self.logger.error("Message queueing failed", error=str(e))
    
    async def notify_task_update(self, task_id: int, changes: Dict[str, Any], 
                                changed_by: str = "system"):
        """Public method to notify about task updates"""
        try:
            update_message = {
                "type": "task_update",
                "task_id": task_id,
                "changes": changes,
                "changed_by": changed_by,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.message_queue.put(update_message)
            
        except Exception as e:
            self.logger.error("Task update notification failed", error=str(e))
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status information"""
        try:
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            agent_status = {}
            for name, agent in self.agents.items():
                agent_status[name] = agent.status.value if hasattr(agent, 'status') else "unknown"
            
            return {
                "status": self.status.value,
                "uptime_seconds": uptime_seconds,
                "queue_size": self.message_queue.qsize(),
                "agents": agent_status,
                "metrics": self.metrics,
                "is_running": self.is_running
            }
            
        except Exception as e:
            self.logger.error("Status retrieval failed", error=str(e))
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        try:
            health_results = {}
            
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, '_health_check'):
                        agent_health = await agent._health_check()
                        health_results[agent_name] = {
                            "healthy": True,
                            "details": agent_health
                        }
                    else:
                        health_results[agent_name] = {
                            "healthy": True,
                            "details": {"status": "no_health_check"}
                        }
                except Exception as e:
                    health_results[agent_name] = {
                        "healthy": False,
                        "error": str(e)
                    }
            
            return health_results
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {"error": str(e)}


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


async def get_orchestrator() -> AgentOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator


async def shutdown_orchestrator():
    """Shutdown global orchestrator instance"""
    global _orchestrator
    if _orchestrator is not None:
        await _orchestrator.shutdown()
        _orchestrator = None