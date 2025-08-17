#!/usr/bin/env python3
"""
Integration test script for WhatsApp Task Tracker
Tests critical components without requiring full WhatsApp connection
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("WHATSAPP_MONITORED_GROUPS", '["Test Group"]')
os.environ.setdefault("ENABLE_NOTIFICATIONS", "true")
os.environ.setdefault("NOTIFICATION_CHANNELS", '["console", "cache"]')
os.environ.setdefault("DATABASE_HOST", "localhost")
os.environ.setdefault("DATABASE_PASSWORD", "test-password")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")  # Use different DB for testing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integration_test")


async def test_agent_initialization():
    """Test agent initialization without external dependencies"""
    try:
        logger.info("🧪 Testing agent initialization...")
        
        # Test individual agent imports
        from agents.message_analysis_agent import MessageAnalysisAgent
        from agents.task_management_agent import TaskManagementAgent  
        from agents.notification_agent import NotificationAgent
        from agents.whatsapp_command_handler import WhatsAppCommandHandler
        
        logger.info("✅ All agent classes imported successfully")
        
        # Test agent instantiation (without full initialization)
        test_config = {"test": True}
        
        message_agent = MessageAnalysisAgent("test_analyzer", test_config)
        task_agent = TaskManagementAgent("test_task_manager", test_config)
        notification_agent = NotificationAgent("test_notifier", test_config)
        
        logger.info("✅ All agent instances created successfully")
        
        # Test command handler (lightweight)
        class MockOrchestrator:
            def __init__(self):
                self.agents = {"task_manager": task_agent}
        
        mock_orchestrator = MockOrchestrator()
        command_handler = WhatsAppCommandHandler(mock_orchestrator)
        
        # Test command detection
        test_message = {"text": "!help", "author": "test_user", "chat_name": "Test Group"}
        is_command = await command_handler.is_command("!help")
        
        assert is_command == True, "Command detection failed"
        logger.info("✅ Command detection working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent initialization test failed: {e}")
        return False


async def test_orchestrator_setup():
    """Test orchestrator configuration without full agent initialization"""
    try:
        logger.info("🧪 Testing orchestrator setup...")
        
        from core.settings import get_settings
        
        # Test settings loading
        settings = get_settings()
        logger.info(f"✅ Settings loaded: {settings.environment}")
        
        # Test that we can import orchestrator components without errors
        from core.orchestrator import AgentOrchestrator, OrchestratorStatus
        
        # Test orchestrator creation (without agent registration to avoid browser deps)
        orchestrator = AgentOrchestrator()
        
        assert orchestrator.status == OrchestratorStatus.INITIALIZING, "Wrong initial status"
        assert isinstance(orchestrator.agents, dict), "Agents dict not initialized"
        assert isinstance(orchestrator.metrics, dict), "Metrics dict not initialized"
        
        logger.info("✅ Orchestrator structure validated")
        
        # Test global orchestrator functions exist
        from core.orchestrator import get_orchestrator, shutdown_orchestrator
        
        logger.info("✅ Global orchestrator functions available")
        return True
        
    except Exception as e:
        logger.error(f"❌ Orchestrator setup test failed: {e}")
        return False


async def test_message_flow():
    """Test message processing workflow without external services"""
    try:
        logger.info("🧪 Testing message processing flow...")
        
        from agents.whatsapp_command_handler import WhatsAppCommandHandler
        from agents.notification_agent import NotificationAgent
        
        # Create mock components
        class MockTaskAgent:
            async def process(self, data):
                return {
                    "success": True, 
                    "task_id": 123,
                    "task_data": {"descripcion": "Test task"}
                }
        
        class MockOrchestrator:
            def __init__(self):
                self.agents = {"task_manager": MockTaskAgent()}
        
        # Test command handling
        orchestrator = MockOrchestrator()
        command_handler = WhatsAppCommandHandler(orchestrator)
        
        test_commands = [
            {"text": "!help", "author": "test_user", "chat_name": "Test Group"},
            {"text": "!create Test task description", "author": "test_user", "chat_name": "Test Group"},
            {"text": "!tasks", "author": "test_user", "chat_name": "Test Group"},
        ]
        
        for test_message in test_commands:
            if await command_handler.is_command(test_message["text"]):
                result = await command_handler.handle_command(test_message)
                assert result.get("success") is not None, f"Command handling failed for {test_message['text']}"
                logger.info(f"✅ Command processed: {test_message['text']}")
        
        # Test notification formatting
        notification_agent = NotificationAgent("test_notifier", {"enabled": True})
        
        test_notification = {
            "type": "task_created",
            "title": "Test Notification",
            "message": "Test message",
            "priority": "medium"
        }
        
        formatted_message = await notification_agent._format_whatsapp_message(test_notification)
        assert "Test Notification" in formatted_message, "Notification formatting failed"
        logger.info("✅ Notification formatting working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Message flow test failed: {e}")
        return False


async def test_configuration_loading():
    """Test environment configuration and settings"""
    try:
        logger.info("🧪 Testing configuration loading...")
        
        from core.settings import get_settings
        
        settings = get_settings()
        
        # Test required settings
        assert settings.anthropic_api_key == "test-key", "Anthropic API key not loaded"
        assert "Test Group" in settings.whatsapp_monitored_groups, "Monitored groups not loaded"
        assert settings.enable_notifications == True, "Notifications setting not loaded"
        
        logger.info("✅ Environment variables loaded correctly")
        
        # Test default values
        assert settings.whatsapp_qr_timeout == 60, "Default QR timeout incorrect"
        # Note: headless can be false for development, true for production
        assert isinstance(settings.whatsapp_headless, bool), "Headless setting should be boolean"
        
        logger.info("✅ Default configuration values correct")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_database_models():
    """Test database model definitions"""
    try:
        logger.info("🧪 Testing database models...")
        
        from database.models import Task, TaskStatus, TaskPriority, Conversation
        from datetime import datetime
        
        # Test model creation
        task = Task(
            descripcion="Test task",
            responsable="test_user",
            estado=TaskStatus.PENDING,
            prioridad=TaskPriority.MEDIUM,
            timestamp=datetime.utcnow()
        )
        
        conversation = Conversation(
            mensaje="Test message",
            autor="test_user",
            timestamp=datetime.utcnow(),
            grupo_nombre="Test Group",
            mensaje_id="test_123"
        )
        
        # Test model serialization
        task_dict = task.to_dict()
        assert task_dict["descripcion"] == "Test task", "Task serialization failed"
        
        logger.info("✅ Database models working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database model test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting WhatsApp Task Tracker Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("Orchestrator Setup", test_orchestrator_setup), 
        ("Message Flow", test_message_flow),
        ("Configuration Loading", test_configuration_loading),
        ("Database Models", test_database_models),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Integration Test Results")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        logger.error("⚠️  Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(run_integration_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Test runner failed: {e}")
        sys.exit(1)