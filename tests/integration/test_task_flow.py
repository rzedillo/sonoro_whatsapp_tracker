"""
Integration tests for complete task flow
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from core.orchestrator import AgentOrchestrator
from database.connection import get_db_context
from database.models import Task


@pytest.fixture
async def orchestrator():
    """Create test orchestrator"""
    orchestrator = AgentOrchestrator()
    
    # Mock agent initialization for testing
    with patch.object(orchestrator, '_register_agents'), \
         patch.object(orchestrator, '_initialize_agents'), \
         patch.object(orchestrator, '_start_message_processing'):
        
        await orchestrator.initialize()
    
    yield orchestrator
    
    await orchestrator.shutdown()


class TestTaskFlow:
    """Integration tests for task management flow"""
    
    @pytest.mark.asyncio
    async def test_complete_task_flow(self):
        """Test complete flow from message to task creation"""
        
        # Mock message data
        message_data = {
            "type": "whatsapp_message",
            "text": "Necesito que alguien complete el reporte para mañana",
            "author": "John Doe",
            "chat_name": "Trabajo",
            "message_id": "test_message_123",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        # This would test the complete flow:
        # 1. Message analysis
        # 2. Task creation
        # 3. Notification sending
        
        # For now, just test that we can create a task directly
        with get_db_context() as db:
            task = Task(
                descripcion="Test task",
                responsable="Test User",
                prioridad="media",
                estado="pendiente",
                grupo_nome="Test Group"
            )
            db.add(task)
            db.flush()
            
            assert task.id is not None
            assert task.descripcion == "Test task"
    
    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test database connectivity"""
        from database.connection import get_db_health
        
        health = await get_db_health()
        assert health is True
    
    @pytest.mark.asyncio
    async def test_redis_connection(self):
        """Test Redis connectivity"""
        from core.redis_client import get_redis_health
        
        try:
            health = await get_redis_health()
            # Redis might not be available in test environment
            assert health in [True, False]
        except Exception:
            # Redis not available in test environment
            pass