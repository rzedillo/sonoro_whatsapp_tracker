"""
Unit tests for base agent functionality
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.base_agent import BaseAgent, AgentStatus


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""
    
    async def _initialize_agent(self):
        self.initialized = True
    
    async def _cleanup_agent(self):
        self.cleaned_up = True
    
    async def _process_data(self, data, context):
        if data.get("fail"):
            raise Exception("Test failure")
        return {"success": True, "processed": True}
    
    async def _health_check(self):
        return {"test_health": True}


@pytest.fixture
async def test_agent():
    """Create test agent fixture"""
    agent = TestAgent("test_agent", {"test_config": True})
    await agent.initialize()
    yield agent
    await agent.shutdown()


class TestBaseAgent:
    """Test cases for BaseAgent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization"""
        agent = TestAgent("test_agent", {"test": True})
        
        assert agent.name == "test_agent"
        assert agent.status == AgentStatus.IDLE
        assert not agent.is_running
        
        # Initialize
        success = await agent.initialize()
        
        assert success
        assert agent.status == AgentStatus.IDLE
        assert agent.is_running
        assert hasattr(agent, "initialized")
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_processing(self, test_agent):
        """Test data processing"""
        # Successful processing
        result = await test_agent.process({"test": "data"})
        
        assert result["success"] is True
        assert result["processed"] is True
        assert test_agent.metrics["tasks_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, test_agent):
        """Test error handling"""
        # Failed processing
        result = await test_agent.process({"fail": True})
        
        assert result["success"] is False
        assert "error" in result
        assert test_agent.metrics["errors_count"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_health_check(self, test_agent):
        """Test health check"""
        health = await test_agent.health_check()
        
        assert health["healthy"] is True
        assert "test_health" in health
    
    @pytest.mark.asyncio
    async def test_agent_status(self, test_agent):
        """Test status reporting"""
        status = test_agent.get_status()
        
        assert status["name"] == "test_agent"
        assert status["is_running"] is True
        assert "metrics" in status
        assert "agent_id" in status
    
    @pytest.mark.asyncio
    async def test_agent_shutdown(self):
        """Test agent shutdown"""
        agent = TestAgent("test_agent", {})
        await agent.initialize()
        
        assert agent.is_running
        
        await agent.shutdown()
        
        assert not agent.is_running
        assert agent.status == AgentStatus.STOPPED
        assert hasattr(agent, "cleaned_up")