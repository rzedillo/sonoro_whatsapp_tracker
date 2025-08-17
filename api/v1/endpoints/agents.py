"""
Agents API endpoints for WhatsApp Task Tracker
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import structlog

from core.orchestrator import get_orchestrator

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/")
async def get_agents_status():
    """Get status of all agents"""
    try:
        orchestrator = get_orchestrator()
        agents_status = orchestrator.get_agent_status()
        
        return {
            "success": True,
            "data": {
                "agents": agents_status,
                "orchestrator": orchestrator.get_orchestrator_status()
            }
        }
        
    except Exception as e:
        logger.error("Get agents status failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{agent_name}")
async def get_agent_status(agent_name: str):
    """Get status of specific agent"""
    try:
        orchestrator = get_orchestrator()
        agent_status = orchestrator.get_agent_status(agent_name)
        
        if "error" in agent_status:
            raise HTTPException(status_code=404, detail=agent_status["error"])
        
        return {
            "success": True,
            "data": {
                "agent": agent_status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get agent status failed", agent=agent_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{agent_name}/health")
async def get_agent_health(agent_name: str):
    """Get health check for specific agent"""
    try:
        orchestrator = get_orchestrator()
        agent = orchestrator.agents.get(agent_name)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        health_status = await agent.health_check()
        
        return {
            "success": True,
            "data": {
                "agent_name": agent_name,
                "health": health_status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent health check failed", agent=agent_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{agent_name}/metrics")
async def get_agent_metrics(agent_name: str):
    """Get performance metrics for specific agent"""
    try:
        orchestrator = get_orchestrator()
        agent = orchestrator.agents.get(agent_name)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        status = agent.get_status()
        
        return {
            "success": True,
            "data": {
                "agent_name": agent_name,
                "metrics": status.get("metrics", {}),
                "performance": {
                    "status": status.get("status"),
                    "is_running": status.get("is_running"),
                    "config": status.get("config", {})
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent metrics failed", agent=agent_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")