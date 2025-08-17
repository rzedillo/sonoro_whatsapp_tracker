"""
API Router for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Any, Optional
import structlog

from api.v1.endpoints import tasks, whatsapp, analytics, notifications, agents
from api.middleware.auth import get_current_user  # Placeholder for auth
from core.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    tasks.router,
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[] if settings.is_development else []  # Add auth in production
)

api_router.include_router(
    whatsapp.router,
    prefix="/whatsapp",
    tags=["whatsapp"],
    dependencies=[] if settings.is_development else []
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"],
    dependencies=[] if settings.is_development else []
)

api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["notifications"],
    dependencies=[] if settings.is_development else []
)

api_router.include_router(
    agents.router,
    prefix="/agents",
    tags=["agents"],
    dependencies=[] if settings.is_development else []
)


@api_router.get("/")
async def api_root():
    """API root endpoint"""
    return {
        "message": "WhatsApp Task Tracker API v1",
        "version": "1.0.0",
        "environment": settings.environment,
        "endpoints": {
            "tasks": "/api/v1/tasks",
            "whatsapp": "/api/v1/whatsapp",
            "analytics": "/api/v1/analytics",
            "notifications": "/api/v1/notifications",
            "agents": "/api/v1/agents",
        }
    }


@api_router.get("/status")
async def api_status():
    """API status endpoint"""
    try:
        from core.orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        status = orchestrator.get_orchestrator_status()
        
        return {
            "api_status": "healthy",
            "timestamp": status.get("uptime_seconds", 0),
            "orchestrator": status
        }
        
    except Exception as e:
        logger.error("Status check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")