"""
WhatsApp API endpoints for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import structlog

from core.orchestrator import get_orchestrator

logger = structlog.get_logger(__name__)
router = APIRouter()


class WhatsAppResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SendMessageRequest(BaseModel):
    chat_name: str
    message: str
    message_type: str = "text"


@router.get("/status", response_model=WhatsAppResponse)
async def get_whatsapp_status():
    """
    Get WhatsApp connection status
    """
    try:
        orchestrator = get_orchestrator()
        whatsapp_agent = orchestrator.agents.get("whatsapp")
        
        if not whatsapp_agent:
            return WhatsAppResponse(
                success=False,
                error="WhatsApp service unavailable"
            )
        
        result = await whatsapp_agent.process({
            "command": "get_status"
        })
        
        if result.get("success"):
            return WhatsAppResponse(
                success=True,
                message="WhatsApp status retrieved",
                data=result
            )
        else:
            return WhatsAppResponse(
                success=False,
                error=result.get("error", "Status check failed")
            )
            
    except Exception as e:
        logger.error("WhatsApp status check failed", error=str(e))
        return WhatsAppResponse(
            success=False,
            error="Internal server error"
        )


@router.get("/qr-code", response_model=WhatsAppResponse)
async def get_qr_code():
    """
    Get QR code for WhatsApp authentication
    """
    try:
        orchestrator = get_orchestrator()
        whatsapp_agent = orchestrator.agents.get("whatsapp")
        
        if not whatsapp_agent:
            return WhatsAppResponse(
                success=False,
                error="WhatsApp service unavailable"
            )
        
        result = await whatsapp_agent.process({
            "command": "get_qr_code"
        })
        
        if result.get("success"):
            return WhatsAppResponse(
                success=True,
                message="QR code retrieved",
                data={
                    "qr_code": result.get("qr_code"),
                    "session_id": result.get("session_id"),
                    "instructions": "Scan this QR code with your WhatsApp mobile app"
                }
            )
        else:
            return WhatsAppResponse(
                success=False,
                error=result.get("error", "QR code not available")
            )
            
    except Exception as e:
        logger.error("QR code retrieval failed", error=str(e))
        return WhatsAppResponse(
            success=False,
            error="Internal server error"
        )


@router.get("/chats", response_model=WhatsAppResponse)
async def get_chats():
    """
    Get list of available WhatsApp chats
    """
    try:
        orchestrator = get_orchestrator()
        whatsapp_agent = orchestrator.agents.get("whatsapp")
        
        if not whatsapp_agent:
            return WhatsAppResponse(
                success=False,
                error="WhatsApp service unavailable"
            )
        
        result = await whatsapp_agent.process({
            "command": "get_chats"
        })
        
        if result.get("success"):
            return WhatsAppResponse(
                success=True,
                message="Chats retrieved",
                data=result
            )
        else:
            return WhatsAppResponse(
                success=False,
                error=result.get("error", "Failed to get chats")
            )
            
    except Exception as e:
        logger.error("Get chats failed", error=str(e))
        return WhatsAppResponse(
            success=False,
            error="Internal server error"
        )


@router.post("/send-message", response_model=WhatsAppResponse)
async def send_message(message_request: SendMessageRequest):
    """
    Send message through WhatsApp
    """
    try:
        orchestrator = get_orchestrator()
        whatsapp_agent = orchestrator.agents.get("whatsapp")
        
        if not whatsapp_agent:
            return WhatsAppResponse(
                success=False,
                error="WhatsApp service unavailable"
            )
        
        result = await whatsapp_agent.process({
            "command": "send_message",
            "chat_name": message_request.chat_name,
            "message": message_request.message,
            "message_type": message_request.message_type
        })
        
        if result.get("success"):
            return WhatsAppResponse(
                success=True,
                message="Message sent successfully",
                data=result
            )
        else:
            return WhatsAppResponse(
                success=False,
                error=result.get("error", "Message sending failed")
            )
            
    except Exception as e:
        logger.error("Send message failed", error=str(e))
        return WhatsAppResponse(
            success=False,
            error="Internal server error"
        )


@router.post("/reconnect", response_model=WhatsAppResponse)
async def reconnect_whatsapp():
    """
    Reconnect to WhatsApp
    """
    try:
        orchestrator = get_orchestrator()
        whatsapp_agent = orchestrator.agents.get("whatsapp")
        
        if not whatsapp_agent:
            return WhatsAppResponse(
                success=False,
                error="WhatsApp service unavailable"
            )
        
        result = await whatsapp_agent.process({
            "command": "reconnect"
        })
        
        if result.get("success"):
            return WhatsAppResponse(
                success=True,
                message="WhatsApp reconnection successful",
                data=result
            )
        else:
            return WhatsAppResponse(
                success=False,
                error=result.get("error", "Reconnection failed")
            )
            
    except Exception as e:
        logger.error("WhatsApp reconnection failed", error=str(e))
        return WhatsAppResponse(
            success=False,
            error="Internal server error"
        )


@router.get("/health", response_model=Dict[str, Any])
async def whatsapp_health_check():
    """
    Detailed WhatsApp service health check
    """
    try:
        orchestrator = get_orchestrator()
        whatsapp_agent = orchestrator.agents.get("whatsapp")
        
        if not whatsapp_agent:
            return {
                "healthy": False,
                "error": "WhatsApp agent not available"
            }
        
        health_status = await whatsapp_agent.health_check()
        
        return {
            "healthy": health_status.get("authenticated", False),
            "details": health_status,
            "timestamp": health_status.get("last_message_time"),
        }
        
    except Exception as e:
        logger.error("WhatsApp health check failed", error=str(e))
        return {
            "healthy": False,
            "error": str(e)
        }


@router.get("/session", response_model=WhatsAppResponse)
async def get_session_info():
    """
    Get current WhatsApp session information
    """
    try:
        orchestrator = get_orchestrator()
        whatsapp_agent = orchestrator.agents.get("whatsapp")
        
        if not whatsapp_agent:
            return WhatsAppResponse(
                success=False,
                error="WhatsApp service unavailable"
            )
        
        status = await whatsapp_agent.health_check()
        
        return WhatsAppResponse(
            success=True,
            message="Session info retrieved",
            data={
                "session_id": status.get("session_id"),
                "authenticated": status.get("authenticated", False),
                "phone_number": status.get("phone_number"),
                "last_activity": status.get("last_message_time"),
                "monitored_groups": status.get("monitored_groups", 0),
            }
        )
        
    except Exception as e:
        logger.error("Get session info failed", error=str(e))
        return WhatsAppResponse(
            success=False,
            error="Internal server error"
        )


@router.get("/conversations", response_model=Dict[str, Any])
async def get_recent_conversations(limit: int = 50):
    """
    Get recent WhatsApp conversations
    """
    try:
        from database.connection import get_db_context
        from database.models import Conversation
        from sqlalchemy import desc
        
        with get_db_context() as db:
            conversations = db.query(Conversation).order_by(
                desc(Conversation.timestamp)
            ).limit(limit).all()
            
            return {
                "success": True,
                "data": {
                    "conversations": [conv.to_dict() for conv in conversations],
                    "count": len(conversations),
                    "limit": limit
                }
            }
            
    except Exception as e:
        logger.error("Get conversations failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/groups", response_model=Dict[str, Any])
async def get_monitored_groups():
    """
    Get list of monitored WhatsApp groups
    """
    try:
        from database.connection import get_db_context
        from database.models import Task
        from sqlalchemy import text
        
        with get_db_context() as db:
            # Get distinct group names from tasks
            result = db.execute(
                text("SELECT DISTINCT grupo_nome FROM tasks WHERE grupo_nome IS NOT NULL")
            ).fetchall()
            
            groups = [row[0] for row in result]
            
            return {
                "success": True,
                "data": {
                    "groups": groups,
                    "count": len(groups)
                }
            }
            
    except Exception as e:
        logger.error("Get monitored groups failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")