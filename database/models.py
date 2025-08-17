"""
Database models for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
Ported from SQLite schema to PostgreSQL with SQLAlchemy
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from database.connection import Base


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pendiente"
    IN_PROGRESS = "en_progreso"
    COMPLETED = "completado"
    CANCELLED = "cancelado"


class TaskPriority(str, Enum):
    """Task priority enumeration"""
    LOW = "baja"
    MEDIUM = "media"
    HIGH = "alta"
    URGENT = "urgente"


class Task(Base):
    """Main tasks table - ported from 'tareas' table"""
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    descripcion = Column(Text, nullable=False, index=True)
    responsable = Column(String(255), index=True)
    fecha_limite = Column(String(50))  # Keeping as string for compatibility
    prioridad = Column(String(20), default=TaskPriority.MEDIUM, index=True)
    estado = Column(String(20), default=TaskStatus.PENDING, index=True)
    mensaje_original = Column(Text)
    autor_mensaje = Column(String(255), index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    grupo_id = Column(String(255), index=True)
    grupo_nombre = Column(String(255), index=True)
    mensaje_id = Column(String(255), unique=True, index=True)
    confidence_score = Column(Float, default=0.8)
    analysis_metadata = Column(Text)  # JSON stored as text
    completion_date = Column(String(50))
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    history = relationship("TaskHistory", back_populates="task", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "descripcion": self.descripcion,
            "responsable": self.responsable,
            "fecha_limite": self.fecha_limite,
            "prioridad": self.prioridad,
            "estado": self.estado,
            "mensaje_original": self.mensaje_original,
            "autor_mensaje": self.autor_mensaje,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "grupo_id": self.grupo_id,
            "grupo_nombre": self.grupo_nombre,
            "mensaje_id": self.mensaje_id,
            "confidence_score": self.confidence_score,
            "analysis_metadata": self.analysis_metadata,
            "completion_date": self.completion_date,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class Conversation(Base):
    """Conversation history table - ported from 'conversaciones' table"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    mensaje = Column(Text, nullable=False)
    autor = Column(String(255), index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    grupo_id = Column(String(255), index=True)
    grupo_nombre = Column(String(255), index=True)
    mensaje_id = Column(String(255), unique=True, index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary"""
        return {
            "id": self.id,
            "mensaje": self.mensaje,
            "autor": self.autor,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "grupo_id": self.grupo_id,
            "grupo_nombre": self.grupo_nombre,
            "mensaje_id": self.mensaje_id,
        }


class TaskHistory(Base):
    """Task change history table"""
    __tablename__ = "task_history"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    previous_state = Column(String(255))
    new_state = Column(String(255))
    changed_by = Column(String(255))
    timestamp = Column(DateTime, default=func.now(), index=True)
    notes = Column(Text)
    
    # Relationships
    task = relationship("Task", back_populates="history")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task history to dictionary"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "action": self.action,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "changed_by": self.changed_by,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "notes": self.notes,
        }


class UserPattern(Base):
    """User productivity patterns table"""
    __tablename__ = "user_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(255), nullable=False, index=True)
    grupo_nombre = Column(String(255), index=True)
    pattern_type = Column(String(50), nullable=False)
    pattern_data = Column(Text)  # JSON stored as text
    calculated_date = Column(DateTime, default=func.now(), index=True)
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    average_completion_time = Column(Float)
    most_common_priority = Column(String(20))
    productivity_score = Column(Float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user pattern to dictionary"""
        return {
            "id": self.id,
            "user_name": self.user_name,
            "grupo_nombre": self.grupo_nombre,
            "pattern_type": self.pattern_type,
            "pattern_data": self.pattern_data,
            "calculated_date": self.calculated_date.isoformat() if self.calculated_date else None,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "average_completion_time": self.average_completion_time,
            "most_common_priority": self.most_common_priority,
            "productivity_score": self.productivity_score,
        }


class WhatsAppSession(Base):
    """WhatsApp session management table - new addition"""
    __tablename__ = "whatsapp_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    qr_code = Column(Text)  # Base64 encoded QR code
    is_authenticated = Column(Boolean, default=False)
    phone_number = Column(String(50))
    session_data = Column(Text)  # JSON session data
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "is_authenticated": self.is_authenticated,
            "phone_number": self.phone_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class AgentMetrics(Base):
    """Agent performance metrics table - new addition"""
    __tablename__ = "agent_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)
    metric_value = Column(Float)
    metric_data = Column(Text)  # JSON additional data
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "metric_type": self.metric_type,
            "metric_value": self.metric_value,
            "metric_data": self.metric_data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }