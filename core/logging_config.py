"""
Structured logging configuration for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

import structlog
import logging
import sys
from pathlib import Path
from typing import Any, Dict
from core.settings import get_settings

settings = get_settings()


def add_app_context(logger: Any, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add application context to log entries"""
    event_dict["app"] = "whatsapp-tracker"
    event_dict["version"] = "1.0.0"
    event_dict["environment"] = settings.environment
    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application"""
    
    # Ensure logs directory exists
    logs_path = Path(settings.logs_path)
    logs_path.mkdir(exist_ok=True)
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_app_context,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.environment == "production" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # File handler for persistent logging
    if not settings.is_testing:
        file_handler = logging.FileHandler(logs_path / "application.log")
        file_handler.setLevel(getattr(logging, settings.log_level))
        
        # JSON formatter for file logs
        json_formatter = structlog.processors.JSONRenderer()
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Add file handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Structured file logging
        if settings.environment == "production":
            structured_handler = logging.FileHandler(logs_path / "structured.jsonl")
            structured_handler.setLevel(getattr(logging, settings.log_level))
            root_logger.addHandler(structured_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)