"""
WhatsApp Task Tracker - Main FastAPI Application
Enhanced Framework V3.1 Implementation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import structlog
import uvicorn

from core.settings import get_settings
from core.logging_config import setup_logging
from database.connection import engine, create_tables
from api.v1.router import api_router


# Configure structured logging
setup_logging()
logger = structlog.get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting WhatsApp Task Tracker", environment=settings.environment)
    
    # Create database tables
    await create_tables()
    
    # Initialize agents and orchestrator
    from core.orchestrator import get_orchestrator
    orchestrator = get_orchestrator()
    await orchestrator.initialize()
    
    logger.info("Application startup complete")
    
    yield
    
    # Cleanup
    logger.info("Shutting down application")
    await orchestrator.shutdown()


# Create FastAPI application
app = FastAPI(
    title="WhatsApp Task Tracker",
    description="Multi-agent WhatsApp task management system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts,
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    from database.connection import get_db_health
    from core.redis_client import get_redis_health
    
    health_status = {
        "status": "healthy",
        "environment": settings.environment,
        "version": "1.0.0",
        "services": {}
    }
    
    # Check database
    try:
        db_healthy = await get_db_health()
        health_status["services"]["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        redis_healthy = await get_redis_health()
        health_status["services"]["redis"] = "healthy" if redis_healthy else "unhealthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check WhatsApp connection
    try:
        from agents.whatsapp_agent import get_whatsapp_status
        whatsapp_status = await get_whatsapp_status()
        health_status["services"]["whatsapp"] = whatsapp_status
    except Exception as e:
        health_status["services"]["whatsapp"] = f"error: {str(e)}"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WhatsApp Task Tracker API",
        "version": "1.0.0",
        "environment": settings.environment,
        "docs": "/docs" if settings.environment != "production" else "disabled",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )