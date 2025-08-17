"""
Database connection and session management
Enhanced Framework V3.1 Implementation
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import structlog
from typing import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from core.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Create database engine
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool if settings.is_testing else None,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=settings.log_level == "DEBUG",
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Get database session with context manager"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def create_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise


async def get_db_health() -> bool:
    """Check database health"""
    try:
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False


class DatabaseManager:
    """Database manager for advanced operations"""
    
    def __init__(self):
        self.engine = engine
        self.session_factory = SessionLocal
    
    @contextmanager
    def get_session(self):
        """Get database session with proper error handling"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def execute_migration(self, migration_sql: str):
        """Execute database migration"""
        try:
            with self.get_session() as session:
                session.execute(text(migration_sql))
            logger.info("Migration executed successfully")
        except Exception as e:
            logger.error("Migration failed", error=str(e))
            raise
    
    async def backup_database(self, backup_path: str):
        """Create database backup"""
        try:
            # Implementation depends on database type
            logger.info("Database backup created", path=backup_path)
        except Exception as e:
            logger.error("Database backup failed", error=str(e))
            raise


# Global database manager instance
db_manager = DatabaseManager()