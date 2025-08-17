"""
Database setup and initialization
Enhanced Framework V3.1 Implementation
"""

import structlog
from sqlalchemy import text
from database.connection import engine, Base, get_db_context
from database.models import Task, Conversation, TaskHistory, UserPattern, WhatsAppSession, AgentMetrics

logger = structlog.get_logger(__name__)


def setup_database():
    """Initialize database with tables and indexes"""
    try:
        logger.info("Setting up database")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Create additional indexes for performance
        create_performance_indexes()
        
        # Insert default data if needed
        create_default_data()
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error("Database setup failed", error=str(e))
        raise


def create_performance_indexes():
    """Create additional performance indexes"""
    try:
        with get_db_context() as db:
            # Composite indexes for common queries
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_priority ON tasks(estado, prioridad)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_group_status ON tasks(grupo_nombre, estado)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_responsable_status ON tasks(responsable, estado)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_timestamp_desc ON tasks(timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_conversations_group_timestamp ON conversations(grupo_nombre, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_task_history_task_timestamp ON task_history(task_id, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_user_patterns_user_group ON user_patterns(user_name, grupo_nombre)",
                "CREATE INDEX IF NOT EXISTS idx_agent_metrics_name_timestamp ON agent_metrics(agent_name, timestamp DESC)",
            ]
            
            for index_sql in indexes:
                db.execute(text(index_sql))
            
        logger.info("Performance indexes created")
        
    except Exception as e:
        logger.error("Failed to create performance indexes", error=str(e))


def create_default_data():
    """Create default data if needed"""
    try:
        with get_db_context() as db:
            # Check if we have any tasks
            result = db.execute(text("SELECT COUNT(*) FROM tasks")).scalar()
            
            if result == 0:
                logger.info("No existing data found, database is ready for use")
            else:
                logger.info("Found existing data", task_count=result)
                
    except Exception as e:
        logger.error("Failed to check existing data", error=str(e))


def migrate_from_sqlite(sqlite_path: str):
    """Migrate data from SQLite database to PostgreSQL"""
    try:
        import sqlite3
        import json
        from datetime import datetime
        
        logger.info("Starting migration from SQLite", path=sqlite_path)
        
        # Connect to SQLite database
        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        
        with get_db_context() as db:
            # Migrate tasks
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("SELECT * FROM tareas")
            
            task_count = 0
            for row in sqlite_cursor.fetchall():
                task = Task(
                    descripcion=row['descripcion'],
                    responsable=row['responsable'],
                    fecha_limite=row['fecha_limite'],
                    prioridad=row['prioridad'],
                    estado=row['estado'],
                    mensaje_original=row['mensaje_original'],
                    autor_mensaje=row['autor_mensaje'],
                    timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
                    grupo_id=row['grupo_id'],
                    grupo_nombre=row['grupo_nombre'],
                    mensaje_id=row['mensaje_id'],
                    confidence_score=row['confidence_score'],
                    analysis_metadata=row['analysis_metadata'],
                    completion_date=row['completion_date'],
                )
                db.add(task)
                task_count += 1
            
            # Migrate conversations
            sqlite_cursor.execute("SELECT * FROM conversaciones")
            
            conversation_count = 0
            for row in sqlite_cursor.fetchall():
                conversation = Conversation(
                    mensaje=row['mensaje'],
                    autor=row['autor'],
                    timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
                    grupo_id=row['grupo_id'],
                    grupo_nombre=row['grupo_nombre'],
                    mensaje_id=row['mensaje_id'],
                )
                db.add(conversation)
                conversation_count += 1
            
            # Migrate task history if it exists
            try:
                sqlite_cursor.execute("SELECT * FROM task_history")
                history_count = 0
                for row in sqlite_cursor.fetchall():
                    history = TaskHistory(
                        task_id=row['task_id'],
                        action=row['action'],
                        previous_state=row['previous_state'],
                        new_state=row['new_state'],
                        changed_by=row['changed_by'],
                        timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
                        notes=row['notes'],
                    )
                    db.add(history)
                    history_count += 1
                
                logger.info("Migrated task history", count=history_count)
            except sqlite3.OperationalError:
                logger.info("No task history table found in SQLite database")
            
            # Migrate user patterns if it exists
            try:
                sqlite_cursor.execute("SELECT * FROM user_patterns")
                pattern_count = 0
                for row in sqlite_cursor.fetchall():
                    pattern = UserPattern(
                        user_name=row['user_name'],
                        grupo_nombre=row['grupo_nombre'],
                        pattern_type=row['pattern_type'],
                        pattern_data=row['pattern_data'],
                        calculated_date=datetime.fromisoformat(row['calculated_date']) if row['calculated_date'] else None,
                        total_tasks=row.get('total_tasks', 0),
                        completed_tasks=row.get('completed_tasks', 0),
                        average_completion_time=row.get('average_completion_time'),
                        most_common_priority=row.get('most_common_priority'),
                        productivity_score=row.get('productivity_score'),
                    )
                    db.add(pattern)
                    pattern_count += 1
                
                logger.info("Migrated user patterns", count=pattern_count)
            except sqlite3.OperationalError:
                logger.info("No user patterns table found in SQLite database")
        
        sqlite_conn.close()
        
        logger.info(
            "Migration completed successfully",
            tasks=task_count,
            conversations=conversation_count
        )
        
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        raise


def reset_database():
    """Reset database by dropping and recreating all tables"""
    try:
        logger.warning("Resetting database - all data will be lost")
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        
        # Recreate tables
        setup_database()
        
        logger.info("Database reset completed")
        
    except Exception as e:
        logger.error("Database reset failed", error=str(e))
        raise


if __name__ == "__main__":
    setup_database()