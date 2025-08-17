"""
Core settings configuration for WhatsApp Task Tracker
Based on Enhanced Framework V3.1 patterns
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings with environment-based configuration"""
    
    # Application
    environment: str = Field(default="development", env="WHATSAPP_TRACKER_ENV")
    log_level: str = Field(default="INFO", env="WHATSAPP_TRACKER_LOG_LEVEL")
    debug: bool = Field(default=False, env="WHATSAPP_TRACKER_DEBUG")
    
    # Database
    database_host: str = Field(default="localhost", env="WHATSAPP_TRACKER_DATABASE_HOST")
    database_port: int = Field(default=5432, env="WHATSAPP_TRACKER_DATABASE_PORT")
    database_name: str = Field(default="whatsapp_tracker", env="WHATSAPP_TRACKER_DATABASE_NAME")
    database_user: str = Field(default="whatsapp_user", env="WHATSAPP_TRACKER_DATABASE_USER")
    database_password: str = Field(default="", env="WHATSAPP_TRACKER_DATABASE_PASSWORD")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="WHATSAPP_TRACKER_REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="WHATSAPP_TRACKER_REDIS_PASSWORD")
    
    # API Configuration
    secret_key: str = Field(default="dev-secret-key", env="WHATSAPP_TRACKER_SECRET_KEY")
    access_token_expire_minutes: int = Field(default=1440, env="WHATSAPP_TRACKER_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_algorithm: str = Field(default="HS256", env="WHATSAPP_TRACKER_JWT_ALGORITHM")
    
    # AI Services
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # WhatsApp Configuration
    whatsapp_session_path: str = Field(default="./data/whatsapp_session", env="WHATSAPP_SESSION_PATH")
    whatsapp_qr_timeout: int = Field(default=60, env="WHATSAPP_QR_TIMEOUT")
    whatsapp_headless: bool = Field(default=True, env="WHATSAPP_HEADLESS")
    whatsapp_monitored_groups: List[str] = Field(default=[], env="WHATSAPP_MONITORED_GROUPS")
    whatsapp_rate_limit_delay: int = Field(default=1000, env="WHATSAPP_RATE_LIMIT_DELAY")
    
    # Security
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1", "*"], env="WHATSAPP_TRACKER_ALLOWED_HOSTS")
    allowed_origins: List[str] = Field(default=["http://localhost:8501", "http://127.0.0.1:8501"], env="WHATSAPP_TRACKER_ALLOWED_ORIGINS")
    
    # Features
    enable_web_interface: bool = Field(default=True, env="WHATSAPP_TRACKER_ENABLE_WEB_INTERFACE")
    enable_notifications: bool = Field(default=True, env="ENABLE_NOTIFICATIONS")
    auto_backup: bool = Field(default=True, env="WHATSAPP_TRACKER_AUTO_BACKUP")
    notification_channels: List[str] = Field(default=["console", "cache", "whatsapp"], env="NOTIFICATION_CHANNELS")
    
    # Paths
    data_path: str = Field(default="./data", env="WHATSAPP_TRACKER_DATA_PATH")
    logs_path: str = Field(default="./logs", env="WHATSAPP_TRACKER_LOGS_PATH")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ["development", "test", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in test mode"""
        return self.environment == "test"
    
    class Config:
        env_file = [".env", ".env.local"]
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()