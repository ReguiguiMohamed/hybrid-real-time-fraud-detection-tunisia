import os
import sqlite3
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker


# Register adapters for Python 3.12+ sqlite3 deprecation
def _adapt_datetime(value):
    return value.isoformat() if isinstance(value, datetime) else value


sqlite3.register_adapter(datetime, _adapt_datetime)

# Default to SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/feedback.db")

# Create engine.
# SQLite needs check_same_thread disabled for FastAPI/TestClient.
# Neon/serverless Postgres can close idle SSL connections, so pre-ping and
# short recycling keep pooled connections from being reused after the server
# has already dropped them.
connect_args = {}
engine_kwargs = {
    "pool_pre_ping": True,
    "pool_recycle": int(os.getenv("DATABASE_POOL_RECYCLE_SECONDS", "300")),
}
if DATABASE_URL.startswith("sqlite"):
    database_path = make_url(DATABASE_URL).database
    if database_path and database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
    engine_kwargs = {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class HighRiskAlert(Base):
    __tablename__ = "high_risk_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(256), unique=True, nullable=False, index=True)
    user_id = Column(String(256))
    amount_tnd = Column(Float)
    governorate = Column(String(128))
    payment_method = Column(String(128))
    branch_id = Column(String(128), index=True)
    timestamp = Column(String(64))
    ml_probability = Column(Float, default=0.0)
    sar_report = Column(Text)
    alert_type = Column(String(64), default="high_risk", index=True)
    shap_top5 = Column(Text)
    anomaly_score = Column(Float)
    anomaly_model_version = Column(String(128))
    ingestion_latency = Column(Float)
    status = Column(String(64), default="pending")
    risk_factors = Column(Text)
    user_id_hashed = Column(String(256), index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class FeedbackLabel(Base):
    __tablename__ = "feedback_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(256), nullable=False, index=True)
    analyst_label = Column(String(64))
    analyst_comment = Column(Text)
    analyst_id = Column(String(128))
    branch_id = Column(String(128), index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    version_id = Column(String(256), primary_key=True)
    model_path = Column(Text, nullable=False)
    f1_score = Column(Float)
    auc = Column(Float)
    is_champion = Column(Integer, default=0, index=True)
    promoted_at = Column(DateTime)
    training_samples_count = Column(Integer)
    feature_importance = Column(Text)
    last_training_success_at = Column(DateTime)
    last_training_failure_at = Column(DateTime)
    last_training_error = Column(Text)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(128), index=True)
    entity_id = Column(String(256), index=True)
    action = Column(String(128))
    user_id = Column(String(256))
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    previous_state = Column(Text)
    new_state = Column(Text)
