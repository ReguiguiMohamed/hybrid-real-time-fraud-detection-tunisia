"""
Migration 0005: Shadow model tables via SQLAlchemy

Changes:
- Adds shadow_model_registry and shadow_score_log tables to the
  shared SQLAlchemy-managed database (replaces raw SQLite).
- Includes latency, label, and sample-count columns for enriched
  shadow comparison analysis.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from shared.database import Base, engine


def upgrade():
    ShadowModelRegistry.__table__.create(engine, checkfirst=True)
    ShadowScoreLog.__table__.create(engine, checkfirst=True)
    print("Migration 0005 applied: shadow model tables created via SQLAlchemy.")


def downgrade():
    ShadowScoreLog.__table__.drop(engine, checkfirst=True)
    ShadowModelRegistry.__table__.drop(engine, checkfirst=True)
    print("Migration 0005 downgraded: shadow model tables dropped.")


class ShadowModelRegistry(Base):
    __tablename__ = "shadow_model_registry"
    version_id = Column(String(256), primary_key=True)
    model_path = Column(Text, nullable=False)
    registered_at = Column(DateTime, server_default=func.now())
    unregistered_at = Column(DateTime)
    status = Column(String(64), default="shadow", index=True)
    total_comparisons = Column(Integer, default=0)
    avg_score_diff = Column(Float, default=0.0)
    shadow_wins = Column(Integer, default=0)
    champion_wins = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)
    avg_latency_ms = Column(Float, default=0.0)


class ShadowScoreLog(Base):
    __tablename__ = "shadow_score_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(256), nullable=False, index=True)
    champion_score = Column(Float, nullable=False)
    shadow_score = Column(Float, nullable=False)
    score_diff = Column(Float, nullable=False)
    champion_label = Column(Integer)
    shadow_label = Column(Integer)
    analyst_label = Column(String(64))
    latency_ms = Column(Float)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
