"""Persistence boundary for model lifecycle operations."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from shared.database import AuditLog, Base, FeedbackLabel, HighRiskAlert, ModelRegistry, SessionLocal, engine


class ModelRepository:
    """Store feedback, model registry entries, and lifecycle audit events."""

    def __init__(self, session_factory=SessionLocal, bind=engine):
        self.session_factory = session_factory
        self.bind = bind

    @classmethod
    def for_sqlite_path(cls, database_path):
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bind = create_engine(
            f"sqlite:///{path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
        return cls(session_factory=factory, bind=bind)

    @contextmanager
    def session(self):
        db = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ensure_schema(self):
        Base.metadata.create_all(bind=self.bind)

    def count_verified_feedback(self) -> int:
        with self.session() as db:
            return (
                db.query(func.count(FeedbackLabel.id))
                .filter(FeedbackLabel.analyst_label.in_(("Confirmed Fraud", "False Positive")))
                .scalar()
                or 0
            )

    def get_verified_feedback_rows(self):
        with self.session() as db:
            rows = (
                db.query(
                    FeedbackLabel.transaction_id,
                    HighRiskAlert.user_id,
                    HighRiskAlert.amount_tnd,
                    HighRiskAlert.governorate,
                    HighRiskAlert.payment_method,
                    HighRiskAlert.ml_probability,
                    FeedbackLabel.analyst_label,
                )
                .join(HighRiskAlert, HighRiskAlert.transaction_id == FeedbackLabel.transaction_id)
                .filter(FeedbackLabel.analyst_label.in_(("Confirmed Fraud", "False Positive")))
                .all()
            )
        return [
            (
                transaction_id,
                user_id,
                amount_tnd,
                governorate,
                payment_method,
                ml_probability,
                1 if analyst_label == "Confirmed Fraud" else 0,
            )
            for (
                transaction_id,
                user_id,
                amount_tnd,
                governorate,
                payment_method,
                ml_probability,
                analyst_label,
            ) in rows
        ]

    def get_current_champion(self):
        with self.session() as db:
            row = (
                db.query(ModelRegistry)
                .filter(ModelRegistry.is_champion == 1)
                .order_by(ModelRegistry.promoted_at.desc())
                .first()
            )
            if not row:
                return None
            return {
                "version_id": row.version_id,
                "model_path": row.model_path,
                "f1_score": row.f1_score,
                "auc": row.auc,
                "promoted_at": row.promoted_at,
            }

    def record_model(
        self,
        *,
        version_id,
        model_path,
        f1_score,
        auc,
        is_champion,
        promoted_at,
        training_samples_count,
        feature_importance,
    ):
        if promoted_at and isinstance(promoted_at, str):
            promoted_at = datetime.fromisoformat(promoted_at)

        with self.session() as db:
            if is_champion:
                db.query(ModelRegistry).filter(ModelRegistry.is_champion == 1).update(
                    {ModelRegistry.is_champion: 0},
                    synchronize_session=False,
                )
            db.add(
                ModelRegistry(
                    version_id=version_id,
                    model_path=model_path,
                    f1_score=f1_score,
                    auc=auc,
                    is_champion=1 if is_champion else 0,
                    promoted_at=promoted_at,
                    training_samples_count=training_samples_count,
                    feature_importance=feature_importance,
                )
            )

    def log_audit_event(
        self,
        *,
        entity_type,
        entity_id,
        action,
        user_id,
        previous_state,
        new_state,
    ):
        with self.session() as db:
            db.add(
                AuditLog(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    action=action,
                    user_id=user_id,
                    previous_state=previous_state,
                    new_state=new_state,
                )
            )
