"""Read model lifecycle status from the shared database."""

from contextlib import contextmanager

from shared.database import ModelRegistry, SessionLocal


class ModelRepository:
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    @contextmanager
    def session(self):
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()

    def get_champion_training_status(self):
        with self.session() as db:
            row = (
                db.query(ModelRegistry)
                .filter(ModelRegistry.is_champion == 1)
                .order_by(ModelRegistry.promoted_at.desc())
                .first()
            )

        if not row:
            return {"last_success_at": None, "last_failure_at": None, "last_error": None}

        return {
            "last_success_at": row.last_training_success_at.isoformat() if row.last_training_success_at else None,
            "last_failure_at": row.last_training_failure_at.isoformat() if row.last_training_failure_at else None,
            "last_error": row.last_training_error,
        }
