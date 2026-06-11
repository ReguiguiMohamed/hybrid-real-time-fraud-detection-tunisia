# dashboard/api.py
import hashlib
import json
import logging
import os
import time as _time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, field_validator
from sqlalchemy import text

from compliance.deadlines import ctaf_filing_deadline
from dashboard.monitoring import ForensicAnalyticEngine
from shared.database import DATABASE_URL, Base, SessionLocal, engine
from shared.logging_config import setup_logging
from shared.version import RELEASE_CHANNEL, __version__

# Initialize structured logging
setup_logging(service_name="fraud-api")
logger = logging.getLogger(__name__)

# Authentication setup
security = HTTPBearer(auto_error=False)

# Load API tokens from environment variables
ANALYST_TOKEN = os.getenv("ANALYST_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
METRICS_TOKEN = os.getenv("METRICS_TOKEN")
METRICS_ALLOW_PUBLIC = os.getenv("METRICS_ALLOW_PUBLIC", "").lower() in {"1", "true", "yes"}

if not ANALYST_TOKEN:
    raise RuntimeError("ANALYST_TOKEN must be set via environment variable.")
if not ADMIN_TOKEN:
    raise RuntimeError("ADMIN_TOKEN must be set via environment variable.")

ANALYST_TOKEN_HASH = hashlib.sha256(ANALYST_TOKEN.encode()).hexdigest()
ADMIN_TOKEN_HASH = hashlib.sha256(ADMIN_TOKEN.encode()).hexdigest()


def require_scopes(scopes):
    """Verify the API token against required scopes."""

    def verifier(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        if token_hash == ADMIN_TOKEN_HASH:
            role = "admin"
        elif token_hash == ANALYST_TOKEN_HASH:
            role = "analyst"
        else:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if role not in scopes:
            raise HTTPException(status_code=403, detail="Forbidden")
        return {"role": role}

    return verifier


# Database setup
# The app uses shared.database.SessionLocal for all DB operations
# which connects to either local SQLite or a remote Postgres URL.
# The prototype creates its schema through SQLAlchemy at startup.


@contextmanager
def get_db_session():
    """Context manager for SQLAlchemy database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def parse_datetime_value(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    raw_value = str(value).strip()
    candidates = [raw_value]
    if raw_value.endswith("Z"):
        candidates.append(f"{raw_value[:-1]}+00:00")
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            continue
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw_value, pattern)
        except ValueError:
            continue
    return None


def format_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def metrics_path_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    if route is None and path.startswith("/api/v1/"):
        return "api_v1_unmatched"
    if path.startswith("/api/v1/"):
        return path
    if path in {"/", "/health/", "/metrics", "/docs", "/openapi.json"}:
        return path
    return "unmatched"


def require_metrics_token(request: Request):
    if not METRICS_TOKEN and (DATABASE_BACKEND == "sqlite" or METRICS_ALLOW_PUBLIC):
        return
    expected = f"Bearer {METRICS_TOKEN}"
    if request.headers.get("authorization") != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def refresh_database_metrics():
    try:
        with get_db_session() as db:
            alert_rows = db.execute(
                text(
                    """
                SELECT COALESCE(alert_type, 'high_risk') AS alert_type, COUNT(*)
                FROM high_risk_alerts
                GROUP BY COALESCE(alert_type, 'high_risk')
            """
                )
            ).fetchall()
            for alert_type, count in alert_rows:
                db_alerts_total.labels(alert_type=alert_type or "unknown").set(count)

            feedback_count = db.execute(text("SELECT COUNT(*) FROM feedback_labels")).scalar() or 0
            review_queue_count = db.execute(text("SELECT COUNT(*) FROM high_risk_alerts")).scalar() or 0
            champion = db.execute(
                text(
                    """
                SELECT version_id, f1_score, auc
                FROM model_registry
                WHERE is_champion = 1
                ORDER BY promoted_at DESC
                LIMIT 1
            """
                )
            ).fetchone()

        db_feedback_total.set(feedback_count)
        db_review_queue_total.set(review_queue_count)
        model_champion_info.clear()
        if champion:
            version_id, f1_score, auc = champion
            model_champion_f1.set(f1_score or 0.0)
            model_champion_auc.set(auc or 0.0)
            model_champion_info.labels(version_id=version_id).set(1)
        else:
            model_champion_f1.set(0.0)
            model_champion_auc.set(0.0)
        db_metrics_scrape_success.set(1)
    except Exception:
        db_metrics_scrape_success.set(0)
        logger.exception("Unable to refresh database metrics")


RANDOM_SAMPLE_RATE = max(0.0, min(parse_float_env("RANDOM_SAMPLE_RATE", 0.01), 1.0))
ENVIRONMENT = os.getenv("DEPLOYMENT_ENVIRONMENT", "production")
DATABASE_BACKEND = "postgresql" if DATABASE_URL.startswith("postgresql") else "sqlite"

prometheus_registry = CollectorRegistry()
api_info = Gauge(
    "amastan_api_info",
    "Static information about the verified command-center API deployment.",
    ["service", "environment", "database_backend", "runtime"],
    registry=prometheus_registry,
)
api_requests_total = Counter(
    "amastan_api_requests_total",
    "Total HTTP requests handled by the FastAPI command-center API.",
    ["method", "path", "status_code"],
    registry=prometheus_registry,
)
api_request_latency_seconds = Histogram(
    "amastan_api_request_latency_seconds",
    "FastAPI command-center request latency in seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=prometheus_registry,
)
alerts_ingested_total = Counter(
    "amastan_alerts_ingested_total",
    "Alerts accepted through the authenticated command-center ingestion endpoint.",
    ["alert_type", "result"],
    registry=prometheus_registry,
)
db_readback_total = Counter(
    "amastan_db_readback_total",
    "Database-backed API readback operations.",
    ["endpoint", "result"],
    registry=prometheus_registry,
)
db_alerts_total = Gauge(
    "amastan_db_alerts_total",
    "Current persisted alerts by alert type.",
    ["alert_type"],
    registry=prometheus_registry,
)
db_feedback_total = Gauge(
    "amastan_db_feedback_total",
    "Current persisted analyst feedback labels.",
    registry=prometheus_registry,
)
db_review_queue_total = Gauge(
    "amastan_db_review_queue_total",
    "Current persisted alerts awaiting review.",
    registry=prometheus_registry,
)
db_metrics_scrape_success = Gauge(
    "amastan_db_metrics_scrape_success",
    "Whether the latest metrics scrape could read the SQLAlchemy database.",
    registry=prometheus_registry,
)
model_champion_f1 = Gauge(
    "amastan_model_champion_f1_score",
    "Registered F1 score for the current champion model.",
    registry=prometheus_registry,
)
model_champion_auc = Gauge(
    "amastan_model_champion_auc",
    "Registered AUC score for the current champion model.",
    registry=prometheus_registry,
)
model_champion_info = Gauge(
    "amastan_model_champion_info",
    "Current champion model identity.",
    ["version_id"],
    registry=prometheus_registry,
)

api_info.labels(
    service="fraud-detection-command-center-api",
    environment=ENVIRONMENT,
    database_backend=DATABASE_BACKEND,
    runtime="huggingface-space",
).set(1)

FEATURE_LABELS = {
    "v_count": "High velocity (v_count)",
    "g_dist": "Multi-governorate travel (g_dist)",
    "avg_amount": "High value transfer (avg_amount)",
    "is_smurfing": "Structuring pattern (is_smurfing)",
    "high_velocity_flag": "D17 velocity cap (high_velocity_flag)",
    "velocity_risk": "Velocity risk flag",
    "travel_risk": "Travel risk flag",
    "high_value_risk": "High value risk flag",
    "d17_risk": "D17 Flouci risk flag",
    "risk_score": "Composite risk score",
}


class FeedbackRequest(BaseModel):
    transaction_id: str
    analyst_label: str
    analyst_comment: Optional[str] = None
    branch_id: Optional[str] = None

    @field_validator("analyst_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        allowed = {"Confirmed Fraud", "False Positive"}
        if v not in allowed:
            raise ValueError(f"analyst_label must be one of {allowed}")
        return v

    @field_validator("transaction_id")
    @classmethod
    def validate_transaction_id(cls, v: str) -> str:
        if not v or len(v) > 256:
            raise ValueError("transaction_id must be non-empty and at most 256 characters")
        return v.strip()


class BatchFeedbackRequest(BaseModel):
    feedback_items: List[FeedbackRequest]


def log_audit_event(entity_type, entity_id, action, user_id, previous_state, new_state):
    with get_db_session() as db:
        db.execute(
            text(
                """
            INSERT INTO audit_logs
            (entity_type, entity_id, action, user_id, previous_state, new_state)
            VALUES (:entity_type, :entity_id, :action, :user_id, :previous_state, :new_state)
        """
            ),
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "user_id": user_id,
                "previous_state": previous_state,
                "new_state": new_state,
            },
        )
        db.commit()


class TransactionAlert(BaseModel):
    transaction_id: str
    user_id: str
    amount_tnd: float
    governorate: str
    payment_method: str
    branch_id: Optional[str] = None
    timestamp: str
    ml_probability: float
    sar_report: Optional[str] = None
    alert_type: Optional[str] = "high_risk"
    shap_top5: Optional[List[Dict[str, Any]]] = None
    anomaly_score: Optional[float] = None
    anomaly_model_version: Optional[str] = None
    ingestion_latency: Optional[float] = None


def parse_feature_importance(feature_payload, limit=3):
    if not feature_payload:
        return []
    try:
        feature_items = json.loads(feature_payload)
    except json.JSONDecodeError:
        return []

    normalized = []
    for item in feature_items:
        if isinstance(item, dict):
            feature_name = item.get("feature")
            score = item.get("score")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            feature_name, score = item
        else:
            continue
        if feature_name is None:
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None
        normalized.append((feature_name, score_value))

    normalized.sort(key=lambda item: item[1] if item[1] is not None else 0, reverse=True)
    top_features = normalized[:limit]
    return [
        {
            "feature": feature_name,
            "description": FEATURE_LABELS.get(feature_name, feature_name),
            "score": round(score, 4) if score is not None else None,
        }
        for feature_name, score in top_features
    ]


def parse_shap_top5(shap_payload):
    if not shap_payload:
        return []
    if isinstance(shap_payload, list):
        items = shap_payload
    else:
        try:
            items = json.loads(shap_payload)
        except (TypeError, json.JSONDecodeError):
            return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        feature_name = item.get("feature")
        if not feature_name:
            continue
        try:
            value = float(item.get("value", 0.0))
            impact = float(item.get("impact", 0.0))
            abs_impact = float(item.get("abs_impact", abs(impact)))
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                "feature": feature_name,
                "description": FEATURE_LABELS.get(feature_name, feature_name),
                "value": value,
                "impact": impact,
                "abs_impact": abs_impact,
                "direction": item.get("direction", "increases_risk" if impact >= 0 else "decreases_risk"),
                "confidence": item.get("confidence"),
            }
        )

    normalized.sort(key=lambda item: item["abs_impact"], reverse=True)
    return normalized[:5]


monitoring_engine = ForensicAnalyticEngine(SessionLocal)


# ---------------------------------------------------------------------------
# Rate Limiter (simple in-memory, per-IP)
# ---------------------------------------------------------------------------
_rate_limit_store: dict = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


# --- Database initialisation ---
# The database schema is managed via Base.metadata.create_all(engine)
# triggered in the lifespan context manager.


# ---------------------------------------------------------------------------
# Lifespan (modern startup/shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_application: FastAPI):
    logger.info("Starting Fraud Detection Command Center API")
    Base.metadata.create_all(bind=engine)
    yield
    logger.info("Fraud Detection Command Center API shut down")


app = FastAPI(
    title="Tunisian Fraud Detection - Command Center API",
    description="Fraud alert review and reporting for Tunisian digital payments",
    version=__version__,
    lifespan=lifespan,
)

CORS_ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Rate Limiting Middleware ---
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = _time.time()
    start_time = _time.perf_counter()

    # Clean old entries
    _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]

    path_label = metrics_path_label(request)
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        api_requests_total.labels(
            method=request.method,
            path=path_label,
            status_code="429",
        ).inc()
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    _rate_limit_store[client_ip].append(now)
    try:
        response = await call_next(request)
    except Exception:
        api_requests_total.labels(
            method=request.method,
            path=path_label,
            status_code="500",
        ).inc()
        api_request_latency_seconds.labels(
            method=request.method,
            path=path_label,
        ).observe(_time.perf_counter() - start_time)
        raise

    path_label = metrics_path_label(request)
    api_requests_total.labels(
        method=request.method,
        path=path_label,
        status_code=str(response.status_code),
    ).inc()
    elapsed = _time.perf_counter() - start_time
    api_request_latency_seconds.labels(
        method=request.method,
        path=path_label,
    ).observe(elapsed)
    monitoring_engine.record_inference_latency(elapsed * 1000)
    return response


# ---------------------------------------------------------------------------
# API Router (all business endpoints under /api/v1/)
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.get("/auth/whoami")
async def whoami(user_id: Optional[str] = Header(None), auth=Depends(require_scopes({"analyst", "admin"}))):
    """Return information about the authenticated user"""
    role = auth["role"]
    return {
        "user_id": user_id or "unknown",
        "role": role.upper(),
        "authenticated": True,
        "_links": {
            "self": "/api/v1/auth/whoami",
            "stats": "/api/v1/stats/",
            "review_queue": "/api/v1/alerts/review-queue/",
            "branches": "/api/v1/branches/",
            "compliance_kpis": "/api/v1/compliance/kpis/",
            "model_performance": "/api/v1/monitoring/model-performance/",
        },
    }


@router.post("/feedback/")
async def submit_feedback(
    feedback: FeedbackRequest, user_id: Optional[str] = Header(None), auth=Depends(require_scopes({"analyst", "admin"}))
):
    """Endpoint to receive analyst feedback on fraud predictions"""
    try:
        with get_db_session() as db:
            branch_id = feedback.branch_id
            if not branch_id:
                alert = db.execute(
                    text("SELECT branch_id FROM high_risk_alerts WHERE transaction_id = :transaction_id"),
                    {"transaction_id": feedback.transaction_id},
                ).fetchone()
                branch_id = alert[0] if alert else None

            previous_feedback = db.execute(
                text(
                    """
                SELECT analyst_label, analyst_comment
                FROM feedback_labels
                WHERE transaction_id = :transaction_id
                ORDER BY timestamp DESC
                LIMIT 1
            """
                ),
                {"transaction_id": feedback.transaction_id},
            ).fetchone()

            db.execute(
                text(
                    """
                INSERT INTO feedback_labels
                (transaction_id, analyst_label, analyst_comment, analyst_id, branch_id)
                VALUES (:transaction_id, :analyst_label, :analyst_comment, :analyst_id, :branch_id)
            """
                ),
                {
                    "transaction_id": feedback.transaction_id,
                    "analyst_label": feedback.analyst_label,
                    "analyst_comment": feedback.analyst_comment,
                    "analyst_id": user_id or auth["role"],
                    "branch_id": branch_id,
                },
            )
            db.commit()

        previous_state = None
        if previous_feedback:
            previous_state = json.dumps(
                {"analyst_label": previous_feedback[0], "analyst_comment": previous_feedback[1]}
            )
        new_state = json.dumps(
            {
                "analyst_label": feedback.analyst_label,
                "analyst_comment": feedback.analyst_comment,
                "branch_id": branch_id,
            }
        )
        log_audit_event("ALERT", feedback.transaction_id, "CLASSIFY", user_id or "unknown", previous_state, new_state)

        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "_links": {
                "feedback": "/api/v1/feedback/",
                "alert": f"/api/v1/alerts/{feedback.transaction_id}/explain",
                "stats": "/api/v1/stats/",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/batch/")
async def submit_batch_feedback(
    batch: BatchFeedbackRequest,
    user_id: Optional[str] = Header(None),
    auth=Depends(require_scopes({"analyst", "admin"})),
):
    """Submit multiple feedback entries at once."""
    results = []
    for item in batch.feedback_items:
        try:
            with get_db_session() as db:
                branch_id = item.branch_id
                if not branch_id:
                    alert = db.execute(
                        text("SELECT branch_id FROM high_risk_alerts WHERE transaction_id = :transaction_id"),
                        {"transaction_id": item.transaction_id},
                    ).fetchone()
                    branch_id = alert[0] if alert else None

                previous_feedback = db.execute(
                    text(
                        """
                    SELECT analyst_label, analyst_comment
                    FROM feedback_labels
                    WHERE transaction_id = :transaction_id
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                    ),
                    {"transaction_id": item.transaction_id},
                ).fetchone()

                db.execute(
                    text(
                        """
                    INSERT INTO feedback_labels
                    (transaction_id, analyst_label, analyst_comment, analyst_id, branch_id)
                    VALUES (:transaction_id, :analyst_label, :analyst_comment, :analyst_id, :branch_id)
                """
                    ),
                    {
                        "transaction_id": item.transaction_id,
                        "analyst_label": item.analyst_label,
                        "analyst_comment": item.analyst_comment,
                        "analyst_id": user_id or auth["role"],
                        "branch_id": branch_id,
                    },
                )
                db.commit()

            previous_state = None
            if previous_feedback:
                previous_state = json.dumps(
                    {"analyst_label": previous_feedback[0], "analyst_comment": previous_feedback[1]}
                )
            new_state = json.dumps(
                {"analyst_label": item.analyst_label, "analyst_comment": item.analyst_comment, "branch_id": branch_id}
            )
            log_audit_event("ALERT", item.transaction_id, "CLASSIFY", user_id or "unknown", previous_state, new_state)
            results.append({"transaction_id": item.transaction_id, "status": "success"})
        except Exception as e:
            results.append({"transaction_id": item.transaction_id, "status": "error", "detail": str(e)})

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    return {
        "status": "success" if error_count == 0 else "partial",
        "total": len(results),
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
        "_links": {
            "feedback": "/api/v1/feedback/",
            "feedback_batch": "/api/v1/feedback/batch/",
            "stats": "/api/v1/stats/",
        },
    }


@router.get("/alerts/high-risk/")
async def get_high_risk_alerts(
    limit: int = 50, branch_id: Optional[str] = None, auth=Depends(require_scopes({"analyst", "admin"}))
):
    """Endpoint to fetch high-risk alerts for the dashboard"""
    try:
        query = "SELECT transaction_id, user_id, amount_tnd, governorate, payment_method, branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk') AS alert_type, shap_top5 FROM high_risk_alerts WHERE COALESCE(alert_type, 'high_risk') = 'high_risk' AND ml_probability > 0.85"
        params = {"limit": limit}
        if branch_id:
            query += " AND branch_id = :branch_id"
            params["branch_id"] = branch_id
        query += " ORDER BY ml_probability DESC LIMIT :limit"

        with get_db_session() as db:
            rows = db.execute(text(query), params).fetchall()

        alerts = []
        for row in rows:
            alerts.append(
                {
                    "transaction_id": row[0],
                    "user_id": row[1],
                    "amount_tnd": row[2],
                    "governorate": row[3],
                    "payment_method": row[4],
                    "branch_id": row[5],
                    "timestamp": row[6],
                    "ml_probability": row[7],
                    "sar_report": row[8],
                    "alert_type": row[9],
                    "shap_top5": parse_shap_top5(row[10]),
                }
            )
        db_readback_total.labels(endpoint="alerts_high_risk", result="success").inc()
        return alerts
    except Exception as e:
        db_readback_total.labels(endpoint="alerts_high_risk", result="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/review-queue/")
async def get_review_queue(
    limit: int = 100,
    alert_type: Optional[str] = None,
    branch_id: Optional[str] = None,
    auth=Depends(require_scopes({"analyst", "admin"})),
):
    """Endpoint to fetch review queue alerts, including random samples"""
    try:
        query = "SELECT transaction_id, user_id, amount_tnd, governorate, payment_method, branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk') AS alert_type, shap_top5 FROM high_risk_alerts"
        conditions = []
        params = {"limit": limit}
        if alert_type:
            conditions.append("COALESCE(alert_type, 'high_risk') = :alert_type")
            params["alert_type"] = alert_type
        if branch_id:
            conditions.append("branch_id = :branch_id")
            params["branch_id"] = branch_id
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT :limit"

        with get_db_session() as db:
            rows = db.execute(text(query), params).fetchall()
        alerts = []
        for row in rows:
            alerts.append(
                {
                    "transaction_id": row[0],
                    "user_id": row[1],
                    "amount_tnd": row[2],
                    "governorate": row[3],
                    "payment_method": row[4],
                    "branch_id": row[5],
                    "timestamp": row[6],
                    "ml_probability": row[7],
                    "sar_report": row[8],
                    "alert_type": row[9],
                    "shap_top5": parse_shap_top5(row[10]),
                }
            )
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/branches/")
async def list_branches(auth=Depends(require_scopes({"analyst", "admin"}))):
    """List all distinct branch IDs that have triggered alerts."""
    try:
        with get_db_session() as db:
            rows = db.execute(
                text(
                    "SELECT DISTINCT branch_id FROM high_risk_alerts WHERE branch_id IS NOT NULL AND branch_id != '' ORDER BY branch_id"
                )
            ).fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/")
async def get_system_stats(branch_id: Optional[str] = None, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Get system statistics for monitoring"""
    try:
        with get_db_session() as db:
            params = {"branch_id": branch_id} if branch_id else {}
            branch_where = "WHERE branch_id = :branch_id" if branch_id else ""
            total_feedback = (
                db.execute(
                    text(f"SELECT COUNT(*) FROM feedback_labels {branch_where}"),
                    params,
                ).scalar()
                or 0
            )

            label_counts = dict(
                db.execute(
                    text(f"SELECT analyst_label, COUNT(*) FROM feedback_labels {branch_where} GROUP BY analyst_label"),
                    params,
                ).fetchall()
            )

            if branch_id:
                label_rows = db.execute(
                    text(
                        """
                SELECT COALESCE(hra.alert_type, 'unknown') AS alert_type,
                       fl.analyst_label,
                       COUNT(*)
                FROM feedback_labels fl
                LEFT JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                  AND fl.branch_id = :branch_id
                GROUP BY alert_type, fl.analyst_label
                """
                    ),
                    params,
                ).fetchall()
            else:
                label_rows = db.execute(
                    text(
                        """
                SELECT COALESCE(hra.alert_type, 'unknown') AS alert_type,
                       fl.analyst_label,
                       COUNT(*)
                FROM feedback_labels fl
                LEFT JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                GROUP BY alert_type, fl.analyst_label
                """
                    )
                ).fetchall()
            label_counts_by_type = {}
            for alert_type_value, analyst_label, count in label_rows:
                label_counts_by_type.setdefault(alert_type_value, {})[analyst_label] = count

            branch_alert_clause = "AND branch_id = :branch_id" if branch_id else ""
            high_risk_count = (
                db.execute(
                    text(
                        f"""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'high_risk'
                  AND ml_probability > 0.85
                  {branch_alert_clause}
            """
                    ),
                    params,
                ).scalar()
                or 0
            )
            random_sample_count = (
                db.execute(
                    text(
                        f"""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'random_sample'
                  {branch_alert_clause}
            """
                    ),
                    params,
                ).scalar()
                or 0
            )
            uncertainty_sample_count = (
                db.execute(
                    text(
                        f"""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'uncertainty_sample'
                  {branch_alert_clause}
            """
                    ),
                    params,
                ).scalar()
                or 0
            )
            review_where = "WHERE branch_id = :branch_id" if branch_id else ""
            review_queue_total = (
                db.execute(
                    text(f"SELECT COUNT(*) FROM high_risk_alerts {review_where}"),
                    params,
                ).scalar()
                or 0
            )

        # Calculate precision based on high-risk alerts only
        high_risk_counts = label_counts_by_type.get("high_risk", {})
        confirmed_fraud = high_risk_counts.get("Confirmed Fraud", 0)
        false_positive = high_risk_counts.get("False Positive", 0)
        high_risk_precision = (
            confirmed_fraud / (confirmed_fraud + false_positive) if (confirmed_fraud + false_positive) > 0 else 0
        )

        random_sample_counts = label_counts_by_type.get("random_sample", {})
        random_sample_fraud = random_sample_counts.get("Confirmed Fraud", 0)
        random_sample_non_fraud = random_sample_counts.get("False Positive", 0)
        random_sample_fraud_rate = (
            random_sample_fraud / (random_sample_fraud + random_sample_non_fraud)
            if (random_sample_fraud + random_sample_non_fraud) > 0
            else 0
        )

        return {
            "total_feedback": total_feedback,
            "high_risk_alerts": high_risk_count,
            "random_sample_alerts": random_sample_count,
            "uncertainty_sample_alerts": uncertainty_sample_count,
            "review_queue_total": review_queue_total,
            "random_sample_rate": RANDOM_SAMPLE_RATE,
            "feedback_breakdown": label_counts,
            "feedback_breakdown_by_type": label_counts_by_type,
            "precision": round(high_risk_precision, 3),
            "precision_scope": "high_risk_only",
            "high_risk_precision": round(high_risk_precision, 3),
            "random_sample_fraud_rate": round(random_sample_fraud_rate, 3),
            "_links": {
                "self": "/api/v1/stats/",
                "review_queue": "/api/v1/alerts/review-queue/",
                "compliance_kpis": "/api/v1/compliance/kpis/",
                "model_performance": "/api/v1/monitoring/model-performance/",
                "ctaf_export": "/api/v1/alerts/ctaf-export",
                "branches": "/api/v1/branches/",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/kpis/")
async def get_compliance_kpis(branch_id: Optional[str] = None, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Return compliance KPIs derived from recorded alerts, SARs, and feedback."""
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        since_30d = now - timedelta(days=30)

        branch_clause = "AND branch_id = :branch_id" if branch_id else ""
        params = {"branch_id": branch_id, "since_30d": since_30d} if branch_id else {"since_30d": since_30d}

        with get_db_session() as db:
            sar_reports_30d = (
                db.execute(
                    text(
                        f"""
            SELECT COUNT(*)
            FROM high_risk_alerts
            WHERE sar_report IS NOT NULL
              AND TRIM(sar_report) != ''
              AND created_at >= :since_30d
              {branch_clause}
            """
                    ),
                    params,
                ).scalar()
                or 0
            )

            filed_rows = db.execute(
                text(
                    f"""
            SELECT transaction_id, timestamp, created_at
            FROM high_risk_alerts
            WHERE sar_report IS NOT NULL
              AND TRIM(sar_report) != ''
              {branch_clause}
            """
                ),
                params,
            ).fetchall()

            on_time = 0
            evaluated_sars = 0
            for _, detection_timestamp, created_at in filed_rows:
                detected_at = parse_datetime_value(detection_timestamp)
                filed_at = parse_datetime_value(created_at)
                if detected_at is None or filed_at is None:
                    continue
                evaluated_sars += 1
                if filed_at <= ctaf_filing_deadline(from_date=detected_at, business_days=10):
                    on_time += 1

            pending_rows = db.execute(
                text(
                    f"""
            SELECT transaction_id, timestamp
            FROM high_risk_alerts
            WHERE COALESCE(alert_type, 'high_risk') IN ('high_risk', 'SANCTIONS_HIT')
              AND (sar_report IS NULL OR TRIM(sar_report) = '')
              {branch_clause}
            """
                ),
                params,
            ).fetchall()

            overdue_sars = []
            for transaction_id, detection_timestamp in pending_rows:
                detected_at = parse_datetime_value(detection_timestamp)
                if detected_at is None:
                    continue
                deadline = ctaf_filing_deadline(from_date=detected_at, business_days=10)
                if now > deadline:
                    overdue_sars.append(
                        {
                            "transaction_id": transaction_id,
                            "detected_at": detected_at.isoformat(),
                            "deadline": deadline.isoformat(),
                        }
                    )

            sanctions_hits_30d = (
                db.execute(
                    text(
                        f"""
            SELECT COUNT(*)
            FROM high_risk_alerts
            WHERE COALESCE(alert_type, 'high_risk') = 'SANCTIONS_HIT'
              AND created_at >= :since_30d
              {branch_clause}
            """
                    ),
                    params,
                ).scalar()
                or 0
            )

            if branch_id:
                feedback_rows = db.execute(
                    text(
                        """
                SELECT fl.analyst_label, COUNT(*)
                FROM feedback_labels fl
                JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IN ('Confirmed Fraud', 'False Positive')
                  AND fl.branch_id = :branch_id
                GROUP BY fl.analyst_label
                """
                    ),
                    params,
                ).fetchall()
            else:
                feedback_rows = db.execute(
                    text(
                        """
                SELECT analyst_label, COUNT(*)
                FROM feedback_labels
                WHERE analyst_label IN ('Confirmed Fraud', 'False Positive')
                GROUP BY analyst_label
                """
                    )
                ).fetchall()
            feedback_counts = dict(feedback_rows)
            reviewed_total = sum(feedback_counts.values())
            false_positives = feedback_counts.get("False Positive", 0)

            accounts_by_tier = dict(
                db.execute(
                    text(
                        f"""
            SELECT
                CASE
                    WHEN ml_probability >= 0.85 THEN 'CRITICAL'
                    WHEN ml_probability >= 0.70 THEN 'HIGH'
                    WHEN ml_probability >= 0.30 THEN 'MEDIUM'
                    ELSE 'LOW'
                END AS risk_tier,
                COUNT(DISTINCT user_id)
            FROM high_risk_alerts
            WHERE user_id IS NOT NULL
              AND user_id != ''
              AND COALESCE(alert_type, 'high_risk') IN ('high_risk', 'SANCTIONS_HIT')
              AND created_at >= :since_30d
              {branch_clause}
            GROUP BY risk_tier
            """
                    ),
                    params,
                ).fetchall()
            )

        return {
            "window_days": 30,
            "sar_reports_generated": sar_reports_30d,
            "sar_on_time_percent": format_percent(on_time, evaluated_sars),
            "sar_on_time_sample_count": evaluated_sars,
            "overdue_sar_count": len(overdue_sars),
            "overdue_sars": overdue_sars,
            "sanctions_hits": sanctions_hits_30d,
            "false_positive_rate": format_percent(false_positives, reviewed_total),
            "reviewed_alerts": reviewed_total,
            "high_risk_accounts_by_tier": accounts_by_tier,
            "branch_id": branch_id,
            "generated_at": now.isoformat(),
            "basis": "Recorded alerts, generated SAR text, analyst feedback, and sanctions alerts.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/add/")
async def add_high_risk_alert(alert: TransactionAlert, auth=Depends(require_scopes({"admin"}))):
    """Store a fraud alert for analyst review."""
    try:
        alert_type = alert.alert_type or "high_risk"
        shap_payload = json.dumps(alert.shap_top5 or [])

        with get_db_session() as db:
            existing = db.execute(
                text("SELECT id FROM high_risk_alerts WHERE transaction_id = :transaction_id"),
                {"transaction_id": alert.transaction_id},
            ).fetchone()
            if existing:
                alerts_ingested_total.labels(alert_type=alert_type, result="duplicate").inc()
                return {"status": "success", "message": "Alert already exists"}

            db.execute(
                text(
                    """
                INSERT INTO high_risk_alerts
                (transaction_id, user_id, amount_tnd, governorate, payment_method, branch_id,
                 timestamp, ml_probability, sar_report, alert_type, shap_top5,
                 anomaly_score, anomaly_model_version, ingestion_latency)
                VALUES (:transaction_id, :user_id, :amount_tnd, :governorate, :payment_method,
                        :branch_id, :timestamp, :ml_probability, :sar_report, :alert_type,
                        :shap_top5, :anomaly_score, :anomaly_model_version, :ingestion_latency)
            """
                ),
                {
                    "transaction_id": alert.transaction_id,
                    "user_id": alert.user_id,
                    "amount_tnd": alert.amount_tnd,
                    "governorate": alert.governorate,
                    "payment_method": alert.payment_method,
                    "branch_id": alert.branch_id,
                    "timestamp": alert.timestamp,
                    "ml_probability": alert.ml_probability,
                    "sar_report": alert.sar_report,
                    "alert_type": alert_type,
                    "shap_top5": shap_payload,
                    "anomaly_score": alert.anomaly_score,
                    "anomaly_model_version": alert.anomaly_model_version,
                    "ingestion_latency": alert.ingestion_latency,
                },
            )
            db.commit()

        alerts_ingested_total.labels(alert_type=alert_type, result="success").inc()
        return {"status": "success", "message": "Alert added successfully"}
    except Exception as e:
        alerts_ingested_total.labels(alert_type=alert.alert_type or "high_risk", result="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/model-performance/")
async def get_model_performance(branch_id: Optional[str] = None, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Get model performance metrics based on human feedback"""
    try:
        with get_db_session() as db:
            if branch_id:
                prob_label_pairs = db.execute(
                    text(
                        """
                SELECT hra.ml_probability, COALESCE(hra.alert_type, 'high_risk'), fl.analyst_label
                FROM high_risk_alerts hra
                JOIN feedback_labels fl ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                  AND COALESCE(hra.alert_type, 'high_risk') IN ('high_risk', 'random_sample')
                  AND fl.branch_id = :branch_id
                """
                    ),
                    {"branch_id": branch_id},
                ).fetchall()
            else:
                prob_label_pairs = db.execute(
                    text(
                        """
                SELECT hra.ml_probability, COALESCE(hra.alert_type, 'high_risk'), fl.analyst_label
                FROM high_risk_alerts hra
                JOIN feedback_labels fl ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                AND COALESCE(hra.alert_type, 'high_risk') IN ('high_risk', 'random_sample')
                """
                    )
                ).fetchall()

        if not prob_label_pairs:
            return {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "total_evaluated": 0,
                "note": "Metrics calculated only on reviewed alerts, not overall model performance",
                "warning": "Cannot calculate true model performance without sampling negative cases",
            }

        # Calculate performance metrics properly for the reviewed subset
        # High-risk alerts are model-flagged fraud; random samples are low-risk reviews
        tp = sum(
            1 for _, alert_type, label in prob_label_pairs if alert_type == "high_risk" and label == "Confirmed Fraud"
        )
        fp = sum(
            1 for _, alert_type, label in prob_label_pairs if alert_type == "high_risk" and label == "False Positive"
        )
        tn = sum(
            1
            for _, alert_type, label in prob_label_pairs
            if alert_type == "random_sample" and label == "False Positive"
        )
        fn_sampled = sum(
            1
            for _, alert_type, label in prob_label_pairs
            if alert_type == "random_sample" and label == "Confirmed Fraud"
        )

        # Calculate precision based only on reviewed alerts where model predicted fraud
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Calculate recall using sampled false negatives
        random_sample_reviewed = tn + fn_sampled
        if random_sample_reviewed == 0:
            reviewed_recall = 0
            estimated_fn = 0
            estimated_recall = 0
        else:
            reviewed_recall = tp / (tp + fn_sampled) if (tp + fn_sampled) > 0 else 0
            estimated_fn = fn_sampled / RANDOM_SAMPLE_RATE if RANDOM_SAMPLE_RATE > 0 else 0
            estimated_recall = tp / (tp + estimated_fn) if (tp + estimated_fn) > 0 else 0

        f1_score = (
            2 * (precision * estimated_recall) / (precision + estimated_recall)
            if (precision + estimated_recall) > 0
            else 0
        )

        return {
            "precision": round(precision, 3),
            "recall": round(estimated_recall, 3),
            "reviewed_recall": round(reviewed_recall, 3),
            "f1_score": round(f1_score, 3),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn_sampled,
            "estimated_false_negatives": round(estimated_fn, 3),
            "random_sample_reviewed": random_sample_reviewed,
            "random_sample_rate": RANDOM_SAMPLE_RATE,
            "total_evaluated": len(prob_label_pairs),
            "note": "Metrics combine high-risk reviews with random-sample reviews to estimate recall.",
            "warning": "Estimated recall assumes random samples represent low-risk traffic.",
            "interpretation": "Precision reflects alert performance; recall is sampling-adjusted.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{transaction_id}/explain")
async def explain_alert(transaction_id: str, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Explain the top risk factors for a transaction using model feature importance."""
    try:
        with get_db_session() as db:
            row = db.execute(
                text(
                    """
            SELECT transaction_id, COALESCE(alert_type, 'high_risk'), shap_top5
            FROM high_risk_alerts
            WHERE transaction_id = :transaction_id
            """
                ),
                {"transaction_id": transaction_id},
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Transaction not found")

            registry_row = db.execute(
                text(
                    """
            SELECT feature_importance
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
            """
                )
            ).fetchone()

        shap_top5 = parse_shap_top5(row[2])
        if shap_top5:
            return {
                "transaction_id": transaction_id,
                "alert_type": row[1],
                "shap_top5": shap_top5,
                "top_risk_factors": shap_top5,
            }

        if not registry_row or not registry_row[0]:
            return {
                "transaction_id": transaction_id,
                "alert_type": row[1],
                "shap_top5": [],
                "top_risk_factors": [],
                "note": "No champion feature importance registered yet.",
            }

        factors = parse_feature_importance(registry_row[0], limit=3)

        return {"transaction_id": transaction_id, "alert_type": row[1], "shap_top5": [], "top_risk_factors": factors}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{transaction_id}/export")
async def export_alert(transaction_id: str, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Export a single alert for compliance filing."""
    try:
        with get_db_session() as db:
            alert_row = db.execute(
                text(
                    """
            SELECT transaction_id, user_id, amount_tnd, governorate, payment_method,
                   branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk'), shap_top5
            FROM high_risk_alerts
            WHERE transaction_id = :transaction_id
            """
                ),
                {"transaction_id": transaction_id},
            ).fetchone()
            if not alert_row:
                raise HTTPException(status_code=404, detail="Transaction not found")

            feedback_row = db.execute(
                text(
                    """
            SELECT analyst_label, analyst_comment, timestamp
            FROM feedback_labels
            WHERE transaction_id = :transaction_id
            ORDER BY timestamp DESC
            LIMIT 1
            """
                ),
                {"transaction_id": transaction_id},
            ).fetchone()

            registry_row = db.execute(
                text(
                    """
            SELECT feature_importance
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
            """
                )
            ).fetchone()

        shap_top5 = parse_shap_top5(alert_row[10])
        factors = shap_top5 or parse_feature_importance(registry_row[0] if registry_row else None, limit=3)

        analyst_payload = None
        if feedback_row:
            analyst_payload = {"label": feedback_row[0], "comment": feedback_row[1], "timestamp": feedback_row[2]}

        return {
            "transaction_id": alert_row[0],
            "user_id": alert_row[1],
            "amount_tnd": alert_row[2],
            "governorate": alert_row[3],
            "payment_method": alert_row[4],
            "branch_id": alert_row[5],
            "timestamp": alert_row[6],
            "ml_probability": alert_row[7],
            "sar_report": alert_row[8],
            "alert_type": alert_row[9],
            "shap_top5": shap_top5,
            "top_risk_factors": factors,
            "analyst_review": analyst_payload,
            "exported_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/ctaf-export")
async def export_ctaf(days: int = 7, branch_id: Optional[str] = None, auth=Depends(require_scopes({"admin"}))):
    """Export confirmed fraud alerts for CTAF reporting."""
    try:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        with get_db_session() as db:
            if branch_id:
                rows = db.execute(
                    text(
                        """
                SELECT hra.transaction_id, hra.user_id, hra.amount_tnd, hra.governorate, hra.payment_method,
                       hra.branch_id, hra.timestamp, hra.ml_probability, hra.sar_report, hra.shap_top5,
                       fl.analyst_label, fl.analyst_comment, fl.timestamp
                FROM feedback_labels fl
                JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label = 'Confirmed Fraud'
                  AND fl.timestamp >= :cutoff
                  AND fl.branch_id = :branch_id
                ORDER BY fl.timestamp DESC
                """
                    ),
                    {"cutoff": cutoff, "branch_id": branch_id},
                ).fetchall()
            else:
                rows = db.execute(
                    text(
                        """
                SELECT hra.transaction_id, hra.user_id, hra.amount_tnd, hra.governorate, hra.payment_method,
                       hra.branch_id, hra.timestamp, hra.ml_probability, hra.sar_report, hra.shap_top5,
                       fl.analyst_label, fl.analyst_comment, fl.timestamp
                FROM feedback_labels fl
                JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label = 'Confirmed Fraud'
                  AND fl.timestamp >= :cutoff
                ORDER BY fl.timestamp DESC
                """
                    ),
                    {"cutoff": cutoff},
                ).fetchall()

            registry_row = db.execute(
                text(
                    """
            SELECT feature_importance
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
            """
                )
            ).fetchone()

        factors = parse_feature_importance(registry_row[0] if registry_row else None, limit=3)

        cases = []
        for row in rows:
            cases.append(
                {
                    "transaction_id": row[0],
                    "user_id": row[1],
                    "amount_tnd": row[2],
                    "governorate": row[3],
                    "payment_method": row[4],
                    "branch_id": row[5],
                    "timestamp": row[6],
                    "ml_probability": row[7],
                    "sar_report": row[8],
                    "shap_top5": parse_shap_top5(row[9]),
                    "analyst_label": row[10],
                    "analyst_comment": row[11],
                    "analyst_timestamp": row[12],
                    "top_risk_factors": parse_shap_top5(row[9]) or factors,
                }
            )

        return {
            "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "days": days,
            "branch_id": branch_id,
            "total_cases": len(cases),
            "cases": cases,
            "_links": {
                "self": "/api/v1/alerts/ctaf-export",
                "stats": "/api/v1/stats/",
                "review_queue": "/api/v1/alerts/review-queue/",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/performance")
async def get_performance_metrics(auth=Depends(require_scopes({"analyst", "admin"}))):
    """Get model performance metrics including precision, recall, and drift indicators."""
    return monitoring_engine.get_performance_metrics()


@router.get("/metrics/feedback")
async def get_feedback_analysis(auth=Depends(require_scopes({"analyst", "admin"}))):
    """Return analyst feedback breakdown and label distribution metrics."""
    return monitoring_engine.get_feedback_analysis()


@router.get("/metrics/threshold-analysis")
async def get_threshold_analysis(auth=Depends(require_scopes({"analyst", "admin"}))):
    """Analyze ML probability threshold trade-offs and recommend optimal cutoffs."""
    return monitoring_engine.get_ml_threshold_analysis()


@router.get("/metrics/drift")
async def get_drift_analysis(auth=Depends(require_scopes({"analyst", "admin"}))):
    """Assess model drift and recommend retraining based on feature distribution shifts."""
    return monitoring_engine.get_drift_retraining_assessment()


@router.get("/metrics/system-overview")
async def get_system_overview(auth=Depends(require_scopes({"analyst", "admin"}))):
    """Aggregated system overview combining performance, feedback, threshold, and drift metrics."""
    return {
        "performance": monitoring_engine.get_performance_metrics(),
        "feedback": monitoring_engine.get_feedback_analysis(),
        "threshold_recommendation": monitoring_engine.get_ml_threshold_analysis(),
        "drift": monitoring_engine.get_drift_retraining_assessment(),
    }


@router.get("/model/training-summary")
async def get_retraining_summary(auth=Depends(require_scopes({"analyst", "admin"}))):
    """Return persistent training status from the model registry."""
    try:
        from ml.model_repository import ModelRepository

        repo = ModelRepository()
        summary = repo.get_champion_training_status()
        return {"status": "available", **summary}
    except Exception as e:
        return {"status": "unavailable", "detail": str(e)}


app.include_router(router)


@app.get("/")
async def root_status():
    """Root status endpoint for hosted environments and browser checks."""
    return {
        "status": "healthy",
        "service": "fraud-detection-command-center-api",
        "version": __version__,
        "release_channel": RELEASE_CHANNEL,
        "routes": {
            "health": "/health/",
            "docs": "/docs",
            "api": "/api/v1/",
        },
    }


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request):
    """Prometheus/OpenMetrics endpoint for Grafana Cloud Metrics Endpoint scraping."""
    require_metrics_token(request)
    refresh_database_metrics()
    return Response(content=generate_latest(prometheus_registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/health/")
async def health_check():
    """Health check endpoint for the API (always at root, not versioned)"""
    return {
        "status": "healthy",
        "service": "fraud-detection-command-center-api",
        "version": __version__,
        "release_channel": RELEASE_CHANNEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
