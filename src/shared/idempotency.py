"""
Idempotency Layer for Amastan Fraud Shield Guard

Prevents duplicate processing of the same transaction when Kafka delivers
messages more than once ("at-least-once" semantics).

Uses a time-bounded deduplication cache:
- Redis (preferred, production)
- SQLite LRU cache (fallback, development)

Every transaction is checked against the cache before scoring.
If already processed, it is silently skipped.

Usage:
    from src.shared.idempotency import DedupCache

    dedup = DedupCache()
    if dedup.is_duplicate("tx-id-001"):
        print("Already processed, skipping")
    else:
        dedup.mark_processed("tx-id-001")
        # ... score transaction
"""

import logging
import os
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = int(os.getenv("DEDUP_CACHE_TTL_SECONDS", "600"))  # 10 min default
MAX_CACHE_SIZE = int(os.getenv("DEDUP_MAX_CACHE_SIZE", "100000"))


class IdempotencyError(Exception):
    pass


class DedupCache:
    """
    Thread-safe deduplication cache with TTL eviction.

    Supports two backends:
    1. Redis (if available and configured)
    2. SQLite LRU cache (default fallback)
    """

    _instance = None

    def __new__(cls, backend: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend: str = None):
        if hasattr(self, "_initialized"):
            return

        self.backend = backend or self._detect_backend()
        self._redis_client = None
        self._db_path = str(Path(__file__).parent.parent.parent / "data" / "dedup_cache.db")

        if self.backend == "redis":
            self._init_redis()
        else:
            self._init_sqlite()

        self._initialized = True

    @staticmethod
    def _detect_backend() -> str:
        """Auto-detect whether Redis is available."""
        if os.getenv("REDIS_URL") or os.getenv("REDIS_HOST"):
            return "redis"
        return "sqlite"

    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            import redis

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis_client = redis.Redis.from_url(redis_url, socket_timeout=5, decode_responses=True)
            self._redis_client.ping()
            logger.info(f"Redis dedup cache initialized: {redis_url}")
        except ImportError:
            logger.warning("redis package not installed. Falling back to SQLite.")
            self.backend = "sqlite"
            self._init_sqlite()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to SQLite.")
            self.backend = "sqlite"
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite dedup cache with LRU eviction."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dedup_cache (
                tx_id TEXT PRIMARY KEY,
                processed_at REAL NOT NULL,
                score REAL DEFAULT 0.0,
                alert_triggered INTEGER DEFAULT 0
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dedup_expiry ON dedup_cache(processed_at)")
        conn.commit()
        conn.close()
        logger.info(f"SQLite dedup cache initialized: {self._db_path}")

    def is_duplicate(self, tx_id: str) -> bool:
        """
        Check if a transaction ID has already been processed.

        Args:
            tx_id: The unique transaction identifier

        Returns:
            True if already processed (within TTL window).
        """
        if self.backend == "redis" and self._redis_client:
            return self._redis_client.exists(f"dedup:{tx_id}") == 1

        return self._sqlite_is_duplicate(tx_id)

    def _sqlite_is_duplicate(self, tx_id: str) -> bool:
        """Check SQLite cache for duplicate."""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM dedup_cache WHERE tx_id = ? AND processed_at > ?",
                (tx_id, time.time() - CACHE_TTL_SECONDS),
            )
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except Exception as e:
            logger.error(f"SQLite dedup check failed: {e}")
            return False  # Fail open: process the tx rather than skip it

    def mark_processed(self, tx_id: str, score: float = 0.0, alert_triggered: bool = False):
        """
        Mark a transaction as processed.

        Args:
            tx_id: The unique transaction identifier
            score: The fraud score assigned
            alert_triggered: Whether an alert was generated
        """
        now = time.time()

        if self.backend == "redis" and self._redis_client:
            try:
                pipe = self._redis_client.pipeline()
                pipe.set(f"dedup:{tx_id}", json.dumps({"score": score, "alert": alert_triggered}), ex=CACHE_TTL_SECONDS)
                pipe.execute()
            except Exception as e:
                logger.error(f"Redis mark_processed failed: {e}")
                self._sqlite_mark_processed(tx_id, score, alert_triggered)
            return

        self._sqlite_mark_processed(tx_id, score, alert_triggered)

    def _sqlite_mark_processed(self, tx_id: str, score: float = 0.0, alert_triggered: bool = False):
        """Mark transaction as processed in SQLite."""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            # Insert or update (in case of retry)
            cursor.execute(
                """
                INSERT INTO dedup_cache (tx_id, processed_at, score, alert_triggered)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tx_id) DO UPDATE SET
                    processed_at = excluded.processed_at,
                    score = excluded.score,
                    alert_triggered = excluded.alert_triggered
            """,
                (tx_id, time.time(), score, 1 if alert_triggered else 0),
            )

            # Evict expired entries periodically (every 100th call)
            if int(time.time()) % 100 == 0:
                cursor.execute(
                    "DELETE FROM dedup_cache WHERE processed_at < ?",
                    (time.time() - CACHE_TTL_SECONDS,),
                )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"SQLite mark_processed failed: {e}")

    def get_processed_info(self, tx_id: str) -> Optional[dict]:
        """Get the processing info for a previously processed transaction."""
        if self.backend == "redis" and self._redis_client:
            data = self._redis_client.get(f"dedup:{tx_id}")
            if data:
                import json

                return json.loads(data)
            return None

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT score, alert_triggered FROM dedup_cache WHERE tx_id = ? AND processed_at > ?",
                (tx_id, time.time() - CACHE_TTL_SECONDS),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {"score": row[0], "alert_triggered": bool(row[1])}
            return None
        except Exception as e:
            logger.error(f"SQLite get_processed_info failed: {e}")
            return None

    def stats(self) -> dict:
        """Get cache statistics."""
        if self.backend == "redis" and self._redis_client:
            info = self._redis_client.info("memory")
            return {
                "backend": "redis",
                "ttl_seconds": CACHE_TTL_SECONDS,
                "memory_bytes": info.get("used_memory", 0),
            }

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dedup_cache")
            total = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*) FROM dedup_cache WHERE processed_at > ?",
                (time.time() - CACHE_TTL_SECONDS,),
            )
            active = cursor.fetchone()[0]
            conn.close()
            return {
                "backend": "sqlite",
                "total_entries": total,
                "active_entries": active,
                "ttl_seconds": CACHE_TTL_SECONDS,
            }
        except Exception as e:
            return {"backend": "sqlite", "error": str(e)}

    def clear_expired(self):
        """Manually clear expired entries."""
        if self.backend == "redis" and self._redis_client:
            # Redis handles this automatically via TTL
            return

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            deleted = cursor.execute(
                "DELETE FROM dedup_cache WHERE processed_at < ?",
                (time.time() - CACHE_TTL_SECONDS,),
            ).rowcount
            conn.commit()
            conn.close()
            logger.info(f"Cleared {deleted} expired dedup entries")
        except Exception as e:
            logger.error(f"Failed to clear expired dedup entries: {e}")


import json

# Module-level singleton
_dedup_cache = None


def get_dedup_cache() -> DedupCache:
    """Get the dedup cache singleton."""
    global _dedup_cache
    if _dedup_cache is None:
        _dedup_cache = DedupCache()
    return _dedup_cache
