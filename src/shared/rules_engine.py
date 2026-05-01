"""
Dynamic Rules Engine for Amastan Fraud Shield Guard
Replaces hard-coded risk_config.py with database-driven, hot-reloadable rules.

Features:
- Reads risk weights, thresholds, and governorate profiles from SQLite
- Hot-reload on every batch (cached with TTL for performance)
- Audit trail for every rule change
- Fallback to compiled defaults if database is unavailable
"""
import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Default compiled fallbacks (matching risk_config.py defaults)
DEFAULT_RISK_WEIGHTS = {
    "velocity": 0.3,
    "travel": 0.3,
    "high_value": 0.2,
    "d17_limit": 0.2,
}

DEFAULT_CBDC_GOVERNORATES = ["Tunis", "Sfax"]
DEFAULT_D17_SOFT_LIMIT = 1500.0
DEFAULT_D17_VELOCITY_CAP = 5
# Finance Law 2026: TND 5,000 cash cap repealed. Threshold is now a general
# large-transaction AML monitoring trigger, not a structuring-cap signal.
DEFAULT_HIGH_VALUE_THRESHOLD = 15000.0
# Velocity-based smurfing defaults (independent of any hard cash cap)
DEFAULT_SMURFING_UNIT_CAP = 3000.0    # per-tx amount ceiling to qualify as smurfing unit
DEFAULT_SMURFING_AGG_MIN = 9000.0     # window aggregate must exceed this to fire
DEFAULT_SMURFING_MIN_COUNT = 3        # minimum qualifying tx count in window


class RulesEngine:
    """
    Database-driven rules engine with hot-reload capability.
    Risk officers can update thresholds without code deployment.
    """

    _instance = None
    _cache = {}
    _cache_ttl = 30  # seconds
    _cache_timestamp = 0

    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = None):
        if db_path:
            self.db_path = db_path
        elif not hasattr(self, "db_path"):
            self.db_path = str(Path(__file__).parent.parent.parent / "data" / "feedback.db")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection with WAL mode and read-only safety."""
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=rw", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError:
            # Fallback to read-only if database is locked
            try:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5)
                conn.row_factory = sqlite3.Row
                return conn
            except sqlite3.OperationalError:
                logger.warning("Rules engine: database not available, using compiled defaults")
                return None

    def _is_cache_valid(self) -> bool:
        """Check if the cache is still within TTL."""
        return (time.time() - self._cache_timestamp) < self._cache_ttl

    def _refresh_cache(self):
        """Reload all rules from database into cache."""
        conn = self._get_connection()
        if conn is None:
            return

        try:
            cursor = conn.cursor()

            # Load risk rules
            cursor.execute("SELECT rule_name, weight, threshold, is_active FROM risk_rules WHERE is_active = 1")
            risk_rules = {}
            for row in cursor.fetchall():
                risk_rules[row["rule_name"]] = {
                    "weight": row["weight"],
                    "threshold": row["threshold"],
                    "is_active": row["is_active"],
                }

            # Load governorate profiles
            cursor.execute(
                "SELECT governorate_name, risk_multiplier, is_cbdc_pilot, is_high_risk_zone FROM governorate_risk_profiles"
            )
            governorates = {}
            for row in cursor.fetchall():
                governorates[row["governorate_name"]] = {
                    "risk_multiplier": row["risk_multiplier"],
                    "is_cbdc_pilot": bool(row["is_cbdc_pilot"]),
                    "is_high_risk_zone": bool(row["is_high_risk_zone"]),
                }

            # Load D17 rules
            cursor.execute(
                "SELECT rule_name, ewallet_provider, threshold_amount, velocity_limit, window_minutes, risk_boost, is_active FROM d17_rules WHERE is_active = 1"
            )
            d17_rules = {}
            for row in cursor.fetchall():
                d17_rules[row["rule_name"]] = {
                    "ewallet_provider": row["ewallet_provider"],
                    "threshold_amount": row["threshold_amount"],
                    "velocity_limit": row["velocity_limit"],
                    "window_minutes": row["window_minutes"],
                    "risk_boost": row["risk_boost"],
                    "is_active": row["is_active"],
                }

            self._cache = {
                "risk_rules": risk_rules,
                "governorates": governorates,
                "d17_rules": d17_rules,
            }
            self._cache_timestamp = time.time()

        except Exception as e:
            logger.error(f"Rules engine: failed to refresh cache: {e}")
        finally:
            conn.close()

    def _ensure_cache(self):
        """Ensure cache is loaded and valid."""
        if not self._cache or not self._is_cache_valid():
            self._refresh_cache()

    # ==================== Public API ====================

    def get_risk_weights(self) -> dict:
        """Get current risk weights from database or fallback."""
        self._ensure_cache()
        risk_rules = self._cache.get("risk_rules", {})

        if not risk_rules:
            return DEFAULT_RISK_WEIGHTS

        weights = {}
        for rule_name, rule_data in risk_rules.items():
            if rule_name in DEFAULT_RISK_WEIGHTS:
                weights[rule_name] = rule_data["weight"]

        # Ensure all default keys exist
        for key in DEFAULT_RISK_WEIGHTS:
            if key not in weights:
                weights[key] = DEFAULT_RISK_WEIGHTS[key]

        return weights

    def get_threshold(self, rule_name: str) -> Optional[float]:
        """Get a specific threshold value."""
        self._ensure_cache()
        risk_rules = self._cache.get("risk_rules", {})
        if rule_name in risk_rules:
            return risk_rules[rule_name]["threshold"]
        return DEFAULT_RISK_WEIGHTS.get(rule_name)

    def get_governorate_profile(self, governorate: str) -> dict:
        """Get risk profile for a governorate."""
        self._ensure_cache()
        governorates = self._cache.get("governorates", {})
        if governorate in governorates:
            return governorates[governorate]

        # Default profile for unknown governorates
        return {"risk_multiplier": 1.0, "is_cbdc_pilot": False, "is_high_risk_zone": False}

    def get_cbdc_pilot_governorates(self) -> list:
        """Get list of CBDC pilot governorates."""
        self._ensure_cache()
        governorates = self._cache.get("governorates", {})
        return [name for name, profile in governorates.items() if profile.get("is_cbdc_pilot")]

    def get_d17_rule(self, rule_name: str) -> Optional[dict]:
        """Get a specific D17 rule."""
        self._ensure_cache()
        d17_rules = self._cache.get("d17_rules", {})
        return d17_rules.get(rule_name)

    def get_all_d17_rules(self) -> dict:
        """Get all active D17 rules."""
        self._ensure_cache()
        return self._cache.get("d17_rules", {})

    def get_high_value_threshold(self) -> float:
        """Get the large-transaction AML monitoring threshold (not a cash cap — repealed 2026)."""
        threshold = self.get_threshold("high_value")
        return threshold if threshold is not None else DEFAULT_HIGH_VALUE_THRESHOLD

    def get_smurfing_params(self) -> dict:
        """Get velocity-based smurfing detection parameters."""
        self._ensure_cache()
        risk_rules = self._cache.get("risk_rules", {})
        return {
            "unit_cap": risk_rules.get("smurfing_velocity_unit_cap", {}).get(
                "threshold", DEFAULT_SMURFING_UNIT_CAP
            ),
            "agg_min": risk_rules.get("smurfing_velocity_agg_min", {}).get(
                "threshold", DEFAULT_SMURFING_AGG_MIN
            ),
            "min_count": int(
                risk_rules.get("smurfing_velocity_min_count", {}).get(
                    "threshold", DEFAULT_SMURFING_MIN_COUNT
                )
            ),
        }

    def force_refresh(self):
        """Force a cache refresh (e.g., after rule update)."""
        self._cache = {}
        self._refresh_cache()

    def get_all_rules_summary(self) -> dict:
        """Get a summary of all current rules for the API/dashboard."""
        self._ensure_cache()
        return {
            "risk_weights": self.get_risk_weights(),
            "high_value_threshold": self.get_high_value_threshold(),
            "cbdc_governorates": self.get_cbdc_pilot_governorates(),
            "d17_rules": self.get_all_d17_rules(),
            "cache_age_seconds": round(time.time() - self._cache_timestamp, 1),
        }

    def update_rule(self, rule_name: str, weight: Optional[float] = None, threshold: Optional[float] = None, changed_by: str = "system") -> bool:
        """
        Update a risk rule in the database.
        This triggers an immediate cache refresh.

        Args:
            rule_name: Name of the rule to update
            weight: New weight value (optional)
            threshold: New threshold value (optional)
            changed_by: User/system that made the change

        Returns:
            True if update was successful.
        """
        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()

            # Get current values for audit log
            cursor.execute("SELECT weight, threshold FROM risk_rules WHERE rule_name = ?", (rule_name,))
            old_row = cursor.fetchone()
            if not old_row:
                return False

            old_weight = old_row["weight"]
            old_threshold = old_row["threshold"]

            # Build update query
            updates = []
            params = []
            if weight is not None:
                updates.append("weight = ?")
                params.append(weight)
            if threshold is not None:
                updates.append("threshold = ?")
                params.append(threshold)

            if not updates:
                return False

            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(rule_name)

            query = f"UPDATE risk_rules SET {', '.join(updates)} WHERE rule_name = ?"
            cursor.execute(query, params)

            # Log the change
            change_desc = []
            if weight is not None:
                change_desc.append(f"weight: {old_weight} -> {weight}")
            if threshold is not None:
                change_desc.append(f"threshold: {old_threshold} -> {threshold}")

            cursor.execute(
                """
                INSERT INTO rule_change_log (rule_table, rule_name, change_type, old_value, new_value, changed_by, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "risk_rules",
                    None,
                    rule_name,
                    "UPDATE",
                    f"weight={old_weight}, threshold={old_threshold}",
                    f"weight={weight}, threshold={threshold}",
                    changed_by,
                    "; ".join(change_desc),
                ),
            )

            conn.commit()

            # Force cache refresh
            self.force_refresh()

            logger.info(f"Rule '{rule_name}' updated by {changed_by}: {'; '.join(change_desc)}")
            return True

        except Exception as e:
            logger.error(f"Failed to update rule '{rule_name}': {e}")
            conn.rollback()
            return False
        finally:
            conn.close()


# Module-level singleton for drop-in replacement of risk_config.py
_rules_engine = None


def get_rules_engine() -> RulesEngine:
    """Get the rules engine singleton."""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = RulesEngine()
    return _rules_engine


# Backward-compatible module-level properties
# These allow the rules engine to act as a drop-in replacement for risk_config.py
def _get_property(prop_name):
    engine = get_rules_engine()
    if prop_name == "RISK_WEIGHTS":
        return engine.get_risk_weights()
    elif prop_name == "CBDC_PILOT_GOVERNORATES":
        return engine.get_cbdc_pilot_governorates()
    elif prop_name == "D17_SOFT_LIMIT":
        d17_rule = engine.get_d17_rule("d17_soft_limit_audit")
        if d17_rule and d17_rule.get("threshold_amount"):
            return d17_rule["threshold_amount"]
        return DEFAULT_D17_SOFT_LIMIT
    elif prop_name == "D17_VELOCITY_CAP":
        d17_rule = engine.get_d17_rule("flouci_velocity")
        if d17_rule and d17_rule.get("velocity_limit"):
            return d17_rule["velocity_limit"]
        return DEFAULT_D17_VELOCITY_CAP
    return None


# Lazy property access via module __getattr__ (Python 3.7+)
def __getattr__(name):
    if name in ("RISK_WEIGHTS", "CBDC_PILOT_GOVERNORATES", "D17_SOFT_LIMIT", "D17_VELOCITY_CAP"):
        return _get_property(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
