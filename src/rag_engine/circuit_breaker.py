"""
Circuit Breaker for RAG/LLM Layer in Amastan Fraud Shield Guard

Prevents the LLM from becoming a bottleneck when it degrades in performance.
If Ollama latency exceeds the threshold for N consecutive requests, the circuit
"trips" and all SAR generation falls back to the deterministic template.

State Machine:
    CLOSED ──► OPEN ──► HALF_OPEN
       ▲                      │
       │                      ▼ (if probe succeeds)
       └────────────────── CLOSED
                          │
                          ▼ (if probe fails)
                        OPEN

Usage:
    from src.rag_engine.circuit_breaker import rag_circuit

    @rag_circuit
    def generate_sar(tx_data, ml_score):
        # This will be skipped if the circuit is OPEN
        return ollama_generate(tx_data, ml_score)

    # The circuit automatically falls back to deterministic SAR
    # when tripped, and probes for recovery after cooldown.
"""
import logging
import time
import threading
from enum import Enum
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"         # Normal operation, LLM is healthy
    OPEN = "open"             # Tripped, using fallback
    HALF_OPEN = "half_open"   # Testing if LLM recovered


class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""
    def __init__(
        self,
        failure_threshold: int = 5,         # Consecutive failures to trip
        latency_threshold_ms: float = 2000,  # Max acceptable latency (2s default)
        recovery_timeout_s: float = 60,      # Time before probing recovery
        name: str = "rag_circuit",
    ):
        self.failure_threshold = failure_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.recovery_timeout_s = recovery_timeout_s
        self.name = name


class CircuitBreakerError(Exception):
    """Raised when the circuit is OPEN and fallback is not available."""
    pass


class CircuitBreaker:
    """
    Production-grade circuit breaker for the RAG/LLM layer.

    Tracks consecutive latency violations and failures.
    When the threshold is crossed, trips to OPEN state and stops calling the LLM.
    After recovery_timeout_s, transitions to HALF_OPEN and sends a probe request.
    If the probe succeeds, returns to CLOSED. If it fails, returns to OPEN.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._last_success_time = 0.0
        self._total_calls = 0
        self._total_fallbacks = 0
        self._total_timeouts = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, with automatic HALF_OPEN transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.config.recovery_timeout_s:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"[{self.config.name}] Circuit transitioning OPEN → HALF_OPEN")
            return self._state

    def record_success(self, latency_ms: float):
        """Record a successful call."""
        with self._lock:
            self._total_calls += 1
            self._last_success_time = time.time()

            if latency_ms > self.config.latency_threshold_ms:
                # Latency violation counts as a "failure"
                self._consecutive_failures += 1
                self._total_timeouts += 1
                logger.warning(
                    f"[{self.config.name}] Latency violation: {latency_ms:.0f}ms > {self.config.latency_threshold_ms:.0f}ms "
                    f"(consecutive: {self._consecutive_failures}/{self.config.failure_threshold})"
                )

                if self._consecutive_failures >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_failure_time = time.time()
                    logger.error(f"[{self.config.name}] Circuit TRIPPED: LLM latency exceeded {self.config.failure_threshold} consecutive times")
            else:
                # True success resets the counter
                if self._state == CircuitState.HALF_OPEN:
                    logger.info(f"[{self.config.name}] Circuit transitioning HALF_OPEN → CLOSED (probe succeeded)")
                    self._state = CircuitState.CLOSED
                self._consecutive_failures = 0

    def record_failure(self, error: str = ""):
        """Record a failed call (connection error, timeout, etc.)."""
        with self._lock:
            self._total_calls += 1
            self._consecutive_failures += 1
            self._last_failure_time = time.time()
            logger.error(f"[{self.config.name}] Call failed: {error} (consecutive: {self._consecutive_failures}/{self.config.failure_threshold})")

            if self._consecutive_failures >= self.config.failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._state = CircuitState.OPEN
                    logger.error(f"[{self.config.name}] Circuit TRIPPED: {self.config.failure_threshold} consecutive failures")

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through to the LLM.

        Returns:
            True if the request should be sent to the LLM.
            False if the circuit is OPEN and fallback should be used.
        """
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            # Allow one probe request through
            return True
        else:  # OPEN
            self._total_fallbacks += 1
            return False

    def record_fallback_used(self):
        """Record that a fallback was used."""
        with self._lock:
            self._total_fallbacks += 1

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_calls": self._total_calls,
            "total_fallbacks": self._total_fallbacks,
            "total_timeouts": self._total_timeouts,
            "fallback_rate": self._total_fallbacks / max(self._total_calls, 1),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "latency_threshold_ms": self.config.latency_threshold_ms,
                "recovery_timeout_s": self.config.recovery_timeout_s,
            },
        }

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            logger.info(f"[{self.config.name}] Circuit manually reset to CLOSED")


# ==================== Decorator for Wrapping LLM Calls ====================

class RagCircuitDecorator:
    """
    Decorator that wraps LLM generation calls with circuit breaker logic.
    If the circuit is OPEN, raises CircuitBreakerError so the caller can
    use the deterministic fallback.
    """

    def __init__(self, breaker: CircuitBreaker):
        self.breaker = breaker

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.breaker.allow_request():
                self.breaker.record_fallback_used()
                logger.warning(f"[{self.breaker.config.name}] Circuit OPEN: skipping LLM call, using fallback")
                raise CircuitBreakerError("LLM circuit breaker is OPEN")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                self.breaker.record_success(latency_ms)
                return result
            except CircuitBreakerError:
                raise
            except Exception as e:
                self.breaker.record_failure(str(e))
                raise

        return wrapper


# ==================== Module-Level Singleton ====================

_rag_circuit = None


def get_rag_circuit() -> CircuitBreaker:
    """Get the RAG circuit breaker singleton."""
    global _rag_circuit
    if _rag_circuit is None:
        _rag_circuit = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=5,
            latency_threshold_ms=2000,  # 2 seconds
            recovery_timeout_s=60,      # 1 minute cooldown
            name="rag_llm",
        ))
    return _rag_circuit


def rag_circuit(func: Callable) -> Callable:
    """
    Decorator for LLM generation functions.
    Automatically enforces the circuit breaker pattern.

    Usage:
        @rag_circuit
        def call_ollama(prompt):
            return requests.post(url, json={"prompt": prompt})

    If the circuit is OPEN, raises CircuitBreakerError.
    """
    decorator = RagCircuitDecorator(get_rag_circuit())
    return decorator(func)
