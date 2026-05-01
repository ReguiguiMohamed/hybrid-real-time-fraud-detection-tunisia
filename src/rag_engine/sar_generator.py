# src/rag_engine/sar_generator.py
import json
import logging
import os
import time
import requests
from datetime import datetime
from rag_engine.vector_store import CTAFVectorStore
from rag_engine.sar_validator import (
    validate_sar_output,
    generate_deterministic_fallback,
    format_sar_report,
    SARReport,
)

logger = logging.getLogger(__name__)


class SARGenerator:
    """
    Generates CTAF-compliant Suspicious Activity Reports (SARs) using
    RAG (Retrieval-Augmented Generation) with Ollama for local LLM inference.

    Features:
    - RAG-based regulatory context retrieval
    - LLM output validated against strict Pydantic schema
    - Deterministic fallback template when LLM fails
    - Structured JSON output requirement for validation
    """

    MAX_RETRIES = 3
    BASE_BACKOFF_SECONDS = 2

    def __init__(self, ollama_url=None):
        self.vector_store = CTAFVectorStore()
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_URL", "http://localhost:11434/api/generate"
        )
        self._stats = {"total": 0, "validated": 0, "fallback": 0}

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with exponential backoff retry logic AND circuit breaker."""
        from rag_engine.circuit_breaker import get_rag_circuit, CircuitBreakerError

        circuit = get_rag_circuit()

        # Check if circuit is tripped
        if not circuit.allow_request():
            circuit.record_fallback_used()
            logger.warning("RAG circuit OPEN: skipping LLM, using fallback")
            raise CircuitBreakerError("LLM circuit breaker is OPEN")

        start_time = time.time()

        try:
            response = requests.post(
                self.ollama_url,
                json={"model": "llama3.1", "prompt": prompt, "stream": False},
                timeout=120,
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                circuit.record_success(latency_ms)
                return response.json().get("response", "No response generated")

            # HTTP error
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            circuit.record_failure(last_error)
            logger.warning("Ollama HTTP error: %s", last_error)
            raise RuntimeError(last_error)

        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            circuit.record_success(latency_ms) if latency_ms < circuit.config.latency_threshold_ms else circuit.record_failure("Timeout")
            raise
        except requests.exceptions.ConnectionError as e:
            circuit.record_failure(f"Connection error: {e}")
            raise
        except CircuitBreakerError:
            raise
        except Exception as e:
            circuit.record_failure(str(e))
            raise

    def generate_report(self, tx_data: dict, ml_score: float) -> str:
        """
        Generate a CTAF-compliant SAR report.

        Flow:
        1. Retrieve regulatory context via RAG
        2. Query LLM for structured JSON SAR (circuit breaker protected)
        3. Validate output against Pydantic schema
        4. Fall back to deterministic template if validation fails OR circuit tripped
        """
        from rag_engine.circuit_breaker import CircuitBreakerError

        self._stats["total"] += 1

        # 1. Retrieve Regulatory Context
        query_text = (
            f"rules for {tx_data.get('payment_method', 'unknown')} "
            f"in {tx_data.get('governorate', 'unknown')}"
        )
        context_result = self.vector_store.query(query_text, n_results=3)

        context_parts = []
        if context_result and "documents" in context_result:
            docs = context_result["documents"]
            if docs and len(docs) > 0:
                context_parts = [d for d in docs[0] if d]
        context = "\n".join(context_parts) if context_parts else "No specific regulatory context found."

        # 2. LLM SAR Prompt (requests structured JSON output)
        from rag_engine.sar_validator import ctaf_filing_deadline
        from datetime import datetime as _dt
        _filing_deadline = ctaf_filing_deadline(from_date=_dt.utcnow(), business_days=10).strftime("%Y-%m-%d")

        prompt = f"""You are a compliance officer generating Suspicious Activity Reports (SARs) for CTAF filing.

CTAF FILING RULES (mandatory):
- Deadline: 10 business days (jours ouvrables) from detection = {_filing_deadline}
- Non-compliance penalty: up to TND 50,000 fine or license revocation
- "filing_deadline" in your response MUST be exactly: "{_filing_deadline}"

Transaction Details (use ONLY these values — do not invent or infer):
- Transaction ID: {tx_data.get('transaction_id', 'UNKNOWN')}
- User ID: {tx_data.get('user_id', 'UNKNOWN')}
- ML Fraud Probability Score: {ml_score:.4f}
- Amount: {tx_data.get('amount_tnd', 0):.2f} TND
- Governorate: {tx_data.get('governorate', 'UNKNOWN')}
- Payment Method: {tx_data.get('payment_method', 'UNKNOWN')}
- Branch: {tx_data.get('branch_id', 'UNKNOWN')}
- Timestamp: {tx_data.get('timestamp', 'UNKNOWN')}

Regulatory Context:
{context}

Respond ONLY with a valid JSON object (no markdown, no commentary) with this exact structure:
{{
  "executive_summary": "Brief summary (30-500 chars) of suspicious activity",
  "risk_factors": [
    {{"factor": "Risk factor description", "severity": "HIGH|MEDIUM|LOW", "evidence": "Specific evidence from the transaction details above"}}
  ],
  "regulatory_violations": [
    {{"regulation": "Regulation name", "description": "What was violated", "article": "Article number if applicable"}}
  ],
  "recommended_next_steps": ["Step 1", "Step 2", "Step 3"],
  "urgency_assessment": {{"urgency_level": "IMMEDIATE|HIGH|STANDARD|LOW", "filing_deadline": "{_filing_deadline}", "reason": "Why"}}
}}"""

        # 3. Call LLM (circuit breaker may raise CircuitBreakerError)
        raw_llm_output = None
        llm_failed = False
        try:
            raw_llm_output = self._call_ollama(prompt)
        except CircuitBreakerError:
            llm_failed = True
            raw_llm_output = f"Circuit breaker tripped: LLM unavailable (latency threshold exceeded)"
        except Exception as e:
            llm_failed = True
            raw_llm_output = f"LLM error: {e}"

        # 4. Validate and fallback
        report = validate_sar_output(raw_llm_output or "", tx_data, ml_score)

        if report.validation_passed and not llm_failed:
            self._stats["validated"] += 1
        else:
            self._stats["fallback"] += 1
            logger.warning(
                "SAR fallback triggered for tx %s (reason: %s)",
                tx_data.get("transaction_id", "unknown"),
                "circuit_breaker" if llm_failed and isinstance(report.validation_passed, bool) and not report.validation_passed else "LLM validation failed",
            )

        # 5. Format as human-readable text
        return format_sar_report(report)

    def generate_report_structured(self, tx_data: dict, ml_score: float) -> SARReport:
        """
        Generate a SAR and return the validated SARReport object (not text).
        Useful for API responses that need structured data.
        """
        self._stats["total"] += 1

        # Retrieve regulatory context
        query_text = f"rules for {tx_data.get('payment_method', 'unknown')} in {tx_data.get('governorate', 'unknown')}"
        context_result = self.vector_store.query(query_text, n_results=3)

        context_parts = []
        if context_result and "documents" in context_result:
            docs = context_result["documents"]
            if docs and len(docs) > 0:
                context_parts = [d for d in docs[0] if d]
        context = "\n".join(context_parts) if context_parts else "No specific regulatory context found."

        prompt = f"""Generate a SAR as JSON for transaction {tx_data.get('transaction_id')} with ML score {ml_score}.
Context: {context}
Respond with ONLY valid JSON matching the SARReport schema."""

        raw_llm_output = self._call_ollama(prompt)
        report = validate_sar_output(raw_llm_output, tx_data, ml_score)

        if report.validation_passed:
            self._stats["validated"] += 1
        else:
            self._stats["fallback"] += 1

        return report

    def save_report(self, tx_data: dict, report: str, ml_score: float) -> str:
        """Save the SAR report to file system."""
        reports_dir = os.path.join(".", "data", "reports")
        os.makedirs(reports_dir, exist_ok=True)

        tx_id = tx_data.get("transaction_id", "unknown")
        safe_id = tx_id.replace(":", "_").replace("-", "_")
        filename = f"SAR_{safe_id}.txt"
        filepath = os.path.join(reports_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("SAR report saved to %s", filepath)
        return filepath

    def get_stats(self) -> dict:
        """Return SAR generation statistics."""
        from rag_engine.circuit_breaker import get_rag_circuit

        total = max(self._stats["total"], 1)
        circuit = get_rag_circuit()
        return {
            "total_generated": self._stats["total"],
            "llm_validated": self._stats["validated"],
            "deterministic_fallback": self._stats["fallback"],
            "validation_rate": round(self._stats["validated"] / total, 4),
            "circuit_breaker": circuit.get_stats(),
        }
