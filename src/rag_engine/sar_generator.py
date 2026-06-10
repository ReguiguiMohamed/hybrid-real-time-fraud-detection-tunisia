# src/rag_engine/sar_generator.py
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from rag_engine.sar_validator import (
    SARReport,
    ctaf_filing_deadline,
    format_sar_report,
    generate_deterministic_fallback,
    validate_sar_output,
)
from rag_engine.vector_store import CTAFVectorStore

logger = logging.getLogger(__name__)


class SARGenerator:
    """
    Generates CTAF-compliant Suspicious Activity Reports (SARs).

    The LLM path is narrative enrichment only. Every call is grounded in a
    source-of-truth JSON object, schema-validated, fact-checked, and audited.
    If any step fails, the deterministic SAR template remains the filing path.
    """

    FACT_CHECK_RETRIES = 2

    def __init__(self, ollama_url=None):
        self.vector_store = CTAFVectorStore()
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.audit_log_path = Path(os.getenv("SAR_LLM_AUDIT_LOG", "./data/audit/sar_llm_audit.jsonl"))
        self._stats = {"total": 0, "validated": 0, "fallback": 0}

    @staticmethod
    def _canonical_json(value) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _sha256(cls, value) -> str:
        if not isinstance(value, str):
            value = cls._canonical_json(value)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with circuit-breaker protection."""
        from rag_engine.circuit_breaker import CircuitBreakerError, get_rag_circuit

        circuit = get_rag_circuit()
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

            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            circuit.record_failure(last_error)
            logger.warning("Ollama HTTP error: %s", last_error)
            raise RuntimeError(last_error)
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            if latency_ms < circuit.config.latency_threshold_ms:
                circuit.record_success(latency_ms)
            else:
                circuit.record_failure("Timeout")
            raise
        except requests.exceptions.ConnectionError as exc:
            circuit.record_failure(f"Connection error: {exc}")
            raise
        except CircuitBreakerError:
            raise
        except Exception as exc:
            circuit.record_failure(str(exc))
            raise

    def _retrieve_context(self, tx_data: dict) -> tuple[str, list[dict]]:
        query_text = (
            f"rules for {tx_data.get('payment_method', 'unknown')} " f"in {tx_data.get('governorate', 'unknown')}"
        )
        context_result = self.vector_store.query(query_text, n_results=3)
        documents = context_result.get("documents") or [] if context_result else []
        ids = context_result.get("ids") or [] if context_result else []
        first_doc_batch = documents[0] if documents else []
        first_id_batch = ids[0] if ids else []

        chunks = []
        for index, document in enumerate(first_doc_batch):
            if not document:
                continue
            chunk_id = (
                first_id_batch[index] if index < len(first_id_batch) and first_id_batch[index] else f"retrieved_{index}"
            )
            chunks.append({"id": chunk_id, "text": document, "hash": self._sha256(document)})

        if not chunks:
            return "No specific regulatory context found.", []
        return "\n".join(chunk["text"] for chunk in chunks), chunks

    def _build_source_of_truth(
        self,
        tx_data: dict,
        ml_score: float,
        filing_deadline: str,
        retrieved_chunks: list[dict],
    ) -> dict:
        transaction = {
            "transaction_id": tx_data.get("transaction_id", "UNKNOWN"),
            "user_id": tx_data.get("user_id", "UNKNOWN"),
            "amount_tnd": float(tx_data.get("amount_tnd") or 0.0),
            "governorate": tx_data.get("governorate", "UNKNOWN"),
            "payment_method": tx_data.get("payment_method", "UNKNOWN"),
            "branch_id": tx_data.get("branch_id", "UNKNOWN"),
            "timestamp": tx_data.get("timestamp", "UNKNOWN"),
        }

        for key in (
            "sender_account",
            "receiver_account",
            "source_account",
            "destination_account",
            "merchant_id",
            "device_id",
            "ttn_invoice_id",
            "ttn_clearance_token",
            "tunicheque_token",
        ):
            if tx_data.get(key) not in (None, ""):
                transaction[key] = tx_data[key]

        rule_flags = {
            key: value
            for key, value in tx_data.items()
            if key.endswith("_flag") or key.endswith("_risk") or key.startswith("rule_")
        }

        return {
            "transaction": transaction,
            "model": {
                "fraud_probability": round(float(ml_score), 6),
                "shap_top5": tx_data.get("shap_top5") or [],
            },
            "rule_flags": rule_flags,
            "compliance": {
                "filing_deadline": filing_deadline,
                "deadline_business_days": 10,
                "non_compliance_penalty_tnd": 50000,
                "enhanced_monitoring_threshold_tnd": 15000,
                "analyst_review_hours": 24,
                "record_retention_years": 5,
                "human_approval_required": True,
            },
            "retrieval": [{"id": chunk["id"], "hash": chunk["hash"]} for chunk in retrieved_chunks],
        }

    def _build_prompt(
        self,
        source_truth: dict,
        context: str,
        fact_check_issues: list[str] | None = None,
    ) -> str:
        correction = ""
        if fact_check_issues:
            correction = (
                "\nPrevious draft was rejected by fact checks. Correct these issues "
                "using only the Source of Truth JSON:\n- " + "\n- ".join(fact_check_issues) + "\n"
            )

        source_json = json.dumps(source_truth, indent=2, sort_keys=True, default=str)
        filing_deadline = source_truth["compliance"]["filing_deadline"]
        return f"""You are a compliance officer generating Suspicious Activity Reports (SARs) for CTAF filing.

Use ONLY the values in the Source of Truth JSON. Do not invent or infer account numbers, transaction IDs, dates, amounts, scores, rule flags, or reasons not present there.
Every identifier, date, score, and amount in your output must exactly match the Source of Truth JSON when it refers to this case.
The LLM narrative is advisory only. A human compliance officer must approve the SAR before CTAF submission.
{correction}
Source of Truth JSON:
{source_json}

Regulatory Context:
{context}

Mandatory CTAF filing facts:
- Deadline: 10 business days (jours ouvrables) from detection = {filing_deadline}
- Non-compliance penalty: up to TND 50,000 fine or license revocation
- "filing_deadline" in your response MUST be exactly: "{filing_deadline}"

Respond ONLY with a valid JSON object (no markdown, no commentary) with this exact structure:
{{
  "executive_summary": "Brief summary (30-500 chars) of suspicious activity",
  "risk_factors": [
    {{"factor": "Risk factor description", "severity": "HIGH|MEDIUM|LOW", "evidence": "Specific evidence from the Source of Truth JSON"}}
  ],
  "regulatory_violations": [
    {{"regulation": "Regulation name", "description": "What was violated", "article": "Article number if applicable"}}
  ],
  "recommended_next_steps": ["Step 1", "Step 2", "Step 3"],
  "urgency_assessment": {{"urgency_level": "IMMEDIATE|HIGH|STANDARD|LOW", "filing_deadline": "{filing_deadline}", "reason": "Why"}}
}}"""

    def _fact_check_report(self, report: SARReport, source_truth: dict) -> dict:
        transaction = source_truth["transaction"]
        model = source_truth["model"]
        compliance = source_truth["compliance"]
        issues = []

        expected_strings = {
            "transaction_id": str(transaction["transaction_id"]),
            "user_id": str(transaction["user_id"]),
            "governorate": str(transaction["governorate"]),
            "payment_method": str(transaction["payment_method"]),
        }
        for field, expected in expected_strings.items():
            actual = str(getattr(report, field))
            if actual != expected:
                issues.append(f"{field} mismatch: expected {expected}, got {actual}")

        expected_amount = float(transaction["amount_tnd"])
        if abs(float(report.amount_tnd) - expected_amount) > 0.01:
            issues.append(f"amount_tnd mismatch: expected {expected_amount:.2f}, got {float(report.amount_tnd):.2f}")

        expected_score = float(model["fraud_probability"])
        if abs(float(report.ml_score) - expected_score) > 0.0001:
            issues.append(f"ml_score mismatch: expected {expected_score:.4f}, got {float(report.ml_score):.4f}")

        expected_deadline = str(compliance["filing_deadline"])
        actual_deadline = str(report.urgency_assessment.filing_deadline)
        if actual_deadline != expected_deadline:
            issues.append(f"filing_deadline mismatch: expected {expected_deadline}, got {actual_deadline}")

        text = format_sar_report(report)
        allowed_identifiers = {
            str(value)
            for value in transaction.values()
            if isinstance(value, (str, int, float)) and str(value) not in {"", "UNKNOWN"}
        }
        id_pattern = re.compile(r"\b(?:TXN|USER|ACC|ACCT|MERCHANT|DEVICE|INV|TTN|TNC)[A-Z0-9_:-]{2,}\b")
        for token in sorted(set(id_pattern.findall(text))):
            if token not in allowed_identifiers:
                issues.append(f"identifier not in source of truth: {token}")

        allowed_tnd_values = {
            round(float(transaction["amount_tnd"]), 2),
            float(compliance["non_compliance_penalty_tnd"]),
            float(compliance["enhanced_monitoring_threshold_tnd"]),
        }
        tnd_pattern = re.compile(r"(?<![\d])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*TND\b")
        for raw_value in tnd_pattern.findall(text):
            numeric = float(raw_value.replace(",", ""))
            if all(abs(numeric - allowed) > 0.01 for allowed in allowed_tnd_values):
                issues.append(f"TND amount not in source of truth: {raw_value} TND")

        return {"passed": not issues, "issues": issues}

    def _append_audit_event(self, event: dict) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        previous_hash = "0" * 64
        if self.audit_log_path.exists():
            try:
                with self.audit_log_path.open("r", encoding="utf-8") as audit_file:
                    for line in audit_file:
                        if line.strip():
                            previous_hash = json.loads(line).get("entry_hash", previous_hash)
            except Exception as exc:
                logger.warning("Could not read previous SAR audit hash: %s", exc)

        event = {**event, "previous_hash": previous_hash}
        event["entry_hash"] = self._sha256(event)
        with self.audit_log_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def _generate_validated_report(self, tx_data: dict, ml_score: float) -> SARReport:
        filing_deadline = ctaf_filing_deadline(
            from_date=datetime.now(timezone.utc).replace(tzinfo=None), business_days=10
        ).strftime("%Y-%m-%d")
        context, retrieved_chunks = self._retrieve_context(tx_data)
        source_truth = self._build_source_of_truth(tx_data, ml_score, filing_deadline, retrieved_chunks)

        audit_attempts = []
        fact_check_issues = []
        fallback_reason = "LLM validation or fact check failed"
        final_report = None

        for attempt in range(1, self.FACT_CHECK_RETRIES + 2):
            prompt = self._build_prompt(source_truth, context, fact_check_issues)
            try:
                raw_llm_output = self._call_ollama(prompt)
                llm_failed = False
            except Exception as exc:
                raw_llm_output = f"LLM error: {exc}"
                llm_failed = True

            candidate = validate_sar_output(raw_llm_output or "", tx_data, ml_score)
            if candidate.validation_passed and not llm_failed:
                fact_check = self._fact_check_report(candidate, source_truth)
            else:
                fact_check = {
                    "passed": False,
                    "issues": ["LLM validation failed or LLM unavailable"],
                }

            audit_attempts.append(
                {
                    "attempt": attempt,
                    "prompt_hash": self._sha256(prompt),
                    "raw_output_hash": self._sha256(raw_llm_output or ""),
                    "raw_output": raw_llm_output,
                    "validation_passed": bool(candidate.validation_passed),
                    "fact_check": fact_check,
                }
            )

            if candidate.validation_passed and fact_check["passed"] and not llm_failed:
                final_report = candidate
                break

            fact_check_issues = fact_check["issues"]
            fallback_reason = "; ".join(fact_check_issues)

        if final_report is None:
            final_report = generate_deterministic_fallback(
                tx_data,
                ml_score,
                raw_llm_output=json.dumps(audit_attempts[-1], sort_keys=True),
                reason=fallback_reason,
            )

        final_fact_check = self._fact_check_report(final_report, source_truth)
        self._append_audit_event(
            {
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "transaction_id": source_truth["transaction"]["transaction_id"],
                "model_version": "llama3.1",
                "source_truth_hash": self._sha256(source_truth),
                "retrieved_chunk_hashes": [chunk["hash"] for chunk in retrieved_chunks],
                "attempts": audit_attempts,
                "final_validation_passed": bool(final_report.validation_passed),
                "final_fact_check": final_fact_check,
                "final_sar_hash": self._sha256(format_sar_report(final_report)),
            }
        )
        return final_report

    def generate_report(self, tx_data: dict, ml_score: float) -> str:
        """
        Generate a human-readable SAR report after LLM validation and fact checks.
        """
        self._stats["total"] += 1
        report = self._generate_validated_report(tx_data, ml_score)
        if report.validation_passed:
            self._stats["validated"] += 1
        else:
            self._stats["fallback"] += 1
            logger.warning(
                "SAR fallback triggered for tx %s",
                tx_data.get("transaction_id", "unknown"),
            )
        return format_sar_report(report)

    def generate_report_structured(self, tx_data: dict, ml_score: float) -> SARReport:
        """
        Generate a SAR and return the validated SARReport object.
        """
        self._stats["total"] += 1
        report = self._generate_validated_report(tx_data, ml_score)
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

        with open(filepath, "w", encoding="utf-8") as report_file:
            report_file.write(report)

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
