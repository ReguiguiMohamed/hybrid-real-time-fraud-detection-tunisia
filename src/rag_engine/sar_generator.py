# src/rag_engine/sar_generator.py
import logging
import os
import time
import requests
from datetime import datetime
from rag_engine.vector_store import CTAFVectorStore

logger = logging.getLogger(__name__)


class SARGenerator:
    """
    Generates CTAF-compliant Suspicious Activity Reports (SARs) using
    RAG (Retrieval-Augmented Generation) with Ollama for local LLM inference.
    """

    MAX_RETRIES = 3
    BASE_BACKOFF_SECONDS = 2

    def __init__(self, ollama_url=None):
        self.vector_store = CTAFVectorStore()
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_URL", "http://localhost:11434/api/generate"
        )

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with exponential backoff retry logic."""
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={"model": "llama3.1", "prompt": prompt, "stream": False},
                    timeout=120,
                )
                if response.status_code == 200:
                    return response.json().get("response", "No response generated")

                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s",
                    attempt, self.MAX_RETRIES, last_error,
                )
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(
                    "Ollama request timed out (attempt %d/%d)", attempt, self.MAX_RETRIES
                )
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning(
                    "Ollama connection error (attempt %d/%d): %s",
                    attempt, self.MAX_RETRIES, e,
                )
            except Exception as e:
                last_error = str(e)
                logger.error(
                    "Unexpected error calling Ollama (attempt %d/%d): %s",
                    attempt, self.MAX_RETRIES, e,
                )

            if attempt < self.MAX_RETRIES:
                backoff = self.BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.info("Retrying in %ds...", backoff)
                time.sleep(backoff)

        return f"SAR generation failed after {self.MAX_RETRIES} attempts. Last error: {last_error}"

    def generate_report(self, tx_data: dict, ml_score: float) -> str:
        """Generate a SAR report using RAG context + Ollama LLM."""
        # 1. Retrieve Regulatory Context
        query_text = (
            f"rules for {tx_data.get('payment_method', 'unknown')} "
            f"in {tx_data.get('governorate', 'unknown')}"
        )
        context_result = self.vector_store.query(query_text, n_results=3)

        # Extract context from results
        context_parts = []
        if context_result and "documents" in context_result:
            docs = context_result["documents"]
            if docs and len(docs) > 0:
                context_parts = [d for d in docs[0] if d]
        context = "\n".join(context_parts) if context_parts else "No specific regulatory context found."

        # 2. Automated SAR Prompting
        prompt = f"""INVESTIGATION: Analyze suspicious activity for user {tx_data.get('user_id', 'UNKNOWN')}.
Transaction ID: {tx_data.get('transaction_id', 'UNKNOWN')}
ML Score: {ml_score}
Amount: {tx_data.get('amount_tnd', 0)} TND
Governorate: {tx_data.get('governorate', 'UNKNOWN')}
Payment Method: {tx_data.get('payment_method', 'UNKNOWN')}
Branch: {tx_data.get('branch_id', 'UNKNOWN')}
Timestamp: {tx_data.get('timestamp', 'UNKNOWN')}

Regulatory Context:
{context}

Generate a professional Suspicious Activity Report (SAR) for CTAF filing. Include:
1. Executive summary of suspicious activity
2. Risk factors observed (transaction amount, velocity, geographic anomaly)
3. Applicable CTAF/BCT regulatory violations
4. Recommended next steps for investigation
5. Filing urgency assessment (immediate / standard 10-day window)"""

        # 3. Local Inference (Data Privacy Compliant)
        return self._call_ollama(prompt)

    def save_report(self, tx_data: dict, report: str, ml_score: float) -> str:
        """Save the SAR report to file system."""
        reports_dir = os.path.join(".", "data", "reports")
        os.makedirs(reports_dir, exist_ok=True)

        tx_id = tx_data.get("transaction_id", "unknown")
        safe_id = tx_id.replace(":", "_").replace("-", "_")
        filename = f"SAR_{safe_id}.txt"
        filepath = os.path.join(reports_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("SUSPICIOUS ACTIVITY REPORT (SAR)\n")
            f.write("CTAF/BCT Compliance Filing\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated:      {datetime.now().isoformat()}\n")
            f.write(f"Transaction ID: {tx_data.get('transaction_id', 'UNKNOWN')}\n")
            f.write(f"User ID:        {tx_data.get('user_id', 'UNKNOWN')}\n")
            f.write(f"ML Score:       {ml_score}\n")
            f.write(f"Amount:         {tx_data.get('amount_tnd', 0)} TND\n")
            f.write(f"Governorate:    {tx_data.get('governorate', 'UNKNOWN')}\n")
            f.write(f"Payment Method: {tx_data.get('payment_method', 'UNKNOWN')}\n")
            f.write(f"Branch:         {tx_data.get('branch_id', 'UNKNOWN')}\n")
            f.write(f"Timestamp:      {tx_data.get('timestamp', 'UNKNOWN')}\n")
            f.write("\n" + "-" * 70 + "\n")
            f.write("REPORT:\n\n")
            f.write(report + "\n")
            f.write("\n" + "=" * 70 + "\n")

        logger.info("SAR report saved to %s", filepath)
        return filepath
