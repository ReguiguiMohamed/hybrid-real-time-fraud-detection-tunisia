import json
import re

from rag_engine import sar_generator as sar_generator_module
from rag_engine.vector_store import CTAF_REGULATIONS


class DeterministicTestVectorStore:
    def query(self, query_text, n_results=3):
        regulation = next(item for item in CTAF_REGULATIONS if item["id"] == "internal_sar_filing_controls")
        return {
            "documents": [[regulation["text"]]],
            "ids": [[regulation["id"]]],
        }


def _tx():
    return {
        "transaction_id": "TXN_HARDEN_001",
        "user_id": "USER_HARDEN_001",
        "amount_tnd": 2500.0,
        "governorate": "Tunis",
        "payment_method": "Flouci",
        "branch_id": "Tunis-GNC",
        "timestamp": "2026-05-01T10:00:00Z",
        "smurfing_velocity_flag": True,
        "shap_top5": [
            {
                "feature": "v_count",
                "value": 5,
                "impact": 0.42,
                "abs_impact": 0.42,
                "confidence": 0.9,
                "description": "High velocity",
            }
        ],
    }


def _deadline_from_prompt(prompt):
    match = re.search(r'"filing_deadline": "(\d{4}-\d{2}-\d{2})"', prompt)
    assert match, prompt
    return match.group(1)


def _llm_payload(prompt, evidence):
    deadline = _deadline_from_prompt(prompt)
    return json.dumps(
        {
            "executive_summary": (
                "Transaction TXN_HARDEN_001 for USER_HARDEN_001 was flagged "
                "for compliance review based on model and rule evidence."
            ),
            "risk_factors": [
                {
                    "factor": "Structured alert evidence",
                    "severity": "HIGH",
                    "evidence": evidence,
                }
            ],
            "regulatory_violations": [
                {
                    "regulation": "CTAF SAR filing requirements",
                    "description": "Suspicious activity requires analyst review and timely SAR handling.",
                    "article": "Article 5",
                }
            ],
            "recommended_next_steps": [
                "Open analyst review.",
                "Preserve transaction evidence.",
                "Do not submit to CTAF until human approval is recorded.",
            ],
            "urgency_assessment": {
                "urgency_level": "HIGH",
                "filing_deadline": deadline,
                "reason": "Model score and structured rule evidence require timely review.",
            },
        }
    )


def test_sar_generator_retries_hallucinated_amount_and_audits(monkeypatch, tmp_path):
    monkeypatch.setattr(sar_generator_module, "CTAFVectorStore", lambda: DeterministicTestVectorStore())
    monkeypatch.setenv("SAR_LLM_AUDIT_LOG", str(tmp_path / "sar_audit.jsonl"))

    prompts = []
    responses = [
        lambda prompt: _llm_payload(prompt, "Invented amount 9999 TND for TXN_HARDEN_001."),
        lambda prompt: _llm_payload(prompt, "Source amount 2500.00 TND for TXN_HARDEN_001."),
    ]

    generator = sar_generator_module.SARGenerator()

    def deterministic_llm_call(prompt):
        prompts.append(prompt)
        return responses[len(prompts) - 1](prompt)

    monkeypatch.setattr(generator, "_call_ollama", deterministic_llm_call)

    report = generator.generate_report_structured(_tx(), 0.87)

    assert report.validation_passed is True
    assert report.transaction_id == "TXN_HARDEN_001"
    assert len(prompts) == 2
    assert "Previous draft was rejected by fact checks" in prompts[1]
    assert '"shap_top5"' in prompts[0]
    assert '"human_approval_required": true' in prompts[0]

    audit_lines = (tmp_path / "sar_audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    audit = json.loads(audit_lines[0])
    assert audit["transaction_id"] == "TXN_HARDEN_001"
    assert len(audit["attempts"]) == 2
    assert audit["attempts"][0]["fact_check"]["passed"] is False
    assert "9999 TND" in audit["attempts"][0]["fact_check"]["issues"][0]
    assert audit["final_validation_passed"] is True
    assert audit["final_fact_check"]["passed"] is True
    assert len(audit["entry_hash"]) == 64


def test_sar_generator_falls_back_after_repeated_fact_check_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sar_generator_module, "CTAFVectorStore", lambda: DeterministicTestVectorStore())
    monkeypatch.setenv("SAR_LLM_AUDIT_LOG", str(tmp_path / "sar_audit.jsonl"))

    generator = sar_generator_module.SARGenerator()

    def hallucinating_llm_call(prompt):
        return _llm_payload(prompt, "Invented account ACC_NOT_IN_SOURCE moved 9999 TND.")

    monkeypatch.setattr(generator, "_call_ollama", hallucinating_llm_call)

    report = generator.generate_report_structured(_tx(), 0.91)

    assert report.validation_passed is False
    assert report.transaction_id == "TXN_HARDEN_001"

    audit = json.loads((tmp_path / "sar_audit.jsonl").read_text(encoding="utf-8"))
    assert len(audit["attempts"]) == 3
    assert audit["final_validation_passed"] is False
    assert audit["final_sar_hash"]
