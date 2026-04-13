"""
SAR Output Validator for Amastan Fraud Shield Guard
Validates LLM-generated SAR reports against a strict Pydantic schema.
Falls back to a deterministic template if the LLM output is malformed or hallucinated.

This ensures CTAF compliance even when the LLM produces gibberish, empty, or malicious output.
"""
import json
import logging
import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SARUrgencyAssessment(BaseModel):
    """Urgency level and filing deadline for the SAR."""
    urgency_level: str = Field(..., description="IMMEDIATE, HIGH, STANDARD, or LOW")
    filing_deadline: str = Field(..., description="ISO-formatted deadline date")
    reason: str = Field(..., min_length=10)

    @field_validator("urgency_level")
    @classmethod
    def validate_urgency(cls, v):
        allowed = {"IMMEDIATE", "HIGH", "STANDARD", "LOW"}
        v = v.upper().strip()
        if v not in allowed:
            raise ValueError(f"Urgency must be one of {allowed}, got: {v}")
        return v


class SARRiskFactor(BaseModel):
    """A single risk factor observed in the transaction."""
    factor: str = Field(..., min_length=5, max_length=200)
    severity: str = Field(..., description="HIGH, MEDIUM, or LOW")
    evidence: str = Field(..., min_length=10)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        allowed = {"HIGH", "MEDIUM", "LOW"}
        v = v.upper().strip()
        if v not in allowed:
            raise ValueError(f"Severity must be one of {allowed}, got: {v}")
        return v


class SARRegulatoryViolation(BaseModel):
    """A regulatory violation cited in the SAR."""
    regulation: str = Field(..., min_length=5, description="e.g., 'CTAF Circular 2024-03'")
    description: str = Field(..., min_length=10)
    article: Optional[str] = Field(None, description="Specific article or section reference")


class SARReport(BaseModel):
    """
    Validated SAR report structure.
    All fields are required for CTAF compliance.
    """
    transaction_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    generated_at: str = Field(..., description="ISO-formatted generation timestamp")
    executive_summary: str = Field(..., min_length=30, max_length=2000, description="Concise summary of suspicious activity")
    risk_factors: list[SARRiskFactor] = Field(..., min_items=1, description="At least 1 risk factor must be cited")
    regulatory_violations: list[SARRegulatoryViolation] = Field(..., min_items=1)
    recommended_next_steps: list[str] = Field(..., min_items=1, min_length=10)
    urgency_assessment: SARUrgencyAssessment = Field(...)
    ml_score: float = Field(..., ge=0.0, le=1.0, description="ML model probability score")
    amount_tnd: float = Field(..., ge=0.0, description="Transaction amount in TND")
    governorate: str = Field(..., min_length=1)
    payment_method: str = Field(..., min_length=1)
    raw_llm_output: Optional[str] = Field(None, description="Original LLM output for audit purposes")
    validation_passed: bool = Field(default=True, description="Whether the output passed schema validation")


def extract_json_from_llm(raw_output: str) -> Optional[dict]:
    """
    Attempt to extract valid JSON from LLM output.
    Handles cases where the LLM wraps JSON in markdown code blocks or adds commentary.
    """
    # Try direct parse first
    try:
        return json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract from markdown code block
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(code_block_pattern, raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to find JSON object in text (heuristic)
    json_start = raw_output.find("{")
    json_end = raw_output.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            return json.loads(raw_output[json_start : json_end + 1])
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def validate_sar_output(raw_llm_output: str, tx_data: dict, ml_score: float) -> SARReport:
    """
    Validate an LLM-generated SAR report against the schema.
    Falls back to a deterministic template if validation fails.

    Args:
        raw_llm_output: Raw string output from the LLM
        tx_data: Transaction data dictionary
        ml_score: ML model probability score

    Returns:
        A validated SARReport (either from LLM or deterministic fallback).
    """
    validation_passed = False
    parsed_data = None

    # Step 1: Try to extract JSON from LLM output
    if raw_llm_output:
        parsed_data = extract_json_from_llm(raw_llm_output)

    if parsed_data is None:
        logger.warning("SAR validation: LLM output contained no parseable JSON, using fallback")
        return generate_deterministic_fallback(tx_data, ml_score, raw_llm_output, reason="No JSON found in LLM output")

    # Step 2: Try to parse into SARReport schema
    try:
        # Add required metadata if not in LLM output
        parsed_data["transaction_id"] = parsed_data.get("transaction_id", tx_data.get("transaction_id", "unknown"))
        parsed_data["user_id"] = parsed_data.get("user_id", tx_data.get("user_id", "unknown"))
        parsed_data["generated_at"] = parsed_data.get("generated_at", datetime.utcnow().isoformat())
        parsed_data["ml_score"] = parsed_data.get("ml_score", ml_score)
        parsed_data["amount_tnd"] = parsed_data.get("amount_tnd", tx_data.get("amount_tnd", 0.0))
        parsed_data["governorate"] = parsed_data.get("governorate", tx_data.get("governorate", "unknown"))
        parsed_data["payment_method"] = parsed_data.get("payment_method", tx_data.get("payment_method", "unknown"))
        parsed_data["raw_llm_output"] = raw_llm_output
        parsed_data["validation_passed"] = True

        report = SARReport(**parsed_data)
        validation_passed = True
        logger.info("SAR validation: LLM output passed schema validation")
        return report

    except Exception as e:
        logger.warning(f"SAR validation: LLM output failed schema validation: {e}")
        return generate_deterministic_fallback(tx_data, ml_score, raw_llm_output, reason=f"Schema validation failed: {e}")


def generate_deterministic_fallback(tx_data: dict, ml_score: float, raw_llm_output: str = None, reason: str = "LLM failure") -> SARReport:
    """
    Generate a deterministic, CTAF-compliant SAR report template.
    This is the safety net when the LLM fails to produce valid output.
    """
    tx_id = tx_data.get("transaction_id", "unknown")
    user_id = tx_data.get("user_id", "unknown")
    amount = tx_data.get("amount_tnd", 0.0)
    governorate = tx_data.get("governorate", "unknown")
    payment_method = tx_data.get("payment_method", "unknown")
    timestamp = tx_data.get("timestamp", "unknown")
    branch_id = tx_data.get("branch_id", "unknown")

    # Determine urgency based on ML score
    if ml_score >= 0.9:
        urgency_level = "IMMEDIATE"
        urgency_reason = f"ML score {ml_score:.2f} indicates extremely high probability of fraud. Immediate investigation required."
    elif ml_score >= 0.75:
        urgency_level = "HIGH"
        urgency_reason = f"ML score {ml_score:.2f} indicates high probability of fraud. Priority investigation recommended."
    else:
        urgency_level = "STANDARD"
        urgency_reason = f"ML score {ml_score:.2f} indicates moderate probability. Standard 10-business-day filing window applies."

    # Determine risk factors based on transaction data
    risk_factors = []

    if amount > 5000:
        risk_factors.append(SARRiskFactor(
            factor="High-value transaction exceeding normal threshold",
            severity="HIGH",
            evidence=f"Transaction amount: {amount:.2f} TND exceeds 5000 TND threshold",
        ))

    if amount >= 1400 and amount <= 1500 and payment_method.lower() == "flouci":
        risk_factors.append(SARRiskFactor(
            factor="Potential smurfing/structuring pattern detected",
            severity="HIGH",
            evidence=f"Flouci payment of {amount:.2f} TND falls within smurfing range (1400-1500 TND)",
        ))

    if payment_method.lower() == "flouci" and amount > 2000:
        risk_factors.append(SARRiskFactor(
            factor="D17 e-wallet threshold exceeded",
            severity="MEDIUM",
            evidence=f"Flouci payment of {amount:.2f} TND exceeds D17 soft limit of 2000 TND",
        ))

    # Always include at least one risk factor
    if not risk_factors:
        risk_factors.append(SARRiskFactor(
            factor="Transaction flagged by ML fraud detection model",
            severity="MEDIUM",
            evidence=f"ML model probability: {ml_score:.2f}. Transaction requires analyst review.",
        ))

    # Default regulatory violations
    regulatory_violations = [
        SARRegulatoryViolation(
            regulation="CTAF Circular 2024-03",
            description="Suspicious transaction monitoring and reporting requirements",
            article="Article 5",
        ),
        SARRegulatoryViolation(
            regulation="BCT Anti-Money Laundering Guidelines 2025",
            description="Threshold-based monitoring for digital payment platforms",
        ),
    ]

    # Default recommended next steps
    recommended_next_steps = [
        "Flag transaction for analyst review within 24 hours",
        "Cross-reference with user's transaction history for pattern analysis",
        "If confirmed: file SAR with CTAF within 10 business days",
        "Consider account-level review for additional suspicious activity",
    ]

    return SARReport(
        transaction_id=tx_id,
        user_id=user_id,
        generated_at=datetime.utcnow().isoformat(),
        executive_summary=(
            f"Automated suspicious activity detection for user {user_id}. "
            f"Transaction of {amount:.2f} TND via {payment_method} in {governorate} "
            f"triggered fraud alert with ML score {ml_score:.2f}. "
            f"This report was auto-generated by deterministic template (LLM fallback: {reason})."
        ),
        risk_factors=risk_factors,
        regulatory_violations=regulatory_violations,
        recommended_next_steps=recommended_next_steps,
        urgency_assessment=SARUrgencyAssessment(
            urgency_level=urgency_level,
            filing_deadline=datetime.utcnow().isoformat(),
            reason=urgency_reason,
        ),
        ml_score=ml_score,
        amount_tnd=amount,
        governorate=governorate,
        payment_method=payment_method,
        raw_llm_output=raw_llm_output,
        validation_passed=False,
    )


def format_sar_report(report: SARReport) -> str:
    """
    Format a validated SARReport into a human-readable text report
    for CTAF filing and storage.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SUSPICIOUS ACTIVITY REPORT (SAR)")
    lines.append("CTAF/BCT Compliance Filing")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Generated:        {report.generated_at}")
    lines.append(f"Transaction ID:   {report.transaction_id}")
    lines.append(f"User ID:          {report.user_id}")
    lines.append(f"ML Score:         {report.ml_score:.4f}")
    lines.append(f"Amount:           {report.amount_tnd:.2f} TND")
    lines.append(f"Governorate:      {report.governorate}")
    lines.append(f"Payment Method:   {report.payment_method}")
    lines.append(f"Validation:       {'PASSED' if report.validation_passed else 'FALLBACK (LLM failed)'}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    lines.append(report.executive_summary)
    lines.append("")
    lines.append("-" * 70)
    lines.append("RISK FACTORS")
    lines.append("-" * 70)
    for i, factor in enumerate(report.risk_factors, 1):
        lines.append(f"  {i}. [{factor.severity}] {factor.factor}")
        lines.append(f"     Evidence: {factor.evidence}")
        lines.append("")
    lines.append("-" * 70)
    lines.append("REGULATORY VIOLATIONS")
    lines.append("-" * 70)
    for i, violation in enumerate(report.regulatory_violations, 1):
        article = f" (Article {violation.article})" if violation.article else ""
        lines.append(f"  {i}. {violation.regulation}{article}")
        lines.append(f"     {violation.description}")
        lines.append("")
    lines.append("-" * 70)
    lines.append("URGENCY ASSESSMENT")
    lines.append("-" * 70)
    lines.append(f"  Level:    {report.urgency_assessment.urgency_level}")
    lines.append(f"  Deadline: {report.urgency_assessment.filing_deadline}")
    lines.append(f"  Reason:   {report.urgency_assessment.reason}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RECOMMENDED NEXT STEPS")
    lines.append("-" * 70)
    for i, step in enumerate(report.recommended_next_steps, 1):
        lines.append(f"  {i}. {step}")
    lines.append("")
    if report.raw_llm_output and not report.validation_passed:
        lines.append("-" * 70)
        lines.append("NOTE: This report was generated using deterministic template fallback.")
        lines.append("LLM output was unavailable or failed validation.")
        lines.append(f"Reason: {report.raw_llm_output[:200] if report.raw_llm_output else 'N/A'}")
        lines.append("-" * 70)
        lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)
