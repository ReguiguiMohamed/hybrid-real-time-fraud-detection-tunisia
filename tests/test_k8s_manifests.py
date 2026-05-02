"""Regression tests for Kubernetes deployment wiring."""
from pathlib import Path

import yaml

from scripts.validate_k8s_manifests import load_documents, validate


def test_k8s_manifests_pass_local_validation():
    validate()


def test_no_static_secrets_are_applied():
    documents = load_documents()
    assert not [
        document
        for _, document in documents
        if document.get("kind") == "Secret"
    ]


def test_compose_consumer_matches_sar_runtime_wiring():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    consumer = compose["services"]["consumer"]
    env = dict(item.split("=", 1) for item in consumer["environment"])
    volumes = set(consumer["volumes"])

    assert env["COMMAND_CENTER_API_VERSION"] == "api/v1"
    assert env["OLLAMA_URL"] == "http://ollama:11434/api/generate"
    assert env["SAR_LLM_AUDIT_LOG"] == "/app/data/audit/sar_llm_audit.jsonl"
    assert "SHAP_MIN_CONFIDENCE_FOR_SAR" in env
    assert "./data:/app/data" in volumes
    assert "./models:/app/models" in volumes
