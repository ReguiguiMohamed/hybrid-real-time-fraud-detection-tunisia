"""Local Kubernetes manifest checks that do not require a cluster."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

K8S_DIR = Path(__file__).resolve().parents[1] / "k8s"
APPLIED_MANIFESTS = sorted(K8S_DIR.glob("*.yml"))


def load_documents():
    documents = []
    for path in APPLIED_MANIFESTS:
        with path.open("r", encoding="utf-8") as manifest_file:
            for document in yaml.safe_load_all(manifest_file):
                if document:
                    documents.append((path.name, document))
    return documents


def find_one(documents, kind, name):
    matches = [
        document
        for _, document in documents
        if document.get("kind") == kind and document.get("metadata", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one {kind}/{name}, found {len(matches)}")
    return matches[0]


def container_env(container):
    env = {}
    for item in container.get("env", []):
        if "value" in item:
            env[item["name"]] = item["value"]
        elif "valueFrom" in item:
            env[item["name"]] = item["valueFrom"]
    return env


def container_mounts(container):
    return {item["name"]: item["mountPath"] for item in container.get("volumeMounts", [])}


def validate():
    documents = load_documents()
    if not documents:
        raise AssertionError("No Kubernetes manifests found")

    applied_secrets = [
        document.get("metadata", {}).get("name") for _, document in documents if document.get("kind") == "Secret"
    ]
    if applied_secrets:
        raise AssertionError("Applied manifests must not include static Secrets; found " + ", ".join(applied_secrets))

    api = find_one(documents, "Deployment", "fraud-api")
    if api["spec"].get("replicas") != 1:
        raise AssertionError("fraud-api must stay single-replica while backed by SQLite")

    hpas = [
        document
        for _, document in documents
        if document.get("kind") == "HorizontalPodAutoscaler"
        and document.get("metadata", {}).get("name") == "fraud-api-hpa"
    ]
    if hpas:
        raise AssertionError("fraud-api HPA must not be enabled while backed by SQLite")

    ingress = find_one(documents, "Ingress", "fraud-api-ingress")
    if not ingress.get("spec", {}).get("tls"):
        raise AssertionError("fraud-api ingress must define TLS")

    consumer = find_one(documents, "Deployment", "fraud-consumer")
    container = consumer["spec"]["template"]["spec"]["containers"][0]
    env = container_env(container)
    mounts = container_mounts(container)

    required_env = {
        "COMMAND_CENTER_API_VERSION",
        "COMMAND_CENTER_API_TOKEN",
        "PKYC_TOPIC",
        "SANCTIONS_CSV_PATH",
        "OLLAMA_URL",
        "SAR_LLM_AUDIT_LOG",
        "SHAP_MIN_CONFIDENCE_FOR_SAR",
        "TUNISIA_ISLAMIC_HOLIDAYS",
    }
    missing_env = sorted(required_env - set(env))
    if missing_env:
        raise AssertionError("fraud-consumer missing env vars: " + ", ".join(missing_env))

    expected_mounts = {
        "checkpoint": "/app/tmp/checkpoint",
        "app-data": "/app/data",
        "models": "/app/models",
    }
    for name, mount_path in expected_mounts.items():
        if mounts.get(name) != mount_path:
            raise AssertionError(f"fraud-consumer mount {name} must be {mount_path}")

    liveness = container.get("livenessProbe", {}).get("exec", {}).get("command", [])
    if "src/streaming/consumer.py" not in " ".join(liveness):
        raise AssertionError("fraud-consumer liveness probe must match the running process")

    print(f"Validated {len(documents)} Kubernetes resources from {len(APPLIED_MANIFESTS)} files")


if __name__ == "__main__":
    try:
        validate()
    except Exception as exc:
        print(f"Kubernetes manifest validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
