"""Regression tests for optional backend import boundaries."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("package_name", "forbidden_modules"),
    [
        ("ml", ("ml.train_model", "ml.train_pipeline")),
        ("rag_engine", ("rag_engine.vector_store", "chromadb", "sentence_transformers")),
        ("shared", ("shared.schemas", "shared.tracing", "shared.vault_client")),
    ],
)
def test_package_import_does_not_load_optional_backends(package_name, forbidden_modules):
    assertions = "\n".join(
        f"assert {module_name!r} not in sys.modules, {module_name!r}" for module_name in forbidden_modules
    )
    script = f"import sys\nimport {package_name}\n{assertions}\n"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
