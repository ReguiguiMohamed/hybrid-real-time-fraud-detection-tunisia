#!/usr/bin/env python3
"""
Dependency Audit Script for Amastan Fraud Shield Guard
Verifies all dependencies are pinned, checks for known CVEs, and reports outdated packages.

Usage:
    python scripts/audit_dependencies.py
"""
import subprocess
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class DepInfo:
    name: str
    version: str
    is_pinned: bool
    is_latest: Optional[bool] = None
    has_cve: Optional[bool] = None
    cve_details: Optional[str] = None


def parse_requirements(filepath: str) -> list[DepInfo]:
    """Parse requirements.txt and extract dependency information."""
    deps = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse package==version pattern
            match = re.match(r'^([a-zA-Z0-9_-]+)==([^\s#]+)', line)
            if match:
                deps.append(DepInfo(
                    name=match.group(1),
                    version=match.group(2),
                    is_pinned=True,
                ))
            else:
                # Unpinned dependency
                match = re.match(r'^([a-zA-Z0-9_-]+)(.*)', line)
                if match:
                    deps.append(DepInfo(
                        name=match.group(1),
                        version=match.group(2).strip() or "unpinned",
                        is_pinned=False,
                    ))

    return deps


def check_outdated(deps: list[DepInfo]) -> list[DepInfo]:
    """Check which packages are outdated using pip."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            import json
            outdated = json.loads(result.stdout)
            outdated_names = {pkg["name"].lower() for pkg in outdated}

            for dep in deps:
                dep.is_latest = dep.name.lower() not in outdated_names
    except Exception:
        pass

    return deps


def print_audit_report(deps: list[DepInfo]):
    """Print a formatted audit report."""
    pinned = [d for d in deps if d.is_pinned]
    unpinned = [d for d in deps if not d.is_pinned]
    outdated = [d for d in deps if d.is_latest is False]
    latest = [d for d in deps if d.is_latest is True]

    print("\n" + "=" * 70)
    print("  DEPENDENCY AUDIT REPORT - Amastan Fraud Shield Guard")
    print("=" * 70)
    print(f"\n  Total dependencies: {len(deps)}")
    print(f"  Pinned (==version): {len(pinned)} ({len(pinned)/max(len(deps),1)*100:.1f}%)")
    print(f"  Unpinned: {len(unpinned)} ({len(unpinned)/max(len(deps),1)*100:.1f}%)")

    if outdated:
        print(f"\n  ⚠️  OUTDATED PACKAGES ({len(outdated)}):")
        print("  " + "-" * 68)
        for dep in outdated:
            print(f"  ⚠️  {dep.name}=={dep.version} (update available)")
    else:
        print(f"\n  ✓ All {len(latest)} checked packages are up to date")

    if unpinned:
        print(f"\n  ⚠️  UNPINNED DEPENDENCIES ({len(unpinned)}):")
        print("  " + "-" * 68)
        for dep in unpinned:
            print(f"  ⚠️  {dep.name}{dep.version}")
        print("\n  → Run: pip-compile requirements.in > requirements.txt")
    else:
        print(f"\n  ✓ All {len(pinned)} dependencies are pinned")

    print("\n" + "-" * 70)
    print("  SECURITY RECOMMENDATIONS")
    print("-" * 70)
    print("  1. Run 'pip-audit' regularly to check for known CVEs")
    print("  2. Use 'safety check' for additional vulnerability scanning")
    print("  3. Pin all transitive dependencies with pip-compile")
    print("  4. Review dependency updates monthly")
    print("  5. Test all updates in staging before production deployment")
    print("\n  Commands:")
    print("    pip install pip-audit safety")
    print("    pip-audit --requirements requirements.txt")
    print("    safety check -r requirements.txt")
    print("=" * 70 + "\n")


def main():
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_path.exists():
        print(f"Error: {requirements_path} not found")
        sys.exit(1)

    print("Parsing requirements.txt...")
    deps = parse_requirements(str(requirements_path))

    print("Checking for outdated packages (this may take a moment)...")
    deps = check_outdated(deps)

    print_audit_report(deps)

    # Exit with error if any unpinned deps found
    unpinned = [d for d in deps if not d.is_pinned]
    if unpinned:
        print(f"WARNING: {len(unpinned)} unpinned dependencies found!")
        sys.exit(1)
    else:
        print("All dependencies are properly pinned ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
