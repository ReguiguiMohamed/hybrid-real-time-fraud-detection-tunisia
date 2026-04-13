#!/usr/bin/env python3
"""
Database Migration Runner for Amastan Fraud Shield Guard
Usage:
    python scripts/migrate.py upgrade    # Apply all pending migrations
    python scripts/migrate.py current    # Show current migration status
    python scripts/migrate.py migrate "description"  # Generate new migration template
"""
import sys
import os
import hashlib
import importlib
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_db_path():
    return PROJECT_ROOT / "data" / "feedback.db"


def get_migration_files():
    """Get all migration files sorted by migration ID."""
    migrations_dir = PROJECT_ROOT / "migrations"
    if not migrations_dir.exists():
        return []

    files = sorted(
        [f for f in migrations_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("test_")]
    )
    return files


def get_applied_migrations(db_path):
    """Get list of already-applied migrations from database."""
    import sqlite3

    if not db_path.exists():
        return set()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check if migration_history table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_history'"
    )
    if not cursor.fetchone():
        conn.close()
        return set()

    cursor.execute("SELECT migration_id FROM migration_history ORDER BY migration_id")
    applied = {row[0] for row in cursor.fetchall()}
    conn.close()
    return applied


def file_checksum(filepath):
    """Calculate MD5 checksum of a migration file for integrity verification."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def cmd_upgrade():
    """Apply all pending migrations."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    applied = get_applied_migrations(db_path)
    migration_files = get_migration_files()

    if not migration_files:
        print("No migration files found.")
        return

    pending = [f for f in migration_files if f.stem not in applied]

    if not pending:
        print("All migrations are up to date.")
        return

    print(f"Applying {len(pending)} pending migration(s)...")

    for migration_file in pending:
        migration_id = migration_file.stem
        checksum = file_checksum(migration_file)

        print(f"  => Applying {migration_id}...")

        # Import and run the migration module
        spec = importlib.util.spec_from_file_location(migration_id, str(migration_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            module.upgrade(str(db_path))
            print(f"  ✓ {migration_id} applied successfully.")
        except Exception as e:
            print(f"  ✗ {migration_id} failed: {e}")
            print("  Migration aborted. Fix the error and retry.")
            sys.exit(1)

    print(f"\nAll {len(pending)} migration(s) applied successfully.")
    cmd_current()


def cmd_current():
    """Show current migration status."""
    db_path = get_db_path()
    applied = get_applied_migrations(db_path)
    migration_files = get_migration_files()

    print("\nMigration Status:")
    print("=" * 70)

    for migration_file in migration_files:
        migration_id = migration_file.stem
        status = "APPLIED" if migration_id in applied else "PENDING"
        marker = "✓" if migration_id in applied else "○"
        print(f"  {marker} {migration_id}: {status}")

    print("=" * 70)
    print(f"Total: {len(migration_files)} | Applied: {len(applied)} | Pending: {len(migration_files) - len(applied)}")
    print()


def cmd_migrate(description):
    """Generate a new migration template."""
    migrations_dir = PROJECT_ROOT / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)

    # Get existing migration count
    existing = get_migration_files()
    next_num = len(existing) + 1

    # Generate migration filename
    migration_id = f"{next_num:04d}_{description.lower().replace(' ', '_')}"
    migration_file = migrations_dir / f"{migration_id}.py"

    if migration_file.exists():
        print(f"Migration {migration_id} already exists. Choose a different description.")
        return

    template = f'''"""
Migration {migration_id}: {description.title()}

TODO: Describe what this migration does.
"""
import sqlite3
from pathlib import Path


def upgrade(db_path: str = None):
    """Apply this migration."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # TODO: Add your schema changes here

    # Record this migration
    cursor.execute("""
        INSERT OR IGNORE INTO migration_history (migration_id, description)
        VALUES (?, ?)
    """, ("{migration_id}", "{description}"))

    conn.commit()
    conn.close()
    print("Migration {migration_id} applied: {description}")


def downgrade(db_path: str = None):
    """Remove this migration."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # TODO: Add your rollback logic here

    cursor.execute("DELETE FROM migration_history WHERE migration_id = ?", ("{migration_id}",))

    conn.commit()
    conn.close()
    print("Migration {migration_id} downgraded.")


if __name__ == "__main__":
    upgrade()
'''

    with open(migration_file, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"Migration generated: migrations/{migration_id}.py")
    print("Edit the file to add your schema changes, then run: make migrate")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/migrate.py upgrade              # Apply pending migrations")
        print("  python scripts/migrate.py current              # Show migration status")
        print('  python scripts/migrate.py migrate "description"  # Generate new migration')
        sys.exit(1)

    command = sys.argv[1]

    if command == "upgrade":
        cmd_upgrade()
    elif command == "current":
        cmd_current()
    elif command == "migrate":
        if len(sys.argv) < 3:
            print("Error: Please provide a migration description")
            print('Usage: python scripts/migrate.py migrate "Add new column"')
            sys.exit(1)
        cmd_migrate(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: upgrade, current, migrate")
        sys.exit(1)


if __name__ == "__main__":
    main()
