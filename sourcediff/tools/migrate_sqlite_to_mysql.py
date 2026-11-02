"""One-time migration of SourceDiff SQLite metadata into MySQL."""
from __future__ import annotations

import argparse
import sqlite3
import os
import sys
from pathlib import Path

import pymysql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sourcediff import history


TABLES = ("users", "submissions", "analysis_cache", "library_registry")


def rows(source: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    source.row_factory = sqlite3.Row
    return source.execute(f"SELECT * FROM {table}").fetchall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", default="data/history.db")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite).resolve()
    if not sqlite_path.is_file():
        raise SystemExit(f"SQLite file not found: {sqlite_path}")
    if not history.mysql_enabled():
        raise SystemExit("Set SOURCEDIFF_DB_BACKEND=mysql and MySQL connection environment variables first.")

    bootstrap_password = os.getenv("SOURCEDIFF_MYSQL_PASSWORD")
    bootstrap = pymysql.connect(
        host=os.getenv("SOURCEDIFF_MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("SOURCEDIFF_MYSQL_PORT", "3306")),
        user=os.getenv("SOURCEDIFF_MYSQL_USER", "sourcediff_app"),
        password=bootstrap_password,
        charset="utf8mb4",
        autocommit=True,
    )
    database = os.getenv("SOURCEDIFF_MYSQL_DATABASE", "sourcediff").replace("`", "")
    with bootstrap.cursor() as cursor:
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    bootstrap.close()

    source = sqlite3.connect(str(sqlite_path))
    target = history.connect()
    target.execute("START TRANSACTION")
    try:
        target.execute(
            "ALTER TABLE users MODIFY user_id "
            "VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL"
        )
        for table in TABLES:
            existing = target.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()
            count = int(existing["n"] if existing else 0)
            if count:
                raise RuntimeError(
                    f"MySQL table {table} already contains {count} rows; migration stopped to prevent duplication."
                )

        user_rows = rows(source, "users")
        for row in user_rows:
            data = dict(row)
            columns = list(data)
            placeholders = ",".join(["%s"] * len(columns))
            target.execute(
                f"INSERT INTO users ({','.join(columns)}) VALUES ({placeholders})",
                tuple(data.values()),
            )

        submission_rows = rows(source, "submissions")
        for row in submission_rows:
            data = dict(row)
            allowed = (
                "id", "user_id", "user_key", "submitted_at", "filename", "sha256",
                "door", "verdict", "label", "reason_code", "case_id", "stub_used",
                "card_sha256", "prev_hash", "row_hash",
            )
            data = {name: data.get(name) for name in allowed}
            columns = list(data)
            placeholders = ",".join(["%s"] * len(columns))
            target.execute(
                f"INSERT INTO submissions ({','.join(columns)}) VALUES ({placeholders})",
                tuple(data.values()),
            )

        cache_rows = rows(source, "analysis_cache")
        for row in cache_rows:
            data = dict(row)
            columns = list(data)
            placeholders = ",".join(["%s"] * len(columns))
            target.execute(
                f"INSERT INTO analysis_cache ({','.join(columns)}) VALUES ({placeholders})",
                tuple(data.values()),
            )

        registry_rows = rows(source, "library_registry")
        for row in registry_rows:
            data = dict(row)
            columns = list(data)
            placeholders = ",".join(["%s"] * len(columns))
            target.execute(
                f"INSERT INTO library_registry ({','.join(columns)}) VALUES ({placeholders})",
                tuple(data.values()),
            )

        target.commit()
        print(
            "Migrated "
            f"{len(user_rows)} users, {len(submission_rows)} submissions, "
            f"{len(cache_rows)} cache rows, and {len(registry_rows)} library rows."
        )
        return 0
    except Exception:
        target.rollback()
        raise
    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    raise SystemExit(main())
