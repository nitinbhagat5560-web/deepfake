"""Submission history: who sent what, when, and what the verdict was. SQLite, one file.

One row per upload: user, UTC time, file name, SHA-256, door, verdict, case folder, card hash.
The rows form a hash chain (audit.py), so editing or removing an old row is detectable.

Privacy: the chain covers only non-personal fields (a hashed user key, the file's SHA-256,
verdict, case id). forget_user() blanks the readable user id and file name and the chain
still verifies. The uploaded media in data/cases/ is a separate thing; delete that folder too.

Swap to PostgreSQL later by replacing connect(); the SQL is plain.
"""
from __future__ import annotations

import sqlite3
import os
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import re

import pymysql

from sourcediff import paths
from sourcediff.audit import GENESIS, chain_hash, sha256_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  user_key TEXT NOT NULL,
  submitted_at TEXT NOT NULL,
  filename TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  door TEXT NOT NULL,
  verdict TEXT NOT NULL,
  label TEXT NOT NULL,
  reason_code TEXT,
  case_id TEXT NOT NULL,
  stub_used INTEGER NOT NULL,
  card_sha256 TEXT,
  prev_hash TEXT NOT NULL,
  row_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sub_user ON submissions(user_key, id DESC);
CREATE TABLE IF NOT EXISTS analysis_cache(
  sha256 TEXT NOT NULL,
  path2_model TEXT NOT NULL,
  path2_mode TEXT NOT NULL,
  engine_version TEXT NOT NULL,
  library_version TEXT NOT NULL,
  case_id TEXT NOT NULL,
  card_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (sha256, path2_model, path2_mode)
);
CREATE TABLE IF NOT EXISTS library_registry(
  kind TEXT NOT NULL,
  door TEXT NOT NULL,
  source_id TEXT NOT NULL,
  title TEXT NOT NULL,
  publisher TEXT NOT NULL,
  license TEXT NOT NULL,
  source_url TEXT NOT NULL,
  notes TEXT NOT NULL,
  filename TEXT NOT NULL,
  disk_path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  added_by TEXT NOT NULL,
  added_at TEXT NOT NULL,
  status TEXT NOT NULL,
  synced_at TEXT NOT NULL,
  PRIMARY KEY (kind, door, source_id)
);
CREATE INDEX IF NOT EXISTS idx_library_registry_sha ON library_registry(sha256);
"""
CHAIN_FIELDS = ("user_key", "submitted_at", "sha256", "door", "verdict", "case_id", "card_sha256", "stub_used")


MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  user_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin PRIMARY KEY,
  created_at VARCHAR(64) NOT NULL,
  password_hash VARCHAR(64) NOT NULL DEFAULT '',
  password_salt VARCHAR(32) NOT NULL DEFAULT '',
  is_admin INTEGER NOT NULL DEFAULT 0,
  last_login_at VARCHAR(64) NULL
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS submissions(
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(64) NOT NULL,
  user_key VARCHAR(16) NOT NULL,
  submitted_at VARCHAR(64) NOT NULL,
  filename TEXT NOT NULL,
  sha256 VARCHAR(64) NOT NULL,
  door VARCHAR(16) NOT NULL,
  verdict VARCHAR(64) NOT NULL,
  label VARCHAR(255) NOT NULL,
  reason_code VARCHAR(64) NULL,
  case_id VARCHAR(128) NOT NULL,
  stub_used INTEGER NOT NULL,
  card_sha256 VARCHAR(64) NULL,
  prev_hash VARCHAR(64) NOT NULL,
  row_hash VARCHAR(64) NOT NULL,
  INDEX idx_sub_user (user_key, id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS analysis_cache(
  sha256 VARCHAR(64) NOT NULL,
  path2_model VARCHAR(128) NOT NULL,
  path2_mode VARCHAR(32) NOT NULL,
  engine_version VARCHAR(32) NOT NULL,
  library_version VARCHAR(64) NOT NULL,
  case_id VARCHAR(128) NOT NULL,
  card_sha256 VARCHAR(64) NOT NULL,
  created_at VARCHAR(64) NOT NULL,
  PRIMARY KEY (sha256, path2_model, path2_mode)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS library_registry(
  kind VARCHAR(16) NOT NULL,
  door VARCHAR(16) NOT NULL,
  source_id VARCHAR(128) NOT NULL,
  title TEXT NOT NULL,
  publisher TEXT NOT NULL,
  license TEXT NOT NULL,
  source_url TEXT NOT NULL,
  notes TEXT NOT NULL,
  filename TEXT NOT NULL,
  disk_path TEXT NOT NULL,
  sha256 VARCHAR(64) NOT NULL,
  added_by VARCHAR(64) NOT NULL,
  added_at VARCHAR(64) NOT NULL,
  status VARCHAR(32) NOT NULL,
  synced_at VARCHAR(64) NOT NULL,
  PRIMARY KEY (kind, door, source_id),
  INDEX idx_library_registry_sha (sha256)
) ENGINE=InnoDB;
"""


class _MySQLCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def execute(self, query, params=None):
        query = str(query).strip()
        upper = query.upper()
        if upper.startswith("BEGIN IMMEDIATE"):
            query = "START TRANSACTION"
        if upper.startswith("PRAGMA TABLE_INFO(USERS)"):
            query = (
                "SELECT COLUMN_NAME AS name FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='users'"
            )
        query = re.sub(r"INSERT\s+OR\s+IGNORE\s+INTO", "INSERT IGNORE INTO", query, flags=re.I)
        replace_match = re.match(
            r"INSERT\s+OR\s+REPLACE\s+INTO\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)\s*VALUES\s*\(([^;]+)\)",
            query,
            flags=re.I | re.S,
        )
        if replace_match:
            table = replace_match.group(1)
            columns = [name.strip() for name in replace_match.group(2).split(",")]
            query = (
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({replace_match.group(3)}) "
                f"ON DUPLICATE KEY UPDATE "
                + ", ".join(f"{name}=VALUES({name})" for name in columns)
            )
        if "?" in query:
            query = query.replace("?", "%s")
        self._cursor.execute(query, params)
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


class _MySQLConnection:
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self._connection = pymysql.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            charset="utf8mb4",
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
        )
        self.executescript(MYSQL_SCHEMA)

    def execute(self, query, params=None):
        return _MySQLCursor(self._connection.cursor()).execute(query, params)

    def executescript(self, script: str):
        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                try:
                    self._connection.cursor().execute(statement)
                except pymysql.err.OperationalError as exc:
                    raise RuntimeError(f"MySQL schema initialization failed: {exc}") from exc
        self._connection.commit()

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


def mysql_enabled() -> bool:
    return os.getenv("SOURCEDIFF_DB_BACKEND", "").strip().lower() == "mysql"


def connect(db=None):
    if mysql_enabled():
        password = os.getenv("SOURCEDIFF_MYSQL_PASSWORD")
        if not password:
            raise RuntimeError(
                "SOURCEDIFF_MYSQL_PASSWORD is required when SOURCEDIFF_DB_BACKEND=mysql"
            )
        return _MySQLConnection(
            host=os.getenv("SOURCEDIFF_MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("SOURCEDIFF_MYSQL_PORT", "3306")),
            database=os.getenv("SOURCEDIFF_MYSQL_DATABASE", "sourcediff"),
            user=os.getenv("SOURCEDIFF_MYSQL_USER", "sourcediff_app"),
            password=password,
        )
    db = db or paths.history_db()
    paths.data_dir().mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db), timeout=30, isolation_level=None)   # we manage transactions
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    columns = {row["name"] for row in con.execute("PRAGMA table_info(users)")}
    for column, definition in (
        ("password_hash", "TEXT NOT NULL DEFAULT ''"),
        ("password_salt", "TEXT NOT NULL DEFAULT ''"),
        ("is_admin", "INTEGER NOT NULL DEFAULT 0"),
        ("last_login_at", "TEXT"),
    ):
        if column not in columns:
            con.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def add_submission(user_id: str, filename: str, sha256: str, door: str, verdict: str, label: str,
                   case_id: str, stub_used: bool, reason_code: str = "", card_sha256: str = "",
                   now: str | None = None, db=None) -> dict:
    now = now or _now()
    user_key = sha256_text(user_id)[:16]
    with closing(connect(db)) as con:
        con.execute("BEGIN IMMEDIATE")                       # one writer at a time keeps the chain linear
        try:
            con.execute("INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)", (user_id, now))
            last = con.execute("SELECT row_hash FROM submissions ORDER BY id DESC LIMIT 1").fetchone()
            prev = last["row_hash"] if last else GENESIS
            core = {"user_key": user_key, "submitted_at": now, "sha256": sha256, "door": door,
                    "verdict": verdict, "case_id": case_id, "card_sha256": card_sha256,
                    "stub_used": int(bool(stub_used))}
            row_hash = chain_hash(prev, core)
            cur = con.execute(
                "INSERT INTO submissions(user_id,user_key,submitted_at,filename,sha256,door,verdict,label,"
                "reason_code,case_id,stub_used,card_sha256,prev_hash,row_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, user_key, now, filename, sha256, door, verdict, label, reason_code, case_id,
                 int(bool(stub_used)), card_sha256, prev, row_hash))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        return dict(con.execute("SELECT * FROM submissions WHERE id=?", (cur.lastrowid,)).fetchone())


def list_submissions(user_id: str | None = None, limit: int = 50, db=None) -> list[dict]:
    with closing(connect(db)) as con:
        if user_id:
            rows = con.execute("SELECT * FROM submissions WHERE user_key=? ORDER BY id DESC LIMIT ?",
                               (sha256_text(user_id)[:16], limit)).fetchall()
        else:
            rows = con.execute("SELECT * FROM submissions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def library_version() -> str:
    """Stable, small fingerprint of trusted library metadata and SHA-256 media hashes."""
    entries = []
    for door in paths.DOORS:
        root = paths.library_door_dir("trusted", door)
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            meta_file = directory / "meta.json"
            if not directory.is_dir() or not meta_file.is_file():
                continue
            meta = paths.load_json(meta_file)
            entries.append({
                "door": door,
                "source_id": meta.get("source_id", directory.name),
                "sha256": meta.get("sha256", ""),
                "file": meta.get("file", ""),
            })
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record_cache(sha256: str, model: str, mode: str, engine_version: str,
                 case_id: str, card_sha256: str, reusable: bool, db=None) -> None:
    if not reusable:
        return
    cache_model = "*" if model == "*" else model
    cache_mode = "*" if mode == "*" else mode
    with closing(connect(db)) as con:
        con.execute(
            "INSERT OR REPLACE INTO analysis_cache "
            "(sha256,path2_model,path2_mode,engine_version,library_version,case_id,card_sha256,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (sha256, cache_model, cache_mode, engine_version, library_version(), case_id,
             card_sha256, _now()),
        )


def find_cache(sha256: str, model: str, mode: str, engine_version: str, db=None) -> dict | None:
    """Return the newest reusable exact or Path-1-wildcard cache entry."""
    lib_version = library_version()
    with closing(connect(db)) as con:
        row = con.execute(
            "SELECT * FROM analysis_cache WHERE sha256=? AND path2_model=? AND path2_mode=? "
            "AND engine_version=? AND library_version=? ORDER BY created_at DESC LIMIT 1",
            (sha256, model, mode, engine_version, lib_version),
        ).fetchone()
        if row is None:
            row = con.execute(
                "SELECT * FROM analysis_cache WHERE sha256=? AND path2_model='*' AND path2_mode='*' "
                "AND engine_version=? AND library_version=? ORDER BY created_at DESC LIMIT 1",
                (sha256, engine_version, lib_version),
            ).fetchone()
        return dict(row) if row else None


def sync_library_registry(db=None) -> dict:
    """Mirror verified disk library entries into SQLite while retaining media on disk."""
    now = _now()
    current = set()
    with closing(connect(db)) as con:
        con.execute("BEGIN IMMEDIATE")
        for kind in paths.LIBRARY_KINDS:
            for door in paths.DOORS:
                root = paths.library_door_dir(kind, door)
                if not root.exists():
                    continue
                for entry in sorted(root.iterdir()):
                    meta_file = entry / "meta.json" if entry.is_dir() else None
                    if not meta_file or not meta_file.is_file():
                        continue
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    filename = str(meta.get("file") or "")
                    media = entry / filename
                    if not filename or not media.is_file():
                        continue
                    recorded_sha = str(meta.get("sha256") or "")
                    if recorded_sha:
                        sha = recorded_sha
                        status = "verified"
                    else:
                        computed_sha = hashlib.sha256(media.read_bytes()).hexdigest() if media.stat().st_size else ""
                        sha = computed_sha
                        status = "verified" if computed_sha else "integrity_error"
                    source_id = str(meta.get("source_id") or entry.name)
                    current.add((kind, door, source_id))
                    try:
                        disk_path = media.resolve().relative_to(paths.library_dir().resolve()).as_posix()
                    except ValueError:
                        disk_path = media.resolve().as_posix()
                    con.execute(
                        "INSERT OR REPLACE INTO library_registry "
                        "(kind,door,source_id,title,publisher,license,source_url,notes,filename,disk_path,"
                        "sha256,added_by,added_at,status,synced_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (kind, door, source_id, str(meta.get("title") or ""),
                         str(meta.get("publisher") or ""), str(meta.get("license") or ""),
                         str(meta.get("url") or ""), str(meta.get("notes") or ""), filename, disk_path,
                         sha, str(meta.get("added_by") or ""), str(meta.get("added_at") or ""),
                         status, now),
                    )
        for row in con.execute("SELECT kind,door,source_id FROM library_registry").fetchall():
            key = (row["kind"], row["door"], row["source_id"])
            if key not in current:
                con.execute(
                    "DELETE FROM library_registry WHERE kind=? AND door=? AND source_id=?",
                    key,
                )
        con.execute("COMMIT")
    with closing(connect(db)) as con:
        count = con.execute(
            "SELECT COUNT(*) AS n FROM library_registry WHERE status='verified'"
        ).fetchone()["n"]
    return {"count": count, "synced_at": now}


def library_registry(db=None) -> list[dict]:
    with closing(connect(db)) as con:
        rows = con.execute(
            "SELECT * FROM library_registry ORDER BY added_at DESC, door, source_id"
        ).fetchall()
        return [dict(row) for row in rows]


def create_user(username: str, password_hash: str, password_salt: str, db=None) -> bool:
    username = clean_user_name(username)
    if len(username) < 3 or len(password_hash) != 64 or len(password_salt) != 32:
        return False
    now = _now()
    with closing(connect(db)) as con:
        try:
            con.execute(
                "INSERT INTO users(user_id,created_at,password_hash,password_salt,is_admin,last_login_at) "
                "VALUES (?,?,?,?,0,NULL)",
                (username, now, password_hash, password_salt),
            )
            con.commit()
            return True
        except (sqlite3.IntegrityError, pymysql.err.IntegrityError):
            return False


def clean_user_name(raw: str) -> str:
    username = re.sub(r"[\x00-\x1f\x7f]", "", raw or "").strip()
    username = re.sub(r"\s+", " ", username)[:64]
    return username


def get_user(username: str, db=None) -> dict | None:
    with closing(connect(db)) as con:
        row = con.execute(
            "SELECT user_id,password_hash,password_salt,is_admin,last_login_at,created_at "
            "FROM users WHERE user_id=?",
            (clean_user_name(username),),
        ).fetchone()
        return dict(row) if row else None


def record_login(username: str, db=None) -> None:
    with closing(connect(db)) as con:
        con.execute(
            "UPDATE users SET last_login_at=? WHERE user_id=?",
            (_now(), clean_user_name(username)),
        )
        con.commit()


def verify_chain(db=None) -> dict:
    with closing(connect(db)) as con:
        prev, n = GENESIS, 0
        for r in con.execute("SELECT * FROM submissions ORDER BY id"):
            n += 1
            core = {k: r[k] for k in CHAIN_FIELDS}
            if r["prev_hash"] != prev or chain_hash(prev, core) != r["row_hash"]:
                return {"ok": False, "rows": n, "first_bad_id": r["id"]}
            prev = r["row_hash"]
        return {"ok": True, "rows": n, "first_bad_id": None}


def forget_user(user_id: str, db=None) -> int:
    """Blank the readable user id and file names. The hash chain still verifies."""
    key = sha256_text(user_id)[:16]
    with closing(connect(db)) as con:
        con.execute("BEGIN IMMEDIATE")
        n = con.execute("UPDATE submissions SET user_id='[deleted]', filename='[deleted]' "
                        "WHERE user_key=? AND user_id != '[deleted]'", (key,)).rowcount
        con.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        con.execute("COMMIT")
        return n


def delete_submission(row_id: int, db=None) -> dict:
    """Delete one history row and its reusable cache entries for that file.

    Rebuilding the remaining row hashes keeps the retained history chain internally
    valid. Case artifacts are retained unless explicitly removed separately.
    """
    with closing(connect(db)) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT id, sha256 FROM submissions WHERE id=?", (row_id,)).fetchone()
        if row is None:
            con.execute("ROLLBACK")
            return {"deleted": False, "deleted_rows": 0, "deleted_cache_rows": 0}

        sha256 = row["sha256"]
        con.execute("DELETE FROM submissions WHERE id=?", (row_id,))
        cache_deleted = con.execute(
            "DELETE FROM analysis_cache WHERE sha256=?", (sha256,)
        ).rowcount

        prev = GENESIS
        for retained in con.execute("SELECT * FROM submissions ORDER BY id ASC").fetchall():
            core = {key: retained[key] for key in CHAIN_FIELDS}
            row_hash = chain_hash(prev, core)
            con.execute(
                "UPDATE submissions SET prev_hash=?, row_hash=? WHERE id=?",
                (prev, row_hash, retained["id"]),
            )
            prev = row_hash
        con.execute("COMMIT")
        return {
            "deleted": True,
            "deleted_rows": 1,
            "deleted_cache_rows": cache_deleted,
        }
