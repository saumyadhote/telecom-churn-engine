"""
accounts.py
-----------
Developer accounts and API keys, stored in SQLite.

Keys look like `ck_<random>` and are shown exactly once, at creation. Only a
SHA-256 hash is stored — keys are 192-bit random tokens, so a fast hash is
safe here (no password-style stretching needed) and lookup stays O(1).

Usage:
    from src.accounts import AccountStore
    store = AccountStore("storage/accounts.db")
    account, key = store.create_account("Alice", tier="standard")
    store.authenticate(key)  # → account dict, or None
"""

import hashlib
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

KEY_PREFIX = "ck_"

# Per-tier limits. "free" is what self-serve signup gets; keys you issue by
# hand default to "standard".
TIERS = {
    "free": {
        "requests_per_minute": 60,
        "max_models":          5,
        "trainings_per_day":   20,
        "max_upload_mb":       20,
    },
    "standard": {
        "requests_per_minute": 300,
        "max_models":          50,
        "trainings_per_day":   200,
        "max_upload_mb":       50,
    },
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT,
    tier        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT PRIMARY KEY,
    account_id   TEXT NOT NULL REFERENCES accounts(id),
    key_hash     TEXT NOT NULL UNIQUE,
    prefix       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at   TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class AccountStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _account(row) -> dict:
        account = dict(row)
        account["limits"] = TIERS[account["tier"]]
        return account

    # ── Accounts ──

    def create_account(self, name: str, email: str | None = None, tier: str = "standard") -> tuple[dict, str]:
        """Create an account and its first key. Returns (account, plaintext key)."""
        if tier not in TIERS:
            raise ValueError(f"Unknown tier {tier!r}; choose from {', '.join(TIERS)}.")
        account_id = "acct_" + uuid.uuid4().hex[:12]
        with self._db() as db:
            db.execute(
                "INSERT INTO accounts (id, name, email, tier, created_at) VALUES (?, ?, ?, ?, ?)",
                (account_id, name, email, tier, _now()),
            )
        key = self.create_key(account_id)
        return self.get_account(account_id), key

    def get_account(self, account_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return self._account(row) if row else None

    def list_accounts(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("""
                SELECT a.*, COUNT(k.id) FILTER (WHERE k.revoked_at IS NULL) AS active_keys
                FROM accounts a LEFT JOIN api_keys k ON k.account_id = a.id
                GROUP BY a.id ORDER BY a.created_at
            """).fetchall()
        return [self._account(r) for r in rows]

    # ── Keys ──

    def create_key(self, account_id: str) -> str:
        key = KEY_PREFIX + secrets.token_urlsafe(24)
        with self._db() as db:
            db.execute(
                "INSERT INTO api_keys (id, account_id, key_hash, prefix, created_at) VALUES (?, ?, ?, ?, ?)",
                ("key_" + uuid.uuid4().hex[:12], account_id, hash_key(key), key[:10], _now()),
            )
        return key

    def authenticate(self, key: str) -> dict | None:
        """Return the account owning an active key, or None."""
        if not key or not key.startswith(KEY_PREFIX):
            return None
        with self._db() as db:
            row = db.execute("""
                SELECT a.*, k.id AS key_id FROM api_keys k JOIN accounts a ON a.id = k.account_id
                WHERE k.key_hash = ? AND k.revoked_at IS NULL
            """, (hash_key(key),)).fetchone()
            if row:
                db.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), row["key_id"]))
        return self._account(row) if row else None

    def list_keys(self, account_id: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                "SELECT id, prefix, created_at, last_used_at, revoked_at FROM api_keys "
                "WHERE account_id = ? ORDER BY created_at", (account_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_key(self, key_or_prefix: str) -> int:
        """
        Revoke a key, given the full key, its 10-character prefix (as shown by
        list_keys), or its key_… id. Returns how many keys were revoked.
        """
        if key_or_prefix.startswith("key_"):
            column, value = "id", key_or_prefix
        elif len(key_or_prefix) > 10:
            column, value = "key_hash", hash_key(key_or_prefix)
        elif len(key_or_prefix) == 10:
            column, value = "prefix", key_or_prefix
        else:
            raise ValueError("Give the full key, its 10-character prefix, or its key_ id.")
        with self._db() as db:
            cur = db.execute(
                f"UPDATE api_keys SET revoked_at = ? WHERE {column} = ? AND revoked_at IS NULL",
                (_now(), value),
            )
        return cur.rowcount
