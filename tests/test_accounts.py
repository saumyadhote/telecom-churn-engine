"""
tests/test_accounts.py
----------------------
Unit tests for account/key storage and the admin CLI.
"""

import sqlite3

import pytest

from src import keys as keys_cli
from src.accounts import TIERS, AccountStore


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db")


def test_create_and_authenticate(store):
    account, key = store.create_account("Ada", "ada@example.com")
    assert key.startswith("ck_") and len(key) > 30
    assert store.authenticate(key)["id"] == account["id"]
    assert account["limits"] == TIERS["standard"]


def test_plaintext_key_is_never_stored(store, tmp_path):
    _, key = store.create_account("Ada")
    dump = "\n".join(sqlite3.connect(tmp_path / "accounts.db").iterdump())
    assert key not in dump


@pytest.mark.parametrize("bad", ["", "ck_wrong", "not-a-key", None])
def test_bad_keys_rejected(store, bad):
    store.create_account("Ada")
    assert store.authenticate(bad) is None


def test_revoke_by_key_prefix_and_id(store):
    account, k1 = store.create_account("Ada")
    k2, k3 = store.create_key(account["id"]), store.create_key(account["id"])
    key_ids = [k["id"] for k in store.list_keys(account["id"])]

    assert store.revoke_key(k1) == 1
    assert store.revoke_key(k2[:10]) == 1
    assert store.revoke_key(key_ids[2]) == 1
    assert all(store.authenticate(k) is None for k in (k1, k2, k3))


def test_revoke_refuses_short_prefix(store):
    """A bare 'ck_' must not match (and revoke) every key."""
    _, key = store.create_account("Ada")
    with pytest.raises(ValueError):
        store.revoke_key("ck_")
    assert store.authenticate(key) is not None


def test_last_used_recorded(store):
    account, key = store.create_account("Ada")
    assert store.list_keys(account["id"])[0]["last_used_at"] is None
    store.authenticate(key)
    assert store.list_keys(account["id"])[0]["last_used_at"] is not None


def test_unknown_tier(store):
    with pytest.raises(ValueError):
        store.create_account("Ada", tier="platinum")


def test_cli_create_list_revoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHURN_STORAGE_DIR", str(tmp_path))
    assert keys_cli.main(["create", "--name", "Ada", "--tier", "free"]) == 0
    key = next(line.split()[-1] for line in capsys.readouterr().out.splitlines() if line.startswith("API key"))

    keys_cli.main(["list"])
    listing = capsys.readouterr().out
    assert "Ada" in listing and key[:10] in listing and key not in listing

    assert keys_cli.main(["revoke", key[:10]]) == 0
    assert AccountStore(tmp_path / "accounts.db").authenticate(key) is None
