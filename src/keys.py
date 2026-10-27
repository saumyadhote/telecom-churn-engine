"""
keys.py
-------
Admin CLI for developer accounts and API keys.

Usage:
    python -m src.keys create --name "Alice" [--email a@x.com] [--tier standard|free]
    python -m src.keys list
    python -m src.keys rotate <account_id>      # issue an extra key for an account
    python -m src.keys revoke <key | prefix | key_id>

Uses the same storage directory as the API (CHURN_STORAGE_DIR, default storage/).
"""

import argparse
import os
import sys
from pathlib import Path

from src.accounts import TIERS, AccountStore


def store() -> AccountStore:
    return AccountStore(Path(os.environ.get("CHURN_STORAGE_DIR", "storage/")) / "accounts.db")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manage Churn Engine API keys")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create an account and its first key")
    create.add_argument("--name", required=True)
    create.add_argument("--email")
    create.add_argument("--tier", default="standard", choices=list(TIERS))

    sub.add_parser("list", help="List accounts and their keys")

    rotate = sub.add_parser("rotate", help="Issue an additional key for an account")
    rotate.add_argument("account_id")

    revoke = sub.add_parser("revoke", help="Revoke a key")
    revoke.add_argument("key", help="Full key, its 10-character prefix, or its key_ id")

    args = parser.parse_args(argv)
    s = store()

    if args.command == "create":
        account, key = s.create_account(args.name, args.email, args.tier)
        print(f"Account  {account['id']}  ({account['name']}, {account['tier']} tier)")
        print(f"API key  {key}")
        print("Store this key now — it can't be shown again.")

    elif args.command == "list":
        accounts = s.list_accounts()
        if not accounts:
            print("No accounts yet. Create one with: python -m src.keys create --name <name>")
        for a in accounts:
            print(f"{a['id']}  {a['name']}  <{a['email'] or '-'}>  {a['tier']}  created {a['created_at']}")
            for k in s.list_keys(a["id"]):
                state = f"revoked {k['revoked_at']}" if k["revoked_at"] else \
                        f"last used {k['last_used_at'] or 'never'}"
                print(f"    {k['id']}  {k['prefix']}…  {state}")

    elif args.command == "rotate":
        if not s.get_account(args.account_id):
            print(f"No account {args.account_id}", file=sys.stderr)
            return 1
        print(f"New API key  {s.create_key(args.account_id)}")
        print("Store this key now. Revoke the old one once you've switched over.")

    elif args.command == "revoke":
        try:
            count = s.revoke_key(args.key)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Revoked {count} key(s).")
        return 0 if count else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
