"""Regenerate dt_ny_snapshot.json (the Draw Territory fallback) from the live NASS cache.

Run with the SNOWFLAKE_* environment variables set (account, user, role, warehouse and
SNOWFLAKE_PRIVATE_KEY_PATH + SNOWFLAKE_PRIVATE_KEY_PWD, or SNOWFLAKE_PASSWORD):

    python build_dt_snapshot.py
"""
import json
import os

import snowflake.connector

import dt_nass


def _private_key():
    path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    pem = os.environ.get("SNOWFLAKE_PRIVATE_KEY", "")
    if not path and not pem.strip():
        return None
    from cryptography.hazmat.primitives import serialization
    raw = open(path, "rb").read() if path else pem.replace("\\n", "\n").encode()
    pwd = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PWD") or None
    key = serialization.load_pem_private_key(raw, password=pwd.encode() if pwd else None)
    return key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def main() -> None:
    kw = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
        role=os.environ.get("SNOWFLAKE_ROLE") or None,
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE") or None,
        database="JSA", schema="NASS_CACHE", login_timeout=30,
    )
    pkey = _private_key()
    if pkey is not None:
        kw["private_key"] = pkey
    else:
        kw["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    conn = snowflake.connector.connect(**kw)
    cur = conn.cursor()

    def fetch(params):
        cur.execute("SELECT data FROM nass_cache WHERE cache_key = %s", (dt_nass.cache_key(params),))
        row = cur.fetchone()
        if not row:
            return None
        raw = row[0] if isinstance(row[0], (dict, list)) else json.loads(row[0])
        recs = raw.get("data", []) if isinstance(raw, dict) else raw
        return [r for r in recs if r.get("state_alpha") == "NY"]

    snap = dt_nass.build_snapshot(fetch)
    conn.close()
    dt_nass.SNAPSHOT_FILE.write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
    n_state = sum(len(v) for m in snap["state"].values() for v in m.values())
    n_county = sum(len(rows) for c in snap["county"].values() for m in c.values() for rows in m.values())
    print(f"wrote {dt_nass.SNAPSHOT_FILE.name}: {n_state} state points, {n_county} county rows, "
          f"{dt_nass.SNAPSHOT_FILE.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
