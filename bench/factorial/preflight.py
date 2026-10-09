#!/usr/bin/env python3
"""Night 2 preflight (B0): back up the gbrain database and prove the backup restores.

1. Count the rows of every table in the live database.
2. `pg_dump -Fc` it, and `pg_dumpall --globals-only`, into a new folder under ~/.gbrain/backups/.
3. Restore the dump into a scratch PostgreSQL 18 cluster that listens only on a Unix socket in its
   own temporary directory (port 5433), count its rows, and compare.
4. Stop the scratch cluster. Its directory is printed so it can be deleted afterwards.

The live database is only read. The password stays in the child process environment.
Writes preflight.json next to this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_measure import pg_env, psql, redact  # noqa: E402

PG = Path("/opt/homebrew/opt/postgresql@18/bin")
BACKUPS = Path.home() / ".gbrain/backups"
SCRATCH_PORT = 5433


def run(cmd: list[str], env: dict, password: str) -> str:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed: {redact(proc.stderr[-2000:], password)}")
    return proc.stdout


def count_sql(env, password) -> str:
    tables = psql("SELECT quote_ident(table_schema) || '.' || quote_ident(table_name) FROM information_schema.tables "
                  "WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema') "
                  "ORDER BY 1", env, password).split()
    return " UNION ALL ".join(f"SELECT '{t}', count(*) FROM {t}" for t in tables) + " ORDER BY 1"


def counts(sql: str, env, password) -> dict[str, int]:
    out = {}
    for line in psql(sql, env, password).splitlines():
        name, _, n = line.rpartition("|")
        out[name] = int(n)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    env, password = pg_env()
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M")
    folder = BACKUPS / f"night2-preflight-{stamp}"
    folder.mkdir(mode=0o700, parents=True)
    sql = count_sql(env, password)
    (folder / "count.sql").write_text(sql + "\n")
    before = counts(sql, env, password)
    (folder / "rowcounts-before.txt").write_text("".join(f"{k} {v}\n" for k, v in before.items()))

    t0 = time.time()
    dump = folder / "postgres.dump"
    run([str(PG / "pg_dump"), "-Fc", "-f", str(dump)], env, password)
    globals_sql = folder / "globals.sql"
    globals_sql.write_text(run([str(PG / "pg_dumpall"), "--globals-only"], env, password))
    os.chmod(globals_sql, 0o600)
    os.chmod(dump, 0o600)
    dump_seconds = time.time() - t0
    after = counts(sql, env, password)
    (folder / "SHA256SUMS").write_text(f"{sha256(dump)}  postgres.dump\n{sha256(globals_sql)}  globals.sql\n")

    scratch = Path(tempfile.mkdtemp(prefix="gbrain-restore-"))
    data = scratch / "data"
    senv = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    senv.update({"PGHOST": str(scratch), "PGPORT": str(SCRATCH_PORT), "PGUSER": env["PGUSER"],
                 "PGDATABASE": "postgres"})
    run([str(PG / "initdb"), "-D", str(data), "-U", env["PGUSER"], "--auth=trust", "--no-sync"], senv, "")
    run([str(PG / "pg_ctl"), "-D", str(data), "-l", str(scratch / "log"), "-w", "-o",
         f"-p {SCRATCH_PORT} -k {scratch} -c listen_addresses=''", "start"], senv, "")
    try:
        t0 = time.time()
        proc = subprocess.run([str(PG / "pg_restore"), "-d", "postgres", "-j", "4", "--no-owner", str(dump)],
                              env=senv, capture_output=True, text=True)
        restore_seconds = time.time() - t0
        restore_errors = [line for line in proc.stderr.splitlines() if "error" in line.lower()]
        restored = counts(sql, senv, "")
    finally:
        run([str(PG / "pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"], senv, "")

    diff_before = {k: [before.get(k), restored.get(k)] for k in set(before) | set(restored)
                   if before.get(k) != restored.get(k)}
    changed_live = {k: [before[k], after.get(k)] for k in before if before[k] != after.get(k)}
    version = psql("SELECT current_setting('server_version'), "
                   "(SELECT extversion FROM pg_extension WHERE extname = 'vector'), "
                   "(SELECT count(*) FROM pages WHERE deleted_at IS NULL), "
                   "(SELECT count(*) FROM content_chunks), "
                   "(SELECT count(*) FROM content_chunks WHERE embedding IS NOT NULL)", env, password).strip().split("|")
    gbrain = subprocess.run([str(Path.home() / ".bun/bin/gbrain"), "--version"], capture_output=True, text=True).stdout.strip()
    report = {
        "taken": stamp, "folder": str(folder), "dump_bytes": dump.stat().st_size, "dump_seconds": round(dump_seconds, 1),
        "restore_seconds": round(restore_seconds, 1), "restore_exit": proc.returncode,
        "restore_error_lines": restore_errors[:20], "tables": len(before),
        "rows_live_before": sum(before.values()), "rows_restored": sum(restored.values()),
        "tables_differing_from_live_before": diff_before,
        "tables_changed_live_during_dump": changed_live,
        "versions": {"postgresql": version[0], "pgvector": version[1], "gbrain": gbrain},
        "corpus": {"live_pages": int(version[2]), "chunks": int(version[3]), "embedded_chunks": int(version[4])},
        "scratch_dir_to_delete": str(scratch),
    }
    (HERE / "preflight.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
