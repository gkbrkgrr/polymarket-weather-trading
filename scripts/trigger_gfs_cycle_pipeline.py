#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_gfs_cycle_pipeline as cycle_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
DEFAULT_STATE_FILE = REPO_ROOT / "live_trading" / "state" / "gfs_cycle_trigger_state.json"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "reports" / "live_probabilities" / "latest_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trigger run_gfs_cycle_pipeline for the latest publish-eligible GFS cycle "
            "and persist cycle-level idempotency state."
        )
    )
    parser.add_argument(
        "--cycle",
        default=None,
        help="Optional cycle token YYYYMMDDHH. If omitted, infer from publish schedule and current UTC time.",
    )
    parser.add_argument(
        "--now-utc",
        default=None,
        help="Override current UTC clock for cycle inference (YYYYMMDDHHMM). Ignored when --cycle is provided.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root path.")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable), help="Python executable for downstream pipeline call.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help=f"Trigger state JSON path (default: {DEFAULT_STATE_FILE}).")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Latest manifest path used for post-run verification (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--calibration-version",
        type=str,
        default="residual_oof_v1",
        help="Calibration version forwarded to run_gfs_cycle_pipeline.py.",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Forward --no-overwrite to run_gfs_cycle_pipeline.py.")
    parser.add_argument("--skip-telegram", action="store_true", help="Forward --skip-telegram to run_gfs_cycle_pipeline.py.")
    parser.add_argument(
        "--telegram-credentials-file",
        type=Path,
        default=cycle_pipeline.DEFAULT_TELEGRAM_CREDENTIALS_FILE,
        help="Forward --telegram-credentials-file to run_gfs_cycle_pipeline.py.",
    )
    parser.add_argument("--force", action="store_true", help="Run even if this cycle already completed successfully.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected cycle/command without executing downstream pipeline.")
    parser.add_argument("--max-history", type=int, default=200, help="Max state history length to retain.")
    return parser.parse_args(argv)


def utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "last_completed_cycle": None,
            "history": [],
            "updated_at_utc": None,
        }

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"State file must be a JSON object: {path}")

    history = loaded.get("history")
    if not isinstance(history, list):
        history = []

    return {
        "schema_version": int(loaded.get("schema_version", 1)),
        "last_completed_cycle": loaded.get("last_completed_cycle"),
        "history": history,
        "updated_at_utc": loaded.get("updated_at_utc"),
    }


def append_history(state: dict[str, Any], record: dict[str, Any], max_history: int) -> None:
    history = state.get("history")
    if not isinstance(history, list):
        history = []
    history.append(record)
    keep = max(1, int(max_history))
    state["history"] = history[-keep:]
    state["updated_at_utc"] = utc_now().isoformat()


def resolve_cycle(args: argparse.Namespace) -> str:
    if args.cycle:
        return cycle_pipeline.parse_cycle_token(str(args.cycle))

    now_utc = cycle_pipeline.parse_now_utc_token(str(args.now_utc)) if args.now_utc else utc_now()
    return cycle_pipeline.infer_cycle_from_time(now_utc)


def verify_latest_manifest(*, manifest_path: Path, expected_cycle: str) -> None:
    if not manifest_path.exists():
        raise RuntimeError(f"latest manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"latest manifest must be a JSON object: {manifest_path}")

    cycle = str(manifest.get("cycle", "")).strip()
    if cycle != expected_cycle:
        raise RuntimeError(
            f"latest manifest cycle mismatch: expected={expected_cycle} got={cycle or '<empty>'} path={manifest_path}"
        )

    status = str(manifest.get("status", "")).strip().lower()
    if status and status != "success":
        raise RuntimeError(f"latest manifest status is not success: status={status} path={manifest_path}")


def build_command(args: argparse.Namespace, cycle: str, repo_root: Path) -> list[str]:
    cmd = [
        str(args.python_bin),
        str(repo_root / "scripts" / "run_gfs_cycle_pipeline.py"),
        "--cycle",
        cycle,
        "--calibration-version",
        str(args.calibration_version),
    ]
    if args.no_overwrite:
        cmd.append("--no-overwrite")
    if args.skip_telegram:
        cmd.append("--skip-telegram")
    cmd.extend(["--telegram-credentials-file", str(args.telegram_credentials_file)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = resolve_path(Path(args.repo_root), REPO_ROOT)
    state_path = resolve_path(Path(args.state_file), repo_root)
    manifest_path = resolve_path(Path(args.manifest_path), repo_root)

    cycle = resolve_cycle(args)
    state = load_state(state_path)

    print(f"Cycle selected: {cycle}")
    print(f"Repo root: {repo_root}")
    print(f"State file: {state_path}")
    print(f"Manifest path: {manifest_path}")
    print(f"Force: {bool(args.force)}")
    print(f"Dry run: {bool(args.dry_run)}")

    if not args.force and str(state.get("last_completed_cycle") or "") == cycle:
        print(f"Skipping cycle {cycle}: already completed successfully (use --force to rerun).")
        return 0

    cmd = build_command(args, cycle, repo_root)
    print("Running command:")
    print(" ".join(cmd))

    started_at = utc_now().isoformat()
    proc = subprocess.run(cmd, cwd=repo_root)
    completed_at = utc_now().isoformat()

    record: dict[str, Any] = {
        "cycle": cycle,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "status": "failed" if proc.returncode != 0 else ("dry_run" if args.dry_run else "success"),
        "return_code": int(proc.returncode),
        "command": cmd,
    }

    if proc.returncode != 0:
        append_history(state, record, max_history=int(args.max_history))
        atomic_write_json(state_path, state)
        return int(proc.returncode)

    if not args.dry_run:
        try:
            verify_latest_manifest(manifest_path=manifest_path, expected_cycle=cycle)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{exc.__class__.__name__}: {exc}"
            append_history(state, record, max_history=int(args.max_history))
            atomic_write_json(state_path, state)
            print(f"Manifest verification failed: {exc}")
            return 1
        state["last_completed_cycle"] = cycle

    append_history(state, record, max_history=int(args.max_history))
    atomic_write_json(state_path, state)

    if args.dry_run:
        print("Dry-run complete; state history updated without manifest verification.")
    else:
        print(f"Cycle {cycle} completed and latest manifest verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
