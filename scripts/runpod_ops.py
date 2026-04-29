#!/usr/bin/env python3
"""RunPod maintenance CLI (list pods, stop active, terminate stopped) — runpod_ops notebook equivalent."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv


def main() -> int:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo)
    load_dotenv(os.path.join(repo, ".env"), override=False)

    from pipeline.integrations.runpod.graphql_helpers import (
        list_my_pods,
        stop_all_active_pods,
        terminate_pods_with_statuses,
        terminate_stopped_pods,
    )

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Print pods (name, id, status, ports)")

    sp = sub.add_parser("stop-active", help="podStop every RUNNING/CREATED/RESTARTING pod")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser(
        "terminate-stopped",
        help="podTerminate EXITED/DEAD/TERMINATED pods (cleanup)",
    )
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("terminate-statuses", help="podTerminate pods in given statuses")
    sp.add_argument(
        "statuses",
        nargs="+",
        help="e.g. EXITED STOPPED",
    )
    sp.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        print("RUNPOD_API_KEY missing in environment or .env", file=sys.stderr)
        return 1

    if args.cmd == "list":
        pods = list_my_pods(key)
        for row in pods:
            print(
                json.dumps(
                    {
                        "name": row.get("name"),
                        "id": row.get("id"),
                        "desiredStatus": row.get("desiredStatus"),
                        "ports": row.get("ports"),
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    if args.cmd == "stop-active":
        stop_all_active_pods(key, dry_run=args.dry_run)
        return 0

    if args.cmd == "terminate-stopped":
        terminate_stopped_pods(key, dry_run=args.dry_run)
        return 0

    if args.cmd == "terminate-statuses":
        st = frozenset(s.strip().upper() for s in args.statuses if s.strip())
        terminate_pods_with_statuses(key, st, dry_run=args.dry_run)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
