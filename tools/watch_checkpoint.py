"""Watch `models/phase1_best.pth` and print an update line when it changes.

This script is intentionally tiny and dependency-free so it can run in the
project virtualenv. It prints lines like:

  CHECKPOINT_UPDATED 2026-05-29T02:57:40.000000Z size=17003883

which are easy to grep in the terminal or CI logs.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

PATH = os.path.join("models", "phase1_best.pth")
SLEEP = 5.0


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def stat_info(p: str):
    st = os.stat(p)
    return st.st_mtime, st.st_size


def main():
    prev_mtime = None
    prev_size = None
    if os.path.exists(PATH):
        try:
            prev_mtime, prev_size = stat_info(PATH)
        except Exception:
            prev_mtime = None

    # Print initial state
    if prev_mtime is not None:
        print(f"FOUND {iso_utc(prev_mtime)} size={prev_size}", flush=True)
    else:
        print("MISSING", flush=True)

    try:
        while True:
            if os.path.exists(PATH):
                try:
                    mtime, size = stat_info(PATH)
                except Exception:
                    time.sleep(SLEEP)
                    continue
                if prev_mtime is None or mtime != prev_mtime or size != prev_size:
                    print(f"CHECKPOINT_UPDATED {iso_utc(mtime)} size={size}", flush=True)
                    prev_mtime, prev_size = mtime, size
            time.sleep(SLEEP)
    except KeyboardInterrupt:
        print("watcher: stopped by user", flush=True)


if __name__ == "__main__":
    main()
