"""Run every validation script listed in registry.toml and save its console output.

    python run_all.py                 # everything
    python run_all.py --only 3.1      # checks whose section starts with "3.1"
    python run_all.py --id spacetime  # a single check

Logs go to outputs/<id>.txt; a one-line summary per check is printed at the end.
"""
import argparse
import os
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="section prefix, e.g. 3.1")
    ap.add_argument("--id", default="", help="single check id")
    args = ap.parse_args()
    with open(os.path.join(ROOT, "registry.toml"), "rb") as f:
        checks = tomllib.load(f)["check"]
    checks = [c for c in checks if c["section"].startswith(args.only) and (not args.id or c["id"] == args.id)]
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", SOURCE_DATE_EPOCH="0")  # reproducible PDF metadata
    summary = []
    for c in checks:
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(ROOT, c["script"]), *c.get("args", [])], cwd=ROOT,
                           capture_output=True, text=True, env=env)
        dt = time.time() - t0
        with open(os.path.join(ROOT, "outputs", f"{c['id']}.txt"), "w") as f:
            f.write(p.stdout)
            if p.stderr:
                f.write("\n--- stderr ---\n" + p.stderr)
        status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
        summary.append(f"{c['section']:>12s}  {c['id']:28s} {dt:6.1f} s  {status}")
        print(summary[-1], flush=True)
    print(f"\n{len(checks)} checks run; logs in outputs/")


if __name__ == "__main__":
    main()
