"""
evaluate.py - run the whole task headless over many objects and mazes and print a results table.

    python evaluate.py                                  # all 15 objects x mazes (seeds) 0-4
    python evaluate.py --seeds 5 6 7 8 9                # other mazes
    python evaluate.py --objects 3 11 --seeds 0 1 --workers 2

A run passes when the object ends within 10 cm of the goal sphere.  Results are appended to results_<tag>.txt.
Note: PyBullet runs are not bit-for-bit repeatable, so a marginal grasp can pass in one run and fail in the next.
"""
import argparse
import time
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def run_one(obj, seed):
    code = ("import headless, runpy, sys; sys.argv=['x', '%d', '%d']; "
            "runpy.run_path('pick_nav_reach_test.py', run_name='__main__')" % (obj, seed))
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1200)
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        out = ""
    found = re.search(r"RESULT (\{.*\})", out)
    if not found:
        return obj, seed, False, "error", None
    result = eval(found.group(1), {"__builtins__": {}}, {"True": True, "False": False, "None": None})
    where = f"{result['dist_to_goal'] * 100:.1f} cm" if result["dist_to_goal"] is not None else f"failed in {result['stage']}"
    return obj, seed, bool(result["success"]), where, result["steps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=int, nargs="*", default=list(range(15)))
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    jobs = [(o, s) for s in args.seeds for o in args.objects]
    rows = []
    with ThreadPoolExecutor(args.workers) as pool, open(f"results_{args.tag}.txt", "a") as log:
        futures = [pool.submit(run_one, *j) for j in jobs]
        pending = set(futures)
        started = time.time()
        done = 0
        while pending:
            finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            if not finished:                                              # heartbeat: still alive, nothing finished yet
                print(f"  ... still running: {done}/{len(jobs)} done, {(time.time() - started) / 60:.0f} min elapsed", flush=True)
                continue
            for future in finished:
                obj, seed, ok, where, steps = future.result()
                done += 1
                line = f"object {obj:2d} seed {seed}: {'PASS' if ok else 'FAIL'}  {where}  steps={steps}"
                print(f"[{done}/{len(jobs)}] {line}", flush=True)
                log.write(line + "\n")
                log.flush()
                rows.append((obj, seed, ok, steps))

    passed = sum(r[2] for r in rows)
    within = sum(r[2] and r[3] is not None and r[3] <= 15000 for r in rows)
    print(f"\n{passed}/{len(rows)} runs delivered within 10 cm ({within} of them within 15000 steps)")
    for obj in sorted(args.objects):
        mine = [r for r in rows if r[0] == obj]
        print(f"  object {obj:2d}: {sum(r[2] for r in mine)}/{len(mine)}")


if __name__ == "__main__":
    main()