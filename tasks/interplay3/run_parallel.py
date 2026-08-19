"""
tasks.interplay3.run_parallel
=============================

Run the three conditions as three processes and merge the results.

The layer-2 mask set is a few megabytes, so a single process spends most of
its time inside one BLAS call that does not scale to every core.  Splitting by
condition turns three sequential runs into three concurrent ones, and each
process keeps its own cache file, which is merged here.

    python -m tasks.interplay3.run_parallel [--seeds 2] [--preset default]
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tasks.interplay3.config import CONDITIONS

OUT_DIR = Path(__file__).resolve().parent


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--preset", default="default")
    p.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    p.add_argument("--threads", type=int, default=5,
                   help="BLAS threads per process")
    args = p.parse_args(argv)

    env_extra = {k: str(args.threads) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS")}

    import os
    procs = []
    t0 = time.time()
    for cond in args.conditions:
        env = dict(os.environ, **env_extra)
        log = open(OUT_DIR / f"run_{cond}.log", "w")
        procs.append((cond, subprocess.Popen(
            [sys.executable, "-u", "-m", "tasks.interplay3.interplay3",
             "--preset", args.preset, "--seeds", str(args.seeds),
             "--conditions", cond, "--tag", f"_{cond}", "--no-figures"],
            cwd=str(_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT),
            log))

    failed = []
    for cond, proc, log in procs:
        rc = proc.wait()
        log.close()
        print(f"  {cond}: exit {rc}  ({time.time() - t0:.0f} s)", flush=True)
        if rc != 0:
            failed.append(cond)
    if failed:
        raise SystemExit(f"conditions failed: {failed}; see run_*.log")

    # ---- merge ----
    store = None
    for cond in args.conditions:
        with open(OUT_DIR / f"results_{cond}.pkl", "rb") as fh:
            part = pickle.load(fh)
        if store is None:
            store = part
        else:
            store["conditions"].update(part["conditions"])
    with open(OUT_DIR / "results.pkl", "wb") as fh:
        pickle.dump(store, fh)

    from tasks.interplay3.figures import make_figures
    make_figures(store)
    print(f"  total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
