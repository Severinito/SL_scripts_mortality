# =====================================================================
# 00 - DRIVER: stages 1 to 3 at both training ends, both Hurst settings,
#      for each country in turn
#
# Stage 1 is H-agnostic and writes the unsuffixed table.  Stage 2 reads
# it, overrides H with 1/2 under --no-lrd, and writes to _lrd / _nolrd,
# leaving the stage 1 table untouched.  Stage 3 needs both ends present,
# the 2003 tables for its backtest and the 2023 tables for its
# projection, so it runs once per Hurst setting after both stage 2 runs.
# Its Monte Carlo covers the 15 backtest years only; the 50-year
# projection is solved by Riccati.
# =====================================================================
import os
import subprocess
import sys
import time

ROOT       = r"C:\Users\sever\Documents\Tu Delft\Thesis\Italy_mortality"
SCRIPT_DIR = os.path.join(ROOT, "scripts_local")
PARAM_DIR  = os.path.join(ROOT, "outputs", "params")

S01 = os.path.join(SCRIPT_DIR, "stage01_fit_abH.py")
S02 = os.path.join(SCRIPT_DIR, "stage02_fit_lsdg.py")
S03 = os.path.join(SCRIPT_DIR, "stage03_projection.py")

COUNTRIES = ["IT", "ES", "FR", "NL", "JP", "US"]
END       = [2003, 2023]
PROJ_END  = 2023                 # origin stage 3 projects from


def run(script, argv, label):
    """Run one script as a subprocess, echoing whatever it printed."""
    cmd = [sys.executable, script] + argv
    print(f"\n{'=' * 78}\n{label}\n{' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r  = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    ok = r.returncode == 0
    print(f"--- {label}: {'ok' if ok else f'FAILED ({r.returncode})'} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return ok


def run_country(country):
    """Stages 1 to 3 for one country; returns the failed labels."""
    failed = []
    for end in END:
        if not run(S01, ["-c", country, "-e", str(end)],
                   f"01 {country} {end}"):
            failed.append(f"01 {country} {end}")
            continue
        for flag, tag in (([], "lrd"), (["--no-lrd"], "nolrd")):
            if not run(S02, ["-c", country, "-e", str(end)] + flag,
                       f"02 {country} {end} {tag}"):
                failed.append(f"02 {country} {end} {tag}")

    if failed:                   # stage 3 needs every table present
        print(f"--- 03 {country}: skipped, stage 1/2 failed", flush=True)
        return failed + [f"03 {country} skipped"]
    for flag, tag in (([], "lrd"), (["--no-lrd"], "nolrd")):
        if not run(S03, ["-c", country, "-e", str(PROJ_END)] + flag,
                   f"03 {country} {tag}"):
            failed.append(f"03 {country} {tag}")
    return failed


t_all    = time.time()
failures = []
for country in COUNTRIES:
    failures += run_country(country)

for country in COUNTRIES:
    print(f"\ntables for {country}:")
    for f in sorted(os.listdir(PARAM_DIR)):
        if f.startswith(f"params_{country}_"):
            p = os.path.join(PARAM_DIR, f)
            print(f"  {f:40s} {os.path.getsize(p):>7} "
                  f"{time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(p)))}")

print(f"\nelapsed {time.time() - t_all:.0f}s")
print("all runs complete" if not failures
      else f"{len(failures)} failures: " + ", ".join(failures))