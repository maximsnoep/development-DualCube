"""Per-mesh outcome of a run directory (argument, default: current directory): writes overview.csv, prints counts per folder."""
import csv, os, re, collections, sys
R = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
STAGES = ["extraction", "certification", "labeling", "placement", "optimization", "drawing", "reconstruction"]
MODULE = {"skeletonize": "extraction", "certify": "certification", "label": "labeling",
          "place_cubes": "placement", "optimize": "optimization", "draw_loops": "drawing",
          "reconstruct": "reconstruction"}
def stage_in_log(name):
    """Running stage of a killed run: the CLI's last "Stage X started" line, else the last log line."""
    try:
        m = re.search(r"Stage (\w+) started", open(f"{R}/logs/{name}.stage").read())
        if m: return m[1]
    except OSError: pass
    try: lines = open(f"{R}/logs/{name}.err", errors="replace").read().splitlines()
    except OSError: return "extraction"
    for line in reversed(lines):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"skeleton::(\w+)", line)
        if m: return MODULE.get(m[1], m[1])
    return "extraction"
def steiner_in_log(name):
    try: m = re.search(r"TetGen added (\d+)", open(f"{R}/logs/{name}.tetgen").read())
    except OSError: return ""
    return m[1] if m else "0"
out = []
for line in open(f"{R}/done.txt"):
    m = re.match(r"(\S+) folder=(\S+) exit=(\d+) seconds=(\d+)(?: killed=(\w+))?", line)
    name, folder, code, secs = m[1], m[2], int(m[3]), int(m[4])
    # Older runs did not record why: SIGKILL then meant the watchdog.
    killed = m[5] or ("timeout" if code == 137 else "none")
    path, row = f"{R}/rows/{name}.csv", None
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        row = rows[0] if rows else None
    if row and row["status"] == "ok":
        reached, outcome, err = "reconstruction", "complete", ""
    elif row:
        failed = row["status"].removeprefix("failed at ")
        reached = STAGES[STAGES.index(failed) - 1] if STAGES.index(failed) else "-"
        outcome, err = f"failed at {failed}", row["error"]
    else:
        failed = stage_in_log(name)
        reached = STAGES[STAGES.index(failed) - 1] if failed in STAGES and STAGES.index(failed) else "-"
        outcome = {"timeout": f"timeout in {failed}", "oom": f"out of memory in {failed}",
                   "other": f"killed in {failed}"}.get(killed, f"crash in {failed} (exit {code})")
        err = ""
    out.append(dict(mesh=name.split("__", 1)[1], folder=folder, outcome=outcome, last_stage_passed=reached,
                    genus=row["genus"] if row else "", faces=row["faces"] if row else "", steiner=row["steiner"] if row else steiner_in_log(name),
                    seconds=secs, peak_mb=row.get("peak_mb", "") if row else "", error=err[:300]))
with open(f"{R}/overview.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
order = ["complete"] + [f"{k} {s}" for s in STAGES[::-1] for k in ("failed at", "timeout in", "out of memory in", "killed in", "crash in")]
key = lambda o: next((i for i, p in enumerate(order) if o.startswith(p)), 99)
counts = collections.defaultdict(collections.Counter)
for o in out: counts[o["folder"]][re.sub(r" \(exit.*", "", o["outcome"])] += 1
cols = sorted({k for c in counts.values() for k in c}, key=key)
print(f"{'folder':<14}" + "".join(f"{c:>{max(len(c),5)+2}}" for c in cols) + "  total")
tot = collections.Counter()
for fo in sorted(counts):
    tot.update(counts[fo])
    print(f"{fo:<14}" + "".join(f"{counts[fo][c]:>{max(len(c),5)+2}}" for c in cols) + f"{sum(counts[fo].values()):>7}")
print(f"{'all':<14}" + "".join(f"{tot[c]:>{max(len(c),5)+2}}" for c in cols) + f"{sum(tot.values()):>7}")
