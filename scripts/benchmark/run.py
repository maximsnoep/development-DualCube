#!/usr/bin/env python3
"""DualCube in the labeling benchmark of polycube-to-hexmesh (../polycube-to-hexmesh/benchmark.py): runs the skeleton
pipeline on the meshes that benchmark selects, and scores and writes the results as benchmark.py does for its methods,
as the method `dualcube`. The scores come from benchmark.py itself (imported), so they are the same statistics.

    run.py [run] [--out RUN] [--filter GLOB] [--limit N] [--genus-zero] [--timeout 1200] [--force | --retry] [--jobs N]
    run.py summary --out RUN [--with ../polycube-to-hexmesh/out]

A run creates its folder (default runs/<date>_<commit>/), builds the CLI into it and records how it was made; started
again with the same --out, it continues where it stopped with that same binary: finished results are kept, unless
--force, or --retry for those that failed (as in benchmark.py).

The folder:
  run.json                  commits of this repository, the benchmark and the dataset, the tools, every invocation
  dualcube.patch            this repository's uncommitted changes (untracked files included) when the run was created
  cli                       the binary every mesh runs with
  results.csv               benchmark.py's results.csv format: append its rows to the benchmark's to combine them
  stages.csv                per mesh: the pipeline's own report (stage reached, seconds per stage, peak memory)
  <category>/<mesh>/        as the benchmark's out/ (copy the category folders into it to merge; benchmark.py then
                            needs "dualcube" in ALL_ALGOS to read them):
    dualcube.json           the scores, or why it failed
    dualcube.dc             the solution, with the skeleton stages: open it in the GUI
    dualcube.obj            the surface the labels are of: our refined mesh (the benchmark scores Fu et al. on fu.obj
                            the same way); the input with the label boundaries cut in, and TetGen's Steiner points
    dualcube.flag           one label per triangle of dualcube.obj
    dualcube.polycube.obj   the benchmark's naive polycube of the labeling (up to its face limit)
    dualcube.log            the pipeline's output
    dualcube.stages.csv     the pipeline's report row

A run that fails has status `timeout` (benchmark.py's word), `failed at <stage>: <error>`, `out of memory in <stage>`,
or `crash in <stage> (exit <code>)`. time_s is the pipeline's wall time, everything included (TetGen too). Run one mesh
at a time (the default) for times comparable to the benchmark's; more jobs share the cores and the memory.

Needs numpy and scipy (for benchmark.py; here in a virtualenv: ~/.cache/dualcube-bench/bin/python run.py ...), TetGen and DiskScissors (TETGEN_EXECUTABLE, DISKSCISSORS_EXECUTABLE; by
default in ~/.local/bin), and cargo.
"""
import argparse, contextlib, csv, datetime, glob, json, os, platform, re, shutil, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALGO = "dualcube"
# The pipeline writes these into a scratch folder, named after the mesh; they are moved to dualcube.<extension>.
SAVED = ("dc", "obj", "flag")
OUTPUTS = [f"{ALGO}.{e}" for e in ("json", "log", "stages.csv", "polycube.obj") + SAVED]
# The processes running, killed on Ctrl-C; their runs then have no result.
LIVE, LIVE_LOCK, STOPPING = set(), threading.Lock(), threading.Event()
# (Before benchmark.py sets its OPENBLAS_NUM_THREADS.)
ENV = dict(os.environ)


def load_benchmark(path):
    sys.path.insert(0, str(Path(path).resolve()))
    try:
        import benchmark
    except ModuleNotFoundError as error:
        raise SystemExit(f"cannot load benchmark.py from {path}: {error} (it needs numpy and scipy)")
    finally:
        sys.path.pop(0)
    return benchmark


def git(folder, *args):
    result = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True)
    return result.stdout.rstrip("\n") if result.returncode == 0 else None


def patch():
    """The uncommitted changes of this repository, as a patch, untracked files included."""
    text = git(REPO, "diff", "HEAD", "--binary") or ""
    for name in (git(REPO, "ls-files", "--others", "--exclude-standard") or "").splitlines():
        # (Exit code 1 when the files differ: always here.)
        text += "\n" + subprocess.run(["git", "-C", str(REPO), "diff", "--no-index", "--binary", "/dev/null", name], capture_output=True, text=True).stdout
    return text


def tools():
    return {
        "TETGEN_EXECUTABLE": ENV.get("TETGEN_EXECUTABLE", str(Path.home() / ".local/bin/tetgen")),
        "DISKSCISSORS_EXECUTABLE": ENV.get("DISKSCISSORS_EXECUTABLE", str(Path.home() / ".local/bin/diskscissors")),
    }


def create(out, args, bench):
    """A new run folder: the binary, the patch, and run.json."""
    out.mkdir(parents=True, exist_ok=True)
    cargo = os.environ.get("CARGO", shutil.which("cargo") or "cargo")
    subprocess.run([cargo, "build", "--release", "-q", "-p", "cli", "--manifest-path", str(REPO / "Cargo.toml")], check=True)
    shutil.copy2(REPO / "target/release/cli", out / "cli")
    (out / f"{ALGO}.patch").write_text(patch())
    dataset = Path(args.dataset)
    record = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "dualcube": {"commit": git(REPO, "rev-parse", "HEAD"), "subject": git(REPO, "log", "-1", "--format=%s"), "branch": git(REPO, "branch", "--show-current"), "uncommitted": (git(REPO, "status", "--short") or "").splitlines()},
        "benchmark": {"path": str(Path(bench.__file__).resolve()), "commit": git(Path(bench.__file__).parent, "rev-parse", "HEAD"), "uncommitted": (git(Path(bench.__file__).parent, "status", "--short") or "").splitlines()},
        "dataset": {"path": str(dataset.resolve()), "commit": git(dataset, "rev-parse", "HEAD"), "tag": git(dataset, "describe", "--tags")},
        "tools": tools(),
        "host": {"node": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count()},
        "invocations": [],
    }
    (out / "run.json").write_text(json.dumps(record, indent=1))


def last_stage(log):
    """The stage a killed run was in: the pipeline's last "Stage X started" line."""
    try:
        found = re.findall(r"Stage (\w+) started", log.read_text(errors="replace"))
    except OSError:
        return "deflation"
    return found[-1] if found else "deflation"


def out_of_memory(pid):
    # (The kernel's line can come after the process is reaped: look again for a few seconds.)
    for attempt in range(10):
        log = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
        if re.search(rf"Out of memory: Killed process {pid} \(|oom-kill:.*pid={pid},", log):
            return True
        time.sleep(0.5)
    return False


def pipeline(cli, mesh, work, timeout, env):
    """Runs the pipeline on a mesh. Returns its status and seconds; a full run leaves dualcube.{dc,obj,flag} in `work`."""
    scratch = work / f"{ALGO}.tmp"
    shutil.rmtree(scratch, ignore_errors=True)
    log, report = work / f"{ALGO}.log", work / f"{ALGO}.stages.csv"
    started = time.time()
    with open(log, "w") as output:
        # (Its own process group, so that a timeout kills DiskScissors and TetGen too.)
        process = subprocess.Popen([cli, "stages", "--input", mesh.resolve(), "--report", report.resolve(), "--save", scratch.resolve()], stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        with LIVE_LOCK:
            LIVE.add(process)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = None
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            with LIVE_LOCK:
                LIVE.discard(process)
    if STOPPING.is_set():
        raise KeyboardInterrupt
    seconds = time.time() - started
    for leftover in glob.glob(f"/tmp/dualcube-certify-{process.pid}-*") + glob.glob(f"/tmp/dualcube-tetgen-{process.pid}-*"):
        shutil.rmtree(leftover, ignore_errors=True)
    rows = list(csv.DictReader(open(report))) if report.is_file() else []
    if code is None:
        status = "timeout"
    elif rows and rows[0]["status"] == "ok":
        for extension in SAVED:
            shutil.move(scratch / f"{mesh.stem}.{extension}", work / f"{ALGO}.{extension}")
        status = "ok"
    elif rows:
        error = rows[0]["error"].strip().splitlines()
        status = f"{rows[0]['status']}: {error[0] if error else ''}"[:300]
    elif code in (-signal.SIGINT, -signal.SIGTERM):
        raise KeyboardInterrupt
    elif code == -signal.SIGKILL and out_of_memory(process.pid):
        status = f"out of memory in {last_stage(log)}"
    else:
        status = f"crash in {last_stage(log)} (exit {code})"
    shutil.rmtree(scratch, ignore_errors=True)
    return status, seconds


def run_mesh(bench, cli, mesh, relative, work, timeout, env):
    """One mesh, as benchmark.py runs one algorithm on it: dualcube.json and the files next to it."""
    work.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (work / name).unlink(missing_ok=True)
    result = {"mesh": relative.name, "category": relative.parent.as_posix(), "algo": ALGO, "status": "ok"}
    status, seconds = pipeline(cli, mesh, work, timeout, env)
    result.update(status=status, time_s=round(seconds, 2))
    if status == "ok":
        try:
            V, F = bench.read_obj(work / f"{ALGO}.obj")
            labels = bench.read_flag(work / f"{ALGO}.flag")
            result.update(bench.score(V, F, labels))
            if len(F) <= bench.POLYCUBE_MAX_FACES:
                result.update(bench._polycube(V, F, labels, work / f"{ALGO}.polycube.obj"))
        except ValueError as failure:
            result["status"] = str(failure)
    (work / f"{ALGO}.json").write_text(json.dumps(result, indent=1))
    return result


def command_run(args):
    bench = load_benchmark(args.benchmark)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = Path(args.out or REPO / "runs" / f"{stamp}_{git(REPO, 'rev-parse', '--short', 'HEAD')}")
    dataset = Path(args.dataset)
    if not (out / "run.json").is_file():
        create(out, args, bench)
    record = json.loads((out / "run.json").read_text())
    if record["invocations"]:
        print(f"continuing the run in {out}, with its binary (of {record['dualcube']['commit'][:7]}{', with uncommitted changes' if record['dualcube']['uncommitted'] else ''})")
    record["invocations"].append({"started": datetime.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:], "dualcube_commit": git(REPO, "rev-parse", "HEAD")})
    (out / "run.json").write_text(json.dumps(record, indent=1))

    env = dict(ENV, **tools())
    env.setdefault("DISKSCISSORS_JOBS", str(max(1, (os.cpu_count() or 4) // args.jobs)))
    meshes = bench._meshes(args)
    todo = []
    for mesh in meshes:
        relative = mesh.relative_to(dataset).with_suffix("")
        target = out / relative / f"{ALGO}.json"
        result = json.loads(target.read_text()) if target.is_file() else None
        if result and not args.force and not (args.retry and result["status"] != "ok"):
            print(f"{relative.as_posix():45s} (kept) {bench._line(result)}")
            continue
        todo.append((mesh, relative))
    print(f"{len(meshes)} meshes, {len(todo)} to run, {args.jobs} at a time, results in {out}", flush=True)

    def one(mesh, relative):
        return relative, run_mesh(bench, (out / "cli").resolve(), mesh, relative, out / relative, args.timeout, env)

    def interrupt(number, frame):
        raise KeyboardInterrupt

    # (A kill stops the run as Ctrl-C does.)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    futures = []
    try:
        futures = [pool.submit(one, *item) for item in todo]
        for number, future in enumerate(as_completed(futures), 1):
            relative, result = future.result()
            print(f"[{number}/{len(todo)}] {relative.as_posix():45s} {bench._line(result)}", flush=True)
    except KeyboardInterrupt:
        # (No result for those cut off: they run again when the run continues.)
        STOPPING.set()
        pool.shutdown(wait=False, cancel_futures=True)
        with LIVE_LOCK:
            for process in LIVE:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        raise SystemExit("stopped")
    pool.shutdown()
    args.out = str(out)
    command_summary(args, bench)


def results_of(folder, algos):
    return [json.loads(path.read_text()) for algo in algos for path in sorted(Path(folder).rglob(f"{algo}.json"))]


def write_results(path, results, bench):
    """As benchmark.py's results.csv."""
    results = sorted(results, key=lambda r: (r["category"], r["mesh"], r["algo"]))
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "mesh", "algo", "status"] + bench.METRICS)
        for r in results:
            writer.writerow([r["category"], r["mesh"], r["algo"], r["status"]] + [r.get(m, "") if not isinstance(r.get(m), float) else f"{r[m]:.4f}" for m in bench.METRICS])


def write_stages(out, results):
    """Per mesh: the status, and the pipeline's report row."""
    rows = []
    for r in results:
        report = out / r["category"] / r["mesh"] / f"{ALGO}.stages.csv"
        row = next(csv.DictReader(open(report)), {}) if report.is_file() else {}
        row.pop("input", None)
        rows.append({"category": r["category"], "mesh": r["mesh"], "outcome": r["status"], "time_s": r.get("time_s", ""), **{f"pipeline_{k}" if k in ("status", "faces") else k: v for k, v in row.items()}})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / "stages.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command_summary(args, bench=None):
    bench = bench or load_benchmark(args.benchmark)
    out = Path(args.out)
    results = results_of(out, [ALGO])
    if not results:
        print("no results")
        return
    write_results(out / "results.csv", results, bench)
    write_stages(out, results)
    kinds = {}
    for r in results:
        kind = "ok" if r["status"] == "ok" and r["invalid_charts"] == 0 else "invalid" if r["status"] == "ok" else re.sub(r":.*| \(exit.*", "", r["status"])
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"\n{len(results)} meshes: " + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items(), key=lambda item: -item[1])))
    print(f"wrote {out / 'results.csv'} and {out / 'stages.csv'}")
    if args.others:
        # The benchmark's table, with dualcube among its methods.
        combined = results + results_of(args.others, bench.ALL_ALGOS)
        bench.ALL_ALGOS = tuple(dict.fromkeys(bench.ALL_ALGOS + (ALGO,)))
        write_results(out / "combined.csv", combined, bench)
        bench._table(combined)
        print(f"\nwrote {out / 'combined.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the pipeline and score its labelings")
    run.add_argument("--dataset", default=str(REPO.parent / "meshes" / "dataset"), help="folder with .obj meshes (searched recursively), as benchmark.py's")
    run.add_argument("--genus-zero", action="store_true", help="also the meshes of genus 0 [default: genus >= 1 only]")
    run.add_argument("--filter", help="only meshes whose path (category/name.obj) contains one of these comma-separated globs")
    run.add_argument("--limit", type=int, help="only the first N meshes")
    run.add_argument("--timeout", type=int, default=1200, help="seconds per mesh [default: 1200, as benchmark.py]")
    run.add_argument("--force", action="store_true", help="run again what already has a result")
    run.add_argument("--retry", action="store_true", help="run again what failed (e.g. with a longer --timeout)")
    run.add_argument("--jobs", type=int, default=1, help="meshes at a time [default: 1, for comparable times]; no memory budget")
    run.set_defaults(function=command_run)
    summary = commands.add_parser("summary", help="write results.csv and stages.csv again")
    summary.set_defaults(function=command_summary)
    for sub in (run, summary):
        sub.add_argument("--out", required=sub is summary, help="the run folder [default for run: runs/<date>_<commit>/]")
        sub.add_argument("--benchmark", default=str(REPO.parent / "polycube-to-hexmesh"), help="the polycube-to-hexmesh repository [default: ../polycube-to-hexmesh]")
        sub.add_argument("--with", dest="others", help="the benchmark's results folder (e.g. ../polycube-to-hexmesh/out): also print its table with dualcube, and write combined.csv")
    argv = sys.argv[1:]
    if not argv or argv[0] not in commands.choices and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
