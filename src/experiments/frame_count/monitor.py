"""Run one pipeline stage and record what it cost: wall time, CPU time, peak RAM and peak GPU memory.

The command runs under /usr/bin/time -v (wall, user and system CPU seconds, the largest single process's peak
RSS), inside a systemd-run scope with a memory cap when systemd-run exists. While it runs, a thread samples its
process tree every interval: RSS summed over the tree (COLMAP and the matchers start helper processes), CPU %
summed over the tree (100 = one core busy), and from nvidia-smi the GPU memory held by the tree's processes and
the GPU's utilisation. The GPU memory of everything else (desktop, editor) is left out, so the peak is the stage's.
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time

import psutil


def gpu_sample():
    """({pid: MiB} of GPU compute processes, utilisation %), or ({}, None) without nvidia-smi."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}, None
    used = {}
    for line in apps.strip().splitlines():
        pid, mib = (part.strip() for part in line.split(","))
        if pid.isdigit() and mib.isdigit():
            used[int(pid)] = int(mib)
    return used, float(util.strip().splitlines()[0]) if util.strip() else None


def parse_time_v(path):
    """The fields of /usr/bin/time -v this module reports."""
    fields = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            key, _, value = line.strip().rpartition(": ")
            fields[key] = value
    wall = fields.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", "0:0")
    seconds = sum(float(part) * 60 ** k for k, part in enumerate(reversed(wall.split(":"))))
    return {"wall_s": round(seconds, 1), "user_s": float(fields.get("User time (seconds)", 0)),
            "system_s": float(fields.get("System time (seconds)", 0)),
            "cpu_percent": float(fields.get("Percent of CPU this job got", "0%").rstrip("%") or 0),
            "max_process_rss_mb": round(int(fields.get("Maximum resident set size (kbytes)", 0)) / 1024, 1),
            "exit_status": int(fields.get("Exit status", 0))}


def run(command, cwd, memory_max=None, interval=0.5, log_path=None):
    """Run command (a list); returns the cost record, with "returncode". Output goes to log_path (and the console)."""
    timing = tempfile.NamedTemporaryFile(prefix="time_v_", suffix=".txt", delete=False).name
    full = ["/usr/bin/time", "-v", "-o", timing, *command]
    if memory_max and shutil.which("systemd-run"):
        full = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={memory_max}", "-p", "MemorySwapMax=0",
                *full]
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    start = time.time()
    proc = subprocess.Popen(full, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    peaks = {"tree_rss_mb": 0.0, "gpu_mib": 0, "cpu_percent": 0.0, "gpu_util_percent": 0.0}
    sums = {"cpu_percent": 0.0, "gpu_util_percent": 0.0, "samples": 0}
    stop = threading.Event()

    def sample():
        tracked = {}
        root = psutil.Process(proc.pid)
        while not stop.is_set():
            try:
                tree = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            rss, cpu = 0, 0.0
            for p in tree:
                p = tracked.setdefault(p.pid, p)
                try:
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(None)
                except psutil.NoSuchProcess:
                    pass
            used, util = gpu_sample()
            gpu = sum(mib for pid, mib in used.items() if pid in tracked)
            peaks["tree_rss_mb"] = max(peaks["tree_rss_mb"], rss / 2 ** 20)
            peaks["gpu_mib"] = max(peaks["gpu_mib"], gpu)
            peaks["cpu_percent"] = max(peaks["cpu_percent"], cpu)
            sums["cpu_percent"] += cpu
            sums["samples"] += 1
            if util is not None:
                peaks["gpu_util_percent"] = max(peaks["gpu_util_percent"], util)
                sums["gpu_util_percent"] += util
            stop.wait(interval)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log:
            log.write(line)
    proc.wait()
    stop.set()
    sampler.join()
    if log:
        log.close()
    record = {"returncode": proc.returncode, "wall_s": round(time.time() - start, 1)}
    if os.path.exists(timing) and os.path.getsize(timing):
        record["time_v"] = parse_time_v(timing)
    os.remove(timing)
    n = max(sums["samples"], 1)
    record.update({"peak_tree_rss_mb": round(peaks["tree_rss_mb"], 1), "peak_gpu_mib": peaks["gpu_mib"],
                   "peak_cpu_percent": round(peaks["cpu_percent"], 1),
                   "mean_cpu_percent": round(sums["cpu_percent"] / n, 1),
                   "peak_gpu_util_percent": peaks["gpu_util_percent"],
                   "mean_gpu_util_percent": round(sums["gpu_util_percent"] / n, 1), "samples": sums["samples"],
                   "memory_max": memory_max, "command": command})
    return record
