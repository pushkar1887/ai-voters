"""Run pipeline commands on a Kaggle GPU (T4 x2, vLLM backend) and bring the caches home.

From the laptop (credentials from ~/.kaggle/kaggle.json):
    python kaggle/job.py push   --name boot --cmd "src.llm --smoke 40 --codes L1" [--cmd ...]
    python kaggle/job.py status --name boot
    python kaggle/job.py pull   --name boot          # merges the job's runs/ into ./runs by hash

`push` bundles code, config, prompts, processed data, pools and any existing caches the commands
will extend into the private dataset <user>/ai-voters-code, then pushes this file as a GPU kernel.
On Kaggle the pushed copy installs vLLM, checks the bundle commit, runs each command in its own
process (one model on the GPUs at a time) and leaves runs/ in /kaggle/working for `pull`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

JOB = None          # filled in the pushed copy: {"cmds": [...], "vllm": "...", "commit": "..."}
DATASET = "ai-voters-code"
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ on Kaggle

def kaggle_main(job: dict) -> None:
    work = Path("/kaggle/working")
    hit = next(Path("/kaggle/input").rglob("src/llm.py"), None)
    if hit is None:
        sys.exit("bundle not found under /kaggle/input")
    code = work / "aiv"
    shutil.copytree(hit.parents[1], code, dirs_exist_ok=True)        # writable copy
    commit = (code / "VERSION").read_text().strip()
    if commit != job["commit"]:
        sys.exit(f"stale bundle: dataset has {commit}, job expects {job['commit']}; re-run push")
    runs = work / "runs"
    if (code / "runs").exists():                                    # resume from shipped caches
        shutil.copytree(code / "runs", runs, dirs_exist_ok=True)
    probe = (f"{sys.executable} -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, "
             f"'gpus', torch.cuda.device_count(), torch.cuda.is_available())\"")
    subprocess.run("nvidia-smi", shell=True)                                    # driver and CUDA version
    subprocess.run(probe, shell=True)                                           # preinstalled torch
    subprocess.run(f"pip install -q {job['vllm']}", shell=True, check=True)
    subprocess.run("pip uninstall -y -q torchaudio", shell=True)   # built for the old torch; nothing here needs audio
    subprocess.run(probe + " && python -c \"import vllm; print('vllm', vllm.__version__)\"", shell=True)
    env = {**os.environ, "AIV_RUNS": str(runs), "AIV_COMMIT": commit, "PYTHONUNBUFFERED": "1",
           "VLLM_RPC_TIMEOUT": "600000"}                                 # long thinking steps; default is short
    for cmd in job["cmds"]:
        for attempt in range(1, 4):                                      # a dead engine resumes from the cache
            t0 = time.time()
            print(f"\n=== {cmd}  (attempt {attempt})", flush=True)
            r = subprocess.run([sys.executable, "-m", *cmd.split()], cwd=code, env=env)
            print(f"=== exit {r.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
            if r.returncode == 0:
                break
    shutil.rmtree(code)                                              # output = runs/ only


# ------------------------------------------------------------------ on the laptop

def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config.yaml", "prompts"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def bundle(user: str, version: str, caches: list[str]) -> Path:
    out = ROOT / "kaggle" / "build" / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for d in ("src", "prompts", "data/processed", "data/pools"):
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, out / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("config.yaml", "requirements.txt"):
        shutil.copy(ROOT / f, out / f)
    for c in caches:                                                 # e.g. "S1/L1"
        src = ROOT / "runs" / f"{c}.jsonl"
        if src.exists():
            (out / "runs" / c).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, out / "runs" / f"{c}.jsonl")
    (out / "VERSION").write_text(version)
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET, "id": f"{user}/{DATASET}", "licenses": [{"name": "other"}]}))
    return out


def push(name: str, cmds: list[str], vllm: str, caches: list[str]) -> None:
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    a = api()
    user = a.get_config_value("username")
    version = commit()
    b = bundle(user, version, caches)
    try:
        a.dataset_create_version(str(b), version_notes=f"{name} @ {version}", dir_mode="zip", quiet=True)
    except Exception:                                                # first push: dataset does not exist
        a.dataset_create_new(str(b), public=False, dir_mode="zip", quiet=True)
    for _ in range(60):                                              # kernel must mount the new version
        if str(a.dataset_status(f"{user}/{DATASET}")).lower() == "ready":
            break
        time.sleep(10)
    job = {"cmds": cmds, "vllm": vllm, "commit": version}
    text = Path(__file__).read_text(encoding="utf-8").replace("JOB = None", f"JOB = {job!r}", 1)
    req = ApiSaveKernelRequest()
    req.slug, req.new_title, req.text = f"{user}/aiv-{name}", f"aiv-{name}", text
    req.language, req.kernel_type, req.is_private = "python", "script", True
    req.enable_gpu, req.enable_internet, req.machine_shape = True, True, "NvidiaTeslaT4"
    req.dataset_data_sources = [f"{user}/{DATASET}"]
    with a.build_kaggle_client() as k:
        res = k.kernels.kernels_api_client.save_kernel(req)
    print(res.error or f"pushed {res.url}  (bundle {version})")


def status(name: str) -> None:
    a = api()
    s = a.kernels_status(f"{a.get_config_value('username')}/aiv-{name}")
    print(getattr(s, "status", s), getattr(s, "failure_message", "") or "")


def pull(name: str) -> None:
    """Download the kernel output and merge its caches into ./runs (dedupe by request hash)."""
    a = api()
    dst = ROOT / "kaggle" / "build" / f"out-{name}"
    shutil.rmtree(dst, ignore_errors=True)
    a.kernels_output(f"{a.get_config_value('username')}/aiv-{name}", str(dst), force=True, quiet=True)
    for f in sorted(dst.rglob("*.jsonl")):
        rel = f.relative_to(next(p for p in f.parents if p.name == "runs"))
        target = ROOT / "runs" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        have = set()
        if target.exists():
            with open(target, encoding="utf-8") as t:
                have = {json.loads(line)["hash"] for line in t if line.strip()}
        new = [line for line in open(f, encoding="utf-8") if line.strip() and json.loads(line)["hash"] not in have]
        with open(target, "a", encoding="utf-8") as t:
            t.writelines(new)
        print(f"{rel}: +{len(new)} records")
    for f in dst.rglob("*.manifest.json"):
        rel = f.relative_to(next(p for p in f.parents if p.name == "runs"))
        shutil.copy(f, ROOT / "runs" / rel)
    log = next(dst.glob("*.log"), None)
    print(f"log: {log}" if log else "no log file")


if __name__ == "__main__":
    if JOB is not None:
        kaggle_main(JOB)
        sys.exit()
    if os.name == "nt" and not sys.flags.utf8_mode:      # the kaggle client writes kernel logs in the locale codec
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "status", "pull"])
    ap.add_argument("--name", required=True, help="kernel name suffix, e.g. s1-l1")
    ap.add_argument("--cmd", action="append", default=[], help="python -m command; repeat for several")
    ap.add_argument("--caches", nargs="*", default=[], help="caches to ship for resume, e.g. S1/L1")
    ap.add_argument("--vllm", default="vllm==0.31.0", help="pip spec for vLLM (0.31.0 boots on T4 x2, Oct 6)")
    a = ap.parse_args()
    if a.action == "push":
        push(a.name, a.cmd, a.vllm, a.caches)
    elif a.action == "status":
        status(a.name)
    else:
        pull(a.name)
