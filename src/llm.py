"""One runner for every LLM call in the project.

A NIM key pool (one pacer per key; a failed call retries on another key), a vLLM
batch backend for the Kaggle GPU models, an append-only hash cache and the reply
parser. Every record lands in runs/<stage>/<name>.jsonl, so re-running anything
costs nothing and a crash resumes where it stopped.

    python -m src.llm --check-keys [MODEL]     one tiny call per key, masked report
    python -m src.llm --models ID [ID ...]     which NIM model strings answer
    python -m src.llm --smoke                  5 claims x every NIM model (gate G1)
    python -m src.llm --probe [--minutes 5]    rate probe: 3 keys at 40 RPM
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
PROMPTS = yaml.safe_load((ROOT / "prompts" / f"{CFG['prompt_version']}.yaml").read_text(encoding="utf-8"))
RUNS = Path(os.environ.get("AIV_RUNS", ROOT / "runs"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render(name: str, key: str | None = None, **slots) -> str:
    """Fill a template from prompts/<version>.yaml; `key` picks a paraphrase or variant."""
    t = PROMPTS[name]
    if isinstance(t, dict):
        t = t[key or "a"]
    return Template(t).substitute(**slots)


def rng(*parts) -> random.Random:
    """Derived randomness: the same parts always give the same draws, so re-runs hit the cache."""
    return random.Random("|".join(str(p) for p in (CFG["seed"], *parts)))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ keys and pool

KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}")


def load_keys() -> list[str]:
    """Keys come from NIM_KEYS or the keys file, matched by pattern so the file layout does
    not matter. NIM_KEY_SLICE="a:b" gives this process its own slice (one writer per key)."""
    raw = os.environ.get("NIM_KEYS")
    if raw is None:
        path = ROOT / CFG["nim"]["keys_file"]
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
    keys = list(dict.fromkeys(KEY_RE.findall(raw)))
    if sl := os.environ.get("NIM_KEY_SLICE"):
        a, b = (int(x) if x else None for x in sl.split(":"))
        keys = keys[a:b]
    if not keys:
        sys.exit("No NIM keys: set NIM_KEYS or put them in the file named by nim.keys_file.")
    return keys


def fp(key: str) -> str:
    """Key fingerprint for logs; key characters are never printed."""
    return hashlib.sha256(key.encode()).hexdigest()[:6]


class BadRequest(Exception):
    """400/404/410/422: the request or the model is wrong, so retrying cannot help."""


class Pool:
    """Every NIM key behind its own pacer; each call takes the key that is free soonest."""

    def __init__(self, keys: list[str], rpm: float):
        self.keys, self.gap = keys, 60.0 / rpm
        self.free_at = [0.0] * len(keys)
        self.dead: set[int] = set()
        self.stats: Counter = Counter()
        self.lock = asyncio.Lock()
        self.http = httpx.AsyncClient(base_url=CFG["nim"]["base_url"], timeout=CFG["nim"]["timeout_s"],
                                      limits=httpx.Limits(max_connections=CFG["nim"]["concurrency"]))

    async def _take_key(self) -> int:
        async with self.lock:
            live = [i for i in range(len(self.keys)) if i not in self.dead]
            if not live:
                raise RuntimeError("every NIM key was rejected (401/403)")
            i = min(live, key=self.free_at.__getitem__)
            t_now = time.monotonic()
            t = max(t_now, self.free_at[i])
            self.free_at[i] = t + self.gap
        await asyncio.sleep(t - t_now)
        return i

    async def post(self, payload: dict, tries: int = 8) -> dict:
        for attempt in range(tries):
            i = await self._take_key()
            try:
                r = await self.http.post("/chat/completions", json=payload,
                                         headers={"Authorization": f"Bearer {self.keys[i]}"})
            except httpx.HTTPError:
                self.stats["net_error"] += 1
            else:
                self.stats[r.status_code] += 1
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (401, 403):
                    self.dead.add(i)
                    continue
                if r.status_code == 429:
                    self.free_at[i] = max(self.free_at[i], time.monotonic() + 20)
                    continue
                if r.status_code in (400, 404, 410, 422):
                    raise BadRequest(f"{r.status_code} {r.text[:200]}")
            await asyncio.sleep(min(30, 2 ** attempt) + random.random())
        raise RuntimeError(f"no answer after {tries} tries")

    async def close(self) -> None:
        await self.http.aclose()


_POOL: Pool | None = None


def get_pool() -> Pool:
    global _POOL
    if _POOL is None:
        _POOL = Pool(load_keys(), float(os.environ.get("NIM_RPM", CFG["nim"]["rpm_per_key"])))
    return _POOL


async def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        await _POOL.close()
        _POOL = None


# ------------------------------------------------------------------ cache

class Cache:
    """Append-only JSONL indexed by request hash. A torn last line from a crash is skipped."""

    def __init__(self, path: Path):
        self.idx: dict[str, dict] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size:
            with open(path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.seek(0, os.SEEK_END)
                    f.write(b"\n")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.idx[rec["hash"]] = rec
        self.f = open(path, "a", encoding="utf-8")

    def get(self, h: str) -> dict | None:
        return self.idx.get(h)

    def put(self, rec: dict) -> None:
        self.idx[rec["hash"]] = rec
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()


_CACHES: dict[Path, Cache] = {}


def cache_for(stage: str, name: str) -> Cache:
    path = RUNS / stage / f"{name}.jsonl"
    if path not in _CACHES:
        _CACHES[path] = Cache(path)
    return _CACHES[path]


# ------------------------------------------------------------------ requests and parsing

def spec(messages: list[dict], *, kind: str = "claim", decoding: str = "greedy", seed_tag="",
         meta: dict | None = None, max_tokens: int | None = None) -> dict:
    """One request as a stage describes it; the runner adds the model's settings."""
    return {"messages": messages, "kind": kind, "decoding": decoding, "seed_tag": str(seed_tag),
            "meta": meta or {}, "max_tokens": max_tokens}


def payload_for(model: str, s: dict) -> dict:
    """The model's `extra` (reasoning switches) goes into the payload, so it is part of the cache key."""
    m = CFG["models"][model]
    dec = CFG["decoding"][s["decoding"]]
    return {"model": m["id"], "messages": s["messages"], "temperature": dec["temperature"], "top_p": dec["top_p"],
            "max_tokens": s["max_tokens"] or m.get("max_tokens") or CFG["decoding"]["max_tokens"],
            **(m.get("extra") or {})}


def req_hash(p: dict, seed_tag: str) -> str:
    """seed_tag separates repeated samples of the same prompt at T > 0."""
    blob = json.dumps({**p, "seed_tag": seed_tag}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()


FIELDS = {   # kind -> required field -> regex fallback when the reply is not clean JSON
    "claim": {"verdict": r'"?verdict"?\s*:\s*"?(TRUE|FALSE)\b', "p_true": r'"?p_true"?\s*:\s*"?(\d{1,3})'},
    "urn": {"urn": r'"?urn"?\s*:\s*"?(A|B)\b', "p_a": r'"?p_a"?\s*:\s*"?(\d{1,3})'},
    "vote": {"vote": r'"?vote"?\s*:\s*"?(UP|DOWN|NONE)\b'},
    "recog": {"seen": r'"?seen"?\s*:\s*"?(YES|NO)\b', "rating": r'"?rating"?\s*:\s*"([^"]{0,40})"'},
    "judge": {"answer": r"\b(YES|NO)\b"},
}
CHOICES = {"verdict": {"TRUE", "FALSE"}, "urn": {"A", "B"}, "vote": {"UP", "DOWN", "NONE"},
           "seen": {"YES", "NO"}, "answer": {"YES", "NO"}}
NUMBERS = {"p_true", "p_a"}
RETRY_MSG = "Your previous reply was not valid JSON. Reply with the JSON object only."


def answer_part(text: str | None) -> str:
    """Drop any reasoning before </think>."""
    return (text or "").rsplit("</think>", 1)[-1]


def first_json(t: str) -> dict:
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = dec.raw_decode(t, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return {str(k).lower(): v for k, v in obj.items()}
    return {}


def parse(text: str | None, kind: str) -> dict | None:
    """JSON first, then one regex per field. None if a required field is missing or invalid."""
    t = answer_part(text)
    obj = first_json(t)
    out = {}
    for field, rx in FIELDS[kind].items():
        v = obj.get(field)
        if v is None or (field in CHOICES and str(v).strip().upper() not in CHOICES[field]):
            m = re.search(rx, t, re.I)
            v = m.group(1) if m else None
        if v is None:
            return None
        if field in NUMBERS:
            num = re.search(r"\d+(?:\.\d+)?", str(v))
            if not num:
                return None
            v = max(0, min(100, round(float(num.group()))))
        elif field in CHOICES:
            v = str(v).strip().upper()
        out[field] = v
    if "reason" in obj:
        out["reason"] = str(obj["reason"])[:500]
    return out


def make_record(h, stage, model, backend, p, s, text, reasoning, finish, usage, parsed, status,
                retry_text, latency, **extra) -> dict:
    return {"hash": h, "stage": stage, "model": model, "model_id": p["model"], "backend": backend,
            "prompt_ver": CFG["prompt_version"], "kind": s["kind"], "decoding": s["decoding"],
            "seed_tag": s["seed_tag"], "meta": s["meta"], "messages": p["messages"], "text": text,
            "retry_text": retry_text, "reasoning_chars": len(reasoning or ""), "finish": finish,
            "usage": usage, "parsed": parsed, "parse_status": status, "latency_s": round(latency, 2),
            "ts": now(), **extra}


# ------------------------------------------------------------------ NIM backend

def _unpack(r: dict):
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return msg.get("content"), reasoning, ch.get("finish_reason"), r.get("usage")


async def _nim_one(pool: Pool, cache: Cache, stage: str, model: str, s: dict) -> tuple[dict, bool]:
    p = payload_for(model, s)
    h = req_hash(p, s["seed_tag"])
    if (hit := cache.get(h)) is not None:
        return hit, True
    t0 = time.monotonic()
    try:
        text, reasoning, finish, usage = _unpack(await pool.post(p))
        parsed, status, retry_text = parse(text, s["kind"]), "ok", None
        if parsed is None:
            msgs = p["messages"] + [{"role": "assistant", "content": text or ""},
                                    {"role": "user", "content": RETRY_MSG}]
            retry_text = _unpack(await pool.post({**p, "messages": msgs}))[0]
            parsed = parse(retry_text, s["kind"])
            status = "retry_ok" if parsed else "fail"
    except (BadRequest, RuntimeError) as e:          # not cached, so a later run tries again
        return {"hash": h, "stage": stage, "model": model, "meta": s["meta"], "parsed": None,
                "parse_status": "error", "error": str(e)[:300]}, False
    rec = make_record(h, stage, model, "nim", p, s, text, reasoning, finish, usage, parsed, status,
                      retry_text, time.monotonic() - t0)
    cache.put(rec)
    return rec, False


class Monitor:
    """A QC line every 1,000 requests: parse rate, TRUE share, cache hits, 429s, calls per minute."""

    def __init__(self, label: str, total: int, pool: Pool | None):
        self.label, self.total, self.pool = label, total, pool
        self.c: Counter = Counter()
        self.t0 = time.monotonic()

    def add(self, rec: dict, hit: bool) -> None:
        c = self.c
        c["done"] += 1
        c["hit"] += hit
        c[rec.get("parse_status", "error")] += 1
        if v := (rec.get("parsed") or {}).get("verdict"):
            c["verdicts"] += 1
            c["true"] += v == "TRUE"
        if c["done"] % 1000 == 0 or c["done"] == self.total:
            self.line()

    def line(self) -> None:
        c, minutes = self.c, max(1e-9, time.monotonic() - self.t0) / 60
        new = c["done"] - c["hit"]
        msg = (f"[{self.label}] {c['done']}/{self.total}  new {new}  hits {c['hit']}  "
               f"parsed {(c['ok'] + c['retry_ok']) / max(1, c['done']):.1%}  fail {c['fail']}  err {c['error']}")
        if c["verdicts"]:
            msg += f"  TRUE {c['true'] / c['verdicts']:.0%}"
        if self.pool:
            msg += f"  429s {self.pool.stats[429]}  rpm {new / minutes:.0f}"
        print(msg, flush=True)


def git_commit() -> str:
    if c := os.environ.get("AIV_COMMIT"):
        return c
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "none"
    except OSError:
        return "none"


def write_manifest(stage: str, name: str, models, mon: Monitor) -> None:
    """runs/<stage>/<name>.manifest.json: what produced the cache file next to it."""
    c = mon.c
    models = [models] if isinstance(models, str) else sorted(models)
    man = {"stage": stage, "name": name, "models": {m: CFG["models"][m]["id"] for m in models},
           "prompt_ver": CFG["prompt_version"], "commit": git_commit(),
           "config_sha1": hashlib.sha1((ROOT / "config.yaml").read_bytes()).hexdigest()[:10],
           "updated": now(), "requests": c["done"], "cache_hits": c["hit"], "parse_ok": c["ok"],
           "parse_retry_ok": c["retry_ok"], "parse_fail": c["fail"], "errors": c["error"],
           "minutes": round((time.monotonic() - mon.t0) / 60, 1),
           "http": {str(k): v for k, v in (mon.pool.stats.items() if mon.pool else [])}}
    path = RUNS / stage / f"{name}.manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=1), encoding="utf-8")


async def acall(stage: str, model: str, s: dict, name: str | None = None) -> dict:
    """One request, for stages whose next prompt depends on this answer (S2, S4)."""
    rec, _ = await _nim_one(get_pool(), cache_for(stage, name or model), stage, model, s)
    return rec


async def arun(stage: str, model: str, specs: list[dict], name: str | None = None) -> list[dict]:
    pool, cache = get_pool(), cache_for(stage, name or model)
    sem = asyncio.Semaphore(CFG["nim"]["concurrency"])
    mon = Monitor(f"{stage}/{name or model}", len(specs), pool)

    async def one(s):
        async with sem:
            rec, hit = await _nim_one(pool, cache, stage, model, s)
        mon.add(rec, hit)
        return rec

    recs = await asyncio.gather(*(one(s) for s in specs))
    write_manifest(stage, name or model, model, mon)
    return list(recs)


def run(stage: str, model: str, specs: list[dict], name: str | None = None) -> list[dict]:
    """Run independent requests; records come back in the order of `specs`."""
    return run_many(stage, {model: specs}, {model: name or model})[model]


def run_many(stage: str, jobs: dict[str, list[dict]], names: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """Several models at once: NIM models share one event loop (each hits its own endpoint, all
    share the key pool); vLLM models run one after another on the local GPUs."""
    names = names or {}
    out = {m: _run_vllm(stage, m, s, names.get(m)) for m, s in jobs.items()
           if CFG["models"][m]["backend"] == "vllm"}
    nim = {m: s for m, s in jobs.items() if m not in out}
    if nim:
        async def go():
            try:
                res = await asyncio.gather(*(arun(stage, m, s, names.get(m)) for m, s in nim.items()))
                return dict(zip(nim, res))
            finally:
                await close_pool()

        out.update(asyncio.run(go()))
    return out


# ------------------------------------------------------------------ vLLM backend (Kaggle GPU)

_ENGINES: dict = {}


def _engine(model_id: str):
    if model_id not in _ENGINES:
        from vllm import LLM
        v = CFG["vllm"]
        _ENGINES[model_id] = LLM(model=model_id, seed=CFG["seed"], **{k: v[k] for k in (
            "tensor_parallel_size", "dtype", "enforce_eager", "disable_custom_all_reduce",
            "max_model_len", "gpu_memory_utilization")})
    return _ENGINES[model_id]


def verdict_prob(g) -> float | None:
    """P(TRUE) from the top log-probs at the token where the verdict value starts."""
    if not g.logprobs:
        return None
    cut = g.text.rfind("</think>")
    start = cut + len("</think>") if cut >= 0 else 0
    m = re.search(r'"verdict"\s*:\s*"?', g.text[start:])
    if not m:
        return None
    target, pos = start + m.end(), 0
    for tid, lps in zip(g.token_ids, g.logprobs):
        pos += len(lps[tid].decoded_token or "") if tid in lps else 0
        if pos > target:
            mass = {"T": 0.0, "F": 0.0}
            for lp in lps.values():
                letter = (lp.decoded_token or "").strip(' "').upper()[:1]
                if letter in mass:
                    mass[letter] += math.exp(lp.logprob)
            total = mass["T"] + mass["F"]
            return mass["T"] / total if total > 0 else None
    return None


def _run_vllm(stage: str, model: str, specs: list[dict], name: str | None = None) -> list[dict]:
    """Offline batch: drop cache hits, generate the rest in one call, retry bad parses once."""
    from vllm import SamplingParams
    cache = cache_for(stage, name or model)
    mon = Monitor(f"{stage}/{name or model}", len(specs), None)
    ps = [payload_for(model, s) for s in specs]
    hs = [req_hash(p, s["seed_tag"]) for p, s in zip(ps, specs)]
    seen: set[str] = set()
    todo = [i for i, h in enumerate(hs) if cache.get(h) is None and not (h in seen or seen.add(h))]
    if todo:
        llm = _engine(CFG["models"][model]["id"])
        kw = {"chat_template_kwargs": ps[0]["chat_template_kwargs"]} if "chat_template_kwargs" in ps[0] else {}

        def generate(idx, msgs):
            sps = [SamplingParams(temperature=ps[i]["temperature"], top_p=ps[i]["top_p"],
                                  max_tokens=ps[i]["max_tokens"], logprobs=CFG["vllm"]["logprobs"],
                                  seed=int(hs[i][:8], 16)) for i in idx]
            return dict(zip(idx, llm.chat(msgs, sps, use_tqdm=True, **kw)))

        t0 = time.monotonic()
        outs = generate(todo, [ps[i]["messages"] for i in todo])
        bad = [i for i in todo if parse(outs[i].outputs[0].text, specs[i]["kind"]) is None]
        retry = {}
        if bad:
            msgs = [ps[i]["messages"] + [{"role": "assistant", "content": outs[i].outputs[0].text},
                                         {"role": "user", "content": RETRY_MSG}] for i in bad]
            retry = {i: o.outputs[0].text for i, o in generate(bad, msgs).items()}
        per_call = (time.monotonic() - t0) / len(todo)
        for i in todo:
            o, g = outs[i], outs[i].outputs[0]
            parsed, status = parse(g.text, specs[i]["kind"]), "ok"
            if parsed is None:
                parsed = parse(retry.get(i), specs[i]["kind"])
                status = "retry_ok" if parsed else "fail"
            reasoning = g.text.split("</think>")[0] if "</think>" in g.text else ""
            usage = {"prompt_tokens": len(o.prompt_token_ids or []), "completion_tokens": len(g.token_ids)}
            extra = {"lp_p_true": verdict_prob(g)} if specs[i]["kind"] == "claim" else {}
            cache.put(make_record(hs[i], stage, model, "vllm", ps[i], specs[i], g.text, reasoning,
                                  g.finish_reason, usage, parsed, status, retry.get(i), per_call, **extra))
    fresh = set(todo)
    recs = [cache.get(h) for h in hs]
    for i, r in enumerate(recs):
        mon.add(r, i not in fresh)
    write_manifest(stage, name or model, model, mon)
    return recs


# ------------------------------------------------------------------ CLI checks

async def _check_keys(model_id: str) -> None:
    keys = load_keys()
    async with httpx.AsyncClient(base_url=CFG["nim"]["base_url"], timeout=60) as http:
        async def one(i, k):
            t0 = time.monotonic()
            try:
                r = await http.post("/chat/completions", headers={"Authorization": f"Bearer {k}"},
                                    json={"model": model_id, "max_tokens": 1,
                                          "messages": [{"role": "user", "content": "Say OK."}]})
                code = r.status_code
            except httpx.HTTPError as e:
                code = type(e).__name__
            return i, fp(k), code, time.monotonic() - t0

        rows = await asyncio.gather(*(one(i, k) for i, k in enumerate(keys)))
    for i, f, code, dt in rows:
        print(f"  key #{i:02d} [{f}]  {code}  {dt:.1f}s")
    print(f"{sum(r[2] == 200 for r in rows)}/{len(keys)} keys answer on {model_id}")


async def _probe_models(ids: list[str], limit_s: float = 120) -> None:
    pool = get_pool()

    async def one(mid):
        p = {"model": mid, "max_tokens": 600, "temperature": 0,
             "messages": [{"role": "user", "content": 'Reply with JSON only: {"ok": true}'}]}
        t0 = time.monotonic()
        try:
            text, reasoning, finish, usage = _unpack(await asyncio.wait_for(pool.post(p, tries=2), limit_s))
        except Exception as e:
            return f"FAIL {mid:44s} {time.monotonic() - t0:5.1f}s {type(e).__name__}: {str(e)[:90]}"
        toks = (usage or {}).get("completion_tokens")
        return (f"OK   {mid:44s} {time.monotonic() - t0:5.1f}s finish={finish} tokens={toks} "
                f"reasoning={len(reasoning)}ch text={(text or '')[:40]!r}")

    try:
        for fut in asyncio.as_completed([one(m) for m in ids]):
            print(await fut, flush=True)
        print(f"http status counts: {dict(pool.stats)}  dead keys: {len(pool.dead)}", flush=True)
    finally:
        await close_pool()


def _smoke(n: int = 5, codes: list[str] | None = None) -> None:
    """G1: n claims through each model (default: every NIM model); NIM models also get a logprobs check."""
    from src.data import load_claims, solo_messages          # lazy: data imports this module
    claims = load_claims("eval")[:n]
    codes = codes or [c for c, m in CFG["models"].items() if m["backend"] == "nim"]
    nim = [c for c in codes if CFG["models"][c]["backend"] == "nim"]
    print(f"{'model':5s} {'parsed':>6s} {'lat_s':>6s} {'out_tok':>7s} {'reason_ch':>9s}  finish / first reply")
    for code in codes:
        t0 = time.monotonic()
        specs = [spec(solo_messages(c, para="a", order="TF"), seed_tag="smoke", meta={"item_id": c["id"]})
                 for c in claims]
        recs = [r for r in run("S0", code, specs, name=f"smoke_{code}") if r.get("parse_status") != "error"]
        if not recs:
            print(f"{code:5s} all requests failed")
            continue
        ok = sum(r["parse_status"] in ("ok", "retry_ok") for r in recs)
        lat = sum(r["latency_s"] for r in recs) / len(recs)
        out = sum((r["usage"] or {}).get("completion_tokens", 0) for r in recs) / len(recs)
        rch = sum(r["reasoning_chars"] for r in recs) / len(recs)
        fin = Counter(r["finish"] for r in recs)
        print(f"{code:5s} {ok}/{len(recs):<4d} {lat:6.1f} {out:7.0f} {rch:9.0f}  {dict(fin)} {recs[0]['parsed']}"
              f"  wall {time.monotonic() - t0:.0f}s"
              + (f"  lp_p_true {[r.get('lp_p_true') for r in recs[:3]]}" if "lp_p_true" in recs[0] else ""))
    if not nim:
        return

    async def logprobs():
        pool = get_pool()
        try:
            for code in nim:
                p = {**payload_for(code, spec([{"role": "user", "content": "Say TRUE or FALSE."}],
                                              max_tokens=1024)), "logprobs": True, "top_logprobs": 5}
                try:
                    r = await pool.post(p, tries=3)
                    print(f"logprobs {code:5s} {'honoured' if r['choices'][0].get('logprobs') else 'absent'}")
                except Exception as e:
                    print(f"logprobs {code:5s} rejected: {str(e)[:80]}")
        finally:
            await close_pool()

    asyncio.run(logprobs())


async def _rate_probe(minutes: float, n_keys: int = 3, rpm: float = 40.0) -> None:
    """Fire at `rpm` per key on the first n_keys for `minutes`; report the 429 share per key."""
    keys = load_keys()[:n_keys]
    model = CFG["models"]["M1"]["id"]
    codes = {fp(k): Counter() for k in keys}
    async with httpx.AsyncClient(base_url=CFG["nim"]["base_url"], timeout=60) as http:
        async def shoot(k):
            try:
                r = await http.post("/chat/completions", headers={"Authorization": f"Bearer {k}"},
                                    json={"model": model, "max_tokens": 1,
                                          "messages": [{"role": "user", "content": "Say OK."}]})
                codes[fp(k)][r.status_code] += 1
            except httpx.HTTPError:
                codes[fp(k)]["net_error"] += 1

        tasks, t_end = [], time.monotonic() + 60 * minutes
        while time.monotonic() < t_end:
            tasks += [asyncio.create_task(shoot(k)) for k in keys]
            await asyncio.sleep(60.0 / rpm)
        await asyncio.gather(*tasks)
    for f, c in codes.items():
        total = sum(c.values())
        print(f"  key [{f}]  {dict(c)}  429 share {c[429] / max(1, total):.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-keys", nargs="?", const=CFG["models"]["M1"]["id"], metavar="MODEL")
    ap.add_argument("--models", nargs="+", metavar="ID")
    ap.add_argument("--smoke", nargs="?", const=5, type=int, metavar="N")
    ap.add_argument("--codes", nargs="+", metavar="CODE", help="model codes for --smoke")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--minutes", type=float, default=5.0)
    a = ap.parse_args()
    if a.check_keys:
        asyncio.run(_check_keys(a.check_keys))
    if a.models:
        asyncio.run(_probe_models(a.models))
    if a.smoke:
        _smoke(a.smoke, a.codes)
    if a.probe:
        asyncio.run(_rate_probe(a.minutes))
