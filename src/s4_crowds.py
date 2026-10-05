"""S4: sequential crowds of 12 agents, with bot seeding (H3; H1 in the wild; F1 at crowd level).

    python -m src.s4_crowds --crowds M1 M2 M3 M6 mixed [--limit 20]        clean (k = 0) and seeded (k = 1-3)
    python -m src.s4_crowds --crowds M1 M2 M3 M6 mixed --f1                 private-first crowd (k = 0)

Agent t makes a solo call (its private signal), then a social call showing every earlier vote.
The first voter in a clean crowd has nothing to see, so its vote is its solo verdict. Solo calls do
not depend on the history, so they go out together and are shared by every k. Seeding: the first
k votes shown are the wrong verdict, unlabelled. The mixed crowd draws each agent's model from MIXED.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

import pandas as pd

from src.data import load_claims, load_personas, private_first_messages, social_messages, solo_messages
from src.llm import ROOT, Monitor, acall, close_pool, get_pool, rng, spec, write_manifest
from src.oracle import informative_lead

AGENTS, MIXED = 12, ["M1", "M2", "M3", "M6"]
SIZES = {"clean": 300, "seeded": 150, "f1": 100}
SCALE = {"M4": 0.5, "M6H": 0.5}            # reasoning variants run on half the claims


def crowd_claims(crowd: str) -> list[dict]:
    """Balanced claims: a model's own pool order (interleaved TRUE/FALSE), or for the mixed crowd the
    claims that sit in at least 3 member pools."""
    members = MIXED if crowd == "mixed" else [crowd]
    pools = {m: json.loads((ROOT / "data" / "pools" / f"{m}.json").read_text(encoding="utf-8"))["claims"]
             for m in members}
    if crowd == "mixed":
        n_in = Counter(r["claim_id"] for rows in pools.values() for r in rows)
        truth = {r["claim_id"]: r["truth"] for rows in pools.values() for r in rows}
        ids = sorted((c for c, n in n_in.items() if n >= 3), key=lambda c: rng("S4", "mixed", c).random())
        t, f = [c for c in ids if truth[c] > 0], [c for c in ids if truth[c] < 0]
        ids = [c for pair in zip(t, f) for c in pair]
    else:
        ids = [r["claim_id"] for r in pools[crowd]]
    claims = {c["id"]: c for split in ("eval", "reserve") for c in load_claims(split)}
    return [claims[i] for i in ids]


def agents(crowd: str, claim_id: str, personas: list[dict]) -> list[dict]:
    g = rng("S4", "agents", crowd, claim_id)            # identical agents for every k and variant
    return [{"t": t, "model": g.choice(MIXED) if crowd == "mixed" else crowd, "persona": g.choice(personas),
             "order": "TF" if g.random() < 0.5 else "FT"} for t in range(AGENTS)]


def verdict(rec: dict) -> int:
    p = rec.get("parsed")
    return 0 if not p else (1 if p["verdict"] == "TRUE" else -1)


async def chain(crowd: str, c: dict, k: int, variant: str, personas: list[dict], mon: Monitor) -> dict:
    ags = agents(crowd, c["id"], personas)
    base = {"item_id": c["id"], "crowd": crowd, "truth": c["truth"]}
    solos = await asyncio.gather(*(acall("S4", a["model"], spec(
        solo_messages(c, a["persona"], "a", a["order"]), kind="claim", decoding="sample",
        seed_tag=f"{crowd}|{a['t']}", meta={**base, "part": "solo", "t": a["t"], "persona_id": a["persona"]["id"]}),
        name=crowd) for a in ags))
    for r in solos:
        mon.add(r, False)
    history, social = [-c["truth"]] * k, []
    for a, solo in zip(ags, solos):
        if not history:                                   # first voter of a clean crowd: nothing to see
            v = verdict(solo)
        else:
            if variant == "f1" and solo.get("parsed"):
                msgs = private_first_messages(c, a["persona"], "a", a["order"], solo["text"], "sequential", history)
            else:
                msgs = social_messages(c, a["persona"], "a", a["order"], "sequential", history)
            rec = await acall("S4", a["model"], spec(
                msgs, kind="claim", decoding="sample", seed_tag=f"{crowd}|{a['t']}|{k}|{variant}",
                meta={**base, "part": variant, "t": a["t"], "k": k, "persona_id": a["persona"]["id"],
                      "history": history[:]}), name=crowd)
            mon.add(rec, False)
            v = verdict(rec)
        if v == 0:                                        # unparseable vote: the chain stops here
            break
        social.append(v)
        history.append(v)
    solo_v = [verdict(r) for r in solos]
    sign = lambda x: (x > 0) - (x < 0)
    shown = [-c["truth"]] * k + social
    first2 = next((i for i in range(1, len(shown) + 1) if abs(sum(shown[:i])) >= 2), None)
    side = sign(sum(shown[:first2])) if first2 else 0
    cascade = side if first2 and len(shown) >= first2 + 3 and all(x == side for x in shown[first2:first2 + 3]) else 0
    return {**base, "k": k, "variant": variant, "models": ",".join(a["model"] for a in ags),
            "complete": len(social) == AGENTS, "solo": solo_v, "social": social,
            "indep_majority": sign(sum(solo_v)), "final_majority": sign(sum(social)),
            "first_lead2": first2, "cascade": cascade, "lead_inf_final": informative_lead(shown)}


async def run_crowds(crowds: list[str], f1: bool, limit: int | None) -> pd.DataFrame:
    get_pool()
    personas = load_personas()
    sem = asyncio.Semaphore(256)
    mons = {cr: Monitor(f"S4/{cr}", 0, get_pool()) for cr in crowds}
    jobs = []
    for cr in crowds:
        claims, scale = crowd_claims(cr), SCALE.get(cr, 1.0)
        n = lambda key: min(len(claims), limit or int(SIZES[key] * scale))
        if f1:
            jobs += [(cr, c, 0, "f1") for c in claims[:n("f1")]]
        else:
            jobs += [(cr, c, 0, "social") for c in claims[:n("clean")]]
            jobs += [(cr, c, k, "social") for k in (1, 2, 3) for c in claims[:n("seeded")]]

    async def one(job):
        async with sem:
            return await chain(*job, personas, mons[job[0]])

    try:
        rows = await asyncio.gather(*(one(j) for j in jobs))
    finally:
        for cr in crowds:
            write_manifest("S4", cr, MIXED if cr == "mixed" else cr, mons[cr])
        await close_pool()
    df = pd.DataFrame(rows)
    path = ROOT / "results" / "S4_sequences.parquet"
    path.parent.mkdir(exist_ok=True)
    if path.exists():                                     # keep other crowds/variants already on disk
        old = pd.read_parquet(path)
        keys = ["crowd", "item_id", "k", "variant"]
        df = pd.concat([old[~old.set_index(keys).index.isin(df.set_index(keys).index)], df], ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.complete].copy()
    acc = lambda maj, truth: (maj == truth).astype(float).where(maj != 0, 0.5)   # 6-6 tie counts as half
    d["social_acc"], d["indep_acc"] = acc(d.final_majority, d.truth), acc(d.indep_majority, d.truth)
    d["wrong_cascade"] = (d.cascade != 0) & (d.cascade != d.truth)
    return (d.groupby(["crowd", "variant", "k"])
            .agg(seqs=("item_id", "size"), social_acc=("social_acc", "mean"), indep_acc=("indep_acc", "mean"),
                 cascades=("cascade", lambda x: (x != 0).mean()), wrong_cascades=("wrong_cascade", "mean"))
            .assign(wisdom_loss=lambda x: x.indep_acc - x.social_acc).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crowds", nargs="+", required=True, help="model codes and/or 'mixed'")
    ap.add_argument("--f1", action="store_true", help="private-first crowd instead of clean + seeded")
    ap.add_argument("--limit", type=int, help="claims per crowd and k (pilot)")
    a = ap.parse_args()
    print(summary(asyncio.run(run_crowds(a.crowds, a.f1, a.limit))))
