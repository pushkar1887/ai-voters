"""S2: Anderson & Holt (1997) urn cascades with LLM agents, the like-for-like human anchor.

    python -m src.s2_urn --models M1 M2 M3 M6 --rounds 20     sanity check (lead 0 -> own draw > 90%)
    python -m src.s2_urn --models M1 M2 M3 M6                 full run (200 rounds) + results/S2_table.csv

Each round a fair coin picks urn A or B. Six agents draw privately (light with probability 2/3 under A)
and guess in turn, seeing only earlier guesses. Bayes and own-draw agents are scored on the same draws.
"""
from __future__ import annotations

import argparse
import asyncio

import pandas as pd

from src.llm import CFG, RUNS, ROOT, Monitor, acall, close_pool, get_pool, read_jsonl, render, rng, spec, write_manifest
from src.oracle import informative_lead, urn_bayes

AGENTS, ROUNDS = 6, 200


def draws_for(r: int) -> tuple[int, list[int]]:
    g = rng("S2", "round", r)
    urn = 1 if g.random() < 0.5 else -1
    return urn, [1 if g.random() < (2 / 3 if urn > 0 else 1 / 3) else -1 for _ in range(AGENTS)]


async def play(model: str, r: int, mon: Monitor) -> None:
    urn, draws = draws_for(r)
    guesses: list[int] = []
    for t, d in enumerate(draws):
        hist = ", ".join("A" if g > 0 else "B" for g in guesses) or "none"
        msg = [{"role": "user", "content": render("urn", color="light" if d > 0 else "dark", guesses=hist)}]
        rec = await acall("S2", model, spec(msg, kind="urn", decoding="sample", seed_tag=f"{r}|{t}",
                                            max_tokens=CFG["s2"]["max_tokens"].get(model),
                                            meta={"round": r, "t": t, "urn": urn, "draw": d, "history": guesses[:]}))
        mon.add(rec, False)
        if not rec.get("parsed"):            # a broken round stops here and is dropped in the table
            return
        guesses.append(1 if rec["parsed"]["urn"] == "A" else -1)


async def run_rounds(models: list[str], rounds: int) -> None:
    get_pool()
    sem = asyncio.Semaphore(CFG["nim"]["concurrency"])
    mons = {m: Monitor(f"S2/{m}", rounds * AGENTS, get_pool()) for m in models}

    async def one(m, r):
        async with sem:
            await play(m, r, mons[m])

    try:
        await asyncio.gather(*(one(m, r) for m in models for r in range(rounds)))
    finally:
        for m in models:
            write_manifest("S2", m, m, mons[m])
        await close_pool()


def decisions(model: str, rounds: int) -> pd.DataFrame:
    """One row per agent decision, with the Bayes reference on the same draw and history."""
    rows = []
    for rec in read_jsonl(RUNS / "S2" / f"{model}.jsonl"):
        m = rec["meta"]
        if m["round"] >= rounds or not rec.get("parsed"):
            continue
        lead = informative_lead(m["history"])
        bayes, p_a = urn_bayes(m["draw"], m["history"])
        guess = 1 if rec["parsed"]["urn"] == "A" else -1
        rows.append({"round": m["round"], "t": m["t"], "urn": m["urn"], "draw": m["draw"], "lead": lead,
                     "guess": guess, "bayes": bayes, "p_a": rec["parsed"]["p_a"] / 100, "bayes_p_a": p_a})
    df = pd.DataFrame(rows).drop_duplicates(["round", "t"])
    full = df.groupby("round")["t"].transform("size") == AGENTS          # drop broken rounds
    return df[full]


def table(models: list[str], rounds: int) -> pd.DataFrame:
    out = []
    for m in models:
        d = decisions(m, rounds)
        against = d[(d.lead.abs() >= 2) & (d.draw != d.lead.clip(-1, 1))]   # cascade opportunities
        tie = d[(d.lead.abs() == 1) & (d.draw != d.lead)]
        rounds_ok = d["round"].nunique()
        casc = rev = 0
        for _, g in d.sort_values("t").groupby("round"):
            leads, gs = list(g.lead), list(g.guess)
            first = next((i for i, l in enumerate(leads) if abs(l) >= 2), None)
            if first is not None and all(x == (1 if leads[first] > 0 else -1) for x in gs[first:]):
                casc += 1
                rev += (1 if leads[first] > 0 else -1) != g.urn.iloc[0]
        out.append({"model": m, "rounds": rounds_ok, "decisions": len(d),
                    "own_draw_at_lead0": round((d[d.lead == 0].guess == d[d.lead == 0].draw).mean(), 3),
                    "cascade_rate": round((against.guess == against.lead.clip(-1, 1)).mean(), 3),
                    "cascade_opportunities": len(against),
                    "tie_own_draw": round((tie.guess == tie.draw).mean(), 3),
                    "inconsistent": round(((d.guess != d.bayes) & (d.guess != d.draw)).mean(), 3),
                    "belief_error": round((d.p_a - d.bayes_p_a).mean(), 3),
                    "cascades": casc, "reverse_cascades": rev, "accuracy": round((d.guess == d.urn).mean(), 3)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    a = ap.parse_args()
    asyncio.run(run_rounds(a.models, a.rounds))
    res = table(a.models, a.rounds)
    print(res.to_string(index=False))
    print("human reference (Anderson & Holt 1997): cascade_rate 0.71 (87/122), inconsistent ~0.04")
    if a.rounds >= ROUNDS:
        (ROOT / "results").mkdir(exist_ok=True)
        res.to_csv(ROOT / "results" / "S2_table.csv", index=False)
