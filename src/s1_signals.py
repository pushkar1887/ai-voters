"""S1: each model's private signal on every claim, then per-model item pools.

    python -m src.s1_signals --models M1 M2 M3 M4 M6 [--limit 100] [--refill 500]

Screening (steps 1-2): 5 solo samples at T = 0.7 without persona, plus a recognition probe.
p_bar = mean probability put on the correct label. Pool (3-4): 0.55 <= p_bar <= 0.90, not
recognized, balanced 50/50, capped at the target. Private signal (5-6): one seeded persona per
claim at T = 0 gives s and p_true; p_hat = accuracy of those verdicts inside the pool.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data import load_claims, load_personas, recog_messages, solo_messages
from src.llm import CFG, ROOT, now, rng, run_many, spec

S = CFG["s1"]
TRUE_SIDE = {"true", "mostly true", "correct", "accurate"}
FALSE_SIDE = {"false", "pants on fire", "pants fire", "mostly false", "barely true", "incorrect", "misleading"}


def order_for(*parts) -> str:
    return "TF" if rng("S1", "order", *parts).random() < 0.5 else "FT"


def screen_specs(c: dict) -> list[dict]:
    out = [spec(solo_messages(c, None, "a", order_for(c["id"], k)), kind="claim", decoding="sample", seed_tag=k,
                meta={"item_id": c["id"], "part": "sample", "k": k, "order": order_for(c["id"], k)})
           for k in range(S["samples"])]
    out.append(spec(recog_messages(c), kind="recog", meta={"item_id": c["id"], "part": "recog"}))
    return out


def persona_of(c: dict, personas: list[dict]) -> dict:
    """One persona and label order per claim, the same for every model."""
    return personas[rng("S1", "persona", c["id"]).randrange(len(personas))]


def persona_spec(c: dict, personas: list[dict]) -> dict:
    p, order = persona_of(c, personas), order_for(c["id"], "persona")
    return spec(solo_messages(c, p, "a", order), kind="claim", decoding="greedy",
                meta={"item_id": c["id"], "part": "persona", "persona_id": p["id"], "order": order})


def said_side(rec: dict) -> int:
    """+1 / -1 if the probe says YES and names a rating on the TRUE / FALSE side, else 0."""
    p = rec.get("parsed") or {}
    if p.get("seen") != "YES":
        return 0
    rating = " ".join(re.sub(r"[^a-z ]", " ", str(p.get("rating", "")).lower()).split())
    return 1 if rating in TRUE_SIDE else -1 if rating in FALSE_SIDE else 0


def probe_is_valid(sides: list[int], truths: list[int]) -> bool:
    """The probe counts only where it discriminates. A model that says YES to more than 20% of claims
    must name the right side at least 10 points more often than a constant guess would."""
    yes = [(s, t) for s, t in zip(sides, truths) if s]
    if len(yes) <= 0.2 * len(sides):
        return True
    acc = sum(s == t for s, t in yes) / len(yes)
    base = max(sum(t > 0 for _, t in yes), sum(t < 0 for _, t in yes)) / len(yes)
    return acc - base >= 0.10


def screen_table(claims: list[dict], recs: list[dict]) -> tuple[pd.DataFrame, bool]:
    """One row per claim (p_bar over parsed samples, probe answer, verdict stats), and whether the
    model's recognition probe is valid enough to filter on."""
    by_claim: dict[str, list[dict]] = {}
    for r in recs:
        by_claim.setdefault(r["meta"]["item_id"], []).append(r)
    rows = []
    for c in claims:
        rs = by_claim.get(c["id"], [])
        samples = [r for r in rs if r["meta"].get("part") == "sample" and r.get("parsed")]
        rec = next((r for r in rs if r["meta"].get("part") == "recog"), {})
        pc = [(r["parsed"]["p_true"] / 100 if c["truth"] > 0 else 1 - r["parsed"]["p_true"] / 100) for r in samples]
        rows.append({"claim_id": c["id"], "truth": c["truth"], "recent": c["recent"], "n_parsed": len(samples),
                     "p_bar": sum(pc) / len(pc) if len(pc) >= 3 else None, "said_side": said_side(rec),
                     "n_true_verdicts": sum(r["parsed"]["verdict"] == "TRUE" for r in samples),
                     "n_disagree": sum((r["parsed"]["verdict"] == "TRUE") != (r["parsed"]["p_true"] >= 50)
                                       for r in samples)})
    t = pd.DataFrame(rows)
    valid = probe_is_valid(list(t.said_side), list(t.truth))
    t["recognized"] = (t.said_side == t.truth) if valid else False
    return t, valid


def select_pool(tab: pd.DataFrame, model: str) -> list[str]:
    lo, hi = S["band"]
    ok = tab[tab.p_bar.between(lo, hi) & ~tab.recognized]
    target = S["pool_cap"].get(model, S["pool_target"])
    order = {cid: rng("S1", "pool", cid).random() for cid in ok.claim_id}
    t = sorted(ok[ok.truth > 0].claim_id, key=order.get)
    f = sorted(ok[ok.truth < 0].claim_id, key=order.get)
    n = min(len(t), len(f), target // 2)
    return [cid for pair in zip(t[:n], f[:n]) for cid in pair]       # interleaved: every prefix is balanced


def ece(samples: list[tuple[float, int]], bins: int = 10) -> float:
    """Expected calibration error of P(TRUE) = p_true/100 against the true label."""
    if not samples:
        return float("nan")
    df = pd.DataFrame(samples, columns=["p", "y"])
    df["bin"] = (df.p * bins).clip(upper=bins - 1).astype(int)
    g = df.groupby("bin").agg(conf=("p", "mean"), acc=("y", "mean"), n=("p", "size"))
    return float((g.n / len(df) * (g.conf - g.acc).abs()).sum())


def main(models: list[str], limit: int | None, refill: int) -> None:
    claims = load_claims("eval")[:limit] if limit else load_claims("eval")
    if refill:
        claims += load_claims("reserve")[:refill]
    personas = load_personas()
    screened = {m: claims[:S["screen"].get(m, len(claims))] for m in models}
    recs = run_many("S1", {m: [s for c in cs for s in screen_specs(c)] for m, cs in screened.items()})

    tabs, pools, valid = {}, {}, {}
    for m in models:
        tabs[m], valid[m] = screen_table(screened[m], recs[m])
        pools[m] = select_pool(tabs[m], m)
    by_id = {c["id"]: c for c in claims}
    prec = run_many("S1", {m: [persona_spec(by_id[i], personas) for i in ids] for m, ids in pools.items()})

    summary = []
    for m in models:
        tab = tabs[m].set_index("claim_id")
        rows = []
        for cid, r in zip(pools[m], prec[m]):
            p = r.get("parsed")
            if not p:
                continue
            s = 1 if p["verdict"] == "TRUE" else -1
            rows.append({"claim_id": cid, "truth": by_id[cid]["truth"], "p_bar": round(float(tab.p_bar[cid]), 4),
                         "persona_id": r["meta"]["persona_id"], "order": r["meta"]["order"], "s": s,
                         "p_true_solo": p["p_true"], "recent": bool(by_id[cid]["recent"])})
        p_hat = sum(x["s"] == x["truth"] for x in rows) / max(1, len(rows))
        out = Path(ROOT / "data" / "pools" / f"{m}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": m, "model_id": CFG["models"][m]["id"], "created": now(),
                                   "screened": len(screened[m]), "p_hat": round(p_hat, 4), "n": len(rows),
                                   "claims": rows}, indent=1), encoding="utf-8")

        samples = [r for r in recs[m] if r["meta"].get("part") == "sample"]
        parsed = [r for r in samples if r.get("parsed")]
        cal = [(r["parsed"]["p_true"] / 100, int(by_id[r["meta"]["item_id"]]["truth"] > 0)) for r in parsed]
        t = tabs[m]
        summary.append({
            "model": m, "screened": len(screened[m]),
            "parse_rate": round(len(parsed) / max(1, len(samples)), 4),
            "true_share": round(sum(r["parsed"]["verdict"] == "TRUE" for r in parsed) / max(1, len(parsed)), 3),
            "verdict_ptrue_disagree": round(t.n_disagree.sum() / max(1, t.n_parsed.sum()), 4),
            "solo_acc": round(sum((r["parsed"]["verdict"] == "TRUE") == (by_id[r["meta"]["item_id"]]["truth"] > 0)
                                  for r in parsed) / max(1, len(parsed)), 3),
            "probe_yes": round((t.said_side != 0).mean(), 3), "probe_valid": valid[m],
            "recognized": round(t.recognized.mean(), 3), "ece": round(ece(cal), 3),
            "in_band": int((t.p_bar.between(*S["band"]) & ~t.recognized).sum()),
            "pool_n": len(rows), "p_hat": round(p_hat, 3),
            "pool_recent": sum(x["recent"] for x in rows)})
    res = pd.DataFrame(summary)
    path = ROOT / "results" / "S1_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    g2 = Counter()
    for r in summary:
        g2["parse"] += r["parse_rate"] >= 0.99
        g2["true_share"] += 0.35 <= r["true_share"] <= 0.65
        g2["pool_1000"] += r["pool_n"] >= 1000
        g2["pool_600"] += r["pool_n"] >= 600
        g2["p_hat"] += 0.60 <= r["p_hat"] <= 0.85
    print(f"G2 counts over {len(summary)} models: {dict(g2)} "
          "(need parse all; TRUE share all; pool>=1000 for >=4 (fallback 600); p_hat in band for >=4)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, help="first N eval claims (pilot)")
    ap.add_argument("--refill", type=int, default=0, help="also screen the first N reserve claims")
    a = ap.parse_args()
    main(a.models, a.limit, a.refill)
