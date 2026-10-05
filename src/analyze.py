"""S6: the paper's numbers, computed only from the stage tables built out of runs/.

    python -m src.analyze all [--boot 2000]       rho, q*, gamma, follow curves, fixes, S2/S4/S5 tables
    python -m src.analyze judge [--n 2000]        silent-herding judge on flips (M6, T = 0) + kappa sheet

Claims are the unit: every interval is a cluster bootstrap over claims.
"""
from __future__ import annotations

import argparse
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit, logit
from statsmodels.stats.proportion import proportion_confint

from src.llm import CFG, ROOT, RUNS, read_jsonl, render, run_many, spec

RES = ROOT / "results"
Q = 0.70
warnings.simplefilter("ignore")    # separation/convergence notices on small cells; failed fits return NaN


def boot(df: pd.DataFrame, stat, B: int, seed: int = CFG["seed"]) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval, resampling claims."""
    est = stat(df)
    if B <= 0 or math.isnan(est):
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("item_id")]
    vals = []
    for _ in range(B):
        v = stat(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True))
        if not math.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return est, float(lo), float(hi)


def rho(df: pd.DataFrame) -> float:
    """Herding ratio: weight of one shown vote in units of the own signal, logit(vote_true) ~ s + lead."""
    try:
        fit = smf.logit("vote_true ~ s + lead_shown", data=df).fit(disp=0, maxiter=200)
        return float(fit.params["lead_shown"] / fit.params["s"])
    except Exception:
        return float("nan")


def slope(df: pd.DataFrame, x: str = "lead_shown") -> float:
    try:
        return float(smf.ols(f"delta ~ {x}", data=df).fit().params[x])
    except Exception:
        return float("nan")


def gamma(df: pd.DataFrame) -> float:
    """Correlation neglect: weight of post-cascade (redundant) votes relative to informative ones."""
    try:
        p = smf.ols("delta ~ lead_inf + extra", data=df).fit().params
        return float(p["extra"] / p["lead_inf"])
    except Exception:
        return float("nan")


def s3_table() -> pd.DataFrame:
    df = pd.read_parquet(RES / "S3_cells.parquet")
    df = df[df.vote.notna()].copy()
    df["vote_true"] = (df.vote > 0).astype(int)
    clip = lambda p: logit(p.clip(1, 99) / 100)
    df["delta"] = clip(df.p_true) - clip(df.p_true_solo)
    return df


def h1_h2(df: pd.DataFrame, B: int) -> pd.DataFrame:
    rows = []
    main = df[df.part == "main"]
    for m, d in main.groupby("model"):
        p_hat = d.p_hat.iloc[0]
        ind, seq = d[d.framing == "independent"], d[d.framing == "sequential"]
        r_un = boot(ind[ind.q.isna()], rho, B)
        r_q = boot(ind[ind.q.notna()], rho, B)
        qs = boot(ind[ind.q.notna()], lambda x: slope(x), B)
        g_ind, g_seq = boot(ind, gamma, B), boot(seq, gamma, B)
        rnd = d[d.framing == "random"]
        rows.append({
            "model": m, "cells": len(d), "claims": d.item_id.nunique(), "p_hat": round(p_hat, 3),
            "rho_unstated": r_un, "rho_bayes_unstated": 1.0,
            "rho_q70": r_q, "rho_bayes_q70": round(logit(Q) / logit(p_hat), 3),
            "q_star": tuple(expit(v) for v in qs), "gamma_independent": g_ind, "gamma_sequential": g_seq,
            "random_slope": round(slope(rnd), 3), "independent_slope": round(slope(ind), 3)})
    out = pd.DataFrame(rows)
    fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if isinstance(t, tuple) else t
    return out.apply(lambda col: col.map(fmt))


def follow_curves(df: pd.DataFrame) -> pd.DataFrame:
    """P(follow the shown majority | it opposes the own verdict) by |lead|, model vs Bayes, Wilson CIs."""
    d = df[(df.part == "main") & df.follow.notna()].assign(abs_lead=lambda x: x.lead_shown.abs())
    g = d.groupby(["model", "framing", "abs_lead"]).agg(n=("follow", "size"), k=("follow", "sum"),
                                                        bayes=("oracle_follow", "mean"))
    lo, hi = proportion_confint(g.k, g.n, method="wilson")
    return g.assign(follow=g.k / g.n, lo=lo, hi=hi).round(3)


def fixes(df: pd.DataFrame) -> pd.DataFrame:
    """H6: each fix cell against the same claim's same main cell (paired)."""
    key = ["model", "item_id", "cell"]
    main = df[df.part == "main"].set_index(key)
    rows = []
    for (m, fix), d in df[df.part == "fixes"].groupby(["model", "fix"]):
        base = main.reindex(d.set_index(key).index)
        ok = base.vote.notna().values
        fx, bs = d[ok], base[ok]
        rows.append({"model": m, "fix": fix, "pairs": int(ok.sum()),
                     "acc_fix": (fx.vote.values == fx.truth.values).mean(),
                     "acc_base": (bs.vote.values == bs.truth.values).mean(),
                     "bayes_agree_fix": (fx.vote.values == fx.oracle_vote.values).mean(),
                     "bayes_agree_base": (bs.vote.values == bs.oracle_vote.values).mean(),
                     "follow_fix": np.nanmean(fx.follow.values), "follow_base": np.nanmean(bs.follow.values)})
    return pd.DataFrame(rows).round(3)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    out, run = {}, 0.0
    for i, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * pvals[h]))
        out[h] = run
    return out


def judge(n: int) -> None:
    """H5: does the reason of a flip toward the majority mention the other votes? M6 judges at T = 0."""
    df = s3_table()
    flips = df[(df.part == "main") & (df.follow == 1)]
    reasons = {}
    for m in flips.model.unique():
        for r in read_jsonl(RUNS / "S3" / f"{m}.jsonl"):
            if (r.get("parsed") or {}).get("reason"):
                reasons[(m, r["meta"]["item_id"], r["meta"]["cell"], r["meta"].get("fix"))] = r["parsed"]["reason"]
    flips = flips.assign(reason=[reasons.get((r.model, r.item_id, r.cell, None)) for r in flips.itertuples()])
    flips = flips[flips.reason.notna()]
    flips = flips.groupby("model", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n // max(1, flips.model.nunique())), random_state=CFG["seed"]))
    specs = [spec([{"role": "user", "content": render("judge", reason=r.reason)}], kind="judge",
                  seed_tag="judge", meta={"model": r.model, "item_id": r.item_id, "cell": r.cell})
             for r in flips.itertuples()]
    recs = run_many("S6", {"M6": specs}, {"M6": "judge"})["M6"]
    flips["mentions"] = [(r.get("parsed") or {}).get("answer") for r in recs]
    silent = flips.groupby("model").mentions.apply(lambda x: (x == "NO").mean()).round(3)
    print("silent-herding rate (flips toward the majority whose reason never mentions the votes):")
    print(silent)
    flips.sample(min(200, len(flips)), random_state=CFG["seed"])[["model", "item_id", "cell", "reason", "mentions"]] \
        .assign(label_pushkar="", label_teammate3="").to_csv(RES / "judge_validation.csv", index=False)
    print("results/judge_validation.csv: 200 reasons for the two human labels (kappa >= 0.7 to use the judge)")


def main(B: int) -> None:
    RES.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    if (RES / "S3_cells.parquet").exists():
        df = s3_table()
        t = h1_h2(df, B)
        t.to_csv(RES / "S6_h1_h2.csv", index=False)
        print("H1/H2 by model (point [95% CI], claims resampled):\n", t.to_string(index=False))
        fc = follow_curves(df)
        fc.to_csv(RES / "S6_follow_curves.csv")
        print("\nfollow curves (independent / sequential / random):\n", fc.to_string())
        if (df.part == "fixes").any():
            fx = fixes(df)
            fx.to_csv(RES / "S6_fixes.csv", index=False)
            print("\nfixes vs paired main cells:\n", fx.to_string(index=False))
    if (RES / "S4_sequences.parquet").exists():
        from src.s4_crowds import summary
        s4 = summary(pd.read_parquet(RES / "S4_sequences.parquet"))
        s4.to_csv(RES / "S6_crowds.csv")
        print("\ncrowds:\n", s4.to_string())
    if (RES / "S5_b1.parquet").exists():
        from src.s5_reddit import lifts
        s5 = lifts(pd.read_parquet(RES / "S5_b1.parquet"))
        s5.to_csv(RES / "S6_muchnik.csv")
        print("\nMuchnik lifts:\n", s5.to_string())
    if (RUNS / "S2").exists():
        from src.s2_urn import table
        models = sorted(p.stem for p in (RUNS / "S2").glob("*.jsonl"))
        s2 = table(models, 200)
        s2.to_csv(RES / "S6_urn.csv", index=False)
        print("\nurn (Anderson & Holt: cascade rate 0.71, inconsistent ~0.04):\n", s2.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "judge"])
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 = point estimates only)")
    ap.add_argument("--n", type=int, default=2000, help="flips to judge")
    a = ap.parse_args()
    if a.what == "all":
        main(a.boot)
    else:
        judge(a.n)
