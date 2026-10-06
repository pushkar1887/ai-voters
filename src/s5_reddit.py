"""S5: Muchnik, Aral & Taylor (2013) first-voter replication on real Reddit comments (H4).

    python -m src.s5_reddit --models M1 M2 M3 M6 [--limit 100]

Each comment is shown under three treatments in separate stateless calls: control (score 1),
up (score 2) and down (score 0). The persona is fixed per comment, so only the score changes.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data import load_comments, load_personas
from src.llm import ROOT, RUNS, read_jsonl, render, rng, run_many, spec

TREATMENTS = {"control": 1, "up": 2, "down": 0}


def specs_for(comments: list[dict], personas: list[dict]) -> list[dict]:
    out = []
    for c in comments:
        p = personas[rng("S5", "persona", c["id"]).randrange(len(personas))]
        for t, score in TREATMENTS.items():
            msgs = [{"role": "system", "content": render("system_reddit", persona=p["text"])},
                    {"role": "user", "content": render("reddit", subreddit=c["subreddit"], title=c["title"],
                                                       comment=c["text"], score=score)}]
            out.append(spec(msgs, kind="vote", decoding="sample", seed_tag="b1",
                            meta={"item_id": c["id"], "treatment": t, "group": c["group"], "tier": c["tier"],
                                  "subreddit": c["subreddit"], "persona_id": p["id"]}))
    return out


def table(models: list[str]) -> pd.DataFrame:
    rows = [{"model": m, **r["meta"], "vote": (r.get("parsed") or {}).get("vote")}
            for m in models for r in read_jsonl(RUNS / "S5" / f"{m}.jsonl")]
    df = pd.DataFrame(rows).drop_duplicates(["model", "item_id", "treatment"])
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_parquet(ROOT / "results" / "S5_b1.parquet", index=False)
    return df


def lifts(df: pd.DataFrame, B: int = 2000) -> pd.DataFrame:
    """Per model: P(UP) by treatment, the up-lift (Muchnik: +32%) and the down-minus-control upvote rate
    (Muchnik: 0.099 vs 0.054, i.e. correction), with 95% intervals from a paired bootstrap over comments."""
    w = (df.assign(up=(df.vote == "UP").astype(float))
         .pivot_table(index=["model", "item_id"], columns="treatment", values="up").dropna())
    rng = np.random.default_rng(13)
    rows = []
    for m, x in w.groupby(level="model"):
        a = x[["control", "up", "down"]].to_numpy()
        boot = np.array([a[rng.integers(0, len(a), len(a))].mean(0) for _ in range(B)])
        lift, diff = boot[:, 1] / boot[:, 0] - 1, boot[:, 2] - boot[:, 0]
        c, u, d = a.mean(0)
        rows.append({"model": m, "comments": len(a), "control": c, "up": u, "down": d,
                     "lift_up": u / c - 1, "lift_up_lo": np.nanpercentile(lift, 2.5),
                     "lift_up_hi": np.nanpercentile(lift, 97.5), "down_vs_control": d - c,
                     "down_lo": np.percentile(diff, 2.5), "down_hi": np.percentile(diff, 97.5)})
    return pd.DataFrame(rows).set_index("model").round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    comments = load_comments()[:a.limit] if a.limit else load_comments()
    run_many("S5", {m: specs_for(comments, load_personas()) for m in a.models})
    print(lifts(table(a.models)))
    print("Muchnik et al. (2013): up-lift +0.32; down-treated upvote 0.099 vs control 0.054 (corrected)")
