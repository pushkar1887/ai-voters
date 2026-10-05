"""S5: Muchnik, Aral & Taylor (2013) first-voter replication on real Reddit comments (H4).

    python -m src.s5_reddit --models M1 M2 M3 M6 [--limit 100]

Each comment is shown under three treatments in separate stateless calls: control (score 1),
up (score 2) and down (score 0). The persona is fixed per comment, so only the score changes.
"""
from __future__ import annotations

import argparse

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


def lifts(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: P(UP) by treatment, the up-lift (Muchnik: +32%) and the down-treatment upvote rate
    (Muchnik: 0.099 vs 0.054 control, i.e. correction)."""
    up = df.assign(up=df.vote == "UP").pivot_table(index="model", columns="treatment", values="up", aggfunc="mean")
    up["lift_up"] = up["up"] / up["control"] - 1
    up["down_vs_control"] = up["down"] - up["control"]
    return up.round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    comments = load_comments()[:a.limit] if a.limit else load_comments()
    run_many("S5", {m: specs_for(comments, load_personas()) for m in a.models})
    print(lifts(table(a.models)))
    print("Muchnik et al. (2013): up-lift +0.32; down-treated upvote 0.099 vs control 0.054 (corrected)")
