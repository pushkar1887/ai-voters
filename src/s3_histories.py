"""S3: scripted histories whose evidential value is known exactly (H1, H2; data for H5, H6).

    python -m src.s3_histories --models M1 M2 M3 M4 M6 M6H --part main [--limit 50]
    python -m src.s3_histories --models M1 M2 M3 M4 M6 M6H --part fixes      (after main)

Each pool claim gets 8 cells sampled (not crossed) from: framing (independent 40%, sequential 40%,
random 20%), q stated or not, list/tally display, paraphrase a/b/c, and a history of 1-6 votes.
Histories for every framing come from one generator, Bayes-consistent sequences with cascades
oversampled, so framings differ only in the vote-block text. 60% of cells disagree with the
agent's own verdict s. Persona, label order and s are the agent's own, fixed in S1.

Fix cells pair with the claim's own main cells: F1 private-first and BR (Bayes-rule prompt) on up
to 4 disagreeing cells, F2 independence cue on the sequential cells; first 300 pool claims only.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.data import load_claims, load_personas, private_first_messages, social_messages
from src.llm import CFG, ROOT, RUNS, read_jsonl, rng, run_many, spec
from src.oracle import bhw_sequences, informative_lead, oracle

CELLS, CELLS_SMALL_POOL, FIX_CLAIMS, Q = 8, 12, 300, 70     # plan fallback: pool < 600 -> 12 cells
FRAMINGS, WEIGHTS = ["independent", "sequential", "random"], [0.4, 0.4, 0.2]
SEQS = {n: bhw_sequences(n) for n in range(1, 7)}


def load_pool(model: str) -> dict:
    return json.loads((ROOT / "data" / "pools" / f"{model}.json").read_text(encoding="utf-8"))


def make_cell(claim_id: str, k: int, s: int) -> dict:
    g = rng("S3", "cell", claim_id, k)
    framing = g.choices(FRAMINGS, WEIGHTS)[0]
    q = Q if framing != "random" and g.random() < 0.5 else None
    display = "list" if framing == "sequential" or g.random() < 0.5 else "tally"
    para = g.choice("abc")
    disagree = g.random() < 0.6
    n = g.randint(1, 6)
    seqs = [x for x in SEQS[n] if (sum(x) * s < 0) == disagree]
    if not seqs:                                            # n = 2k agree cells can be ties only: allowed
        seqs = [x for x in SEQS[n] if sum(x) * s <= 0] if disagree else SEQS[n]
    w = [3 if sum(x) != informative_lead(x) else 1 for x in seqs]   # oversample cascades
    votes = list(g.choices(seqs, w)[0])
    return {"cell": k, "framing": framing, "q": q, "display": display, "para": para, "votes": votes,
            "disagree": sum(votes) * s < 0}


def cell_meta(row: dict, c: dict, cell: dict, p_hat: float, part: str, fix: str | None) -> dict:
    framing = "cue" if fix == "F2" else cell["framing"]
    q_eff = cell["q"] / 100 if cell["q"] else p_hat
    ov, op = oracle(cell["votes"], row["s"], p_hat, q_eff, framing)
    return {"item_id": c["id"], "part": part, "fix": fix, **{k: cell[k] for k in
            ("cell", "q", "display", "para", "votes", "disagree")}, "framing": framing,
            "persona_id": row["persona_id"], "order": row["order"], "s": row["s"],
            "p_true_solo": row["p_true_solo"], "truth": c["truth"], "p_hat": p_hat,
            "lead_inf": informative_lead(cell["votes"]), "oracle_vote": ov, "oracle_p": round(op, 4)}


def specs_for(model: str, part: str, limit: int | None) -> list[dict]:
    pool = load_pool(model)
    rows = pool["claims"][:limit] if limit else pool["claims"]
    claims = {c["id"]: c for split in ("eval", "reserve") for c in load_claims(split)}
    personas = {p["id"]: p for p in load_personas()}
    solo_text = {}
    if part == "fixes":
        rows = rows[:FIX_CLAIMS]
        solo_text = {r["meta"]["item_id"]: r["text"] for r in read_jsonl(RUNS / "S1" / f"{model}.jsonl")
                     if r["meta"].get("part") == "persona" and r.get("parsed")}
    small = pool["n"] < 600 and model not in CFG["s1"]["pool_cap"]   # capped reasoning variants keep 8
    out = []
    for row in rows:
        c, persona = claims[row["claim_id"]], personas[row["persona_id"]]
        cells = [make_cell(c["id"], k, row["s"]) for k in range(CELLS_SMALL_POOL if small else CELLS)]
        if part == "main":
            todo = [(x, None) for x in cells]
        else:
            dis = [x for x in cells if x["disagree"]][:4]
            todo = [(x, "BR") for x in dis] + [(x, "F2") for x in cells if x["framing"] == "sequential"]
            if c["id"] in solo_text:
                todo += [(x, "F1") for x in dis]
        for x, fix in todo:
            if fix == "F1":
                msgs = private_first_messages(c, persona, "a", row["order"], solo_text[c["id"]],
                                              x["framing"], x["votes"], x["q"], x["display"])
            else:
                msgs = social_messages(c, persona, x["para"], row["order"], "cue" if fix == "F2" else x["framing"],
                                       x["votes"], x["q"], x["display"], bayes_rule=fix == "BR")
            out.append(spec(msgs, kind="claim", decoding="greedy", seed_tag=f"{x['cell']}|{fix or ''}",
                            meta=cell_meta(row, c, x, pool["p_hat"], part, fix)))
    return out


def table(models: list[str]) -> pd.DataFrame:
    """One row per S3 call: design, oracle and the model's social answer -> results/S3_cells.parquet."""
    rows = []
    for m in models:
        in_pool = {r["claim_id"] for r in load_pool(m)["claims"]}      # pilot claims may have left the pool
        for r in read_jsonl(RUNS / "S3" / f"{m}.jsonl"):
            meta, p = r["meta"], r.get("parsed") or {}
            if meta["item_id"] not in in_pool:
                continue
            v = meta["votes"]
            shown = sum(v)
            vote = (1 if p["verdict"] == "TRUE" else -1) if p else None
            maj = (1 if shown > 0 else -1) if shown else 0
            against = bool(maj) and maj != meta["s"]           # the shown majority opposes the own verdict
            rows.append({"model": m, **{k: meta[k] for k in meta if k != "votes"},
                         "votes": "".join("T" if x > 0 else "F" for x in v), "n": len(v),
                         "n_true": sum(x > 0 for x in v), "lead_shown": shown, "lead_rel": shown * meta["s"],
                         "extra": shown - meta["lead_inf"], "vote": vote, "p_true": p.get("p_true"),
                         "follow": (vote == maj) if (vote and against) else None,
                         "oracle_follow": (meta["oracle_vote"] == maj) if against else None,
                         "parse_status": r["parse_status"], "reasoning_chars": r.get("reasoning_chars", 0),
                         "lp_p_true": r.get("lp_p_true")})
    df = pd.DataFrame(rows)
    for col in ("follow", "oracle_follow", "vote", "p_true", "q"):
        df[col] = df[col].astype("float")                   # None -> NaN, bool -> 0/1
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_parquet(ROOT / "results" / "S3_cells.parquet", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--part", choices=["main", "fixes"], default="main")
    ap.add_argument("--limit", type=int, help="first N pool claims (pilot)")
    a = ap.parse_args()
    run_many("S3", {m: specs_for(m, a.part, a.limit) for m in a.models})
    df = table(a.models)
    against = df[df.follow.notna()]
    print("follow rate when the shown majority opposes the own verdict (model vs Bayes):")
    print(against.groupby(["model", "part", "fix", "framing"], dropna=False)
          .agg(cells=("follow", "size"), follow=("follow", "mean"), bayes=("oracle_follow", "mean")).round(3))
