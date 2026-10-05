"""S0 data (LIAR2 claims, Reddit comments, personas), loaders, and the claim prompts.

    python -m src.data claims | comments | personas | all
"""
from __future__ import annotations

import html
import io
import json
import sys
import zipfile
from collections import Counter

import httpx
import pandas as pd

from src.llm import CFG, PROMPTS, ROOT, read_jsonl, render, rng, write_jsonl

RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"


def clean(s: str) -> str:
    """Unescape HTML entities (Reddit's &gt; quotes), straighten curly quotes, normalise whitespace."""
    s = html.unescape(str(s)).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    return " ".join(s.split())


# ------------------------------------------------------------------ claims (Setting A)

def build_claims() -> None:
    from huggingface_hub import hf_hub_download
    d, c = CFG["data"], CFG["data"]["claims"]
    df = pd.concat([pd.read_csv(hf_hub_download(d["liar2_repo"], f"{s}.csv", repo_type="dataset",
                                                local_dir=RAW / "liar2")) for s in ("train", "valid", "test")],
                   ignore_index=True)
    counts = df["label"].value_counts().to_dict()
    if counts != {int(k): v for k, v in d["liar2_counts"].items()}:
        sys.exit(f"LIAR2 label counts differ from the published ones: {counts}")
    n0 = len(df)

    # Binary state; the fact-check (justification), rating history (*_counts) and
    # speaker_description never reach an agent, so they are dropped here.
    df = df[df["label"].isin([0, 1, 5])].copy()
    df["truth"] = df["label"].map({5: 1, 1: -1, 0: -1})
    df["label6"] = df["label"].map({int(k): v for k, v in d["liar2_labels"].items()})
    df = df[["id", "statement", "speaker", "date", "context", "subject", "truth", "label6"]]
    df = df.dropna(subset=["statement", "speaker", "date", "context"])
    for col in ("statement", "speaker", "context"):
        df[col] = df[col].map(clean)
    df["speaker"] = df["speaker"].map(lambda s: " ".join(w[:1].upper() + w[1:] for w in s.split()))
    df["dt"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["dt"])

    words = df["statement"].str.split().str.len()
    df = df[words.between(c["min_words"], c["max_words"])]
    df["norm"] = (df["statement"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
                  .str.split().str.join(" "))
    df = df[df.groupby("norm")["truth"].transform("nunique") == 1].drop_duplicates("norm")

    df["year"] = df["dt"].dt.year
    df["subject"] = df["subject"].fillna("none").str.split(";").str[0].str.strip()
    df["recent"] = df["dt"] >= pd.Timestamp(c["recent_from"])

    # Balance: every TRUE claim, plus FALSE claims drawn with weight n_TRUE / n_FALSE of their
    # (year, first subject) cell, so both labels share one year-by-subject distribution.
    T, F = df[df.truth == 1], df[df.truth == -1]
    cell_t = T.groupby(["year", "subject"]).size()
    cell_f = F.groupby(["year", "subject"]).size()
    w = F.apply(lambda r: cell_t.get((r.year, r.subject), 0) / cell_f[(r.year, r.subject)], axis=1)
    if (w > 0).sum() < len(T):                                   # too few matched cells: match on year only
        yt, yf = T.groupby("year").size(), F.groupby("year").size()
        w = F["year"].map(lambda y: yt.get(y, 0) / yf[y])
    F = F.sample(n=len(T), weights=w, random_state=CFG["seed"])

    T = T.sample(frac=1, random_state=CFG["seed"])
    F = F.sample(frac=1, random_state=CFG["seed"])
    ne, nl = c["n_eval"] // 2, c["n_lora"] // 2
    splits = {"eval": (T[:ne], F[:ne]), "lora": (T[ne:ne + nl], F[ne:ne + nl]), "reserve": (T[ne + nl:], F[ne + nl:])}
    report = {"liar2_rows": n0, "binary_after_cleaning": int(len(df)), "true": int(len(T))}
    for name, (t, f) in splits.items():
        part = pd.concat([t, f]).sample(frac=1, random_state=CFG["seed"])
        rows = [{"id": f"liar2-{r.id}", "statement": r.statement, "speaker": r.speaker, "context": r.context,
                 "date": r.date, "year": int(r.year), "subject": r.subject, "truth": int(r.truth),
                 "label": "TRUE" if r.truth > 0 else "FALSE", "label6": r.label6, "recent": bool(r.recent)}
                for r in part.itertuples()]
        write_jsonl(OUT / f"claims_{name}.jsonl", rows)
        report[name] = {"n": len(rows), "true": sum(r["truth"] > 0 for r in rows),
                        "recent": sum(r["recent"] for r in rows)}
    print(json.dumps(report, indent=1))
    _note("claims", report)


# ------------------------------------------------------------------ comments (Setting B)

def download(url: str, path) -> None:
    """Resumable download: the ConvoKit server is slow, so a partial file continues where it stopped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = path.stat().st_size if path.exists() else 0
    with httpx.stream("GET", url, headers={"Range": f"bytes={done}-"} if done else {},
                      follow_redirects=True, timeout=600) as r:
        if r.status_code == 416:                       # nothing left to fetch
            return
        r.raise_for_status()
        with open(path, "ab" if r.status_code == 206 else "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def build_comments() -> None:
    c = CFG["data"]["comments"]
    zpath = RAW / "reddit-corpus-small.corpus.zip"
    if not (zpath.exists() and zipfile.is_zipfile(zpath)):
        download(CFG["data"]["reddit_url"], zpath)
    rows = []
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        convs = json.loads(z.read(next(n for n in names if n.endswith("conversations.json"))))
        if jl := [n for n in names if n.endswith("utterances.jsonl")]:            # ConvoKit v3 layout
            with z.open(jl[0]) as f:
                utts = [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8")]
        else:                                                                     # older layout: one JSON array
            utts = json.loads(z.read(next(n for n in names if n.endswith("utterances.json"))))
        for u in utts:
            m, text = u.get("meta") or {}, clean(u.get("text") or "")
            root = u.get("conversation_id") or u.get("root")
            if (u["id"] == root or u.get("reply_to") is None                       # thread posts
                    or text in ("[deleted]", "[removed]") or m.get("stickied")
                    or (u.get("speaker") or u.get("user")) == "AutoModerator"
                    or not c["min_words"] <= len(text.split()) <= c["max_words"]):
                continue
            conv = convs.get(root) or {}
            conv = conv.get("meta", conv)
            rows.append({"id": u["id"], "text": text, "score": m.get("score"),
                         "subreddit": m.get("subreddit") or conv.get("subreddit"),
                         "title": " ".join(clean(conv.get("title") or "").split()[:c["title_words"]])})
        del utts
    df = pd.DataFrame(rows).dropna(subset=["score", "subreddit"])
    df = df[df["title"].str.len() > 0]
    # Quality tier: score percentile within subreddit (top half = high).
    df["tier"] = df.groupby("subreddit")["score"].rank(pct=True).map(lambda p: "high" if p > 0.5 else "low")

    groups = {s: g for g, subs in (c.get("groups") or {}).items() for s in subs}
    subs = df["subreddit"].value_counts()
    if unmapped := sorted(set(subs.index) - set(groups)):
        print(f"unmapped subreddits (dropped): {unmapped}")
    if not groups:
        subs.rename_axis("subreddit").reset_index(name="comments").to_csv(OUT / "subreddits.csv", index=False)
        sys.exit(f"{len(subs)} subreddits written to data/processed/subreddits.csv; "
                 "map them to Muchnik's groups under data.comments.groups in config.yaml, then re-run.")
    df["group"] = df["subreddit"].map(groups)
    df = df[df["group"].notna() & (df["group"] != "other")]

    out = []
    for g, part in df.groupby("group"):
        n = min(c["per_group"], len(part))
        # stratified by subreddit: proportional allocation, rounded, topped up at random
        take = (part.groupby("subreddit").size() / len(part) * n).round().astype(int)
        picked = pd.concat([sub.sample(n=min(len(sub), take[s]), random_state=CFG["seed"])
                            for s, sub in part.groupby("subreddit")])
        if len(picked) < n:
            rest = part.drop(picked.index)
            picked = pd.concat([picked, rest.sample(n=n - len(picked), random_state=CFG["seed"])])
        out.append(picked.sample(n=n, random_state=CFG["seed"]))
    res = pd.concat(out).sample(frac=1, random_state=CFG["seed"])
    write_jsonl(OUT / "comments.jsonl", res[["id", "group", "subreddit", "title", "text", "score", "tier"]]
                .to_dict("records"))
    report = {"utterances_kept_after_filters": len(rows), "comments": len(res),
              "per_group": res["group"].value_counts().to_dict(), "tiers": res["tier"].value_counts().to_dict()}
    print(json.dumps(report, indent=1))
    _note("comments", report)


# ------------------------------------------------------------------ personas

def build_personas() -> None:
    """60 personas with balanced margins: each occupation 5x, each region and age band 12x."""
    p = CFG["data"]["personas"]
    r = rng("personas")
    k = p["n"]
    occ = (p["occupations"] * k)[:k]
    reg = (p["regions"] * k)[:k]
    band = (list(range(len(p["age_bands"]))) * k)[:k]
    for lst in (occ, reg, band):
        r.shuffle(lst)
    out = []
    for i, (o, g, b) in enumerate(zip(occ, reg, band)):
        age = r.randint(*p["age_bands"][b])
        art = "an" if age == 18 or 80 <= age <= 89 else "a"
        out.append({"id": f"P{i:02d}", "age": age, "occupation": o, "region": g,
                    "text": f"{art} {age}-year-old {o} from {g} who reads the news most days"})
    (OUT / "personas.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} personas, e.g. {out[0]['text']!r}")


def _note(key: str, report: dict) -> None:
    """Keep counts and seeds next to the data (data/processed/build_report.json)."""
    path = OUT / "build_report.json"
    rep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rep[key] = {**report, "seed": CFG["seed"]}
    path.write_text(json.dumps(rep, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ loaders

def load_claims(split: str) -> list[dict]:
    return read_jsonl(OUT / f"claims_{split}.jsonl")


def load_personas() -> list[dict]:
    return json.loads((OUT / "personas.json").read_text(encoding="utf-8"))


def load_comments() -> list[dict]:
    return read_jsonl(OUT / "comments.jsonl")


# ------------------------------------------------------------------ claim prompts

def labels(order: str) -> dict:
    """order 'TF' asks 'TRUE or FALSE'; 'FT' asks 'FALSE or TRUE'."""
    return {"A": "TRUE", "B": "FALSE"} if order == "TF" else {"A": "FALSE", "B": "TRUE"}


def claim_text(c: dict) -> str:
    return render("claim", statement=c["statement"], speaker=c["speaker"], context=c["context"], date=c["date"])


def system_text(persona: dict | None, para: str, bayes_rule: bool = False) -> str:
    s = (render("system_claims", para, persona=persona["text"]) if persona
         else render("system_claims_nopersona", para))
    return f"{s} {PROMPTS['bayes_rule']}" if bayes_rule else s


def solo_messages(c: dict, persona: dict | None = None, para: str = "a", order: str = "TF") -> list[dict]:
    user = f"{claim_text(c)}\n\n{render('question', para, **labels(order))}\n{PROMPTS['json_claim']}"
    return [{"role": "system", "content": system_text(persona, para)}, {"role": "user", "content": user}]


def recog_messages(c: dict) -> list[dict]:
    return [{"role": "user", "content": render("recog", statement=c["statement"], speaker=c["speaker"],
                                               context=c["context"], date=c["date"])}]


def vote_block(framing: str, votes: list[int], q: int | None = None, display: str = "list") -> str:
    """votes are +1 (TRUE) / -1 (FALSE) in the order they were cast."""
    head = render("vote_block", framing, n=len(votes), q=render("q_stated", q=q) if q else "")
    n_true = sum(v > 0 for v in votes)
    body = render("display", display, noun="Labels" if framing == "random" else "Ratings",
                  lines="\n".join(f"{i}. {'TRUE' if v > 0 else 'FALSE'}" for i, v in enumerate(votes, 1)),
                  n_true=n_true, n_false=len(votes) - n_true)
    return f"{head}\n{body}"


def social_messages(c: dict, persona: dict | None, para: str, order: str, framing: str, votes: list[int],
                    q: int | None = None, display: str = "list", bayes_rule: bool = False) -> list[dict]:
    user = (f"{claim_text(c)}\n\n{vote_block(framing, votes, q, display)}\n\n"
            f"{render('social_close', para, **labels(order))}\n{PROMPTS['json_claim']}")
    return [{"role": "system", "content": system_text(persona, para, bayes_rule)}, {"role": "user", "content": user}]


def private_first_messages(c: dict, persona: dict | None, para: str, order: str, solo_reply: str,
                           framing: str, votes: list[int], q: int | None = None,
                           display: str = "list") -> list[dict]:
    """F1: turn 1 is the agent's own cached solo reply; turn 2 shows the votes."""
    turn2 = f"{vote_block(framing, votes, q, display)}\n\n{PROMPTS['private_close']}\n{PROMPTS['json_claim']}"
    return solo_messages(c, persona, para, order) + [{"role": "assistant", "content": solo_reply},
                                                     {"role": "user", "content": turn2}]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("personas", "all"):
        build_personas()
    if what in ("claims", "all"):
        build_claims()
    if what in ("comments", "all"):
        build_comments()
