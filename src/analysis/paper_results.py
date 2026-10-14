"""S6: the paper's numbers, computed only from the stage tables built out of runs/.

    python -m src.analysis.paper_results all [--boot 2000]       rho, q*, gamma, follow curves, fixes, S2/S4/S5 tables
    python -m src.analysis.paper_results judge [--n 2000]        silent-herding judge on flips (M6, T = 0) + blind kappa sheet
    python -m src.analysis.paper_results kappa                   H5 gate: judge vs human labels on that sheet
    python -m src.analysis.paper_results holm [--boot 2000]      pooled Holm-corrected tests for H1-H4 and H6 (writes S6_holm.csv only)
    python -m src.analysis.paper_results judge_rates [--n 2000] [--models ...]  H5 rates from cached judge answers (no API calls)
    python -m src.analysis.paper_results muchnik_topic [--boot 2000]          S5 Muchnik panel within each topic group
    python -m src.analysis.paper_results summary                              H2 median across models (from S6_h1_h2.csv)

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

from src.core.nim_client import CFG, ROOT, RUNS, read_jsonl, render, run_many, spec

RES = ROOT / "results"
Q = 0.70
warnings.simplefilter("ignore")    # separation/convergence notices on small cells; failed fits return NaN


def boot(df: pd.DataFrame, stat, B: int, seed: int = CFG["seed"]) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval, resampling claims."""
    est = stat(df)
    if B <= 0 or math.isnan(est):
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = list(df.groupby("item_id").indices.values())        # positional row indices per claim
    vals = []
    for _ in range(B):
        take = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = stat(df.iloc[take])
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


def gamma_parts(df: pd.DataFrame) -> tuple[float, float]:
    """(beta_inf, beta_extra): belief shift per informative vote and per redundant post-cascade vote."""
    try:
        p = smf.ols("delta ~ lead_inf + extra", data=df).fit().params
        return float(p["lead_inf"]), float(p["extra"])
    except Exception:
        return float("nan"), float("nan")


def gamma(df: pd.DataFrame) -> float:
    """Correlation neglect: weight of post-cascade (redundant) votes relative to informative ones.
    Unstable when beta_inf is near zero (a model that barely moves), so read it with gamma_parts."""
    b_inf, b_extra = gamma_parts(df)
    return b_extra / b_inf if b_inf else float("nan")


def s3_table(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not (RES / "S3_cells.parquet").exists():   # from every model's S3 cache (NIM and Kaggle)
        from src.stages.stage3_histories import table
        table()
    df = pd.read_parquet(RES / "S3_cells.parquet")
    df = df[df.vote.notna()].copy()
    df["vote_true"] = (df.vote > 0).astype(int)
    clip = lambda p: logit(p.clip(1, 99) / 100)
    df["delta"] = clip(df.p_true) - clip(df.p_true_solo)
    return df


def h1_h2(df: pd.DataFrame, B: int) -> pd.DataFrame:
    """Per model: rho (q unstated / q = 70%), q*, gamma by framing, each as est, lo, hi columns."""
    rows = []
    for m, d in df[df.part == "main"].groupby("model"):
        p_hat = d.p_hat.iloc[0]
        ind, seq, rnd = (d[d.framing == f] for f in ("independent", "sequential", "random"))
        row = {"model": m, "cells": len(d), "claims": d.item_id.nunique(), "p_hat": p_hat,
               "rho_bayes_unstated": 1.0, "rho_bayes_q70": logit(Q) / logit(p_hat),
               "random_slope": slope(rnd), "independent_slope": slope(ind)}
        row["seq_beta_inf"], row["seq_beta_extra"] = gamma_parts(seq)
        row["ind_beta_inf"], row["ind_beta_extra"] = gamma_parts(ind)
        stats = {"rho_unstated": (ind[ind.q.isna()], rho), "rho_q70": (ind[ind.q.notna()], rho),
                 "q_star": (ind[ind.q.notna()], lambda x: float(expit(slope(x)))),
                 "gamma_independent": (ind, gamma), "gamma_sequential": (seq, gamma)}
        for name, (data, fn) in stats.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = boot(data, fn, B)
        rows.append(row)
    return pd.DataFrame(rows)


def readable(t: pd.DataFrame) -> pd.DataFrame:
    """est [lo, hi] strings for printing."""
    out = t[["model", "cells", "claims"]].copy()
    out["p_hat"] = t.p_hat.round(3)
    for name in ("rho_unstated", "rho_q70", "q_star", "gamma_independent", "gamma_sequential"):
        out[name] = [f"{e:.2f} [{lo:.2f}, {hi:.2f}]" for e, lo, hi in zip(t[name], t[f"{name}_lo"], t[f"{name}_hi"])]
    out["rho_bayes_q70"] = t.rho_bayes_q70.round(2)
    out["random_slope"], out["independent_slope"] = t.random_slope.round(3), t.independent_slope.round(3)
    for c in ("ind_beta_inf", "ind_beta_extra", "seq_beta_inf", "seq_beta_extra"):
        out[c] = t[c].round(3)
    return out


def follow_curves(df: pd.DataFrame) -> pd.DataFrame:
    """P(follow the shown majority | it opposes the own verdict) by |lead|, model vs Bayes, Wilson CIs."""
    d = df[(df.part == "main") & df.follow.notna()].assign(abs_lead=lambda x: x.lead_shown.abs())
    g = d.groupby(["model", "framing", "abs_lead"]).agg(n=("follow", "size"), k=("follow", "sum"),
                                                        bayes=("oracle_follow", "mean"))
    lo, hi = proportion_confint(g.k, g.n, method="wilson")
    return g.assign(follow=g.k / g.n, lo=lo, hi=hi).round(3)


def fixes(df: pd.DataFrame) -> pd.DataFrame:
    """H6 in S3: each fix cell against the same claim's same main cell (paired). S3's scripted votes are
    built from the agent's own verdict, not the truth, so the S3 outcome is agreement with Bayes; truth
    accuracy is reported only for completeness (ignoring scripted votes is always 'accurate') and the
    accuracy outcome for H6 comes from S4's real crowds."""
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
    df = s3_table(rebuild=True)
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
    label_sheet(flips.sample(min(200, len(flips)), random_state=CFG["seed"]))


def label_sheet(sample: pd.DataFrame) -> None:
    """Blind kappa sheet: the labeller sees only what the judge saw (the reason), in random order.
    The judge's answers go to a separate key file, so they cannot anchor the human label."""
    s = sample.reset_index(drop=True).assign(row=lambda x: x.index + 1)
    s[["row", "reason"]].assign(label_first="", label_second="").to_csv(RES / "judge_labels.csv", index=False)
    s[["row", "model", "item_id", "cell", "mentions"]].to_csv(RES / "judge_key.csv", index=False)
    print(f"results/judge_labels.csv: {len(s)} reasons to label YES/NO (answers in judge_key.csv; do not open first)")


def cohen(a: pd.Series, b: pd.Series) -> tuple[int, float, float]:
    """Rows labelled by both, raw agreement, Cohen's kappa for YES/NO labels."""
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in ("YES", "NO"))
    return int(ok.sum()), po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa() -> None:
    """H5 gate: kappa >= 0.7 between the judge and the human keeps the judge's rates on all flips;
    below that, H5 is reported from the human labels alone."""
    d = pd.read_csv(RES / "judge_labels.csv").merge(pd.read_csv(RES / "judge_key.csv"), on="row")
    yn = lambda s: s.map(lambda x: {"Y": "YES", "N": "NO"}.get(str(x).strip().upper()[:1]) if pd.notna(x) else None)
    labels = {c: yn(d[c]) for c in d.columns if c.startswith("label_")}
    labels = {c: h for c, h in labels.items() if h.notna().any()}
    if not labels:
        print("no labels yet: write YES or NO in label_first in results/judge_labels.csv")
    for c, h in labels.items():
        n, po, k = cohen(d.mentions, h)
        print(f"judge vs {c}: {n} rows, agreement {po:.3f}, kappa {k:.3f} -> "
              f"{'keep the judge' if k >= 0.7 else 'report H5 from the human labels'}")
        print(f"  silent share (NO): human {(h.dropna() == 'NO').mean():.3f}, judge on the same rows "
              f"{(d.mentions[h.notna()] == 'NO').mean():.3f}")
        print(pd.crosstab(d.mentions.rename("judge"), h.rename(c)).to_string())
    if len(labels) == 2:
        n, po, k = cohen(*labels.values())
        print(f"{' vs '.join(labels)}: {n} rows, agreement {po:.3f}, kappa {k:.3f}")


def main(B: int, models: list[str] | None = None) -> None:
    """models: restrict every table to these models (S4 keeps the mixed crowd); None = all cached models."""
    RES.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    figs = Figures()
    if (RUNS / "S3").exists():
        df = s3_table(rebuild=True)
        if models:
            df = df[df.model.isin(models)]
        t = h1_h2(df, B)
        t.to_csv(RES / "S6_h1_h2.csv", index=False)
        print("H1/H2 by model (point [95% CI], claims resampled):\n", readable(t).to_string(index=False))
        fc = follow_curves(df)
        fc.to_csv(RES / "S6_follow_curves.csv")
        print("\nfollow curves (independent / sequential / random):\n", fc.to_string())
        figs.follow(fc)
        figs.rho(t)
        figs.gamma(t)
        if (df.part == "fixes").any():
            fx = fixes(df)
            fx.to_csv(RES / "S6_fixes.csv", index=False)
            print("\nfixes vs paired main cells:\n", fx.to_string(index=False))
    if (RES / "S4_sequences.parquet").exists():
        from src.stages.stage4_crowds import paired, summary
        seqs = pd.read_parquet(RES / "S4_sequences.parquet")
        if models:
            seqs = seqs[seqs.crowd.isin(list(models) + ["mixed"])]
        s4 = summary(seqs)
        s4.to_csv(RES / "S6_crowds.csv")
        print("\ncrowds:\n", s4.to_string())
        figs.crowds(s4)
        pf = paired(seqs, "f1", B)
        pf.to_csv(RES / "S6_private_first.csv", index=False)
        br = paired(seqs, "br", B)
        if not br.empty:
            br.to_csv(RES / "S6_bayes_rule_crowds.csv", index=False)
            print("\nBayes-rule crowd vs social crowd (same claims, same k):\n", br.to_string(index=False))
        print("\nH6 private-first crowd vs sequential crowd (same claims):\n", pf.to_string(index=False))
    if (RES / "S5_b1.parquet").exists():
        from src.stages.stage5_reddit import lifts
        s5 = lifts(pd.read_parquet(RES / "S5_b1.parquet"))
        s5.to_csv(RES / "S6_muchnik.csv")
        print("\nMuchnik lifts:\n", s5.to_string())
        figs.muchnik(s5)
    if (RUNS / "S2").exists():
        from src.stages.stage2_urn import table
        models = sorted(p.stem for p in (RUNS / "S2").glob("*.jsonl") if not models or p.stem in models)
        s2 = table(models, 200)
        s2.to_csv(RES / "S6_urn.csv", index=False)
        print("\nurn (Anderson & Holt: cascade rate 0.71, inconsistent ~0.04):\n", s2.to_string(index=False))
    print(f"\nfigures: {', '.join(figs.saved) or 'none'} -> paper/figures/")


# ------------------------------------------------------------------ figures

class Figures:
    """Paper figures (PDF for LaTeX, PNG for preview). Reference palette slots in fixed order:
    slot 1 = the model, slot 2 = a second framing or the human reference; Bayes is ink, dashed."""

    INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    S1, S2 = "#2a78d6", "#eb6834"

    def __init__(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt, self.saved = plt, []
        self.out = ROOT / "paper" / "figures"
        self.out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
            "axes.edgecolor": self.AXIS, "axes.labelcolor": self.INK2, "axes.titlecolor": self.INK,
            "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
            "grid.color": self.GRID, "grid.linewidth": 0.5, "xtick.color": self.MUTED, "ytick.color": self.MUTED,
            "xtick.labelcolor": self.INK2, "ytick.labelcolor": self.INK2, "legend.frameon": False,
            "legend.fontsize": 7, "lines.linewidth": 1.5, "lines.markersize": 4, "savefig.bbox": "tight"})

    def save(self, fig, name: str) -> None:
        for ext in ("pdf", "png"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=200)
        self.plt.close(fig)
        self.saved.append(name)

    def follow(self, fc: pd.DataFrame) -> None:
        """Fig. 1: P(follow | majority opposes own verdict) by |lead|, independent framing, against Bayes."""
        from matplotlib.ticker import MaxNLocator
        d = fc.xs("independent", level="framing")
        models = list(d.index.get_level_values("model").unique())
        ncols = min(5, len(models))
        nrows = -(-len(models) // ncols)
        fig, axes = self.plt.subplots(nrows, ncols, figsize=(1.45 * ncols, 1.75 * nrows), sharey=True,
                                      sharex=True, squeeze=False)
        for ax, m in zip(axes.flat, models):
            x = d.loc[m]
            ax.fill_between(x.index, x.lo, x.hi, color=self.S1, alpha=0.15, linewidth=0)
            ax.plot(x.index, x.follow, color=self.S1, marker="o", label="model")
            ax.plot(x.index, x.bayes, color=self.INK2, linestyle="--", linewidth=1, label="Bayes")
            ax.set_title(m)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(-0.03, 1.03)
        for ax in axes.flat[len(models):]:
            ax.set_visible(False)
        for row in axes:
            row[0].set_ylabel("P(follow majority)")
        fig.supxlabel("|lead| shown against the own verdict", fontsize=8, color=self.INK2, y=-0.02)
        axes[0][0].legend(loc="upper left")
        self.save(fig, "fig1_follow_curves")

    def _forest(self, t: pd.DataFrame, cols: list[tuple[str, str, str]], refs: list[tuple[float, str]],
                xlabel: str, xlim: tuple[float, float], name: str) -> None:
        """Point + 95% CI per model; intervals are clipped to xlim. A point outside xlim sits on the edge and is
        labelled with its true value; the CSV holds the unclipped numbers."""
        fig, ax = self.plt.subplots(figsize=(3.3, 0.35 * len(t) + 0.9))
        y = np.arange(len(t))
        for j, (col, color, label) in enumerate(cols):
            off = (j - (len(cols) - 1) / 2) * 0.22
            est, lo, hi = (t[c].clip(*xlim) for c in (col, f"{col}_lo", f"{col}_hi"))
            ax.errorbar(est, y + off, xerr=[est - lo, hi - est], fmt="o", color=color, ecolor=color,
                        elinewidth=1, capsize=0, label=label)
            for yi, v in zip(y + off, t[col]):               # off-scale points: label them with the true value
                if pd.notna(v) and not (xlim[0] <= v <= xlim[1]):
                    edge = xlim[0] if v < xlim[0] else xlim[1]
                    side = "left" if v < xlim[0] else "right"
                    ax.annotate(f"{v:.2f}", xy=(edge, yi), xytext=(10 if side == "left" else -10, 0),
                                textcoords="offset points", ha=side, va="center", fontsize=6, color=color)
        for i, (x, lab) in enumerate(refs):                  # first label sits left of its line, the next right
            ax.axvline(x, color=self.INK2, linestyle="--", linewidth=1)
            ax.text(x, 1.01, lab, transform=ax.get_xaxis_transform(), color=self.INK2, fontsize=7,
                    ha=("right" if i == 0 and len(refs) > 1 else "left"), va="bottom")
        ax.set_yticks(y, t.model)
        ax.set_xlim(*xlim)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel(xlabel)
        if len(cols) > 1:
            ax.legend(loc="lower right")
        self.save(fig, name)

    def rho(self, t: pd.DataFrame) -> None:
        """Fig. 2: rho per model (q unstated), against its Bayes value 1."""
        self._forest(t, [("rho_unstated", self.S1, "q unstated")], [(1.0, "Bayes")],
                     "ρ: one shown vote in units of the own signal", (-0.1, 1.3), "fig2_rho")

    def gamma(self, t: pd.DataFrame) -> None:
        """Fig. 3: gamma under independent vs sequential framing, against Bayes 1 and 0."""
        self._forest(t, [("gamma_independent", self.S1, "independent"), ("gamma_sequential", self.S2, "sequential")],
                     [(0.0, "Bayes: sequential"), (1.0, "Bayes: independent")],
                     "γ: redundant votes ÷ informative votes", (-1.0, 2.5), "fig3_gamma")

    def crowds(self, s4: pd.DataFrame) -> None:
        """Fig. 4: final-majority accuracy by seeded wrong votes k, against the independent crowd."""
        d = s4.reset_index()
        d = d[d.variant == "social"]
        crowds = list(d.crowd.unique())
        fig, axes = self.plt.subplots(1, len(crowds), figsize=(1.45 * len(crowds), 1.8), sharey=True, squeeze=False)
        for ax, cr in zip(axes[0], crowds):
            x = d[d.crowd == cr].sort_values("k")
            ax.plot(x.k, x.social_acc, color=self.S1, marker="o", label="sequential crowd")
            ax.axhline(x.indep_acc.iloc[0], color=self.INK2, linestyle="--", linewidth=1, label="independent crowd")
            ax.set_title(cr)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_ylim(-0.03, 1.03)
        axes[0][0].set_ylabel("final-majority accuracy")
        fig.supxlabel("seeded wrong votes k", fontsize=8, color=self.INK2, y=-0.08)
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
        self.save(fig, "fig4_crowds")

    def muchnik(self, s5: pd.DataFrame) -> None:
        """Fig. 5: first-voter up-lift per model against Muchnik et al.'s +32%."""
        t = s5.reset_index()
        fig, ax = self.plt.subplots(figsize=(3.3, 0.35 * len(t) + 0.8))
        y = np.arange(len(t))
        ax.errorbar(t.lift_up, y, xerr=[t.lift_up - t.lift_up_lo, t.lift_up_hi - t.lift_up], fmt="o",
                    color=self.S1, ecolor=self.S1, elinewidth=1, capsize=0)
        ax.axvline(0.32, color=self.S2, linestyle="--", linewidth=1)
        ax.text(0.32, 1.01, "humans, +32% (Muchnik et al. 2013)", transform=ax.get_xaxis_transform(),
                color=self.INK2, fontsize=7, ha="right", va="bottom")
        ax.axvline(0, color=self.AXIS, linewidth=0.8)
        ax.set_yticks(y, t.model)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("up-treatment lift in P(upvote)")
        self.save(fig, "fig5_muchnik")


# ------------------------------------------------------------------ pooled tests (B3)

MODELS_SIX = ["M1", "M2", "M3", "M4", "M6", "M7"]     # the six crowd models; "mixed" is a crowd, not a model
CLAIM = {"H1": "rho > 1", "H2": "gamma_sequential > 0.5", "H3": "visible votes cost at least 5 pp (k = 0)",
         "H4": "no downvote correction (down minus control <= 0)", "H6": "private-first gain over social > 0"}
NULL_SIDE = {"H1": lambda d: np.mean(d <= 1.0), "H2": lambda d: np.mean(d <= 0.5),
             "H3": lambda d: np.mean(d < 0.05), "H4": lambda d: np.mean(d > 0), "H6": lambda d: np.mean(d <= 0)}


def draws(df: pd.DataFrame, stat, B: int, seed: int = CFG["seed"]) -> np.ndarray:
    """Bootstrap distribution of stat over claims: the resampling of boot(), kept as draws."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby("item_id").indices.values())
    out = np.full(B, np.nan)
    for b in range(B):
        take = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        out[b] = stat(df.iloc[take])
    return out


def vec_draws(vals: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean of per-claim values."""
    return vals[rng.integers(0, len(vals), size=(B, len(vals)))].mean(axis=1)


def pooled_holm(B: int = 2000) -> pd.DataFrame:
    """Plan row 9: one pooled test per hypothesis. Each model or crowd is resampled over its own claims, the
    draws are averaged with equal weight (model as a fixed effect), and Holm corrects across the tests.
    p is the share of bootstrap draws on the null side of the claim, so a small p supports the claim.
    H5 has no test yet: it waits for the T1 gate. Writes results/S6_holm.csv only."""
    rng = np.random.default_rng(CFG["seed"])
    pooled, est = {}, {}

    s3 = s3_table()                                                       # H1, H2: S3 main cells
    main = s3[(s3.part == "main") & s3.model.isin(MODELS_SIX)]
    h1, h2, e1, e2 = {}, {}, {}, {}
    for m, d in main.groupby("model"):
        ind, seq = d[d.framing == "independent"], d[d.framing == "sequential"]
        ind_u = ind[ind.q.isna()]
        h1[m], e1[m] = draws(ind_u, rho, B), rho(ind_u)
        h2[m], e2[m] = draws(seq, gamma, B), gamma(seq)
    pooled["H1"], est["H1"] = np.nanmean(np.vstack(list(h1.values())), axis=0), e1
    pooled["H2"], est["H2"] = np.nanmean(np.vstack(list(h2.values())), axis=0), e2

    s4 = pd.read_parquet(RES / "S4_sequences.parquet")                    # H3, H6: S4 crowds at k = 0
    s4 = s4[s4.complete & s4.crowd.isin(MODELS_SIX) & (s4.k == 0)]
    acc = lambda maj, truth: (maj == truth).astype(float).where(maj != 0, 0.5).to_numpy()
    gap, gain, gap_d, gain_d = {}, {}, [], []
    for cr, d in s4.groupby("crowd"):
        soc = d[d.variant == "social"].set_index("item_id")
        f1 = d[d.variant == "f1"].set_index("item_id")
        f1 = f1.loc[f1.index.intersection(soc.index)]
        gap[cr] = acc(soc.indep_majority, soc.truth) - acc(soc.final_majority, soc.truth)
        sp = soc.loc[f1.index]                                            # paired with the private-first claims
        gain[cr] = acc(f1.final_majority, f1.truth) - acc(sp.final_majority, sp.truth)
        gap_d.append(vec_draws(gap[cr], B, rng))
        gain_d.append(vec_draws(gain[cr], B, rng))
    pooled["H3"], est["H3"] = np.mean(gap_d, axis=0), {c: v.mean() for c, v in gap.items()}
    pooled["H6"], est["H6"] = np.mean(gain_d, axis=0), {c: v.mean() for c, v in gain.items()}

    s5 = pd.read_parquet(RES / "S5_b1.parquet")                           # H4: S5 comments (M1, M2, M3, M6)
    w = (s5.assign(up=(s5.vote == "UP").astype(float))
         .pivot_table(index=["model", "item_id"], columns="treatment", values="up").dropna())
    dn, de = [], {}
    for m, x in w.groupby(level="model"):
        diff = x["down"].to_numpy() - x["control"].to_numpy()
        dn.append(vec_draws(diff, B, rng))
        de[m] = diff.mean()
    pooled["H4"], est["H4"] = np.mean(dn, axis=0), de

    # Robustness: claim-weighted pooling (each claim counts once, so bigger crowds weigh more).
    cw = []
    for h, per in (("H3", gap), ("H6", gain)):
        vals = np.concatenate(list(per.values()))
        lo, hi = np.percentile(vec_draws(vals, B, rng), [2.5, 97.5])
        cw.append({"check": f"{h} claim-weighted", "estimate": vals.mean(), "lo": lo, "hi": hi, "claims": len(vals)})
    pd.DataFrame(cw).round(4).to_csv(RES / "S6_holm_robustness.csv", index=False)

    rows = []
    for h in ["H1", "H2", "H3", "H4", "H6"]:
        d = pooled[h][~np.isnan(pooled[h])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"hypothesis": h, "claim": CLAIM[h], "units": " ".join(est[h]),
                     "estimate": np.mean(list(est[h].values())), "lo": lo, "hi": hi, "p": NULL_SIDE[h](d)})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.hypothesis.map(holm(dict(zip(out.hypothesis, out["p"]))))
    out.round(4).to_csv(RES / "S6_holm.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"Pooled tests (equal weight across units, B = {B}):\n", out.round(3).to_string(index=False))
    for h in est:
        print(f"  {h} per unit:", {k: round(float(v), 3) for k, v in est[h].items()})
    print("H5: no pooled test yet (waits for the T1 gate). Adding it makes the Holm family six tests.")
    all_gap, all_gain = np.concatenate(list(gap.values())), np.concatenate(list(gain.values()))
    print(f"Register check, six crowds, claims-weighted:\n"
          f"  H3 visible-vote cost at k = 0: {all_gap.mean() * 100:.1f} pp over {len(all_gap)} claim rows "
          f"(register §12.3: 6.0 pp over 1,565 crowds)\n"
          f"  H6 private-first gain: {all_gain.mean() * 100:.1f} pp over {len(all_gain)} claims "
          f"(register §12.3: +6.4 pp over 549 claims)")
    return out


def judge_rates(n: int = 2000, models: list[str] | None = None) -> pd.DataFrame:
    """H5 rates from the cached judge answers only. Rebuilds the flip sample that judge() draws (same seed and
    per-model cap), looks up each reason in runs/S6/judge.jsonl, and makes no API calls. Writes
    results/S6_judge_rates.csv. Unlike judge(), it never rewrites the blind labelling sheet."""
    df = s3_table()
    flips = df[(df.part == "main") & (df.follow == 1)]
    if models:
        flips = flips[flips.model.isin(models)]
    reasons = {}
    for m in flips.model.unique():
        for r in read_jsonl(RUNS / "S3" / f"{m}.jsonl"):
            if (r.get("parsed") or {}).get("reason"):
                reasons[(m, r["meta"]["item_id"], r["meta"]["cell"], r["meta"].get("fix"))] = r["parsed"]["reason"]
    flips = flips.assign(reason=[reasons.get((r.model, r.item_id, r.cell, None)) for r in flips.itertuples()])
    flips = flips[flips.reason.notna()]
    flips = flips.groupby("model", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n // max(1, flips.model.nunique())), random_state=CFG["seed"]))
    judged = {r["messages"][0]["content"]: (r.get("parsed") or {}).get("answer")
              for r in read_jsonl(RUNS / "S6" / "judge.jsonl")}
    flips = flips.assign(mentions=[judged.get(render("judge", reason=x)) for x in flips.reason])
    missing = int(flips.mentions.isna().sum())
    if missing:
        raise SystemExit(f"{missing} sampled flips have no cached judge answer; nothing was requested")
    rates = (flips.groupby("model")
             .agg(flips=("mentions", "size"),
                  silent_share=("mentions", lambda x: (x == "NO").mean()),
                  mention_share=("mentions", lambda x: (x == "YES").mean()))
             .round(3).reset_index())
    rates.to_csv(RES / "S6_judge_rates.csv", index=False)
    print(rates.to_string(index=False))
    print(f"flips judged: {len(flips)}; silent share over all flips: {(flips.mentions == 'NO').mean():.3f}")
    return rates


def muchnik_by_topic(B: int = 2000) -> pd.DataFrame:
    """Muchnik panel by topic group (the approved subreddit map, T2): the S5 lifts recomputed inside each group.
    Writes results/S6_muchnik_by_topic.csv."""
    from src.stages.stage5_reddit import lifts
    s5 = pd.read_parquet(RES / "S5_b1.parquet")
    cols = ["group", "model", "comments", "control", "up", "down", "lift_up", "lift_up_lo", "lift_up_hi",
            "down_vs_control", "down_lo", "down_hi"]
    out = pd.concat([lifts(d, B).reset_index().assign(group=g) for g, d in s5.groupby("group")])[cols]
    out.round(4).to_csv(RES / "S6_muchnik_by_topic.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


def summary() -> None:
    """H2 robustness: the median sequential-framing gamma across the models in S6_h1_h2.csv (no bootstrap)."""
    t = pd.read_csv(RES / "S6_h1_h2.csv")
    t[["model", "gamma_sequential", "gamma_sequential_lo", "gamma_sequential_hi"]].round(4).to_csv(
        RES / "S6_h2_by_model.csv", index=False)
    print(f"H2 median gamma_sequential across {len(t)} models: {t.gamma_sequential.median():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "judge", "kappa", "holm", "judge_rates", "muchnik_topic", "summary"])
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 = point estimates only)")
    ap.add_argument("--n", type=int, default=2000, help="flips to judge")
    ap.add_argument("--models", nargs="+", default=None, help="restrict 'all' to these models")
    a = ap.parse_args()
    if a.what == "all":
        main(a.boot, a.models)
    elif a.what == "judge":
        judge(a.n)
    elif a.what == "holm":
        pooled_holm(a.boot)
    elif a.what == "judge_rates":
        judge_rates(a.n, a.models)
    elif a.what == "muchnik_topic":
        muchnik_by_topic(a.boot)
    elif a.what == "summary":
        summary()
    else:
        kappa()
