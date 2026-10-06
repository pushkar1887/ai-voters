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


def s3_table() -> pd.DataFrame:
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
    figs = Figures()
    if (RES / "S3_cells.parquet").exists():
        df = s3_table()
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
        from src.s4_crowds import summary
        s4 = summary(pd.read_parquet(RES / "S4_sequences.parquet"))
        s4.to_csv(RES / "S6_crowds.csv")
        print("\ncrowds:\n", s4.to_string())
        figs.crowds(s4)
    if (RES / "S5_b1.parquet").exists():
        from src.s5_reddit import lifts
        s5 = lifts(pd.read_parquet(RES / "S5_b1.parquet"))
        s5.to_csv(RES / "S6_muchnik.csv")
        print("\nMuchnik lifts:\n", s5.to_string())
        figs.muchnik(s5)
    if (RUNS / "S2").exists():
        from src.s2_urn import table
        models = sorted(p.stem for p in (RUNS / "S2").glob("*.jsonl"))
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
        d = fc.xs("independent", level="framing")
        models = list(d.index.get_level_values("model").unique())
        fig, axes = self.plt.subplots(1, len(models), figsize=(1.45 * len(models), 1.8), sharey=True, squeeze=False)
        from matplotlib.ticker import MaxNLocator
        for ax, m in zip(axes[0], models):
            x = d.loc[m]
            ax.fill_between(x.index, x.lo, x.hi, color=self.S1, alpha=0.15, linewidth=0)
            ax.plot(x.index, x.follow, color=self.S1, marker="o", label="model")
            ax.plot(x.index, x.bayes, color=self.INK2, linestyle="--", linewidth=1, label="Bayes")
            ax.set_title(m)
            ax.set_xlabel("|lead| shown")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(-0.03, 1.03)
        axes[0][0].set_ylabel("P(follow majority)")
        axes[0][0].legend(loc="upper left")
        self.save(fig, "fig1_follow_curves")

    def _forest(self, t: pd.DataFrame, cols: list[tuple[str, str, str]], refs: list[tuple[float, str]],
                xlabel: str, xlim: tuple[float, float], name: str) -> None:
        """Point + 95% CI per model; intervals are clipped to xlim (an arrow-free cap keeps the scale honest
        for the well-estimated models; the CSV holds the unclipped numbers)."""
        fig, ax = self.plt.subplots(figsize=(3.3, 0.35 * len(t) + 0.9))
        y = np.arange(len(t))
        for j, (col, color, label) in enumerate(cols):
            off = (j - (len(cols) - 1) / 2) * 0.22
            est, lo, hi = (t[c].clip(*xlim) for c in (col, f"{col}_lo", f"{col}_hi"))
            ax.errorbar(est, y + off, xerr=[est - lo, hi - est], fmt="o", color=color, ecolor=color,
                        elinewidth=1, capsize=0, label=label)
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
                     "ρ: one shown vote in units of the own signal", (-0.5, 3.0), "fig2_rho")

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
