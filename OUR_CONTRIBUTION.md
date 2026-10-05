# Our Contribution, and Why the Paper Is Needed

**Do LLM Voters Cascade?** · companion to `DOMAIN_GAPS.md` and `EXPERIMENT_PLAN.md` (v2) · Oct 5, 2026

> **Tags** as in `DOMAIN_GAPS.md`: **[V]** verified · **[I]** inference · **[C]** check before the Oct 7 freeze. Stage names follow the v2 plan (S0–S6, optional X1–X2). Gap IDs (G1–G10) refer to `DOMAIN_GAPS.md`.

---

## Thesis

*The one sentence everything else defends.*

We give the first [C] normative account of how LLM agents weigh other participants' votes on real web content. Every decision is scored against a Bayesian oracle, an empirical optimum and published human data, at both the individual and the crowd level, and every proposed fix is scored against the same three references.

---

## Why the paper is needed

*Three arguments, each tied to gaps in `DOMAIN_GAPS.md`.*

1. **AI participants are already inside crowd systems** (G6, §3 of the gaps file). X has piloted AI-written Community Notes since July 2025, vetted like human notes [V], and shopping agents respond to ratings and endorsements [V]. Platforms need to know whether an AI vote adds information or echoes the votes before it.
2. **The current yardstick is broken** (G1, G2, G8). Conformity rates cannot separate good updating from bad, and they mix social influence with plain repetition. A paper that supplies the normative yardstick changes what the field measures and what it trains models toward.
3. **LLM simulations need a mechanism check** (G5). If LLM crowds are used to test platform rules, the step that creates cascades, one agent reading others' votes, must be validated first. We validate exactly that step.

---

## Gap → what we do → evidence → where

*How each gap is closed, by which stage, and which number in the paper shows it.*

| Gap | What we do | Stage | Evidence produced | Paper |
|---|---|---|---|---|
| G1 · no normative reference | Score every vote against the BHW oracle: herding ratio ρ against ρ_Bayes, implied rater accuracy q\*, follow curves against the Bayes step | S3 (S4 replicates) | ρ and q\* per model, claim-clustered 95% CIs | §4 |
| G2 · repetition and anchoring | Random-labeler control showing the same tallies; randomized label order | S3 | Belief shift per vote, social versus non-social | §4 |
| G3 · no theory or human comparison | BHW oracle; empirical optimum (Weizsäcker's method); like-for-like urn replication | S2, S3, S4 | Cascade rate against 71%; payoff share against 53% | §4–5 |
| G4 · correlation neglect | The same votes framed as independent or as sequential; cascade sequences split into informative and redundant votes | S3 | γ per framing, against the Bayes values 1 and 0 | §4 |
| G5 · micro versus macro | The same models measured per decision (S3) and in 12-agent crowds (S4); solo verdicts give a free independent-crowd counterfactual | S3, S4 | Wisdom loss with CIs | §5 |
| G6 · manipulation | Bot seeding with k = 1–3 wrong votes; Muchnik-style random prior vote on real comments | S4, S5 | Seeding flip rate; lifts against +32% and +25%; downvote correction | §5–6 |
| G7 · undisclosed influence | Silent-herding rate on 2,000 stratified flips, judged by a model validated at κ ≥ 0.7 | S3, S4, S6 | Rate per model | §4 |
| G8 · fixes without a target | Private-first rating, independence cue, hidden counts, Bayes-rule prompt; optional belief-trace LoRA | S3/S4 fix cells, X1 | Accuracy recovered; ρ and γ after each fix, judged in both directions | §7 |
| G9 · label leakage | Drop the justification and rating-history fields; recognition probe; recent split | S0, S1 | Recognition rate per model | §3 |
| G10 · unused human data | Anderson–Holt urn design replicated with LLM agents | S2 | Like-for-like table against the lab numbers | §4 |

---

## What is new against the closest work

*Where a reviewer will look first, and our answer.*

| Closest work | What it shows | What we add |
|---|---|---|
| Jain & Krishnamurthy (arXiv 2411.01271) | LLM agents herd on real text; stochastic control delays herding | A Bayes oracle and human references; the correlation test; platform voting and seeding; current model families |
| BenchForm (Weng et al., ICLR 2025) | Conformity rates; personas and reflection reduce them | A normative target, so a fix is judged by accuracy, in both directions |
| Hu & Qu (2026) | Most "conformity" survives with no speaker | That control is built in, and we measure the *weight* of a vote, not only whether answers change |
| Liu et al. (2026) | Herding versus wisdom in synthetic LLM networks | Real claims and real comments; human anchors; manipulation |
| de Arruda et al. (2026); Li & Ji (2026) | Aggregate or causal validity of LLM simulations | Validity of the social-signal step itself, micro and macro on the same models |
| Allouah et al. (WWW 2026) | Shopping agents respond to platform signals | AI participants reading signals that other AIs produced; seeding attacks |
| Anderson & Holt; Weizsäcker; Muchnik et al. | Human herding, in the lab and in the field | The same designs and numbers, applied to LLM agents |

---

## Contributions, as they will read in the paper

1. **A normative evaluation** of LLM voters on real web content, scored against a Bayes oracle, an empirical optimum and human data.
2. **Two model-level diagnostics,** the herding ratio ρ (with its belief version q\*) and the correlation-neglect coefficient γ, across six models from four families.
3. **Crowd-level evidence** on wisdom loss and bot seeding, plus a randomized prior-vote replication of Muchnik et al. on real Reddit comments.
4. **Fixes judged by accuracy, not by independence,** with a release of code, prompts, item IDs, the oracle and every model output.

---

## What we can claim, and what we will not

*Fixed now, so the writing does not drift toward the results we hope for.*

**Can claim, if the data support it:**

- How much one shown vote moves each model, relative to Bayes, and whether its error points the same way as people's.
- Whether models discount correlated votes.
- Whether visible votes cost AI crowds accuracy, and how far k seeded votes move them.
- Whether cheap protocol changes recover that accuracy.

**Will not claim:**

- That LLM agents "behave like people". The human numbers come from urns and comment voting, not fact-checking, so we compare directions and curve shapes, and claim numeric equivalence only in the urn replication (S2).
- Effects in deployed systems; we have no deployment logs.
- Generality beyond English, US political claims and Reddit.
- Anything about normative (approval-seeking) influence.

---

## Why WWW, and which track

WWW 2027 runs a *Social Networks and Social Media* track and a *Responsible Web* track [V], and WWW 2026 published Allouah et al. on agentic commerce [V]. Our object is the web's own machinery (votes, ratings, crowd verdicts) with AI participants inside it. Primary track: Social Networks and Social Media. Fallback: Responsible Web.

---

## Elevator pitch (about 60 words)

> AI agents now vote, rate and fact-check alongside people. A crowd signal is only useful if each participant adds information instead of copying it. We test six LLMs against a Bayesian oracle and human cascade data, on real fact-checked claims and real Reddit comments, and find [RESULT]. Visible votes cost AI crowds [X] points of accuracy; a private-first protocol recovers [Y].

## Abstract skeleton (numbers filled on Oct 17)

1. **Context.** AI participants in crowd judgment: AI-written Community Notes, the agentic web.
2. **Gap.** Conformity is measured without a normative reference (G1, G2).
3. **Method.** Bayes oracle, empirical optimum and human references; S2–S5 on LIAR2 claims and Reddit comments; six models from four families.
4. **Findings.** ρ against ρ_Bayes [..]; γ under the sequential framing [..]; wisdom loss [..] pp; seeding flip rate [..]; silent-herding rate [..]%.
5. **Fix.** Private-first rating recovers [..]% of the lost accuracy.
6. **Release.** Code, prompts, items, oracle, outputs.

---

## The objections reviewers will raise to the *need*

| Objection | Answer |
|---|---|
| "AI agents don't really vote on platforms yet." | X has piloted AI-written Community Notes since July 2025 [V], and shopping agents already act on ratings [V]. The paper is about the rules to set *before* scale |
| "This is conformity research again." | Conformity research has no normative target (G1). We show when following the crowd is right, and score models on that |
| "Urns are not fact-checking." | Urns are used only for the like-for-like comparison (S2); the web results come from claims and comments |
| "Same-model crowds are trivially correlated." | Yes, and we measure it: the empirical optimum prices that correlation in, and the mixed crowd varies it (G4) |
| "Why not just tell agents to ignore votes?" | That throws away real information when following is optimal (G8). Hidden counts are one of our fixes, and are scored for exactly that cost |

---

## Before the Oct 7 freeze

- [ ] Re-run the novelty search for "first normative account" and for G4 and G10; downgrade the wording if anything close appears [C].
- [ ] Open and check the two [C] items in `DOMAIN_GAPS.md` §1 (arXiv 2508.18321; Arghal et al., ICLR 2026).
- [ ] Align H1's benchmark with ρ_Bayes in the plan PDF (see `EXPERIMENT_PLAN.md` §11, row 8).

Sources: see `DOMAIN_GAPS.md` and the WWW 2027 track list — https://www2027.thewebconf.org/research-track-papers/
