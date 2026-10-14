# Do LLM Voters Cascade?

Do LLM agents that vote after seeing earlier votes weigh them the way a Bayesian would, the way people do, or by blindly following the crowd?

This repository holds the code, configuration, prompts, claim pools and analysis behind a WWW 2027 research-track paper. Every number in the paper is produced by the commands in [Running the pipeline](#running-the-pipeline).

> **Status (14 October 2026).** Ten models have completed every stage and form the basis of the results. Three later models (GLM-5.3, DeepSeek-v4.1-flash and Kimi-K3) are unfinished and excluded. See [Status](#status).

## The question

In a cascade, later voters copy the votes they can see instead of trusting their own judgement. We ask whether LLM voters cascade, and if so, whether they do it for the right reasons. Each model is compared with a Bayesian benchmark: a rational voter who weighs a shown vote by how reliable it is, and treats repeated or post-cascade votes as redundant.

## Design

Two experiments share one pipeline.

- **Urn experiment (S2).** The Anderson and Holt (1997) urn cascade. A fair coin picks one of two urns. Each agent draws one ball, and the earlier guesses are visible. The human reference cascade rate is 0.71.
- **Claim experiment (S1, S3, S4).** Fact-checked political claims from LIAR2. Each model judges a claim alone, which gives its private signal. It then judges the same claim again after seeing other reviewers' votes, scripted in several framings (independent, sequential, random and cue). Crowds are built from these responses, both clean and seeded with bot votes. S4 also tests a private-first condition, where a model commits to its own view before it sees any votes, and a Bayes-rule prompt.
- **Reddit (S5).** Whether an upvote or downvote on a comment changes later voters' behaviour, replicating Muchnik et al. (2013) across seven topic groups.
- **Analysis (S6).** Pooled tests, tables and figures, the silent-herding judge (H5) and its validation.

| Stage | Question | Module | Output |
|---|---|---|---|
| S0 | Claims, comments, personas; Bayes oracle checks | `src/core/prepare_data.py`, `src/core/bayes_oracle.py` | `data/processed/` |
| S1 | Each model's private judgement of every claim; claim pools | `src/stages/stage1_private_signals.py` | `data/pools/`, `runs/S1/` |
| S2 | Urn cascade | `src/stages/stage2_urn.py` | `runs/S2/` |
| S3 | Scripted vote histories and fix cells | `src/stages/stage3_histories.py` | `runs/S3/` |
| S4 | Crowds: clean, seeded, private-first, Bayes-rule prompt | `src/stages/stage4_crowds.py` | `runs/S4/` |
| S5 | Reddit first-voter experiment | `src/stages/stage5_reddit.py` | `runs/S5/` |
| S6 | Every number, table and figure in the paper | `src/analysis/paper_results.py` | `results/`, `paper/figures/` |

## Hypotheses

The hypotheses were pre-registered in the project plan and frozen on 7 October 2026. H1 and H2 carry the paper.

| | Hypothesis | Refuted if |
|---|---|---|
| H1 | Over-herding: a shown vote counts for more than Bayes allows | the 95% CI covers ρ_Bayes for at least 4 of 6 base models |
| H2 | Correlation neglect: sequential (correlated) votes are not discounted | γ_seq ≤ 0.5 |
| H3 | Wisdom loss: visible votes hurt crowd accuracy, and seeded bot votes flip outcomes | the gap is below 5 pp, and the seeding effect is no larger than a Bayesian crowd's |
| H4 | Symmetric herding: no Muchnik-style correction after a downvote | downvoted comments are corrected, as in humans |
| H5 | Silent herding: most vote-driven flips never mention the votes | 50% or fewer of the flips are unacknowledged |
| H6 | Cheap fixes work: a private-first step beats prompting with the Bayes rule | no fix beats the Bayes-rule prompt |

ρ is the weight a shown vote gets, measured against the model's own signal. ρ_Bayes = logit(q) / logit(p̂) is the Bayesian weight, where q is the reliability of the shown votes and p̂ is the model's own accuracy. γ is the ratio of redundant to informative votes.

## Models

The ten models with completed results:

| Name | Model | Backend | Notes |
|---|---|---|---|
| M1 | `meta/llama-3.2-11b-vision-instruct` | NIM | |
| M2 | `nvidia/nemotron-3-super-120b-a12b` | NIM | thinking off |
| M3 | `nvidia/nemotron-3.5-lightning-30b-a3b` | NIM | thinking off |
| M4 | `nvidia/nemotron-3-super-120b-a12b` | NIM | thinking on |
| M6 | `openai/gpt-oss-20b` | NIM | reasoning low; also the H5 judge |
| M6H | `openai/gpt-oss-20b` | NIM | reasoning medium |
| M7 (Ultra) | `nvidia/nemotron-3-ultra-550b-a55b` | NIM | thinking off |
| L1 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | vLLM on Kaggle | |
| L2 | `Qwen/Qwen3-8B-AWQ` | vLLM on Kaggle | thinking off |
| L2T | `Qwen/Qwen3-8B-AWQ` | vLLM on Kaggle | thinking on |

Pending and excluded: M8 `z-ai/glm-5.3`, M9 `deepseek-ai/deepseek-v4.1-flash`, M10 `google/gemma-4-31b-it` (parked) and M11 `moonshotai/kimi-k3`. `config.yaml` holds the settings for all of them.

## Status

As of 14 October 2026, on the ten completed models:

| | Verdict | Key numbers |
|---|---|---|
| H1 | Not supported | Pooled ρ − ρ_Bayes = −0.24 [−0.27, −0.21] at q = 70%. Only M3 weights shown votes above Bayes (ρ 0.81 [0.70, 0.97] against 0.58). Eight of ten under-weight. Not refuted by its count rule: 1 of 6 base models cover ρ_Bayes |
| H2 | Refuted | Median γ_seq across the ten models is 0.15. M1 alone is −6.5, so the pooled mean is not used |
| H3 | Borderline | Visible votes cost 4.7 pp pooled [3.0, 6.5] against a 5 pp threshold. Claim-weighted: 5.2 pp [3.5, 7.0] |
| H4 | Model-specific | M1 corrects after downvotes; M2, M3 and M6 do not. The pooled test cannot show the effect |
| H5 | Supported | Pooled silent share 93.3% [91.9, 94.7]. The judge agrees with 200 blind human labels (κ = 0.97) |
| H6 | Supported | Private-first recovers the loss: +5.4 pp [2.4, 8.5] pooled, Holm p = 0.0025 |

These values come from the `paper_results` commands run on 14 October. Re-run them after any change.

Known limitations:
- The pooled tests cover six crowd models with equal weight. A claim-weighted check is reported beside them.
- The per-topic Muchnik panel has 300 comments per model per topic, which is too few for topic-level claims.
- Ultra recognises 56% of its claims. Contamination has not been checked.
- Pooled figures from earlier drafts (H3 at 6.0 pp, H6 at +6.4 pp) did not reproduce. The values above are the reproduced ones.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt    # Windows; on Linux use .venv/bin/python
```

- **NIM keys.** Put the keys in `nvidia_keys.txt` at the repo root, in any layout. The file is git-ignored, and `src/core/nim_client.py` finds keys by their `nvapi-` pattern. Check them with `python -m src.core.nim_client --check-keys`.
- **Run-time settings.** `NIM_KEY_SLICE` chooses which keys a process uses (for example `0:8,9:16`). `NIM_RPM` sets requests per minute per key, and `NIM_CONCURRENCY` sets requests in flight per process.
- **Kaggle.** Put credentials in `~/.kaggle/kaggle.json`. The local models (L1, L2, L2T) run on a Kaggle T4 × 2 machine through `kaggle/kaggle_job.py`.

## Running the pipeline

```bash
python -m src.core.prepare_data all                                                   # S0: claims, comments, personas
python -m src.core.bayes_oracle                                                       # S0: oracle checks (hand-worked and exhaustive)
python -m src.core.nim_client --smoke                                                 # G1: every NIM model answers and parses
python -m src.stages.stage1_private_signals --models M1 M2 M3 M4 M6 M6H --refill 2078 # S1: private signals -> data/pools/
python -m src.stages.stage2_urn --models M1 M2 M3 M4 M6 M6H                           # S2: urn anchor
python -m src.stages.stage3_histories --models M1 M2 M3 M4 M6 M6H --part main         # S3: scripted histories
python -m src.stages.stage3_histories --models M1 M2 M3 M4 M6 M6H --part fixes        # S3: fix cells
python -m src.stages.stage4_crowds --crowds M1 M2 M3 M4 M6 mixed                      # S4: clean and seeded crowds
python -m src.stages.stage4_crowds --crowds M1 M2 M3 M4 M6 mixed --f1                 # S4: private-first crowd
python -m src.stages.stage5_reddit --models M1 M2 M3 M6                               # S5: Reddit first voter
python -m src.analysis.paper_results judge            # H5: first run only; draws the blind labelling sheet, never overwrites one
python -m src.analysis.paper_results judge_fill       # H5: judge calls for the flip sample (uncached reasons only)
python -m src.analysis.paper_results judge_rates      # H5: silent-herding rates from the cache
python -m src.analysis.paper_results kappa            # H5 gate, once the sheet is labelled
python -m src.analysis.paper_results all              # S6: tables and figures
python -m src.analysis.paper_results holm             # pooled Holm tests and the claim-weighted check
python -m src.analysis.paper_results muchnik_topic    # S5 Muchnik panel within each topic group
python -m src.analysis.paper_results summary          # H2 median across models
```

`all`, `holm`, `judge_rates`, `muchnik_topic` and `summary` make no API calls. They read the cache only. Those are the commands that produce the paper's numbers.

Local models run the same stage commands on Kaggle, then come home as cache files:

```bash
python kaggle/kaggle_job.py push --name s1-l1 --cmd "src.stages.stage1_private_signals --models L1 --refill 2078"
python kaggle/kaggle_job.py pull --name s1-l1
python -m src.stages.stage1_private_signals --models L1 --refill 2078      # all cache hits: rebuilds the pool locally
```

**The cache.** Every API call is stored in `runs/<stage>/<model>.jsonl`, keyed by a hash of the request. Re-running is free, and an interrupted run resumes where it stopped. The cache is git-ignored and large, and it is needed to regenerate any number.

## Repository layout

```
config.yaml                  every setting for every stage and model
prompts/templates_v1.yaml    prompt templates, frozen at G2
src/core/                    services used by every stage
  nim_client.py              NIM calls, key pool, cache
  bayes_oracle.py            Bayes benchmark and urn agents
  prepare_data.py            claims, comments, personas
src/stages/                  the five experiment stages, one module each
  stage1_private_signals.py  S1
  stage2_urn.py              S2
  stage3_histories.py        S3
  stage4_crowds.py           S4
  stage5_reddit.py           S5
src/analysis/
  paper_results.py           every number, table and figure in the paper (S6)
kaggle/kaggle_job.py         runs a stage on a Kaggle GPU machine
data/pools/                  claim pools per model (from S1)
data/processed/              prepared claims, comments, personas, subreddit map
```

## Data

- **Claims.** LIAR2, a Hugging Face dataset of PolitiFact statements. `prepare_data.py` cleans them and splits them into pools and evaluation sets.
- **Comments.** Reddit comments from the ConvoKit corpus, mapped to seven topic groups in `config.yaml`.
- **Personas.** Sixty synthetic personas (age band, occupation, region).

Check the source terms before sharing raw claim or comment text outside the team.

## Working rules

- **No keys in git.** `.gitignore` is an allowlist, so only listed paths can enter git. Before each commit, run `git grep -n -E "nvapi-[A-Za-z0-9_-]{20,}"`. It should print nothing.
- **Prompts are frozen.** `templates_v1.yaml` froze at G2. Any change goes into `templates_v2.yaml` and needs a re-run.
- **The labelling sheet is never overwritten.** `judge` creates `results/judge_labels.csv` only if it is missing. `judge_fill` and `judge_rates` never touch it.
- **Numbers come from the commands.** The paper's numbers come only from the `paper_results` commands listed above.
- **Log decisions and runs.** Record every run, stop, restart and decision in the project log, which is kept outside this repository.

## Known issues

- NIM retired the original runbook models (HTTP 410) on 6 October 2026. `config.yaml` holds the current roster.
- The NIM gateway cuts requests at about 300 seconds (HTTP 504), so very long reasoning outputs fail. Reasoning effort is set per model to avoid this.
- Some endpoints (GLM-5.3, DeepSeek-v4.1-flash) returned few or no answers for long stretches, and Kimi-K3 is slow. Completed calls are kept in the cache.
- `src/s1_signals.py` to `src/s4_crowds.py` are temporary forwarding files for runs started under the old module names. Delete them once no run uses those names.

## Licence

Not yet chosen. Decide before any public release.
