# Do LLM Voters Cascade?

In a cascade, later voters copy the votes they can see instead of trusting their own judgement. This repository asks whether LLM voters cascade, and if so, whether they do it for the right reasons. Each model is compared with a Bayesian benchmark: a rational voter who weighs a shown vote by how reliable it is and treats repeated or post-cascade votes as redundant. Three experiments test this: the Anderson and Holt (1997) urn cascade; fact-checked political claims from LIAR2, judged alone and again after seeing other reviewers' ratings; and a Reddit first-voter experiment that replicates Muchnik et al. (2013) across seven topic groups. The code, configuration, prompts, claim pools and analysis behind the paper are all here. The paper's numbers come only from the commands in [Section 5](#5-reproduce-the-results).

## Repository map

```
Datasets/
  prepare_data.py              claims, comments, personas
  processed/                   prepared claims, Reddit comments, personas, subreddit list, build report
  pools/                       per-model claim pools (output of S1)
Experiments/
  core/                        NIM client (key pool, cache); Bayes oracle
  stage1_private_signals.py    S1: private signals and claim pools
  stage2_urn.py                S2: urn cascade
  stage3_histories.py          S3: scripted vote histories and fix cells
  stage4_crowds.py             S4: crowds, seeding, private-first and Bayes-rule prompts
  stage5_reddit.py             S5: Reddit first voter
  kaggle/                      launcher for the local models on Kaggle GPUs
Analysis/
  paper_results.py             every number, table and figure in the paper (S6)
config.yaml                    settings for every stage and model
prompts/                       frozen prompt templates
```

Each stage is a module run from the repository root with `python -m`. Every model call goes through one runner that caches the reply, so a stage can be rerun without new calls. S0 prepares the data, S1 builds the per-model claim pools that S3 and S4 draw on, and S6 reads the tables written by S2 to S5.

| Stage | Question | Module | Output |
|---|---|---|---|
| S0 | Claims, comments, personas; Bayes oracle checks | `Datasets/prepare_data.py`, `Experiments/core/bayes_oracle.py` | `Datasets/processed/` |
| S1 | Each model's private judgement of every claim; claim pools | `Experiments/stage1_private_signals.py` | `Datasets/pools/`, `runs/S1/`, `results/S1_summary.csv` |
| S2 | Urn cascade (Anderson and Holt 1997) | `Experiments/stage2_urn.py` | `runs/S2/`, `results/S2_table.csv` |
| S3 | Scripted vote histories and fix cells | `Experiments/stage3_histories.py` | `runs/S3/`, `results/S3_cells.parquet` |
| S4 | Crowds: clean, seeded, private-first, Bayes-rule prompt | `Experiments/stage4_crowds.py` | `runs/S4/`, `results/S4_sequences.parquet` |
| S5 | Reddit first-voter experiment | `Experiments/stage5_reddit.py` | `runs/S5/`, `results/S5_b1.parquet` |
| S6 | Every number, table and figure in the paper | `Analysis/paper_results.py` | `results/`, `paper/figures/` |

H1 and H2 carry the paper. The hypotheses were fixed before the runs.

| | Hypothesis | Refuted if |
|---|---|---|
| H1 | Over-herding: a shown vote counts for more than Bayes allows | the 95% CI covers ρ_Bayes for at least 4 of 6 base models |
| H2 | Correlation neglect: sequential (correlated) votes are not discounted | γ_seq ≤ 0.5 |
| H3 | Wisdom loss: visible votes hurt crowd accuracy, and seeded bot votes flip outcomes | the gap is below 5 pp, and the seeding effect is no larger than a Bayesian crowd's |
| H4 | Symmetric herding: no Muchnik-style correction after a downvote | downvoted comments are corrected, as in humans |
| H5 | Silent herding: most vote-driven flips never mention the votes | 50% or fewer of the flips are unacknowledged |
| H6 | Cheap fixes work: a private-first step beats prompting with the Bayes rule | no fix beats the Bayes-rule prompt |

ρ is the weight a shown vote gets, measured against the model's own signal. ρ_Bayes = logit(q) / logit(p̂) is the Bayesian weight, where q is the reliability of the shown votes and p̂ is the model's own accuracy. γ is the ratio of redundant to informative votes. Section 4 gives the exact definitions and denominators.

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

M8, M9, M10 and M11 are also configured in `config.yaml` but are not part of the results.

Large or sensitive files (API keys, the run caches in `runs/`, raw downloads in `Datasets/raw/`, and the human labelling sheet for H5) are excluded from the repository.

## 1. Prepare the Environment

Clone the repository and create a virtual environment with Python 3.10:

```bash
export ANONYMOUS_REPOSITORY_URL=https://github.com/OWNER/REPOSITORY.git
git clone "$ANONYMOUS_REPOSITORY_URL" ai-voters
cd ai-voters
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python -m pip install -r requirements.txt
```

Run all commands from the repository root. `requirements.txt` includes `pyarrow`, which the parquet outputs need. The Kaggle launcher also imports the `kaggle` package, which is not in `requirements.txt` (`pip install kaggle`).

- **NIM keys.** Put the keys in `nvidia_keys.txt` at the repository root, in any layout. The file is git-ignored, and keys are found by their `nvapi-` pattern. The `NIM_KEYS` environment variable can hold them instead. Check them with `python -m Experiments.core.nim_client --check-keys`, which makes one small call per key and prints a masked report.
- **Run-time settings.** `NIM_KEY_SLICE` chooses which keys a process uses (for example `0:8,9:16`). `NIM_RPM` sets requests per minute per key, and `NIM_CONCURRENCY` sets requests in flight per process; both default to the values in `config.yaml`. `AIV_RUNS` sets a different cache folder (default `runs/`).
- **Kaggle.** Put the API credentials in `~/.kaggle/kaggle.json`.

## 2. Prepare the Data

```bash
python -m Datasets.prepare_data all          # claims, comments, personas
python -m Experiments.core.bayes_oracle      # Bayes oracle checks
```

The processed files are already in `Datasets/processed/`, so these steps only need to run to rebuild them.

- **Claims.** LIAR2, a Hugging Face dataset of PolitiFact statements (`chengxuphd/liar2`). The script checks the label counts against the published ones and keeps the ratings true, false and pants-fire, mapped to TRUE and FALSE. It keeps claims of 6 to 60 words and writes three disjoint sets: `eval` (1,500 claims), `lora` (1,500, written but not used by any stage) and `reserve` (2,078, the source of the `--refill` claims).
- **Comments.** The ConvoKit Reddit small corpus (`reddit-corpus-small`). The script keeps comments of 15 to 120 words, maps subreddits to seven topic groups (the `data.comments.groups` map in `config.yaml`; other subreddits are dropped), and samples 300 comments per group.
- **Personas.** 60 synthetic personas (age band, occupation and region), built from the lists in `config.yaml` with seed 13.

Raw downloads go to `Datasets/raw/`, which is git-ignored. Check the source terms of LIAR2 and the ConvoKit corpus before redistributing raw claim or comment text.

Expected contents of `Datasets/processed/`, as checked out (record counts, and SHA-256 of each file's bytes):

| File | Records | SHA-256 |
|---|---|---|
| `build_report.json` | 2 top-level keys (`claims`, `comments`) | `d6192f06730d07f756e8717e3520f4947f19d65a93dc1c606fe62b6e9d334b87` |
| `claims_eval.jsonl` | 1,500 claims (750 TRUE, 750 FALSE) | `23304c6b84af6c93d781076ef2a955cb258641ddb0c336c27e7d2d11b4b83c91` |
| `claims_lora.jsonl` | 1,500 claims (750 TRUE, 750 FALSE) | `c9f8083624b3822fc3909d1ee752c195d907b1b5b2fe2a28aa61f7052b9b6dd0` |
| `claims_reserve.jsonl` | 2,078 claims (1,039 TRUE, 1,039 FALSE) | `02133dec1d4febff1c3fa47a7b0dafcff866459f95670bbf5369da705f10278c` |
| `comments.jsonl` | 2,100 comments (300 in each of 7 groups) | `3e8e259c2681abb85a3d918b2da0db8808348dba3dfa0a111c077257fa0436a3` |
| `personas.json` | 60 personas | `83331769d602bf91a6e21ddff33961828edef42e807bb52d8298c7483152adb8` |
| `subreddits.csv` | 100 subreddits (rows after the header) | `115b56c8dcf61a884eab2101b1a578527bfde9c7a07928d2901d7d98c78cd13f` |

Compare the checksums after a rebuild to confirm that it reproduces these files.

## 3. Generate Model Answers (S1–S5)

The commands below call the NIM models over HTTP. The local models (L1, L2, L2T) run on Kaggle instead, as shown after them. Before a run, check that every NIM model answers and parses with `python -m Experiments.core.nim_client --smoke` (five claims per model).

```bash
python -m Experiments.stage1_private_signals --models M1 M2 M3 M4 M6 M6H --refill 2078   # S1: private signals -> Datasets/pools/
python -m Experiments.stage2_urn --models M1 M2 M3 M4 M6 M6H                             # S2: urn cascade (200 rounds)
python -m Experiments.stage3_histories --models M1 M2 M3 M4 M6 M6H --part main           # S3: scripted histories
python -m Experiments.stage3_histories --models M1 M2 M3 M4 M6 M6H --part fixes          # S3: fix cells (after main)
python -m Experiments.stage4_crowds --crowds M1 M2 M3 M4 M6 mixed                        # S4: clean and seeded crowds
python -m Experiments.stage4_crowds --crowds M1 M2 M3 M4 M6 mixed --f1                   # S4: private-first crowd
python -m Experiments.stage4_crowds --crowds M2 M3 --br                                  # S4: Bayes-rule prompt crowd
python -m Experiments.stage5_reddit --models M1 M2 M3 M6                                 # S5: Reddit first voter
```

M7 (Ultra) uses the same modules with its own code (for example `--models M7`).

The local models run the same stage commands on a Kaggle GPU (T4 × 2) and bring their caches home:

```bash
python Experiments/kaggle/kaggle_job.py push --name s1-l1 --cmd "Experiments.stage1_private_signals --models L1 --refill 2078"
python Experiments/kaggle/kaggle_job.py status --name s1-l1
python Experiments/kaggle/kaggle_job.py pull --name s1-l1
python -m Experiments.stage1_private_signals --models L1 --refill 2078      # all cache hits: rebuilds the pool locally
```

`push` bundles the code, `config.yaml`, prompts, processed data, pools and any caches named with `--caches` (for example `--caches S1/L1`) into a private Kaggle dataset, then starts a GPU kernel. `--vllm` pins the vLLM version (default `vllm==0.31.0`). `pull` merges the kernel's caches into `runs/`, skipping requests already present. L2 and L2T follow the same pattern.

**The cache.** Every API call is stored in `runs/<stage>/<model>.jsonl`, with a `.manifest.json` beside it. Each record is keyed by a SHA-1 hash of the full request (model, messages, decoding settings and the model's extra options) plus a seed tag for repeated samples. A rerun reads the cache first, so it costs nothing, and an interrupted run resumes where it stopped. The cache is git-ignored and is needed to regenerate any number.

## 4. Metric Settings

The pooled tests use the six crowd models M1, M2, M3, M4, M6 and M7. H4 uses the S5 models M1, M2, M3 and M6.

- **ρ (herding ratio).** The weight of one shown vote in units of the model's own signal. For each model, on the independent-framing cells, fit `logit(vote_true) ~ s + lead_shown`, where `vote_true` is 1 for TRUE, `s` is the own signal (±1) and `lead_shown` is the sum of the shown votes. ρ = coefficient of `lead_shown` / coefficient of `s`. Denominator: the coefficient of `s`. Reported with q unstated and with q = 70% stated.
- **ρ_Bayes = logit(q) / logit(p̂).** q is the reliability of the shown votes: 0.70 when the prompt states it (`Q` in `Analysis/paper_results.py`). With q unstated, the benchmark is 1, used only as a robustness check. p̂ is the model's accuracy with its private signal on its claim pool (`p_hat` in `Datasets/pools/<model>.json`). Denominator: logit(p̂).
- **`q*`.** expit(β), where β is the OLS slope of `delta` on `lead_shown`, over the q-stated independent cells. `delta` is the change in the model's logit P(TRUE) from its solo answer to its answer after the votes, with probabilities clipped to 1–99%.
- **γ (correlation neglect).** On the sequential-framing cells, OLS of `delta` on `lead_inf` (the lead before a cascade, the informative part) and `extra` (the shown lead minus `lead_inf`, the redundant post-cascade votes). γ = β_extra / β_inf, and γ_seq is this value on the sequential cells. Bayes gives γ = 0 for sequential votes and γ = 1 for independent votes. Denominator: β_inf, which is unstable when it is near zero, so both coefficients are reported.
- **Crowd accuracy gap (pp).** Per claim, the accuracy of the independent crowd's majority minus the accuracy of the sequential crowd's final majority, for the clean crowd (k = 0) of 12 agents. A tie counts as half. Denominator: the claims with a complete chain. Pooled over the six crowd models with equal weight and reported in percentage points; the claim-weighted version is a robustness check. The seeding effect is the k = 0 accuracy minus the accuracy at k = 1 to 3 seeded wrong votes, on the same claims.
- **Silent-herding share.** Flips are S3 main cells in which the shown majority opposes the model's own verdict and the model follows that majority. For each flip, the judge answers YES or NO to whether the model's stated reason mentions or relies on the other reviewers' ratings. The judge is M6 (`openai/gpt-oss-20b`, reasoning low) at temperature 0 (greedy, top_p 1.0, 1,024-token cap). The sample is up to 2,000 flips (`--n 2000`), split equally across models. Denominator: the judged flips. The share is the NO answers over the judged flips.
- **κ (Cohen's kappa).** Agreement between the judge and the blind human YES/NO labels on the sheet of 200 reasons, over the rows labelled by both. The gate is κ ≥ 0.7. Below that, H5 is reported from the human labels alone.
- **Holm correction.** Each model or crowd is resampled over its own claims, and the draws are averaged with equal weight. A test's p-value is the share of draws on the null side. Holm's step-down correction is applied across H1 to H6. The pooled H5 test uses the cached judge answers of the six crowd models.
- **Bootstrap CIs.** Percentile 95% intervals from a cluster bootstrap over claims (resampling by `item_id`), with B = 2,000 resamples (`--boot 2000`, the default) and seed 13 from `config.yaml`.
- **Other fixed settings.** Greedy decoding (temperature 0, top_p 1.0) for the S1 persona verdicts, the S3 cells and the judge. Sampling (temperature 0.7, top_p 0.95) for the S1 screening samples, S2, S4 and S5. The urn signal accuracy is 2/3 (`URN_P` in `Experiments/core/bayes_oracle.py`).

## 5. Reproduce the Results

Run these from the repository root. `all`, `holm`, `judge_rates`, `muchnik_topic` and `summary` make no API calls: they read the cache in `runs/` and the tables in `results/`. The H5 `judge` and `judge_fill` commands call M6 only for judge answers that are not yet cached. `all` also writes the S2 urn table, `results/S6_urn.csv`.

### H1 and H2

```bash
python -m Analysis.paper_results all --boot 2000
python -m Analysis.paper_results summary
```

Writes `results/S3_cells.parquet`, `results/S6_h1_h2.csv` (ρ, ρ_Bayes, `q*` and γ per model, with intervals), `results/S6_follow_curves.csv`, `results/S6_h2_by_model.csv` (from `summary`), and `paper/figures/fig1_follow_curves`, `fig2_rho` and `fig3_gamma` (PDF and PNG).

### H3 and H6

```bash
python -m Analysis.paper_results all --boot 2000
python -m Analysis.paper_results holm --boot 2000
```

Writes `results/S6_crowds.csv`, `results/S6_private_first.csv`, `results/S6_bayes_rule_crowds.csv` (when the Bayes-rule crowd is cached), `results/S6_fixes.csv` (the S3 fix cells) and `paper/figures/fig4_crowds`. `holm` writes `results/S6_holm.csv` (the pooled tests with Holm p-values) and `results/S6_holm_robustness.csv` (the claim-weighted and unstated-benchmark checks).

### H4

```bash
python -m Analysis.paper_results all --boot 2000
python -m Analysis.paper_results muchnik_topic --boot 2000
```

Writes `results/S6_muchnik.csv`, `paper/figures/fig5_muchnik` and `results/S6_muchnik_by_topic.csv` (the Muchnik panel within each of the seven topic groups).

### H5

```bash
python -m Analysis.paper_results judge --n 2000          # first run only: draws the blind labelling sheet
python -m Analysis.paper_results judge_fill --n 2000     # M6 judge calls for uncached flips only
python -m Analysis.paper_results judge_rates --n 2000    # silent-herding rates from the cache
python -m Analysis.paper_results kappa                   # judge vs human labels, once the sheet is labelled
```

`judge_fill` and `judge` write the judge cache `runs/S6/judge.jsonl`, and `judge_rates` writes `results/S6_judge_rates.csv`. `judge` creates `results/judge_labels.csv` (the blind sheet of 200 reasons) and `results/judge_key.csv` (the judge's answers, kept apart so they cannot anchor the human labels) only when the sheet is missing, and never overwrites it. The κ check needs the human labels in `results/judge_labels.csv`, which is private and not in the repository. Use the same `--n` in all four H5 commands, since it sets the flip sample.

## Rules

- **No keys in git.** `.gitignore` is an allowlist, so only the listed paths can enter git. Before each commit, run `git grep -n -E "nvapi-[A-Za-z0-9_-]{20,}"`; it should print nothing.
- **Prompts are frozen.** `prompts/templates_v1.yaml` is frozen. Changes go into a new file, `prompts/templates_v2.yaml`, selected by `prompt_version` in `config.yaml`, and need a rerun.
- **Numbers come only from the analysis commands.** Every number, table and figure in the paper comes from the `Analysis.paper_results` commands in Section 5.

## Known Issues

- The NIM gateway cuts requests at about 300 seconds (HTTP 504), so very long reasoning outputs fail. Reasoning effort and token caps are therefore set per model in `config.yaml`.
- Some endpoints returned few or no answers for long stretches (M8 GLM-5.3 and M9 DeepSeek-v4.1-flash), and M11 Kimi-K3 is slow. Their `timeout_s` and `tries` settings in `config.yaml` allow long waits, and completed calls are kept in the cache.

## Licence

To be added.
