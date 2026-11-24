# Do LLM Voters Cascade?

## Repository map

```
Datasets/
  prepare_data.py              LIAR2 claims, Reddit comments, personas
  processed/                   prepared claims, comments, personas, subreddit map
  pools/                       per-model claim pools (output of S1)
Experiments/
  core/                        NIM client, key pool and cache; Bayes oracle
  stage1_private_signals.py    S1 private verdicts and claim pools
  stage2_urn.py                S2 Anderson–Holt urn
  stage3_histories.py          S3 scripted vote histories and fix cells
  stage4_crowds.py             S4 12-agent crowds, planted votes, fixes
  stage5_reddit.py             S5 Reddit first-voter experiment
  kaggle/                      launcher for the 8B models on Kaggle GPUs
Analysis/
  paper_results.py             tables, figures, judge and pooled tests
config.yaml                    settings for every stage and model
prompts/                       frozen prompt templates
README.md                      setup and step-by-step run guide
```

The primary comparison includes ten completed configurations of seven open-weight models. Seven run on NVIDIA NIM endpoints through `Experiments/core/nim_client.py`; the three 8B configurations run with vLLM on two T4 GPUs through `Experiments/kaggle/kaggle_job.py`. Four further models (M8–M11) are configured in `config.yaml` but are not part of the primary comparison.

| Size group | Completed configurations |
|---|---|
| ~8B | Llama-8B (L1), Qwen-8B (L2), Qwen-8B-R (L2T) |
| ~11B | Llama-11B (M1) |
| ~20B | gpt-oss-L (M6), gpt-oss-M (M6H) |
| 30B-A3B | Lightning (M3) |
| 120B-A12B | Super (M2), Super-R (M4) |
| 550B-A55B | Ultra (M7) |

Large or sensitive files—including API keys, the run cache `runs/` with the raw model outputs, raw dataset downloads (`Datasets/raw/`), and the private H5 labelling sheet `results/judge_labels.csv`—are excluded from the source tree.

## 1. Prepare the Environment

Clone the repository and create a virtual environment with Python 3.10.

```bash
export ANONYMOUS_REPOSITORY_URL=https://github.com/OWNER/REPOSITORY.git
```

Run commands from the repository root; the clone below creates it as `ai-voters`.

```bash
git clone "$ANONYMOUS_REPOSITORY_URL" ai-voters
cd ai-voters
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` gives minimum versions only, and there is no single dependency lockfile. It includes `pyarrow`, which the parquet outputs need. The Kaggle launcher also imports the `kaggle` package, which is not in `requirements.txt`, so run `pip install kaggle` as well.

NIM keys go in `nvidia_keys.txt` at the repository root. The file is git-ignored, and keys are found by their `nvapi-` pattern, so the layout of the file does not matter. The `NIM_KEYS` environment variable can hold the keys instead. Check them with `python3 -m Experiments.core.nim_client --check-keys`, which makes one small call per key and prints one line per key with a fingerprint, never the key itself. `NIM_KEY_SLICE` chooses the keys a process uses (for example `0:8,9:16`). `NIM_RPM` sets requests per minute per key, and `NIM_CONCURRENCY` sets requests in flight per process; both default to the values in `config.yaml`. `AIV_RUNS` sets another cache folder (default `runs/`). Kaggle credentials go in `~/.kaggle/kaggle.json`.

## 2. Install the Dataset Locally

The processed files are already in `Datasets/processed/`; the commands below rebuild them. Claims come from LIAR2 (`chengxuphd/liar2`). The ratings true, false and pants-fire map to TRUE and FALSE, and claims of 6 to 60 words are kept. The build checks the label counts against the published ones and stops if they differ. Claims are split into `eval` (1,500), `lora` (1,500, written but not read by any stage) and `reserve` (2,078, the source of `--refill`). Comments come from the ConvoKit Reddit small corpus (`reddit-corpus-small`): 300 comments of 15 to 120 words in each of seven topic groups, with subreddits mapped to groups by `data.comments.groups` in `config.yaml`. The 60 personas are built from the lists in `config.yaml` with seed 13. Raw downloads go to `Datasets/raw/`, which is git-ignored.

From the repository root:

```bash
python3 -m Datasets.prepare_data all          # claims, comments, personas
python3 -m Experiments.core.bayes_oracle      # Bayes oracle checks
sha256sum Datasets/processed/*                # compare with the table below
python3 -c "import glob; [print(f, sum(1 for _ in open(f, encoding='utf-8'))) for f in sorted(glob.glob('Datasets/processed/*.jsonl'))]"
```

Expected SHA-256 of each file's bytes, and the expected record counts:

| File | Records | SHA-256 |
|---|---|---|
| `build_report.json` | 2 top-level keys (`claims`, `comments`) | `d6192f06730d07f756e8717e3520f4947f19d65a93dc1c606fe62b6e9d334b87` |
| `claims_eval.jsonl` | 1,500 claims (750 TRUE, 750 FALSE) | `23304c6b84af6c93d781076ef2a955cb258641ddb0c336c27e7d2d11b4b83c91` |
| `claims_lora.jsonl` | 1,500 claims (750 TRUE, 750 FALSE) | `c9f8083624b3822fc3909d1ee752c195d907b1b5b2fe2a28aa61f7052b9b6dd0` |
| `claims_reserve.jsonl` | 2,078 claims (1,039 TRUE, 1,039 FALSE) | `02133dec1d4febff1c3fa47a7b0dafcff866459f95670bbf5369da705f10278c` |
| `comments.jsonl` | 2,100 comments (300 in each of 7 groups) | `3e8e259c2681abb85a3d918b2da0db8808348dba3dfa0a111c077257fa0436a3` |
| `personas.json` | 60 personas | `83331769d602bf91a6e21ddff33961828edef42e807bb52d8298c7483152adb8` |
| `subreddits.csv` | 100 subreddits (rows after the header) | `115b56c8dcf61a884eab2101b1a578527bfde9c7a07928d2901d7d98c78cd13f` |

`subreddits.csv` is written only while `data.comments.groups` is empty, so `prepare_data all` does not rewrite it. Compare the checksums after a rebuild to confirm that it reproduces these files.

The H5 κ check needs the private human labelling sheet `results/judge_labels.csv`, which is not in the repository. Place it there before running `judge` or `kappa`, because `judge` draws a new blind sheet whenever that file is missing. Check the source terms of LIAR2 and the ConvoKit corpus before redistributing raw claim or comment text.

## 3. Generate Model Answers

Greedy decoding (temperature 0, top-p 1.0) is used for the S1 persona verdicts and recognition probe, the S3 cells and fix cells, and the H5 judge. Sampling (temperature 0.7, top-p 0.95) is used for the S1 screening samples, the S2 urn, the S4 crowds and the S5 Reddit runs. The local models use the same settings. Token caps come from `config.yaml`: 160 by default; 512 for Ultra; 1,024 for gpt-oss-L and the judge; 2,048 for gpt-oss-M and Qwen-8B-R; 4,096 for Super-R; and in the urn (S2), 16,384 for Super-R, gpt-oss-M and Qwen-8B-R. The urn signal accuracy is 2/3 (`URN_P` in `Experiments/core/bayes_oracle.py`).

### Open-weight example

The local models run on Kaggle GPUs (two T4s) with vLLM. `push` bundles the code, `config.yaml`, the prompts, the processed data, the pools and every local-model cache in `runs/` (plus any named with `--caches`, for example `--caches S1/L1`) into a private Kaggle dataset, then starts a GPU kernel. `--vllm` sets the vLLM package (default `vllm==0.31.0`). This example runs S1 for L1 (Llama-8B) on Kaggle, pulls its cache home and rebuilds the pool locally:

```bash
python3 Experiments/kaggle/kaggle_job.py push --name s1-l1 --cmd "Experiments.stage1_private_signals --models L1 --refill 2078"
python3 Experiments/kaggle/kaggle_job.py status --name s1-l1
python3 Experiments/kaggle/kaggle_job.py pull --name s1-l1
python3 -m Experiments.stage1_private_signals --models L1 --refill 2078      # all cache hits: rebuilds the pool locally, no GPU
```

`pull` merges the kernel's caches into `runs/`, skipping requests already present. For another 8B configuration, replace L1 with L2 or L2T. The S3 cells for these models follow the same pattern, for example `--cmd "Experiments.stage3_histories --models L1 --part main"`, then `--part fixes`.

Every call is cached in `runs/<stage>/<name>.jsonl` (the model code, or the crowd name in S4), with a `.manifest.json` beside it. The key is a SHA-1 hash of the full request (model, messages, decoding settings and the model's extra options) plus a seed tag for repeated samples. A rerun reads the cache first, so it makes no new calls, and an interrupted run resumes where it stopped.

### Hosted-Model Example

Check the NIM keys, then smoke-test the models. `--smoke 5` sends five claims to each model. Pass `--codes` to limit the test to the models in this run; without it, `--smoke` also calls every NIM model in `config.yaml`, including M8–M11.

```bash
python3 -m Experiments.core.nim_client --check-keys
python3 -m Experiments.core.nim_client --smoke 5 --codes M1 M2 M3 M4 M6 M6H M7
python3 -m Experiments.stage1_private_signals --models M1 M2 M3 M4 M6 M6H M7 --refill 2078   # S1: private signals -> Datasets/pools/
python3 -m Experiments.stage2_urn --models M1 M2 M3 M4 M6 M6H M7                             # S2: urn cascade (200 rounds)
python3 -m Experiments.stage3_histories --models M1 M2 M3 M4 M6 M6H M7 --part main           # S3: scripted histories
python3 -m Experiments.stage3_histories --models M1 M2 M3 M4 M6 M6H M7 --part fixes          # S3: fix cells (after main)
python3 -m Experiments.stage4_crowds --crowds M1 M2 M3 M4 M6 M7 mixed                        # S4: clean and seeded crowds
python3 -m Experiments.stage4_crowds --crowds M1 M2 M3 M4 M6 M7 mixed --f1                   # S4: private-first crowd (k = 0 only)
python3 -m Experiments.stage4_crowds --crowds M2 M3 --br                                     # S4: Bayes-rule prompt crowd (k = 0 and 2)
python3 -m Experiments.stage5_reddit --models M1 M2 M3 M6                                    # S5: Reddit first voter
```

For another model, change `--models` or `--crowds` to its code. M7 (Ultra) is in the S1 to S4 lines because the paper's pooled tests and Tables 2 to 4 and 6 include it.

## 4. Metric Settings

Use these definitions and preserve each metric's denominator:

- **ρ, H1:** the weight of one shown vote in units of the model's own signal. On the independent-framing cells, fit `logit(vote_true) ~ s + lead_shown`, where `vote_true` is 1 for TRUE, `s` is the own signal (±1) and `lead_shown` is the sum of the shown votes. ρ is the coefficient of `lead_shown` divided by the coefficient of `s`; the denominator is the coefficient of `s`. It is reported with q unstated and with q = 70% stated. The q-stated cells also give `q*` = expit(β), where β is the OLS slope of `delta` on `lead_shown`, and `delta` is the change in logit P(TRUE) from the solo answer to the answer after the votes, with probabilities clipped to 1–99%.
- **ρ_Bayes, H1:** ρ_Bayes = logit(q) / logit(p̂), with q = 0.70 when the prompt states it. p̂ is the model's accuracy with its private signal on its pool (`p_hat` in `Datasets/pools/<model>.json`); the denominator is logit(p̂). With q unstated, the benchmark is 1, which is a robustness check only. The H1 rule is ρ > ρ_Bayes.
- **γ_seq, H2:** on the sequential-framing cells, fit by OLS `delta` on `lead_inf` (the lead before a cascade, the informative part) and `extra` (the shown lead minus `lead_inf`, the redundant post-cascade votes). γ = β_extra / β_inf. The denominator is β_inf, which is unstable when it is near zero, so both coefficients are reported. Bayes gives γ = 0 for sequential votes and γ = 1 for independent votes. The H2 rule is γ_seq > 0.5.
- **Crowd accuracy loss, H3:** per claim, the accuracy of the independent crowd's majority minus the accuracy of the sequential crowd's final majority, for the clean crowd (k = 0) of 12 agents. A tie counts as half. The denominator is the claims with a complete chain. The estimate is pooled over the six crowd models with equal weight; `results/S6_holm.csv` stores it as a fraction (multiply by 100 for points). The H3 rule is a loss of at least 5 points. The seeding effect is the k = 0 accuracy minus the accuracy with 1 to 3 planted wrong votes on the same claims (`acc_drop_vs_k0` in `results/S6_crowds.csv`).
- **Muchnik up-lift and down-vote correction, H4:** each comment is shown with score 1 (control), 2 (up) or 0 (down), in separate calls. The up-lift is P(UP | up) / P(UP | control) − 1, relative to the control rate. The down-vote correction is P(UP | down) − P(UP | control), a difference in P(UP). The pooled test uses M1, M2, M3 and M6 with equal weight. The H4 rule is no correction, that is, a down-vote difference of at most zero.
- **Silent share, H5:** the flips are S3 main cells in which the shown majority opposes the model's own verdict and the model follows it. For each flip, the judge (M6, `openai/gpt-oss-20b`, reasoning low, temperature 0 with top-p 1.0, 1,024-token cap) answers YES or NO to whether the stated reason mentions or relies on the other reviewers' ratings. The silent share is the NO answers divided by the judged flips, which is the denominator. The sample is up to 2,000 flips (`--n 2000`), split equally across models. κ (Cohen's kappa) compares the judge with the blind human YES/NO labels on the 200-reason sheet, over the rows labelled by both. The gate is κ ≥ 0.7; below it, H5 is reported from the human labels alone. The pooled test uses the six crowd models with equal weight, and the H5 rule is a silent share above 50%.
- **Private-first gain, H6:** on the k = 0 crowds, the private-first crowd (`--f1`) is compared with the visible-vote crowd on the same claims and agents. The gain is the accuracy of the private-first final majority minus that of the visible-vote final majority, with a tie counted as half. The denominator is the claims complete in both protocols. The pooled test uses the crowd models with private-first runs, with equal weight, and the H6 rule is a gain above zero.
- **Intervals and pooling:** every interval is a 95% percentile interval from a claim-clustered bootstrap, resampling by `item_id`, with B = 2,000 (`--boot 2000`, the default) and seed 13 from `config.yaml`. Pooled tests average the models or crowds with equal weight, each resampled over its own claims (for H4, comments; for H5, judged flips). A test's p-value is the share of draws on the null side. Holm's step-down correction is applied across H1 to H6.

## 5. Build the Analysis Tables

The analysis reads the caches in `runs/` and the stage tables in `results/`, so it needs step 3 for all ten configurations and the H5 judge answers. Run the judge commands first:

```bash
python3 -m Analysis.paper_results judge --n 2000         # first run only: draws the blind labelling sheet
python3 -m Analysis.paper_results judge_fill --n 2000    # M6 judge calls for uncached flips only
python3 -m Analysis.paper_results judge_rates --n 2000   # silent-herding rates from the cache
python3 -m Analysis.paper_results kappa                  # judge vs human labels, once the sheet is labelled
```

After the judge commands finish:

```bash
python3 -m Analysis.paper_results all --boot 2000
```

`all`, `holm`, `judge_rates`, `muchnik_topic` and `summary` make no API calls: they read the caches and write only to `results/` and `paper/figures/`. `judge` and `judge_fill` call M6 only for judge answers that are not cached. Use the same `--n` in all the H5 commands, since it sets the flip sample.

## 6. Reproduce the Figures

`all` writes the figures to `paper/figures/` as PDF and PNG: `fig1_follow_curves`, `fig2_rho`, `fig3_gamma`, `fig4_crowds` and `fig5_muchnik`. Those are the code's file names; the figure numbers below follow the paper. Figure 2 in the paper is a study diagram that the code does not draw, and `fig2_rho`, `fig3_gamma` and `fig5_muchnik` have no numbered figure in the paper.

### Figure 1

Figure 1 (planted votes and the private-first fix) is only partly produced by `fig4_crowds`, which draws the visible-vote curve and the independent-crowd line for each crowd. The private-first curve is not drawn, and `--f1` produces it only at k = 0, so the planted-vote points are missing. The Bayesian crowd is not drawn either, although its accuracy is in `results/S6_crowds.csv`.

```bash
python3 -m Analysis.paper_results all --boot 2000
```

### Figure 3

Figure 3 (following real votes against a placebo) corresponds to `fig1_follow_curves`, which draws the independent-framing curve with Wilson bands and the Bayes curve for each model. The random-label placebo curve is not drawn, although `results/S6_follow_curves.csv` holds the random framing.

```bash
python3 -m Analysis.paper_results all --boot 2000
```

### Figure 4

Figure 4 (ρ_crowd, ρ_emp and ρ_priv_emp) is not produced, because no code computes those weights. `all` writes `fig2_rho` and `fig3_gamma`, which show per-model ρ with q unstated and γ, not Figure 4.

## 7. Reproduce the Tables

### Tables 1 and 2

Table 1 is `results/S1_summary.csv`, written in step 3 (`pool_n`, `p_hat`, `true_share` and `recognized`; `probe_valid` marks where the recognition probe is usable). Table 2 comes from `results/S6_urn.csv` (cascade rate), `results/S6_h1_h2.csv` (ρ and ρ_Bayes at q = 70%) and `results/S6_judge_rates.csv` (raw silent share); its f_inf, f_rand, φ, D and bounded silent columns are not computed (see below).

```bash
python3 -m Analysis.paper_results all --boot 2000
```

### Tables 3 and 4

Table 3's accuracies and wisdom loss by crowd and k are in `results/S6_crowds.csv` (point estimates only), and its pooled H3 loss with an interval is in `results/S6_holm.csv`. Table 4's k = 0 columns and gains are in `results/S6_private_first.csv`, and its Bayes-rule rows are in `results/S6_bayes_rule_crowds.csv`.

```bash
python3 -m Analysis.paper_results all --boot 2000
python3 -m Analysis.paper_results holm --boot 2000
```

### Tables 5 and 6

Table 5 is `results/S6_holm.csv` (H1 to H6 with Holm p-values). Its H2 entry is the mean over the six crowd models, whereas the paper reports the median across models; `summary` prints that median and writes the per-model values to `results/S6_h2_by_model.csv`. Table 6 is `results/S6_fixes.csv`, with the following rate and the agreement with the Bayes oracle for each fix. The Reddit lifts by topic (Section C) are in `results/S6_muchnik_by_topic.csv`.

```bash
python3 -m Analysis.paper_results holm --boot 2000
python3 -m Analysis.paper_results judge_rates --n 2000
python3 -m Analysis.paper_results summary
python3 -m Analysis.paper_results muchnik_topic --boot 2000
```

The repository does not compute these paper quantities. The placebo rate f_rand and the following rate f_inf are not reported as single numbers; `results/S6_follow_curves.csv` gives them by |lead|. The vote-caused share φ, the silent-flip bound derived from it and the framing discount D are not computed, although D can be derived from the four β columns in `results/S6_h1_h2.csv`. N_eff, the error correlations, the rate at which an opposing majority is right, ρ_crowd, ρ_emp and ρ_priv_emp are not computed. Private-first runs under planted votes (k ≥ 1) are not produced, because `--f1` runs only k = 0, so the k ≥ 1 columns and the pooled gain in Table 4 are missing. The pooled seeding effects and the per-crowd intervals of Table 3 are not computed either. The pooled rows without Llama-11B, the asymmetric-signal and null-weight checks, and the pairs on shared claims are not computed. The cached model outputs (`runs/`) are not in the repository, so the tables cannot be regenerated from it without rerunning step 3 and the stages.
