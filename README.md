# Do LLM Voters Cascade? (WWW 2027)

Do LLM agents that vote after seeing earlier votes weigh them like Bayesian learners, like people, or like blind herders? The run order and stage outputs are below.

## Layout

- `src/core/`: services used by every stage. `nim_client.py` (NIM calls, key pool, cache), `bayes_oracle.py` (Bayes benchmark and urn agents), `prepare_data.py` (claims, comments, personas).
- `src/stages/`: the experiment stages, one module each: `stage1_private_signals.py` (S1), `stage2_urn.py` (S2), `stage3_histories.py` (S3), `stage4_crowds.py` (S4), `stage5_reddit.py` (S5).
- `src/analysis/paper_results.py`: every number in the paper (S6), including the H5 judge and the kappa check.
- `kaggle/kaggle_job.py`: runs a stage on a Kaggle GPU machine.
- `prompts/templates_v1.yaml`: prompt templates, frozen at G2.
- `config.yaml`: every setting for every stage. `data/pools/` and `data/processed/`: claim pools and prepared inputs.

## Setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows path; bin/ on Linux
```

- **NIM keys:** put them in `nvidia_keys.txt` at the repo root, in any layout; git never sees the file. Check them with `python -m src.core.nim_client --check-keys`.
- **Kaggle:** credentials go in `~/.kaggle/kaggle.json`. GPU jobs run on the T4 × 2 machine through `kaggle/kaggle_job.py`.

## Run order

```bash
python -m src.core.prepare_data all                                               # S0: claims, comments, personas
python -m src.core.bayes_oracle                                                 # S0: oracle checks (hand-worked + exhaustive)
python -m src.core.nim_client --smoke                                            # G1: every NIM model answers and parses
python -m src.stages.stage1_private_signals --models M1 M2 M3 M4 M6 M6H --refill 2078   # S1: private signals -> data/pools/
python -m src.stages.stage2_urn --models M1 M2 M3 M4 M6 M6H                     # S2: urn anchor
python -m src.stages.stage3_histories --models M1 M2 M3 M4 M6 M6H --part main   # S3: scripted histories
python -m src.stages.stage3_histories --models M1 M2 M3 M4 M6 M6H --part fixes  # S3: fix cells
python -m src.stages.stage4_crowds --crowds M1 M2 M3 M4 M6 mixed                # S4: clean and seeded crowds
python -m src.stages.stage4_crowds --crowds M1 M2 M3 M4 M6 mixed --f1           # S4: private-first crowd
python -m src.stages.stage5_reddit --models M1 M2 M3 M6                         # S5: Reddit first voter
python -m src.analysis.paper_results judge            # H5: first run only, draws the blind labelling sheet (never overwrites one)
python -m src.analysis.paper_results judge_fill       # H5: judge calls for the flip sample (only uncached reasons)
python -m src.analysis.paper_results judge_rates      # H5: silent-herding rates from the cache, no API calls
python -m src.analysis.paper_results kappa            # H5 gate, once the sheet is labelled
python -m src.analysis.paper_results all              # S6: tables and figures
python -m src.analysis.paper_results holm             # pooled Holm tests and the claim-weighted check
python -m src.analysis.paper_results muchnik_topic    # S5 Muchnik panel within each topic group
python -m src.analysis.paper_results summary          # H2 median across models
```

Local models run the same commands on Kaggle, then come home as caches:

```bash
python kaggle/kaggle_job.py push --name s1-l1 --cmd "src.stages.stage1_private_signals --models L1 --refill 2078"
python kaggle/kaggle_job.py pull --name s1-l1
python -m src.stages.stage1_private_signals --models L1 --refill 2078      # all cache hits: rebuilds the pool locally
```

## Rules

- **Every call is cached** in `runs/<stage>/<name>.jsonl`, keyed by a hash of the request. Re-running is free, and a crash resumes where it stopped.
- **One process per key slice.** Use `NIM_KEY_SLICE=0:18` and `NIM_KEY_SLICE=18:24` for two jobs at once.
- **Prompts live in `prompts/templates_v1.yaml`** and freeze at G2. Any later edit goes into `templates_v2.yaml`, followed by a re-run.
- **Numbers in the paper come only from the `paper_results` commands:** `all`, `holm`, `judge_rates`, `muchnik_topic` and `summary`.
