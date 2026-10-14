# Do LLM Voters Cascade? (WWW 2027)

Do LLM agents that vote after seeing earlier votes weigh them like Bayesian learners, like people, or like blind herders? The run order and stage outputs are below.

## Setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows path; bin/ on Linux
```

- **NIM keys:** put them in `nvidia_keys.txt` at the repo root, in any layout; git never sees the file. Check them with `python -m src.llm --check-keys`.
- **Kaggle:** credentials go in `~/.kaggle/kaggle.json`. GPU jobs run on the T4 × 2 machine through `kaggle/job.py`.

## Run order

```bash
python -m src.data all                                               # S0: claims, comments, personas
python -m src.oracle                                                 # S0: oracle checks (hand-worked + exhaustive)
python -m src.llm --smoke                                            # G1: every NIM model answers and parses
python -m src.s1_signals --models M1 M2 M3 M4 M6 M6H --refill 2078   # S1: private signals -> data/pools/
python -m src.s2_urn --models M1 M2 M3 M4 M6 M6H                     # S2: urn anchor
python -m src.s3_histories --models M1 M2 M3 M4 M6 M6H --part main   # S3: scripted histories
python -m src.s3_histories --models M1 M2 M3 M4 M6 M6H --part fixes  # S3: fix cells
python -m src.s4_crowds --crowds M1 M2 M3 M4 M6 mixed                # S4: clean and seeded crowds
python -m src.s4_crowds --crowds M1 M2 M3 M4 M6 mixed --f1           # S4: private-first crowd
python -m src.s5_reddit --models M1 M2 M3 M6                         # S5: Reddit first voter
python -m src.analyze judge                                          # H5: silent-herding judge + blind kappa sheet
python -m src.analyze kappa                                          # H5 gate, once the sheet is labelled
python -m src.analyze all                                            # S6: every number, table, figure
```

Local models run the same commands on Kaggle, then come home as caches:

```bash
python kaggle/job.py push --name s1-l1 --cmd "src.s1_signals --models L1 --refill 2078"
python kaggle/job.py pull --name s1-l1
python -m src.s1_signals --models L1 --refill 2078      # all cache hits: rebuilds the pool locally
```

## Rules

- **Every call is cached** in `runs/<stage>/<name>.jsonl`, keyed by a hash of the request. Re-running is free, and a crash resumes where it stopped.
- **One process per key slice.** Use `NIM_KEY_SLICE=0:18` and `NIM_KEY_SLICE=18:24` for two jobs at once.
- **Prompts live in `prompts/v1.yaml`** and freeze at G2. Any later edit goes into `v2.yaml`, followed by a re-run.
- **Numbers in the paper come only from `python -m src.analyze all`.**
