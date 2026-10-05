# Do LLM Voters Cascade? — Experiment Plan

**WWW 2027 · lean execution plan · v2, Oct 5 2026 · lead: Pushkar**

This is the working plan. It covers what we run, in what order, what each stage takes in and produces, and how it all runs on NVIDIA NIM keys and Kaggle. The *Execution Plan* PDF holds the motivation and related work. The *Experiment runbook* holds the exact prompt texts and code snippets. Once this file is approved, it wins wherever the three disagree; every difference is listed in §11.

| At a glance | |
|---|---|
| **Question** | When LLM agents vote after seeing earlier votes, do they weigh those votes like Bayesian learners, like humans, or like blind herders? And what does that do to crowd accuracy? |
| **Shape** | 7 stages (S0–S6) + 2 optional (X1, X2) · 3 gates: Oct 6, Oct 8, Oct 13 · runs end Oct 14 |
| **Budget** | ≈ 260k NIM calls + ≈ 38k local calls · ≈ 15 h of NIM time at a realistic 300 RPM |
| **Deadlines** | abstract Oct 19, paper Oct 26 (both 17:29 IST) · we submit Oct 25 |
| **Rule** | A number goes into the paper only if `python -m src.analyze all` regenerates it from `runs/` |

---

## 1. What we are doing

**Research question.** When LLM agents join crowd judgments (veracity ratings, up/down votes), do they weigh earlier votes against their own reading the way cascade theory says they should?

**Three yardsticks**, so herding is never judged by eye:

1. **Bayes oracle** (Bikhchandani, Hirshleifer & Welch 1992). The vote a rational agent would cast, given its own signal and the *informative* earlier votes.
2. **Empirical optimum** (Weizsäcker 2010). The vote that actually paid off in our own crowds. It needs no independence assumption, which matters because LLM errors are correlated.
3. **Human data.** Anderson & Holt (1997) urn cascades, compared like-for-like in S2, and Muchnik, Aral & Taylor (2013) randomized Reddit votes, mirrored in S5.

**Two settings.**

- **A · Fact-checked claims (LIAR2).** The hidden state is the PolitiFact verdict: TRUE versus FALSE or PANTS-FIRE. Agents see the statement, speaker, context and date. They never see the fact-checker's justification or the speaker's rating history.
- **B · Reddit comments (ConvoKit).** The same comment is shown at score 1, 2 or 0. The real score is never shown.

**Hypotheses.** These are pre-registered and freeze on Oct 7. Each one is publishable whichever way it lands. H1 and H2 carry the paper.

| | Hypothesis | Measured by | Stage | Refuted if |
|---|---|---|---|---|
| H1 | **Over-herding:** a shown vote counts for more than Bayes allows | ρ against ρ_Bayes; q\* | S3 (S4 replicates) | ρ's 95% CI covers ρ_Bayes for ≥ 4 of 6 models |
| H2 | **Correlation neglect:** sequential (correlated) votes are not discounted | γ under the sequential framing; framing gap | S3 | γ_seq ≤ 0.5, i.e. models discount at least half of what Bayes would |
| H3 | **Wisdom loss:** visible votes hurt crowd accuracy, and seeded bot votes flip outcomes | wisdom loss; seeding flip rate | S4 | gap < 5 pp and seeding effect ≤ that of a Bayesian crowd |
| H4 | **Symmetric herding:** no Muchnik-style correction after a downvote | first-voter lifts | S5 | down-treated comments are corrected, as in humans |
| H5 | **Silent herding:** most vote-driven flips never mention the votes | judged flip reasons | S3, S4 | ≤ 50% of flips are unacknowledged |
| H6 | **Cheap fixes work** | accuracy recovered, compared with a Bayes-rule prompt | S3, S4 (+ X1) | no fix beats prompting with the Bayes rule |

**Out of scope:** RAG, images, network topology, RL/GRPO, new human-subject data, and anything taken from the unpublished "Joneses" submission.

---

## 2. How: one method, three moves

Every stage applies the same three moves:

1. **Measure** the private signal. A solo call with no votes visible gives a verdict s and a confidence `p_true`.
2. **Show** social information whose evidential value we control: scripted histories (S3), real earlier votes (S4), one randomized prior vote (S5), or urn draws and guesses (S2).
3. **Score** the social decision against the three yardsticks. This gives ρ, q\*, γ, follow curves and crowd accuracy.

We never infer the private signal from the social run. It comes from a separate, stateless solo call (S1). It is then cross-checked two ways: against stated confidence, and against a *private-first* variant in which the agent commits to a verdict before it sees any votes (S3, S4).

**The three estimators that carry the paper:**

| Estimator | What it answers | Bayes value |
|---|---|---|
| **ρ** = β_lead / β_s, from logit P(TRUE) ~ s + lead | How much one shown vote weighs, in units of the agent's own signal | ρ_Bayes = logit(q) / logit(p̂); equals 1 only when q = p̂ |
| **q\*** = expit(slope of Δ logit `p_true` on lead) | How accurate the model acts as if other raters are | 0.70 when told "right 70% of the time" |
| **γ** = β_extra / β_inf | How much weight redundant post-cascade votes get, relative to informative ones | 1 under the independent framing, 0 under the sequential one |

---

## 3. Architecture and data flow

```
INPUTS (S0: built once, then frozen)
  LIAR2 ──────► data.py ──► claims_eval · claims_lora · claims_reserve
  ConvoKit ───► data.py ──► comments.jsonl
  seed 13 ────► data.py ──► personas.json            + prompts/v1.yaml
                    │
                    ▼
STAGE SCRIPTS       s1 … s5 build each request = template × item × cell × persona × seed
                    │
                    ▼
RUNNER (llm.py)     h = sha1(model, messages, T, top_p, max_tokens, seed_tag)
                    ├─ in cache? → return the stored record
                    ├─ else NIM key pool (laptop / Kaggle CPU)  or  vLLM batch (Kaggle GPU: L1, L2)
                    ├─ parse: JSON → regex → one retry → parse_fail
                    └─ append the record to runs/<stage>/<model>.jsonl
                    │
                    ▼
RAW DATA (runs/)    append-only JSONL, synced to and backed up on a private HF dataset
                    │
                    ▼
ANALYSIS            analyze.py + oracle.py ──► results/*.parquet ──► figures · tables · numbers
```

**Life of one claim** (the main data flow):

1. **S0.** A LIAR2 row is cleaned, balanced, and placed in `claims_eval`.
2. **S1.** For each model, 5 solo samples and a recognition probe give p̄, the mean probability placed on the correct label. The claim enters that model's pool if 0.55 ≤ p̄ ≤ 0.90 and the model does not recognize it. One persona solo call then fixes the agent's private signal (s, `p_true`).
3. **S3.** The same agent sees 8 scripted histories. Each social reply is scored against the oracle.
4. **S4.** Twelve agents vote on the claim in turn, each making a solo call and then a social call. The final majority is compared with the majority of their solo votes, with and without seeded wrong votes.
5. **S6.** Records join on `claim_id` to give ρ, q\*, γ, follow curves and crowd metrics.

A Reddit comment takes the short path S0 → S5 → S6. An urn round takes S0 → S2 → S6.

**Five rules that keep a three-person team out of trouble:**

1. **The cache is the dataset.** Each call is one JSONL record, keyed by its request hash. Re-running a stage is idempotent, a crash resumes for free, and nothing is ever overwritten.
2. **Randomness is derived, never drawn.** Every random choice (label order, persona, cell, history, crowd) comes from `random.Random(f"13|{stage}|{item}|{cell}")`. A re-run therefore renders byte-identical prompts and hits the cache.
3. **One writer per file.** Exactly one process writes each `runs/<stage>/<model>.jsonl`. Machines split work by model and never share a file. Merging is just concatenation, because records dedupe by hash.
4. **Numbers come only from `analyze.py`.** It reads `runs/` and writes `results/`. Figures read `results/` and nothing else.
5. **Prompts freeze at G2 (Oct 8).** Any later edit gets a new version tag (`v2`) and a full re-run of the cells it touches. Paraphrases are a separate factor (`para = a | b | c`), never a version.

**Every record carries** `hash`, `stage`, `model` (exact string), `backend`, `prompt_ver`, `para`, `item_id`, `cell` (all factor levels), `persona_id`, `seed_tag`, `messages`, `text`, `parsed`, `parse_status`, `finish`, `usage`, `latency_s` and `ts`. Each run also writes `runs/<stage>/manifest.json` with the git commit, config hash, model strings and the date they were checked, call count, cache hits and parse rate.

---

## 4. What we use

### Data

| Source | Role | Raw → kept | Notes |
|---|---|---|---|
| LIAR2 `chengxuphd/liar2` (Apache-2.0) | Setting A | 22,962 rows → balanced set: 1,500 eval · 1,500 LoRA-only · the rest held in reserve | drop `justification`, `*_counts`, `speaker_description`; `recent` = dated ≥ 2022-08-01 |
| ConvoKit `reddit-corpus-small` | Setting B | 297,132 utterances → 2,100 comments (300 × 7 topic groups) | quality tier from the real score; the score is never shown |
| Urn rounds (generated) | S2 | 200 rounds × 6 agents | Anderson–Holt parameters, signal accuracy 2/3 |
| Personas (generated, seed 13) | agent diversity | 60 | age × occupation × US region, no political leaning |
| LIAR (UCSB) | backup | 12,836 | used only if LIAR2 fails the label-count check |

### Models

**Roster as frozen at G1 on Oct 6.** All six runbook model strings now return 410 Gone on NIM, so the roster was rebuilt from what NIM actually serves (details in §13). M5 is retired.

| Code | Model string | Where | Why it is in | Runs in |
|---|---|---|---|---|
| M1 | `meta/llama-3.2-11b-vision-instruct` | NIM | small Meta baseline, no reasoning | all |
| M2 | `nvidia/nemotron-3-super-120b-a12b`, thinking off | NIM | large NVIDIA MoE | all |
| M3 | `nvidia/nemotron-3.5-lightning-30b-a3b`, thinking off | NIM | small NVIDIA MoE: scale within one family with M2 | all |
| M4 | M2's model with thinking on | NIM | reasoning twin of M2 (same weights) | subsets (≤ 300 claims) |
| M6 | `openai/gpt-oss-20b`, `reasoning_effort` low | NIM | third family | all |
| M6H | `openai/gpt-oss-20b`, `reasoning_effort` medium | NIM | more reasoning, same weights as M6 | subsets (≤ 300 claims) |
| L1 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | Kaggle vLLM | open weights, token log-probs | S1, S3 |
| L2 | `Qwen/Qwen3-8B-AWQ` (L2T: thinking on) | Kaggle vLLM | fourth family; thinking on/off; LoRA base | S1, S3, X1 |

The six base models for H1's "≥ 4 of 6" test are M1, M2, M3, M6, L1 and L2. The exact strings and switches live in `config.yaml`. Re-check them before every long run, because the catalog changes.

**Decoding.** T = 0 (top_p 1.0) for S1 persona and recognition calls, and for all of S3, including its fix cells. T = 0.7 (top_p 0.95) for S1 solo samples, S2, S4 and S5. `max_tokens` is 160; 1,024 for M6, 2,048 for M6H and L2T, and 4,096 for M4. Reasoning is switched by request parameters, never by prompt text: `chat_template_kwargs.enable_thinking` for Nemotron and Qwen, and `reasoning_effort` for gpt-oss. Parse only the text after `</think>`, and log reasoning length separately.

### Compute

| Resource | Capacity | Used for |
|---|---|---|
| NVIDIA NIM API keys | up to 20 keys, ~40 RPM each on the free tier (confirm in the S0 probe); we pace at 35 | every M1–M6 call |
| Kaggle GPU, T4 × 2 | ~30 GPU-h/week and 12 h/session per account (check your quota page) | L1, L2 batches (#1, #2); X1 LoRA (#3) |
| Kaggle CPU sessions | any account; no GPU quota used | unattended NIM sweeps |
| Laptops | 3 | development, pilots, S2, analysis |
| Private GitHub repo `ai-voters` | — | code, prompts, config, processed data, pools |
| Private HF dataset `ai-voters-runs` | — | sync point and nightly backup for `runs/` |

**Libraries.** `openai` (async client; NIM is OpenAI-compatible), `jinja2`, `orjson`, `pandas`, `pyarrow`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `datasets`, `convokit`, `python-dotenv`, `tqdm` and `huggingface_hub`. On Kaggle GPU, add `vllm`, pinned to whichever version boots in S0. For X1 only, add `unsloth` and `trl`.

---

## 5. Repo layout: 10 Python files, no notebooks

The repo root is `D:\web` (private GitHub `pushkar1887/ai-voters`). Its `.gitignore` is an allowlist: everything is ignored unless listed, so `nvidia_keys.txt`, the PDFs and the run caches can never be committed.

```
D:\web  (git root)
├── config.yaml           models (exact strings + switches), decoding, seeds, sizes, subreddit map
├── requirements.txt      laptop dependencies (vLLM is installed on Kaggle only)
├── nvidia_keys.txt       NIM keys, matched by pattern, never printed      (not in git)
├── prompts/v1.yaml       every template and paraphrase in one file; v2.yaml after any post-freeze edit
├── data/
│   ├── raw/              downloads, never edited                          (not in git)
│   ├── processed/        claims_{eval,lora,reserve}.jsonl · comments.jsonl · personas.json · build_report.json
│   └── pools/            <model>.json, written by S1
├── src/
│   ├── llm.py            key pool, pacer, cache, parser, vLLM backend; --check-keys, --models, --smoke, --probe
│   ├── data.py           S0 builders, loaders, and every claim prompt (solo, social, private-first)
│   ├── oracle.py         BHW informative lead, Bayes vote and posterior, urn reference agents
│   ├── s1_signals.py     S1
│   ├── s2_urn.py         S2, including its table
│   ├── s3_histories.py   S3, including fix cells
│   ├── s4_crowds.py      S4, including seeding and the private-first crowd
│   ├── s5_reddit.py      S5 (X2 behind --b2)
│   └── analyze.py        S6: every estimator, figure and table
├── kaggle/job.py         bundle -> private Kaggle dataset -> T4 x2 kernel -> pull caches (also the kernel)
├── runs/                 <stage>/<name>.jsonl + <name>.manifest.json      (not in git)
├── results/              generated tables                                 (not in git)
└── paper/figures/        generated, never hand-edited
```

Stage scripts share their flags: `--models M1 M2 …` and `--limit N` for pilots. The backend follows the model code (`nim` or `vllm` in `config.yaml`), so the same command runs on the laptop or on Kaggle. One file per stage means one owner per file.

---

## 6. Stage map

```
S0 Setup + data
 │  G1
 ├──► S1 Signals + pools
 │     │  G2
 │     ├──► S3 Scripted histories (+ fix cells) ──G3──► S6
 │     └──► S4 Crowds + seeding ──────────────────────► S6
 ├──► S2 Urn anchor ──────────────────────────────────► S6
 └──► S5 Reddit first voter ──────────────────────────► S6 Analysis ──► paper

Optional:  S1 (L2 on claims_lora) ──► X1 Belief-trace LoRA ──(kept only if G3 passes)──► S6
           S5 ──► X2 Reddit accumulation (idle keys after Oct 12)
```

- **Critical path: S0 → S1 → S3 → S6.** S3 carries H1 and H2, so nothing may delay it.
- **Fillers.** S2 and S5 need only S0. They soak up idle keys and never block anything.
- **NIM priority** when keys are short: S1 > S3 > S4 > S5 > S3 fix cells > S2 > X2.
- **Analysis is written early.** `analyze.py` is built against pilot data from Oct 7. On Oct 14, the paper's numbers are then one command away.

**Interfaces** (the files that link the stages):

| Stage | Reads | Writes | Read by |
|---|---|---|---|
| S0 | LIAR2, ConvoKit, seed 13 | `data/processed/*`, `prompts/v1.yaml`, `config.yaml` | all |
| S1 | `claims_eval` (+ `claims_reserve`), personas | `runs/S1/*`, `data/pools/<model>.json`, `results/S1_summary.csv` | S3, S4, X1, S6 |
| S2 | urn template, `oracle.py` | `runs/S2/*` | S6 |
| S3 | pools, S1 persona replies (via the cache) | `runs/S3/*` | S6, X1 |
| S4 | pools, personas | `runs/S4/*` | S6 |
| S5 | `comments.jsonl`, personas | `runs/S5/*` | S6 |
| S6 | `runs/*` only | `results/*`, `paper/figures/*` | paper |

**The pool file is the key contract.** It has one row per kept claim: `{claim_id, truth (±1), p_bar, persona_id, label_order, s (±1), p_true_solo}`, plus the model-level `p_hat`.

**Owners.**

| Person | Owns |
|---|---|
| Pushkar (lead) | hypotheses and analysis plan, `oracle.py`, S2, S6 statistics, novelty search, writing |
| Teammate 2 (systems) | `data.py`, `llm.py`, S1, S3 (the critical path) |
| Teammate 3 (ML/NLP) | `kaggle/` and L1/L2, S4 (proposed; see §12), S5, the silent-herding judge, figures, X1 |

**Budget.**

| Stage | NIM calls | Local calls | NIM hours at 300 RPM |
|---|---|---|---|
| S0 | ≈ 1k | smoke only | — |
| S1 | ≈ 54k | ≈ 20k | 3.0 |
| S2 | 7.2k | — | 0.4 |
| S3 | ≈ 67k (45k main + 22k fix cells) | ≈ 18k | 3.7 |
| S4 | ≈ 90k | — | 5.0 |
| S5 | ≈ 38k | — | 2.1 |
| S6 | ≈ 2k (judge) | — | 0.1 |
| **Core** | **≈ 260k** | **≈ 38k** | **≈ 14.5** |
| X1 | — | ≈ 30k + ≈ 10 GPU-h of training | — |
| X2 | 18k | — | 1.0 |

---

## 7. Stages

Each card gives the stage's goal, inputs (**In**), outputs (**Out**), links to other stages, steps and gate. The exact prompt texts are in the runbook, under *Prompt templates*.

### S0 · Setup and data — Oct 5–6 · Gate G1

- **Goal:** build everything the later stages need, once, then freeze it.
- **Owner:** Teammate 2 (data, runner); Pushkar (oracle, novelty search).
- **In:** LIAR2, the ConvoKit corpus, NIM keys, Kaggle accounts.
- **Out:** `data/processed/*`, `prompts/v1.yaml`, `config.yaml`, a smoke-test report.
- **Feeds:** every other stage.

1. **Repo and environment.** Create the private GitHub repo. Use Python 3.11 or newer locally (Kaggle ships 3.12). Put `.env` and `runs/` in `.gitignore`.
2. **Claims** (`python -m src.data claims`).
   - Concatenate the LIAR2 splits and check the label counts: pants-fire 3,031, false 6,605, barely-true 3,603, half-true 3,709, mostly-true 3,429, true 2,585. Stop if they differ.
   - TRUE = true. FALSE = false + pants-fire. Drop the three middle labels and the leaky fields.
   - Keep statements of 6–60 words and remove near-duplicates.
   - Balance TRUE and FALSE, stratified by year and first subject. Flag `recent`.
   - Split by claim id with seed 13: 1,500 eval, 1,500 LoRA-only, the rest to reserve.
3. **Comments** (`python -m src.data comments`).
   - Keep comments only (no thread posts) of 15–120 words. Drop anything deleted, removed, stickied or posted by AutoModerator.
   - Pushkar and Teammate 3 each map the 100 subreddits to Muchnik's 7 topic groups, then reconcile.
   - Sample 300 comments per group, stratified by subreddit. The quality tier is the top or bottom half of score within the subreddit.
4. **Personas** (`python -m src.data personas`). 60 personas from age band × occupation × region, seed 13, with no politics.
5. **Prompts.** Copy the runbook templates into `prompts/v1.yaml`, with paraphrases a, b and c. Paraphrases change only the instruction wording; the JSON keys stay the same. Add the Bayes-rule prompt that serves as the H6 baseline.
6. **Oracle.** Pushkar writes `oracle.py` from the runbook. Teammate 3 checks it against 20 hand-worked histories.
7. **Smoke test** (`python -m src.llm --smoke`). Make one solo call on each of 5 claims per model, and print the model string, latency, parsed JSON and token counts. Also record:
   - whether `logprobs` is honoured;
   - whether `Reasoning: high` lengthens M6's reasoning.
8. **Rate probe** (`python -m src.llm --probe`). Run 5 minutes at 40 RPM, first on 3 keys from one account, then on 3 keys from different accounts.
   - Set `NIM_RPM` to the highest rate that gives < 2% 429s.
   - The two runs also tell us whether the limit is per key or per account.
9. **Sync points.** Create the private HF dataset `ai-voters-runs`. Add Kaggle secrets on each account (§8.2). Boot vLLM on #1 (L1) and #2 (L2) with a 5-claim batch.

**G1 passes (Oct 6) if:**
- all six NIM models parse ≥ 99% in the smoke test;
- the data counts match;
- the novelty search (Semantic Scholar, ACL Anthology, arXiv 2025–26) is written up.

**If not:**
- Swap an unavailable model for its nearest listed sibling, and log the swap.
- If the novelty search finds a close prior paper, lead with H2 and Setting B, and drop any "first" claim.

A failed vLLM boot does **not** fail G1. The local models are a robustness track.

### S1 · Private signals and item pools — Oct 6–8 · Gate G2 · critical path

- **Goal:** measure each model's private signal on every claim before it sees any vote. Then keep the claims where that signal is informative but imperfect. Cascades only matter there; Anderson and Holt used p = 2/3.
- **Owner:** Teammate 2.
- **In:** `claims_eval` (plus `claims_reserve` for refills), personas.
- **Out:**
  - `runs/S1/<model>.jsonl`
  - `data/pools/<model>.json`
  - `results/S1_summary.csv`: parse rate, TRUE share, recognition rate, calibration error of `p_true`, pool size, p̂
- **Feeds:** S3 and S4 (pools), X1 (L2 signals), S6 (Table 1).
- **Calls:** ≈ 54k NIM + ≈ 20k local.

0. **Pilot first (Oct 6–7).** Run 100 claims through every model. Then push the same claims through 2 S3 cells each and through ρ in `analyze.py`. This vertical slice catches prompt and schema bugs while they are cheap.
1. **Solo samples.** 5 calls per claim and model at T = 0.7, with no persona. Label order is randomized, and `seed_tag` is the sample index.
2. **Recognition probe.** One call at T = 0. A claim counts as recognized when the model says YES and names a rating that matches the binary label.
3. **Score.** p̄ = the mean, over the 5 samples, of the probability put on the correct label.
4. **Pool.** Keep claims with 0.55 ≤ p̄ ≤ 0.90 that are not recognized. Balance them 50/50 and aim for 1,000 per model. M4 screens only 600 claims, for a 300-claim pool. Refill from `claims_reserve` if a pool is short.
5. **Private signal.** For pool claims only, make one call at T = 0 with one seeded persona. This gives s and `p_true`.
6. **Signal accuracy.** p̂ = the accuracy of those persona verdicts within the pool. The oracle uses it.
7. **Local replicas.** L1 and L2 run steps 1–6 on Kaggle. If X1 is on, L2 also runs step 1 on `claims_lora`.

**G2 passes (Oct 8) if:**
- every model parses ≥ 99%;
- the TRUE share is within 0.50 ± 0.15;
- pools reach ≥ 1,000 claims for ≥ 4 models (fallback: ≥ 600, with more cells per claim in S3);
- p̂ is in [0.60, 0.85] for ≥ 4 models;
- in the pilot, random-labeler cells move beliefs less than independent ones;
- at lead 0, S2 agents follow their own draw > 90% of the time.

**Then** prompts and hypotheses freeze, and the pools are committed to the repo.

**If not:**
- shift the band, or add a short evidence snippet, to tune p;
- fix the prompts;
- drop any model that cannot parse.

### S2 · Urn anchor — Oct 7–12, on idle keys

- **Goal:** the paper's only like-for-like comparison with human cascade data (Anderson & Holt 1997).
- **Owner:** Pushkar.
- **In:** the urn template (no persona) and the reference agents in `oracle.py`.
- **Out:** `runs/S2/<model>.jsonl` → `results/S2_table.csv` → the lab panel of Figure 1.
- **Depends on:** S0 only. **Blocks:** nothing.
- **Calls:** 7.2k (200 rounds × 6 agents × 6 models).

1. **Sanity check** (20 rounds per model). At lead 0, agents must follow their own draw > 90% of the time. If they do not, the prompt is being misread; fix it before scaling.
2. **Rounds.** A fair coin picks urn A or B. Each agent draws a light ball with probability 2/3 under A and 1/3 under B. Agent t sees the guesses of agents 1 to t−1.
   - Rounds run in parallel; agents within a round run in order.
   - T = 0.7.
3. **Reference agents** play on the same draws:
   - **Bayes:** follows the informative lead once it reaches 2; ties go to its own draw.
   - **Own-draw-only:** always follows its own draw.

| Measure | Human value (Anderson & Holt) |
|---|---|
| Cascade rate: follows an informative lead ≥ 2 against its own draw | 87 of 122 possible periods (71%) |
| Reverse cascades: cascades on the wrong urn | 13 of 41 and 18 of 46, by treatment |
| Inconsistent decisions: against both Bayes and the own draw | ≈ 4% of decisions |
| Tie behaviour, and belief error (`p_a` − Bayes posterior) | report only |

### S3 · Scripted histories (the causal core) — Oct 8–11 · Gate G3

- **Goal:** show each agent a history whose evidential value is known exactly, so every reply can be scored against Bayes. Carries **H1 and H2**, and supplies data for H5 and H6.
- **Owner:** Teammate 2.
- **In:** the pools (claim, persona, label order, s and `p_true` from S1) and `oracle.py`.
- **Out:** `runs/S3/<model>.jsonl` → `results/S3_cells.parquet`, the one table every H1/H2 analysis reads.
- **Feeds:** S6 (Figures 1–3, Table 2), X1 (its evaluation design).
- **Calls:** ≈ 45k main + ≈ 22k fix-cell NIM calls; ≈ 18k local.

**Design.** Each pool claim gets 8 cells, sampled from the factors below rather than fully crossed. Adding a factor therefore costs no calls. Persona, label order and private signal stay fixed from S1.

| Factor | Levels (sampling weight) |
|---|---|
| Framing | independent 40% · sequential 40% · random-labeler control 20% |
| q stated | "right about 70% of the time" or nothing, 50/50 (not for random) |
| Display | list or tally, 50/50 (sequential is always a list) |
| Length n | 1–6 votes |
| Lead vs own verdict | −3 … +3, \|lead\| ≤ n, same parity as n; 60% of cells disagree with s |
| Paraphrase | a · b · c, 1/3 each |

- **Independent and random histories.** For length n and lead L relative to s, (n + L)/2 votes equal s and the rest are opposite, shuffled.
- **Sequential histories.** Use only sequences that Bayesian predecessors could have produced (`oracle.bhw_sequences`). Oversample cascades, because that is where the displayed lead (up to 6) and the informative lead (2) differ most.

**Runs.**

1. **Main cells**, at T = 0:
   - M1–M3, M5 and M6 on their full pools;
   - M4 on its 300-claim pool;
   - M6 at high effort on 300 claims;
   - L1 and L2 on Kaggle, with L2's thinking on for 300 claims.
2. **Fix cells** (H6), run after the main pass on 300 claims per model. They use T = 0 and the same histories, so each pairs with a main cell.
   - **F1 private-first:** 4 disagreeing cells. Turn 1 is the cached S1 persona reply, so only turn 2 is new.
   - **F2 independence cue:** 4 sequential cells.
   - **BR Bayes-rule prompt:** 4 disagreeing cells. This is the baseline every fix must beat.

**Record per call:** claim, model, persona, framing, q, display, para, sequence, n_TRUE, n_FALSE, displayed lead, informative lead, s, solo `p_true`, social verdict, social `p_true`, follow flag, oracle vote, oracle P(TRUE).

**G3 passes (Oct 13) if** H1 or H2 holds across ≥ 4 models and all 3 paraphrases. The headline is then herding or correlation neglect, and X1 may enter the paper.

**If not,** the headline becomes "LLM voters are near-Bayesian". That is still a paper, and X1 is cut. Before concluding there is no herding, check that the pools are not too easy, and shift them toward p̄ 0.55–0.75.

### S4 · Sequential crowds and seeding — Oct 9–12

- **Goal:** run real crowds in which each agent sees only real earlier votes, and measure wisdom loss and seeding attacks. Carries **H3**, replicates H1 in the wild, and tests F1 at crowd level.
- **Owner:** Teammate 3 (proposed; the runbook assigns Teammate 2).
- **In:** the pools (for the mixed crowd, claims that appear in ≥ 3 pools) and personas.
- **Out:** `runs/S4/<crowd>.jsonl` → `results/S4_sequences.parquet`.
- **Feeds:** S6 (Figure 4, empirical optimum).
- **Calls:** ≈ 90k NIM (clean ≈ 47k, seeded ≈ 35k, private-first ≈ 8k).

1. **Crowds.** Twelve agents per claim, in seven crowds:
   - six homogeneous crowds, one model each;
   - one mixed crowd, where each agent's model is drawn from M1–M6.

   Each crowd gets 300 claims (M4: 150), split 50/50 TRUE/FALSE. T = 0.7, with the sequential framing and the list display.
2. **Each agent** makes a solo call (its private signal), then a social call that shows all earlier votes. Solo calls do not depend on the history, so all 12 go out at once; only the 12 social calls run in sequence.
3. **Seeding.** On 150 claims per crowd, the first k ∈ {1, 2, 3} votes are set to the wrong verdict before agent 1 arrives, without saying they are bots. Crowd composition is seeded by claim, so the agents are identical across k and their solo calls come from the cache.
4. **F1 private-first crowd.** 100 claims × 7 crowds at k = 0, with the two-turn social prompt.
5. **Free counterfactuals** (no calls):
   - The majority of the 12 solo verdicts is the independent crowd. It also serves as fix F3 (hidden counts).
   - A Bayesian crowd, run by `oracle.py` on the same solo signals, is the H3 benchmark for seeding.

**Record per sequence:** the agents, their solo and social verdicts and `p_true`, k, the final majority, the independent majority, the first position where the lead reached 2, and whether a cascade formed and on which side.

### S5 · Reddit first-voter replication — Oct 7–11

- **Goal:** mirror Muchnik et al. (2013). One comment is shown at score 1 (control), 2 (up) or 0 (down), so any difference in votes is caused by the prior vote alone. Carries **H4**.
- **Owner:** Teammate 3.
- **In:** `comments.jsonl` and personas. Each comment keeps one persona across its three treatments.
- **Out:** `runs/S5/<model>.jsonl` → `results/S5_b1.parquet`.
- **Depends on:** S0 only.
- **Calls:** 37.8k (2,100 comments × 3 treatments × 6 models, one stateless call each at T = 0.7).

| Muchnik et al. (2013) | Our number |
|---|---|
| Up-treatment raised the first upvote probability by 32% | P(UP \| up) / P(UP \| control) − 1 |
| After a downvote, the upvote probability was 0.099, against 0.054 for control (correction) | P(UP \| down) vs P(UP \| control) |
| Herding only in politics, culture and society, and business | lift per topic group, with CIs |

Every analysis is repeated within the high and low quality tiers, so herding cannot be explained by comment quality.

### S6 · Analysis → paper numbers — code from Oct 7, final Oct 14–17

- **Goal:** turn `runs/` into every number, figure and table with one command.
- **Owner:** Pushkar (statistics); Teammate 3 (silent-herding judge, figures).
- **In:** `runs/*` and `oracle.py` only.
- **Out:** `results/*` and `paper/figures/*`.
- **Command:** `python -m src.analyze all`.

| # | Estimator | Hypothesis | Recipe |
|---|---|---|---|
| 1 | ρ | H1 | logit(vote_true ~ s + lead) → β_lead / β_s; cluster bootstrap over claims (2,000 resamples); compare with ρ_Bayes |
| 2 | q\* | H1 | in independent cells with q stated: slope of Δ logit `p_true` on lead → expit. Bayes value 0.70. Also computed on L1/L2 token probabilities |
| 3 | γ | H2 | in cascade sequences: Δ ~ lead_inf + extra, by framing; γ = β_extra / β_inf. H2 holds if γ_seq > 0.5 |
| 4 | Follow curves | H1 | P(follow majority \| own verdict disagrees) by \|lead\|, with Wilson CIs, against the oracle and humans |
| 5 | Empirical optimum | H1, H3 | P(majority right \| model, \|lead\|) from S4; share of payoff earned where following is optimal (human: 53%) |
| 6 | Crowd metrics | H3 | cascade = the lead reaches 2 and the next 3 agents agree; wisdom loss; seeding flip rate against the Bayesian crowd |
| 7 | Muchnik lifts | H4 | paired bootstrap over comments; overall, per topic, per tier |
| 8 | Silent herding | H5 | M6 (low effort, T = 0) judges a stratified sample of 2,000 flips. Used only if Cohen's κ ≥ 0.7 against Pushkar's and Teammate 3's labels on 200 reasons |
| 9 | Tests | all | one pooled p-value per hypothesis (model as a fixed effect), Holm-corrected; per-model results reported as CIs |

| Exhibit | Shows | From |
|---|---|---|
| Fig. 1 | follow curves: each model against the Bayes step and humans | S2, S3 |
| Fig. 2 | ρ per model with 95% CIs, reasoning on and off paired | S3 |
| Fig. 3 | γ: independent vs sequential framing, observed vs Bayes | S3 |
| Fig. 4 | crowd accuracy: independent, sequential, seeded k = 1–3 | S4 |
| Fig. 5 | lift ratios against Muchnik's +32% | S5 |
| Table 1 | models, p̂, pool size, recognition rate | S1 |
| Table 2 | fixes: accuracy recovered, and ρ after each fix | S3, S4 (+ X1) |

### X1 · Belief-trace LoRA (optional)

- **Go:** Oct 9, once L2's `claims_lora` signals exist and S5 is running.
- **Kill:** Oct 13, if G3 fails or no adapter has been trained.
- **Owner:** Teammate 3. Training runs on Kaggle #3; evaluation on #1–#2.
- **Scope:** one base model (Qwen3-8B) and two runs: the belief-trace target, and an answer-only twin.

1. **Data.** `claims_lora` (never evaluated) × 4 S3-style cells = 6,000 examples. The private signal is L2's own solo signal.
2. **Target.** A short trace (own reading, informative lead, weights, total), then the JSON with the oracle's vote and `p_true`. The twin's target is the JSON only.
3. **Train.** Unsloth 4-bit QLoRA: r = 16, α = 32, dropout 0.05, on the attention projections. 2 epochs, learning rate 2e-4 with cosine decay, effective batch 16, loss on completions only, seed 13.
4. **Evaluate** with the tuned model's own signals:
   - the S3 design on 300 pool claims × 8 cells;
   - a 100-claim crowd;
   - S5, as the transfer test.
5. **It must beat:**
   - the answer-only twin;
   - the Bayes-rule prompt;
   - a calculator, where the model outputs s, its confidence and the informative lead, and Python computes the vote.

   If the calculator matches the LoRA, report that as a finding.

### X2 · Reddit accumulation (optional)

300 comments × 3 treatments × 20 sequential voters, on M2 only: 18k calls. Run it only on idle keys after Oct 12. Voter t sees the starting score plus every earlier vote. Report the final mean score and the share of comments that reach 10. Do not test them: the scale differs from Muchnik's.

---

## 8. Running it on NIM keys and Kaggle

### 8.1 NIM keys

**Setup, once:**

1. Keys live in `nvidia_keys.txt` at the repo root, which git never sees. The endpoint is `https://integrate.api.nvidia.com/v1` (OpenAI-compatible).
2. `llm.load_keys()` matches every `nvapi-…` string by pattern, so the file's layout does not matter and no key is ever printed. The `NIM_KEYS` environment variable overrides the file, and `NIM_KEY_SLICE=a:b` gives one process its own slice.
3. `python -m src.llm --check-keys` reports each key by a 6-character fingerprint and its HTTP status, never the key itself.
4. Keys are personal. Use only team members' own keys, keep them out of git and logs, and check that the free tier's terms cover research evaluation at this volume.

**How the pool behaves.**

- Each call takes the live key that is free soonest. A pacer per key spaces its calls 60/RPM seconds apart.
- On 401/403 the key is dropped for the rest of the run. On 429 that key cools down for 20 s and the call moves to another key. On 5xx or a timeout, the call backs off with jitter and retries, up to 8 times.
- On 400, 404, 410 or 422 the call is not retried, because the request or the model itself is wrong. These errors are not cached, so a later run tries again.
- At most 256 calls are in flight at once.

**Key slices.** Each key is used by exactly one running process. If two processes share a key, both pacers fire on it and 429s follow. Slices are reassigned at gates, never mid-run. With 20 keys:

| Job | Keys | Runs | Where |
|---|---|---|---|
| A | K01–K08 | S1, then S3 (the critical path) | Kaggle CPU session |
| B | K09–K14 | S4 | Kaggle CPU session |
| C | K15–K18 | S5, then X2 | Kaggle CPU session or laptop |
| D | K19–K20 | smoke test, probe, pilots, S2, judge | Pushkar's laptop |

With fewer keys, run the jobs one after another in priority order. Even 6 keys, at a realistic 90 RPM, cover the core in about 48 h of unattended runs.

**Watch the 429 rate** that each manifest reports. If it goes above 2%, lower `NIM_RPM` to 25, test each key alone, and drop dead keys.

### 8.2 Kaggle (GPU jobs)

Kaggle runs the local models (L1, L2, L2T, and X1) on the **T4 × 2** machine. Everything is driven from the laptop by `kaggle/job.py`, using the credentials in `~/.kaggle/kaggle.json` (the university account `pushkarsiuuarora`).

1. **`push`** copies the code, config, prompts, processed data, pools, and any caches to resume into the private dataset `<user>/ai-voters-code`. It waits until that version is ready, then pushes `job.py` itself as a private GPU kernel named `aiv-<name>`.
   - The Kaggle CLI (1.7.4.5) cannot choose the machine type, so `job.py` calls `kagglesdk` directly with `machine_shape="NvidiaTeslaT4"`, which is the T4 × 2 machine.
2. **On Kaggle**, the kernel checks the bundle's commit against the job, and stops if the dataset version is stale. It then installs vLLM and runs each `--cmd` as its own process, so only one model is on the GPUs at a time. It leaves `runs/` in `/kaggle/working`.
3. **`pull`** downloads the output and merges the caches into `./runs` by request hash. **`status`** shows the kernel state.

The kernel does not `git clone`, because the GitHub repo is private and a kernel pushed through the API has no token. The dataset bundle needs no secrets, and the commit check keeps it honest.

```bash
python kaggle/job.py push --name s1-l1 --cmd "src.s1_signals --models L1" --caches S1/L1
python kaggle/job.py status --name s1-l1
python kaggle/job.py pull --name s1-l1
```

Keep each job under 10 h; the session limit is 12 h. A killed job loses nothing that was pulled, because `--caches` ships the existing cache and the run resumes from it.

**vLLM on the T4 pair.** The T4 has no bf16 and no FlashAttention 2, so serve fp16 AWQ builds with these settings:

- `tensor_parallel_size=2, dtype="float16", enforce_eager=True, disable_custom_all_reduce=True`
- `max_model_len=4096, gpu_memory_utilization=0.85`
- `logprobs=20`
- for Qwen3, `chat_template_kwargs={"enable_thinking": False}` (or `True`)

Local jobs use offline batching:

1. The stage script builds every request and drops the cache hits.
2. It sends the rest as one `llm.chat` batch.
3. It writes the records in the same schema as NIM records, so analysis never cares where a record came from.

If tensor parallel fails, run one replica per T4. If vLLM will not boot by Oct 8, drop L1 and L2: they add robustness, not the headline.

**GPU budget** (an estimate; measure it on day 1). Each account needs well under one week's quota:

- #1 (L1): ≈ 3 GPU-h.
- #2 (L2): ≈ 5 GPU-h, because thinking-on runs are slower.
- #3 (X1): ≈ 10 GPU-h.

### 8.3 Commands

```bash
python -m src.data all                                   # S0: build the data
python -m src.llm --smoke                                # G1: smoke test
python -m src.llm --probe                                # S0: rate probe
python -m src.s1_signals --models all --limit 100        # S1 pilot
python -m src.s1_signals --models all                    # S1
python -m src.s2_urn --models all                        # S2
python -m src.s3_histories --models all --part main      # S3 main cells
python -m src.s3_histories --models all --part fixes     # S3 fix cells
python -m src.s4_crowds --crowds all --k 0,1,2,3         # S4
python -m src.s5_reddit --models all                     # S5
python -m src.analyze all                                # S6: every number, figure and table
```

---

## 9. Schedule and gates

| Day | Work | Gate or deadline |
|---|---|---|
| Mon Oct 5 | S0: repo, keys, `data.py`, prompts, `llm.py`, smoke test | |
| Tue Oct 6 | rate probe; Kaggle boot; S1 pilot (100 claims); S2 sanity check | **G1** |
| Wed Oct 7 | S1 full run (NIM + Kaggle); vertical slice through S3 and ρ; S5 pilot; oracle cross-check | hypotheses frozen |
| Thu Oct 8 | pools built; S3 main starts; S5 and S2 full runs | **G2**, prompts frozen |
| Fri Oct 9 | S3 main + local replicas; S4 clean crowds; X1 go/no-go | |
| Sat Oct 10 | S3 fix cells; S4 seeding | |
| Sun Oct 11 | S4 private-first crowd; S5 done; analysis on partial data | |
| Mon Oct 12 | refills and re-runs; X2 if keys are idle | |
| Tue Oct 13 | check H1/H2 across models and paraphrases | **G3**: X1 kept or cut |
| Wed Oct 14 | all runs frozen; manifests final | runs end |
| Oct 15–17 | `analyze all`; figures and tables locked | |
| Sun Oct 18 | abstract final | abstract due Oct 19, 17:29 IST |
| Oct 19–24 | writing; every section read cold by someone who did not write it | |
| Sun Oct 25 | submit | paper due Oct 26, 17:29 IST |

There is a daily 15-minute standup at 10:00 IST covering yesterday's runs, today's runs and blockers.

---

## 10. Quality control

**After every 1,000 calls** (the runner prints these; the stage owner checks them):

- [ ] parse rate ≥ 99% per model, and < 1% of replies cut off (`finish == "length"`)
- [ ] TRUE share within 0.50 ± 0.15 on balanced items
- [ ] random-labeler cells move beliefs less than independent cells (S3)
- [ ] 429 rate < 2%, and calls per minute near plan
- [ ] `--dry`: a human reads five random rendered prompts and confirms the vote block is present and correct

| Symptom | Likely cause | Action |
|---|---|---|
| Parse failures > 3% | JSON instruction ignored, thinking text leaking, or truncation | strip text before `</think>`; add one JSON example; raise `max_tokens` for M4 and M6-high |
| Social reply = solo reply everywhere | vote block missing or truncated | run `--dry` and read the rendered prompts |
| Nearly all TRUE (or all FALSE) | label bias | keep pools balanced and label order randomized; analyze `p_true`; report base rates |
| `p_true` only ever 0 or 100 | overconfident stated beliefs | clip to [1, 99]; lean on L1/L2 token probabilities |
| Solo accuracy > 0.9 on most claims | memorized claims | drop recognized claims; refill from reserve; compare the `recent` split |
| 429 storms | pacer too fast, a dead key, or a per-account limit | set `NIM_RPM` to 25; test keys alone; drop bad keys |
| NIM model string changes mid-run | catalog update | stop; re-run that model's cells or report both versions |
| No herding at all in S3 | ceiling from strong private signals | check pool p̄; shift toward 0.55–0.75 before concluding |
| Kaggle session killed | 12 h limit | keep jobs < 10 h; the cache resumes; push hourly |

---

## 11. What changed from the runbook, and why

| # | Change | Why |
|---|---|---|
| 1 | E6 is dissolved. F1, F2 and a Bayes-rule baseline become S3/S4 cells; F3 comes free from S4 | Same science with one stage fewer. The fixes share S3's histories, so every comparison is paired |
| 2 | Fix cells use their parent stage's decoding (T = 0 in S3) | The runbook ran E6 at T = 0.7 against E3 at T = 0, which confounds the fix with temperature |
| 3 | B2 becomes optional X2 (one model, 300 comments) | B2 cost 108k calls, a quarter of the old budget, for a number the runbook says to report, not test. B1 alone tests H4 |
| 4 | The LoRA becomes optional X1: one base, two runs, go Oct 9, kill Oct 13 | The paper stands on H1–H5, and the plan already said to cut the LoRA first |
| 5 | S4 uses 300 clean and 150 seeded claims per crowd (was 400 and 200) | H3 is tested pooled across 7 crowds, where the standard error is ≈ 1 pp either way. Saves ≈ 28k calls |
| 6 | M4 screens only 600 claims in S1, and persona calls run only on pool claims | M4 needs only a 300-claim pool, and thinking calls are slow |
| 7 | Paraphrase is an explicit S3 factor (a, b, c); version tags now mean prompt versions only | The runbook used "v2/v3" for both. G3 needs all three paraphrases measured |
| 8 | H1 is judged against ρ_Bayes = logit(q)/logit(p̂), not against 1 | The two are equal only when q = p̂. The PDF's H1 row still says 1; fix it before the Oct 7 freeze |
| 9 | The silent-herding judge labels 2,000 stratified flips, not every flip | Gives a standard error of ≈ 1 pp without judging every flip |
| 10 | 9 Python files and 2 notebooks, instead of ~25 files | One file per stage, one owner per file, one CLI |
| 11 | NIM sweeps run unattended on Kaggle CPU sessions with fixed key slices; code reaches Kaggle by `git clone`, not dataset uploads | No laptop babysitting, no GPU quota used, no key used twice, no stale code |
| 12 | L1 and L2 become a robustness track, off the critical path | If vLLM will not boot on a T4, the paper still stands |

**Old → new names:** E0 → S0 · E1 → S1 · E2 → S2 · E3 → S3 · E4 → S4 · E5 B1 → S5 · E5 B2 → X2 · E6 → S3/S4 fix cells + X1 · E7 → S6.

---

## 12. Decisions needed by Oct 7

1. **Approve the lean scope** in §11, especially making X1 and X2 optional.
2. **Fix H1's benchmark** in the pre-registration before the freeze (§11, row 8).
3. **S4 owner.** The proposal is Teammate 3, so that Teammate 2 holds only the critical path S0 → S1 → S3.
4. **Contamination guard.** LIAR2's `recent` split (2022-08 to 2023-07) predates every model's training cutoff. It therefore measures "less repeated", not "unseen", and the recognition probe is the real guard. If reviewers are likely to push on this, 150 hand-collected PolitiFact claims from Jul–Sep 2026 (the PDF's original idea) would take about 3 hours.
5. **Throughput.** The S0 probe will tell us whether NIM limits apply per key or per account. Plan the key slices on that answer.

---

## 13. Logs

This log is append-only, with the newest entry last. Each entry says what ran, what it showed, what changed and why, and what comes next. Numbers here are working numbers; paper numbers come only from `src/analyze.py`.

### 2026-10-06 · S0 build and G1

**Built**
- The repo is `D:\web`, pushed to the private GitHub repo `pushkar1887/ai-voters`. The `.gitignore` is an allowlist, and a key-pattern scan runs before every commit (0 hits so far).
- Code: `src/llm.py`, `src/data.py`, `src/oracle.py`, `src/s1_signals.py`, `src/s2_urn.py` and `kaggle/job.py`. Prompts are in `prompts/v1.yaml`, settings in one `config.yaml`. There are no notebooks and no test files: the oracle carries its hand-worked checks in `__main__`.
- The laptop environment is `.venv` (Python 3.10) built from `requirements.txt`.

**S0 data** (full counts in `data/processed/build_report.json`)
- **LIAR2.** Label codes are confirmed against the published counts (0 = pants-fire … 5 = true). The 22,962 rows leave 11,947 binary claims after cleaning, 2,542 of them TRUE. Balanced splits: eval 1,500, LoRA 1,500, reserve 2,084.
- **`recent` split** (dated on or after 2022-08-01): only 24, 18 and 32 claims in eval, LoRA and reserve. That is too few for a contamination split, so the recognition probe is the real guard (supports §12, item 4).
- **Reddit.** The ConvoKit server is slow (about 1 MB per 2 minutes), so the download resumes. Of 297k utterances, 147,568 pass the filters, giving 2,100 comments: 300 in each of 7 groups, split 1,075 low tier and 1,025 high tier. The subreddit-to-group map in `config.yaml` is a DRAFT.
- **Personas:** 60, with balanced margins.

**NIM and gate G1**
- **Keys.** 24 keys parse from `nvidia_keys.txt` (never printed). One returns 403 and is dropped at runtime, leaving 23 live.
- **The catalog has moved.** All six runbook model strings return 410 Gone. These return 404: Mistral, Phi, Gemma-3, Granite, the Llama-3.1-Nemotron line, Nemotron-4, Jamba, Yi and DBRX.
- **Listed but unusable under load:**
  - DeepSeek-v4.1-flash: median 178 s and about 4k reasoning characters per call, 8 of 30 parsed; requests with thinking off all fail.
  - GLM-5.3, GLM-5.3-flash, Kimi-K3 and Gemma-4-31B: no answer within 6 minutes.
  - Llama-3.2-90B answered 3 of 30.
  - Muse-Glimmer-30B: median 139–176 s.
  - Nemotron-3-Ultra and Nano-Omni: most calls fail.
- **Reasoning switches** (knob test on real claim prompts):
  - Nemotron-3 obeys `chat_template_kwargs.enable_thinking`: with `false`, it produces 0 reasoning characters and about 50 tokens. The `/no_think` and "detailed thinking off" system lines do nothing.
  - gpt-oss-20b obeys the `reasoning_effort` parameter: about 90 tokens at low, about 330 at medium. The runbook's "Reasoning: low" system line is ignored. High effort runs past 5k tokens and truncates half the replies at 8,192.
- **G1 smoke test** (10 claims each): M1, M2, M3, M4 and M6 all parsed 10/10, so **G1 passes for the new roster**.
- **Throughput.** The S1 pilot ran at about 640 calls/min overall with zero 429s, which is the pacer's own ceiling (23 keys × 35 RPM). There is no sign of per-account throttling, which answers §12, item 5.

**Decisions**
- **Roster rebuilt** (§4): M1 Llama-3.2-11B; M2 Nemotron-3-Super-120B, thinking off; M3 Nemotron-3.5-Lightning-30B, thinking off; M4 is M2 with thinking on; M6 gpt-oss-20b at low effort; M6H gpt-oss-20b at medium effort. M5 is retired.
- The base models for H1 are M1, M2, M3, M6, L1 and L2, from four families (Meta, NVIDIA, OpenAI, Alibaba).
- **Code reaches Kaggle as a private dataset bundle**, not by `git clone`: the repo is private, and API-pushed kernels have no token. T4 × 2 is selected through `kagglesdk` with `machine_shape="NvidiaTeslaT4"`.

**Kaggle**
- **Boot 1.** T4 × 2 confirmed. The latest vLLM (0.31.0, torch 2.13 with CUDA 13) installed, but import failed because Kaggle's preinstalled torchaudio is built for CUDA 12.8.
- **Boot 2** (pushed) removes torchaudio and prints the driver and torch versions.

**Open**
- **Label bias.** In the S1 pilot, M2 answers TRUE on only 17% of samples (a FALSE bias), and M4 on 31%. Check this against G2's 0.50 ± 0.15 on the full S1. The playbook answer is balanced pools and analysis on `p_true`.
- **Drafts to review.** Pushkar and Teammate 3 review the subreddit map and the Bayes-rule prompt before G2.

**Next:** the S1 pilot summary, then full S1 on NIM, the S2 sanity check, S3 code and its vertical slice, and L1/L2 on Kaggle once vLLM boots.

### 2026-10-06 (later) · Pilots: S1 prompt fix, recognition probe, Kaggle vLLM

**S1 pilot, first pass (100 claims × M1, M2, M3, M4, M6)**
- **Health:** 100% parsed, zero 429s, about 640 calls/min overall.
- **Two problems:**
  - **Prompt validity.** M2 and M4 judged whether the speaker *said* the quote ("no record of Quentin Miller saying this… likely fabricated") instead of whether it is *true*. This drove most of their FALSE answers.
  - **Recognition probe.** M1 says "seen it" to 93 of 100 claims, mostly answering "false" whatever the truth; M6 says YES to 76 and M4 to 62. For these models the probe measures a habit, and filtering on it removed about half the claims, skewed toward FALSE ones.
- **Fixes, all before the G2 freeze:**
  - The claim block now adds "The speaker really said this; judge whether what it says is true."
  - The leading "Says (that)" is stripped from statements, and inner double quotes become single quotes.
  - The recognition filter applies only where the probe discriminates. A model that says YES to more than 20% of claims must name the right side at least 10 points more often than a constant guess.
  - The pre-fix caches moved to `runs/_archive/2026-10-06-pilot0-attribution-prompt/`.
- **Two code bugs fixed:**
  - `config.yaml` turned the YAML labels `false`/`true` into booleans in `label6`; they are now quoted. The binary labels were never affected.
  - Pools listed all TRUE claims first, so any "first N" slice was mostly TRUE. Pools now interleave TRUE and FALSE.

**S1 pilot, second pass** (`results/S1_summary.csv`)

| Model | TRUE share | Solo accuracy | Probe says YES | Probe valid | In band | Pool | p̂ |
|---|---|---|---|---|---|---|---|
| M1 | 0.46 | 0.66 | 0.79 | no | 51 | 34 | 0.88 |
| M2 | 0.11 | 0.59 | 0.02 | yes | 22 | 14 | 0.86 |
| M3 | 0.35 | 0.62 | 0.05 | yes | 32 | 28 | 0.82 |
| M4 | 0.26 | 0.70 | 0.67 | yes (51% recognized) | 16 | 12 | 0.92 |
| M6 | 0.24 | 0.57 | 0.84 | no | 32 | 20 | 0.75 |

- **Reading.** The FALSE lean of the Nemotron models and gpt-oss survives the prompt fix, so it is the models' own skeptical prior, a reportable property. Balanced pools neutralize it, because inside each pool the signal is informative for both labels.
- **G2 TRUE-share check.** M2, M4 and M6 will fail 0.50 ± 0.15. Proposal for the lead: report this check as a property of each model rather than a gate, since its purpose (catching fake herding from label bias) is served by the balanced pools and the analysis on `p_true`.
- **Pool yield** is 12–34% of screened claims. The full S1 therefore screens eval plus all of reserve (3,578 claims) for M1, M2, M3 and M6, and 2,500 claims for M4 and M6H. That is about 116k calls (about 3 h), against the 54k planned.
- **Expected pools:** M1 and M3 reach about 1,000; M6 about 700; M2 about 500 (below the 600 fallback, so it gets 12 cells per claim in S3); M4 and M6H about 300.

**Kaggle T4 × 2: vLLM works** (kernel `aiv-boot`)
- **Stack:** driver 580.178 (CUDA 13.0); the preinstalled torch was 2.11+cu128. vLLM 0.31.0 brings torch 2.13+cu130. Kaggle's stale torchaudio must be removed, and `job.py` now does this. vLLM is pinned at `vllm==0.31.0`.
- **Smoke test, 40 claims each, run on the old prompt, so it validates the pipeline only:**
  - L1 Llama-3.1-8B AWQ: 40/40 parsed, about 2 min including model load.
  - L2 Qwen3-8B, thinking off: 40/40.
  - L2T, thinking on: 40/40, about 550 tokens per reply.
- **Token probabilities** (`lp_p_true`) are extracted for every reply, and their argmax matches the stated verdict 40/40 for both L1 and L2.

**Next:** the S3 vertical slice (10 claims per model, main and fix cells), the S2 sanity check (20 rounds), then the full S1 on NIM and S1 for L1, L2 and L2T on Kaggle.

### 2026-10-06 (evening) · Sanity checks, vertical slice, full runs launched

**S2 sanity check** (20 rounds; at lead 0, agents must follow their own draw more than 90% of the time)
- **First pass failed for three models.** M1, M3 and M6 followed their draw only 70%, 79% and 89% of the time.
  - Their reasons show composition errors: M1 wrote "a dark ball is more likely under A", M3 "equally likely under either urn".
  - M6 showed an output slip: it computed P(A) = 33%, then wrote `"urn": "A"`.
- **Fix** (v1, before the freeze): the prompt now states what each draw points to, as the lab trained its subjects, and asks for `p_a` before the guess. The old caches moved to `runs/_archive/2026-10-06-s2-sanity0-urn-prompt/`.
- **Second pass:**

| Model | Own draw at lead 0 | Inconsistent decisions | Cascade rate (humans 0.71) |
|---|---|---|---|
| M1 | 0.80 | 0.13 | 0.78 |
| M2 | 1.00 | 0.02 | 0.24 |
| M3 | 0.81 | 0.08 | 0.30 |
| M6 | 0.96 | 0.08 | 0.74 |

- **Decision:** M1 and M3 keep about 20% residual errors. That reads as a capability limit of the small models rather than a misread prompt. They stay in the study, and the lead-0 rate is reported per model as a task-comprehension measure. M4 and M6H are added to S2, because reasoning on and off belongs in the human-anchor comparison too.

**S3 vertical slice** (10 pool claims per model; main and fix cells; 100% parsed)
- **The whole path S1 → S3 → S6 runs end to end:** ρ, q\* and γ with claim-bootstrap intervals, follow curves against Bayes, and paired fix effects.
- **First patterns,** too small to claim anything:
  - The Nemotron models over-follow sequential majorities, at 0.69 and 0.92 against Bayes's 0.31 and 0.69. That is the correlation-neglect shape.
  - M1 and M6 under-follow independent majorities.
  - Every model follows random labels 17–33% of the time, where Bayes says 0.
  - Private-first lifts M2's accuracy from 0.54 to 0.89 on paired cells.
- **Design refinement** (logged in the code docstring): every framing draws its history from the same Bayes-consistent generator, with cascades oversampled. Framings therefore differ only in the vote-block text, and the redundant post-cascade votes range from 0 to 4, so γ is identifiable.
- **Fallback built in:** a base model whose pool is under 600 claims gets 12 cells per claim (the plan's fallback). Cells are seeded by claim and index, so the first 8 are unchanged.

**S5 pilot** (100 comments × M1, M2, M3, M6; 100% parsed)
- Voting styles differ sharply. M1 never abstains (79% UP). M2 abstains 67% of the time, close to real readers. M6 downvotes 72% of comments.
- Up-lift ranges from −1% (M1) to +92% (M2). The prompt is neutral, so these base rates are findings, and the lifts are measured within each model.

**Launched (Oct 6, evening)**
- Full S1 on NIM: M1, M2, M3, M4, M6 and M6H over eval plus all reserve (3,578 claims; 2,500 for M4 and M6H), on keys 0–17. About 116k calls.
- Full S2 (200 rounds × 6 models), then full S5 (2,100 comments × 4 models), on keys 18–23.
- Kaggle T4 × 2: kernel `aiv-s1-l1` (L1) and kernel `aiv-s1-l2` (L2 and L2T) run in parallel, so the university account takes at least two GPU sessions at once.

### 2026-10-06 (night) · S4 pilot bug, figures, throughput ceiling

- **S4 pilot** (2 claims × M2, M6 and the mixed crowd, k = 0–3). The code ran end to end, but it caught a design-critical bug: **agents' solo verdicts differed across k**. All four k-chains fired the same solo request at once, before any reached the cache, so each drew its own sample at T = 0.7. The seeding effect must hold each agent's private signal fixed.
  - **Fix in `llm.py`:** in-flight deduplication. A request whose hash is already pending now awaits that result.
  - **Re-run:** solo verdicts are identical across k for 6 of 6 crowd–claim pairs. The bad pilot moved to `runs/_archive/2026-10-06-s4-pilot0-no-inflight-dedup/`.
  - S1, S2, S3 and S5 never send duplicate requests, so nothing else was affected.
- **Throughput ceiling.** With S1 on keys 0–17 and S2/S5 on keys 18–23, the system runs at about 800 calls/min, which is the pacer ceiling (24 × 35). The ~145 s "latency" is the wait for a key slot. Remaining NIM work for the core is about 290k calls, about 6 h at the ceiling.
- **Figures.** `analyze.py` now writes Figs. 1–5 (PDF + PNG) to `paper/figures/`.
  - They use the dataviz reference palette in fixed slot order: model = slot 1, second framing or humans = slot 2, Bayes = dashed ink. Small multiples keep panels at three or fewer series.
  - Node is not installed, so the palette validator did not run here. The slots used are the reference palette's first three, which it documents as validated across all pairs in both modes.
  - All five figures were rendered and inspected; label collisions in Figs. 3–5 were fixed.
- **Kaggle pools.** Pools for L1, L2 and L2T are rebuilt on the laptop from the pulled caches (`s1_signals` with all cache hits). `llm.py` now imports vLLM only when there is something to generate.
- **S3 table** keeps only claims in each model's current pool, so pilot-slice claims that left a pool never enter the analysis.
- **Repo:** `README.md` added with setup and run order.
