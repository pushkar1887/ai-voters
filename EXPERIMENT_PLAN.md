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

**Owner.** Pushkar (lead) does all the work in this plan: code, runs, analysis, the human tasks in §12, and the writing. Teammates have no tasks in it (lead's note, Oct 7). The checks that needed a second person are now automated (the oracle, S0 step 6) or optional (a second κ labeller, §12 D6).

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
- **Owner:** Pushkar.
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
   - Map the 100 subreddits to Muchnik's 7 topic groups. Claude drafted the map in `config.yaml`; Pushkar reviews it (§12, T2).
   - Sample 300 comments per group, stratified by subreddit. The quality tier is the top or bottom half of score within the subreddit.
4. **Personas** (`python -m src.data personas`). 60 personas from age band × occupation × region, seed 13, with no politics.
5. **Prompts.** Copy the runbook templates into `prompts/v1.yaml`, with paraphrases a, b and c. Paraphrases change only the instruction wording; the JSON keys stay the same. Add the Bayes-rule prompt that serves as the H6 baseline.
6. **Oracle.** `oracle.py` follows the runbook. `python -m src.oracle` checks it against hand-worked cases and against exact enumeration of every history that Bayesian voters can produce, up to 6 votes. This replaces the second person's 20 hand-worked histories.
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
- **Owner:** Pushkar.
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
- **Owner:** Pushkar.
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
- **Owner:** Pushkar.
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
- **Owner:** Pushkar.
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
- **Owner:** Pushkar.
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
| 8 | Silent herding | H5 | M6 (low effort, T = 0) judges a stratified sample of 2,000 flips. Used only if Cohen's κ ≥ 0.7 against blind human labels on 200 reasons (§12, T1) |
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
- **Owner:** Pushkar. Training runs on Kaggle #3; evaluation on #1–#2.
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

Run these with the project venv (`.venv/Scripts/python` on Windows). The system Python has no statsmodels.

```bash
.venv/Scripts/python -m src.data all                                              # S0: claims, comments, personas
.venv/Scripts/python -m src.oracle                                                # S0: oracle checks (hand-worked + exhaustive)
.venv/Scripts/python -m src.llm --smoke                                           # G1: every NIM model answers and parses
.venv/Scripts/python -m src.s1_signals --models M1 M2 M3 M4 M6 M6H --refill 2078  # S1: private signals -> data/pools/
.venv/Scripts/python -m src.s1_signals --models M7                                # S1, Ultra (M7): 2,500 screened, 300-claim pool
.venv/Scripts/python -m src.s2_urn --models M1 M2 M3 M4 M6 M6H M7                 # S2: urn anchor
.venv/Scripts/python -m src.s3_histories --models M1 M2 M3 M4 M6 M6H M7 --part main
.venv/Scripts/python -m src.s3_histories --models M1 M2 M3 M4 M6 M6H M7 --part fixes
.venv/Scripts/python -m src.s4_crowds --crowds M1 M2 M3 M4 M6 M7 mixed            # S4: clean and seeded crowds
.venv/Scripts/python -m src.s4_crowds --crowds M1 M2 M3 M4 M6 M7 --f1             # S4: private-first crowd
.venv/Scripts/python -m src.s4_crowds --crowds M2 M3 --br                         # S4: Bayes-rule prompt crowd
.venv/Scripts/python -m src.s5_reddit --models M1 M2 M3 M6                        # S5: Reddit first voter
.venv/Scripts/python -m src.analyze judge                                         # H5: silent-herding judge + blind kappa sheet
.venv/Scripts/python -m src.analyze kappa                                         # H5 gate, once the sheet is labelled
.venv/Scripts/python -m src.analyze all                                           # S6: every number, table, figure
```

Local models (L1, L2, L2T) run the same stage commands on Kaggle through `kaggle/job.py`; see the README.

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

## 12. Human tasks and decisions

Pushkar owns all of it. The lead's calls on D1–D8 are recorded from Oct 8.

### 12.1 Human tasks

**T1 · Label the 200 flip reasons (the H5 gate)** · about 1 h · by Oct 14

- **Where:** open `D:\web\results\judge_labels.csv` in Excel or VS Code. Do **not** open `results/judge_key.csv` until you have finished: it holds the judge's answers.
- **What you label:** the `reason` column. The judge was asked: *does this reason mention or rely on the other reviewers' ratings?*
  - **YES** when the reason refers to the ratings or raters that were shown ("most reviewers said FALSE", "the others think it's true"), or agrees or disagrees with them because of those ratings.
  - **NO** when the reason argues from the claim, the facts, the speaker or the model's own knowledge. A fact-check or news item that the reason cites is not a shown rating, so it is NO.
  - Made-up examples: "Most reviewers rated this false, so I agree." → YES. "The budget figure in the claim is wrong; the deficit was about $1.5 trillion." → NO.
- **How:** type `YES` or `NO` in `label_pushkar` on each row. Save as CSV (Excel: "CSV UTF-8") with the header and the 200 rows intact.
- **Then run**, from `D:\web`:

```bash
.venv/Scripts/python -m src.analyze kappa
```

- **Read the result.** κ ≥ 0.7: the judge's rates on all 1,998 flips go in the paper. κ < 0.7: H5 is reported from your 200 labels, and the paper says the judge failed validation.

**T2 · Review the subreddit → topic map (S5)** · about 20 min
- **Where:** `config.yaml` → `data.comments.groups`, Claude's draft. It maps 77 subreddits to Muchnik's 7 groups and sends 23 to `other`, which is dropped.
- **Check** that each subreddit fits what its group means on a news site. The borderline calls:
  - `conspiracy` → politics; `Bitcoin` and `Frugal` → economics;
  - `canada`, `unitedkingdom`, `australia` and `singapore` → general news;
  - `AskMen`, `AskWomen`, `relationships` and `AmItheAsshole` → culture and society;
  - `books`, `Music`, `movies` and `photography` → culture and society rather than fun; sports and games → fun;
  - `science`, `askscience` and `guns` → other.
- **What it moves:** only the per-topic panel of S5. H4 is tested on all groups pooled. Muchnik found herding only in politics, culture and society, and business, so those groups matter most.
- **If you change it:** run `.venv/Scripts/python -m src.data comments`, then `.venv/Scripts/python -m src.s5_reddit --models M1 M2 M3 M6`, then `.venv/Scripts/python -m src.analyze all`. Only the groups whose subreddit lists changed are re-sampled. Every other comment is a cache hit.

**T3 · Review the Bayes-rule prompt (H6 baseline)** · about 10 min
- **Where:** `prompts/v1.yaml` → `bayes_rule`. It is appended to the social prompt in the S3 BR cells and in the S4 Bayes-rule crowd.
- **Check:**
  - *Correct:* the log-odds values (0.41, 0.85, 1.39 and 2.20 for 60–90%) are right, and the rules for post-cascade and random votes match the oracle. One gap: it does not say what to do on an exact tie. The oracle keeps its own reading.
  - *Neutral:* nothing in it hints at which verdict is right.
- **Already measured:** in S3, the BR cells raised the follow rate in all 8 models tested (M4: 0.29 → 0.68) and lowered their accuracy. In S4 the Bayes-rule crowd is running for M2 and M3 (§13).
- **Recommendation:** approve v1 as run. Editing a baseline after seeing it lose looks like tuning it to lose. Only a real error justifies `v2.yaml` and a re-run.

### 12.2 Decisions (recorded Oct 8)

| # | Decision | The lead's call (Oct 8) |
|---|---|---|
| D1 | Model roster | Approved. No paper paragraph on the swap. Table 1 still lists the models. Ultra (M7) added, see §13. |
| D2 | G2's TRUE-share item | Reported as a finding. |
| D3 | Reporting H1–H6 | In the paper's findings, results and analysis sections (table in §12.3). |
| D4 | Headline and thesis | The lead writes it later, once every experiment is complete. |
| D5 | Extra runs | 1 (re-probe) done. 2 (Bayes-rule crowd, M2 and M3) and 3 (Ultra as M7) running. 4 (F1 crowd to 300 claims) skipped. |
| D6 | Labelling | The lead labels alone (T1). |
| D7 | Contamination check | Not doing. Ultra recognises 59% of the claims (§13), so this is reported as a finding, not checked. |
| D8 | Release for double-blind review | The lead, at the end (by Oct 24). |

### 12.3 Reporting draft (D3)

| | The rule's verdict | What the data say |
|---|---|---|
| H1 | Not supported: the deviation runs the other way | 8 of 9 models give a shown vote less weight than Bayes (ρ 0.05–0.43 with q unstated, where Bayes is 1; the intervals exclude 1). M3 is the exception (0.94). Read literally, the rule ("the CI covers ρ_Bayes") does not refute H1, but it was written for an upward deviation |
| H2 | Refuted (γ_seq ≤ 0.5) | Models stop counting after about two votes in both framings, so there is no framing gap where Bayes has one |
| H3 | Supported | Visible votes cost 6.0 pp [4.1, 7.8] pooled. One bot costs the M2 and M3 crowds about 20 pp. Against Bayes it splits: only the M2 crowd loses more than a Bayesian crowd to one bot, and every LLM crowd does far better at k ≥ 2, where Bayesian crowds collapse to 0 |
| H4 | Supported for M2, M3 and M6; refuted for M1 | No downvote correction for M2, M3 and M6, while M1 corrects downvotes as people do. Herding is one-sided and model-specific |
| H5 | Supported, pending T1 | 76–99.5% of flips never mention the votes |
| H6 | Supported | Private-first recovers 69–100% of the loss (pooled gain +6.4 pp [3.1, 9.7]). In single decisions it beats the Bayes-rule prompt on accuracy in all 8 models. In crowds the Bayes-rule prompt adds nothing without bots (M2 +1.3 pp, M3 −1.5 pp) and makes seeded crowds follow the bots more (k = 2: M2 −11 pp, M3 −28 pp, against the social crowd on the same claims) |

### 12.4 Still open on Claude's side

- **Pooled, Holm-corrected tests** (S6 row 9: one p-value per hypothesis, model as a fixed effect). Not computed; `holm()` exists but nothing calls it. Needs a go.
- **The no-cost checks** from earlier (ρ on TRUE- vs FALSE-leading histories; ρ by p̄ band). Not run. They were tied to D2 and D7, so they are optional now.

**Settled by existing rules:**
- **M1's urn row** stays out of the human comparison. At lead 0 it follows its own draw only 64% of the time, and S2 requires more than 90%. M3 passes at 90.3%.
- **X1** is cut by G3. G3's fallback headline, "near-Bayesian", does not fit, because the models are sub-Bayesian.

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

### 2026-10-06 · Pilots: S1 prompt fix, recognition probe, Kaggle vLLM

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

### 2026-10-06 · Sanity checks, vertical slice, full runs launched

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

**Launched**
- Full S1 on NIM: M1, M2, M3, M4, M6 and M6H over eval plus all reserve (3,578 claims; 2,500 for M4 and M6H), on keys 0–17. About 116k calls.
- Full S2 (200 rounds × 6 models), then full S5 (2,100 comments × 4 models), on keys 18–23.
- Kaggle T4 × 2: kernel `aiv-s1-l1` (L1) and kernel `aiv-s1-l2` (L2 and L2T) run in parallel, so the university account takes at least two GPU sessions at once.

### 2026-10-06 · S4 pilot bug, figures, throughput ceiling

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

### 2026-10-06 · Background limits, vLLM checkpoints, cross-platform parity

- **Background limit.** Long laptop jobs are capped at 2 h each. The first full S1 stopped at 56–90% screening per model, the S2 → S5 chain stopped mid-S5, and the Kaggle watcher stopped too; nothing was lost, because every finished call is in the cache. S2 completed in full. S1 and S5 resumed on the user's "continue", sized to finish inside one window.
- **S2 full** (200 rounds × 6 models): done. `results/S2_table.csv` holds the numbers; the human-anchor reading comes with the S6 analysis.
- **Kaggle L1 S1:** complete, 22,468 records. Rebuilt on the laptop from cache: pool 1,000, p̂ 0.87, TRUE share 0.60, 99.99% parsed. The probe answers YES 91% of the time and is invalid, so it is reported, not filtered.
- **Kaggle L2/L2T S1 failed partway.**
  - L2's screening finished: 21,468 prompts, 100% parsed, TRUE share 0.25.
  - L2T (thinking on) then ran for 2 h 14 m, and vLLM died at 9,839 of 15,000 prompts (`RPC call to sample_tokens timed out`).
  - Because `_run_vllm` cached only after the whole batch, **all 9,839 L2T generations were lost**, and L2's persona calls never ran.
- **Fixes:**
  - vLLM now generates in chunks of 1,000 (`vllm.chunk`), caching each chunk before the next.
  - `job.py` retries a failed command up to 3 times (each retry resumes from the cache) and raises `VLLM_RPC_TIMEOUT`.
  - L2T screens 1,200 claims instead of 2,500, which is enough for its 300-claim pool at a fraction of the thinking cost.
- **Laptop and Kaggle parity.** One L1 claim sat exactly on the band edge. It had p̄ = 0.9000000000000001 on the laptop (Python 3.10) and 0.9 on Kaggle (Python 3.13), because `sum()` compensates float error from 3.12 on. p̄ is now integer percent with one division. Local and Kaggle pools match 1,000/1,000.
- **Bundle race.** Two back-to-back pushes created two dataset versions, and the second held only L2's cache, so the L1 kernel may have mounted the wrong one. The fixes:
  - Every bundle now ships all local-model caches, so each version is a superset.
  - The kernel checks each expected cache's record count and stops on a stale bundle.
  - If `aiv-local-l1` regenerated its S1, the S3 table's pool filter keeps the analysis on the laptop's pool, and any missing cells go in a small follow-up job.
- **Running:**
  - S1 (keys 0–17) and S5 (keys 18–23) on NIM.
  - Kaggle `aiv-local-l1`: S1 hits, then S3 main and fix cells.
  - Kaggle `aiv-local-l2`: L2 persona calls, S3 main and fix cells, then L2T S1 and S3.

### 2026-10-06 · First full S3 result (L1), round-robin keys, how to read S3

- **Keys:** strict round robin over live keys. Dead keys (401/403) are skipped permanently; a key in a 429 cool-down is skipped for that turn; each key keeps its own 35 RPM pacer. Verified offline, and every manifest now carries `key_requests`, per-key counts by fingerprint.
- **Kaggle `aiv-local-l1`: complete.**
  - S1 was all cache hits (21,468 screening + 1,000 persona calls), so the bundle race did not hit L1, and Kaggle's pool equals the laptop's.
  - S3 main: 8,000 cells in 18 min. Fix cells: 3,242 in 9 min. Both 100% parsed.
- **L1 (Llama-3.1-8B, 1,000 claims). First real H1/H2 numbers** (claim bootstrap, 300 resamples):
  - **ρ** = 0.10 [0.06, 0.13] with q unstated (Bayes 1.0), and 0.05 [0.02, 0.09] with q = 70% (Bayes 0.45).
  - **q\*** = 0.55 [0.54, 0.57] (Bayes 0.70).
  - **Follow curve, independent framing:** flat at 0.29–0.39 for leads 1–6, where Bayes rises from 0 at lead 1 to 1.0 at lead 3 and above.
  - **Random labels** are followed 8–25% of the time (Bayes 0), close to the sequential rate (12–25%).
  - **Reading:** L1 strongly *under*-weights other votes and is insensitive to how big the majority is. Whatever does move it looks like anchoring rather than reasoning about evidence. For L1, H1 is refuted in the under-herding direction, which the plan counts as publishable either way.
- **How to read S3** (a design point, now in the code docstrings): S3's scripted votes are built from the agent's own verdict, not from the truth, so they carry no real information about the claim.
  - S3's outcome is therefore **consistency with the stated evidence**: ρ, q\*, γ and agreement with the Bayes oracle.
  - Accuracy against the truth is meaningful only in S4's real crowds. This is why the Bayes-rule prompt "lowers accuracy" in S3 (0.61 vs 0.77) while moving following toward Bayes.
- **γ** is reported with its parts, β_inf and β_extra, because the ratio is unstable when a model barely moves at all. L1's γ_sequential of −1.64 is that case.
- **The S3 table** now includes every model with an S3 cache, so a NIM run never drops the Kaggle models' rows.

### 2026-10-06 · S5 complete (Reddit first voter, H4)

- **Run:** 2,100 comments × 3 treatments × 4 models (M1, M2, M3, M6) = 25,200 calls. 100% parsed, 9 429s in total. `results/S5_b1.parquet`.
- **Results** (paired bootstrap over comments, 2,000 resamples):

| Model | Up-lift in P(UP) | Down minus control | Reading |
|---|---|---|---|
| Humans (Muchnik et al.) | +32% | +4.5 pts | herd after an upvote, correct a downvote |
| M1 Llama-3.2-11B | +0.0% [−1.4, +1.4] | +2.5 [+1.4, +3.7] | no up-herding; corrects downvotes, as humans do |
| M2 Nemotron-Super | +10.6% [+2.8, +18.7] | −0.1 [−1.8, +1.4] | up-herding, no correction |
| M3 Nemotron-Lightning | +12.0% [+4.3, +21.0] | −1.0 [−3.0, +1.1] | up-herding, no correction |
| M6 gpt-oss-20b | −2.7% [−9.1, +4.2] | −2.0 [−4.0, −0.1] | follows the downvote (negative herding) |

- **Reading for H4:**
  - The Nemotron models show exactly H4's predicted asymmetry, positive herding without the human downvote correction, at about a third of the human effect size.
  - gpt-oss shows the mirror image: it follows a downvote and ignores an upvote.
  - Llama corrects downvotes like people do.
  - There is no single "LLM voter" profile.
- **Topics.** By topic group, up-lifts do not concentrate in politics, culture and business the way Muchnik found for people; each group has 300 comments, so these cells are noisy.
- **Code.** `s5_reddit.lifts` now returns the paired-bootstrap intervals, and Fig. 5 draws them.
- **Earlier pilot was noise.** The 100-comment pilot's +92% lift for M2 shrank to +10.6% on the full data.

### 2026-10-06 · S1 complete, gate G2

**S1 final** (`results/S1_summary.csv`, `data/pools/`; L2 and L2T are still running on Kaggle)

| Model | Screened | Parsed | TRUE share | Solo accuracy | Probe | In band | Pool | p̂ |
|---|---|---|---|---|---|---|---|---|
| L1 | 3,578 | 99.99% | 0.60 | 0.64 | invalid (91% YES) | 1,665 | 1,000 | 0.87 |
| M1 | 3,578 | 99.99% | 0.47 | 0.63 | invalid (80% YES) | 1,565 | 1,000 | 0.86 |
| M2 | 3,578 | 100% | 0.10 | 0.57 | valid, 1% recognized | 594 | 418 | 0.85 |
| M3 | 3,578 | 99.98% | 0.39 | 0.64 | valid, 3% recognized | 1,027 | 840 | 0.81 |
| M4 | 2,500 | 99.98% | 0.29 | 0.69 | valid, 49% recognized | 421 | 300 (cap) | 0.84 |
| M6 | 3,578 | 100% | 0.24 | 0.57 | invalid (79% YES) | 1,172 | 804 | 0.74 |
| M6H | 2,500 | 99.70% | 0.28 | 0.59 | invalid (94% YES) | 813 | 297 | 0.77 |

**Gate G2**
- Parse ≥ 99%: ✅ all models.
- Pools: ✅ by the plan's fallback (≥ 600 for M1, M3, M6 and L1). M2 has 418 claims and gets 12 cells per claim in S3.
- p̂ in [0.60, 0.85]: ✅ for 5 of 6 NIM models. M1 (0.86) and L1 (0.87) sit just above the band.
- TRUE share 0.50 ± 0.15: ❌ for M2, M4, M6 and M6H, the skeptical prior. **Proposal: report it as a model property, not a gate.** Pools are balanced by truth, and the analysis conditions on s and `p_true`.
- **G2 passes on the plan's own fallbacks, with the TRUE-share item awaiting the lead's call.** Prompts are frozen at v1 from here; any change goes to `v2.yaml` plus a re-run.

**Other notes**
- **Recognition.** Only M4 (thinking on) shows real recognition: 49% of claims, named on the correct side well above chance. Its pool excludes those claims.
- **Calibration.** ECE is 0.21–0.36, so stated `p_true` is poorly calibrated across the board. This motivates leaning on token probabilities (L1, L2) for the belief analyses.
- **Launched:** S3 main and fix cells on all six NIM models (keys 0–17, round robin), and the S2 re-run of M4 and M6H with a 16k-token budget (keys 18–23).

### 2026-10-06 · S2 complete (urn anchor, all six models)

- **Re-run of M4 and M6H** with a 16k-token budget: 1,200 decisions each, 100% parsed, zero truncations (longest replies 10,122 and 8,428 tokens; p50 1,981 and 937). Under the old caps, more than 10% of these calls were cut off, which had inflated their cascade rates slightly (0.96 → 0.92 for M4, 0.985 → 0.97 for M6H).
- **Final table** (`results/S2_table.csv`, 200 rounds × 6 agents per model):

| Model | Own draw at lead 0 | Cascade rate | Inconsistent | Reverse cascades | Accuracy |
|---|---|---|---|---|---|
| Humans (Anderson & Holt) | — | 0.71 | ~0.04 | — | — |
| Bayes | 1.00 | 1.00 | 0 | — | — |
| M1 Llama-3.2-11B | 0.64 ⚠ | 0.80 | 0.18 | 43 | 0.57 |
| M2 Nemotron-Super, thinking off | 0.97 | 0.18 | 0.02 | 5 | 0.68 |
| M4 Nemotron-Super, thinking on | 1.00 | 0.92 | 0.05 | 45 | 0.69 |
| M3 Nemotron-Lightning | 0.90 | 0.31 | 0.08 | 9 | 0.65 |
| M6 gpt-oss-20b, low effort | 0.97 | 0.67 | 0.06 | 19 | 0.69 |
| M6H gpt-oss-20b, medium effort | 0.98 | 0.97 | 0.06 | 54 | 0.68 |

- **Reading:**
  - **Reasoning switches the same weights from under-herding to near-Bayesian cascading:** 0.18 → 0.92 (Nemotron) and 0.67 → 0.97 (gpt-oss).
  - gpt-oss at low effort is the closest to people (0.67 vs 0.71).
  - Cascading more buys no accuracy here. Accuracy stays at 0.68–0.69, because the extra cascades include more reverse ones (5 → 45, 19 → 54), the classic cost of rational herding.
  - M1 fails the comprehension check (it follows its own draw only 64% of the time at lead 0), so its row carries a caveat.
- **Round robin, checked in the manifest:** the five live keys in the 18–23 slice got 482–483 requests each.
- **Fix:** the dead key (`0eff8f`) still received 42 requests that had been assigned before its first 403 came back. `Pool.post` now re-checks after the pacing wait and moves to another key without spending a retry.

### 2026-10-06 · S4: first crowd (M2) and the Bayesian-crowd benchmark

- **M2 crowd:** 750 sequences (300 clean, plus 150 each at k = 1, 2, 3), all complete. Solo verdicts are identical across k for 300 of 300 claims, which confirms the in-flight dedup fix. Key slice 18–23: 87 429s in about 17k calls (0.5%).
- **New in the summary** (`s4_crowds.summary`):
  - **Bayesian-crowd benchmark (H3).** The same solo signals (each agent's own verdict, with its model's p̂) face the same unlabelled bot votes, in the same order.
  - **Paired seeding effect.** k = 0 accuracy on the same 150 claims that were seeded.

| k | LLM crowd accuracy | Drop vs k = 0 (same claims) | Independent majority | Bayesian crowd | Wrong cascades |
|---|---|---|---|---|---|
| 0 | 0.77 | — | 0.84 | 0.82 | 18% |
| 1 | 0.61 | −0.19 | 0.87 | 0.75 | 31% |
| 2 | 0.50 | −0.30 | 0.87 | 0.00 | 45% |
| 3 | 0.49 | −0.31 | 0.87 | 0.00 | 58% |

- **Reading for H3:**
  - **Wisdom loss holds for M2:** visible votes cost 7 points with no bots, and one bot costs another 19.
  - **The "worse than Bayes" clause splits.** One bot hurts the LLM crowd more than a Bayesian crowd (0.61 vs 0.75). Two bots wipe out the Bayesian crowd entirely, because rational agents who trust the planted votes cascade on them (textbook BHW), while the LLM crowd half-resists.
  - So Bayes is a poor safety benchmark under manipulation. The paper should say "more manipulable than Bayes by one bot, less by two".
- **Running:** the M3 crowd on keys 18–23. The remaining crowds (M1, M4, M6, mixed) and the private-first crowd follow on all keys after S3.

### 2026-10-06 · S3 complete: H1/H2 results (7 models)

- **Run:** S3 main and fix cells for M1, M2, M3, M4, M6 and M6H on NIM, plus L1 on Kaggle. 62,186 cells, about 100% parsed (M6H 99.1–99.5%). M2's 418-claim pool got 12 cells per claim.
- **Two fixes:**
  - **Stale meta.** The 80 pilot cells per model were cache hits that kept the pilot pool's p̂ in their meta. `s3.table()` now recomputes every oracle answer from the current pool.
  - **Faster bootstrap.** It now resamples positional indices instead of concatenating frames, about 10× faster.
- **Results** (`results/S6_h1_h2.csv`; 95% CIs from 2,000 claim resamples):

| Model | ρ, q unstated (Bayes 1.0) | ρ, q = 70% (Bayes) | q\* (Bayes 0.70) | γ independent (Bayes 1) | γ sequential (Bayes 0) |
|---|---|---|---|---|---|
| L1 Llama-3.1-8B | 0.10 [0.06, 0.13] | 0.05 [0.01, 0.09] (0.45) | 0.55 [0.54, 0.57] | 0.06 | unstable (β_inf ≈ 0.15) |
| M1 Llama-3.2-11B | 0.05 [0.02, 0.08] | 0.05 [0.02, 0.08] (0.48) | 0.52 [0.51, 0.54] | 0.26 | unstable (β_inf ≈ 0.07) |
| M2 Nemotron-Super | 0.43 [0.38, 0.49] | 0.42 [0.36, 0.48] (0.49) | 0.62 [0.61, 0.63] | 0.05 [−0.06, 0.18] | 0.20 [0.08, 0.36] |
| M3 Nemotron-Lightning | 0.94 [0.82, 1.11] | 0.81 [0.70, 0.97] (0.58) | 0.65 [0.64, 0.66] | 0.09 [0.01, 0.18] | 0.23 [0.12, 0.34] |
| M4 Nemotron-Super, thinking on | 0.15 [0.08, 0.22] | 0.21 [0.15, 0.27] (0.51) | 0.57 [0.56, 0.59] | 0.24 [−0.01, 0.58] | 0.21 [−0.14, 0.79] |
| M6 gpt-oss-20b, low | 0.09 [0.04, 0.14] | 0.06 [0.01, 0.11] (0.79) | 0.55 [0.54, 0.56] | −0.19 [−0.38, 0.03] | −0.20 [−0.37, −0.00] |
| M6H gpt-oss-20b, medium | 0.10 [0.02, 0.17] | 0.06 [−0.03, 0.16] (0.71) | 0.54 [0.53, 0.56] | −0.25 [−0.49, 0.03] | −0.28 [−0.48, −0.03] |

- **Reading:**
  1. **H1 (over-herding) fails in the opposite direction for 5 of 7 models.** A shown vote carries 5–15% of the weight Bayes gives it, and the intervals exclude Bayes. Only the Nemotron models without thinking come close: M3 matches Bayes when q is unstated and over-weights when q = 70% is stated (0.81 vs 0.58), while M2 sits just under Bayes. The lead must decide how to report this against the pre-registered H1, whose refutation rule (the CI covers ρ_Bayes) assumed the deviation would be upward.
  2. **On claims, reasoning cuts herding.** Nemotron's ρ falls from 0.43 to 0.15 with thinking on, the opposite of the urn game, where thinking made cascades near-Bayesian. With real content to reason about, thinking makes a model trust its own reading.
  3. **Every model under-updates its belief.** q\* is 0.52–0.65 against the 0.70 stated; the models treat other raters as only slightly better than chance.
  4. **H2.** γ is near zero under both framings: models give little weight to votes past a lead of 2, whether those votes are independent (Bayes: count them fully) or sequential (Bayes: ignore them).
     - That is not correlation neglect. It is **saturation and framing insensitivity**: the independent–sequential gap is about 0, where Bayes says 1.
     - The pre-registered refutation (γ_seq ≤ 0.5) is met for M2, M3, M6 and M6H, but for the wrong reason. Report the framing contrast, not γ_seq alone.
  5. **Anchoring.** Every model follows random labels 18–31% of the time. M1, M6 and M6H shift their beliefs nearly as much per random label as per real rating; M2 and M3 tell them apart; L1's beliefs ignore random labels but its verdicts still flip.

### 2026-10-06 · Status at pause (usage limit)

- **Done:**
  - S0 and S1 for all 9 model codes (L2/L2T pools rebuilt from the Kaggle cache).
  - S2 for all six NIM models.
  - S3 main and fix cells for M1, M2, M3, M4, M6, M6H, L1, L2 and L2T. L2T has main cells only; the reasoning variants (M4, M6H, L2T) have no fix cells by design.
  - S5 complete.
  - S4 crowd M2 complete.
- **Still running in the background** (results land in the cache whatever happens):
  - S4 crowd M3 on keys 18–23.
  - S4 crowds M1, M4, M6 and mixed on keys 0–17.
- **Left:**
  1. Resume any S4 crowd that did not finish (`python -m src.s4_crowds --crowds M1 M3 M4 M6 mixed`; cached calls are free).
  2. S4 private-first crowd (`--f1`).
  3. Silent-herding judge (`python -m src.analyze judge`) and the 200-reason κ sheet for Pushkar and Teammate 3.
  4. `python -m src.analyze all` for the final tables and figures. L2 and L2T now enter the H1/H2 table automatically.
  5. The lead's decisions: the new roster, the G2 TRUE-share item, and the H1/H2 reporting (see the S3 entry).

### 2026-10-06 · S4 complete (6 crowds); large-model probe; next steps

- **S4, clean and seeded:** 4,039 of 4,041 sequences complete. Solo verdicts are identical across k for 1,566 of 1,566 claims. The mixed crowd has 216 clean claims, the number of claims that sit in at least 3 member pools.

| Crowd | Sequential, no bots | Independent majority | Accuracy change with 1 / 2 / 3 bots (same claims) | Bayesian crowd at k = 1 / ≥ 2 |
|---|---|---|---|---|
| M3 Nemotron-Lightning | 0.73 | 0.80 | −20 / −28 / −40 | 0.56 / 0.00 |
| M2 Nemotron-Super | 0.77 | 0.84 | −19 / −30 / −31 | 0.75 / 0.00 |
| mixed | 0.80 | 0.84 | −10 / −17 / −19 | 0.64 / 0.00 |
| M6 gpt-oss | 0.78 | 0.81 | −7 / −7 / −9 | 0.69 / 0.00 |
| M4 Nemotron, thinking on | 0.85 | 0.84 | +1 / −9 / −5 | 0.67 / 0.00 |
| M1 Llama-3.2-11B | 0.75 | 0.87 | +4 / +10 / +9 | 0.76 / 0.00 |

- **Reading:**
  - **The followers in S3 are the manipulable crowds in S4.** M2 and M3 lose about 20 points to one bot.
  - **The under-weighters resist bots.** M4, M6 and M1 barely move. M1 even gains: its answers move away from the planted votes, which in S4 are always wrong. Yet M1 still loses 12 points to visible votes with no bots, so its social prompt itself degrades its answers. Treat M1 with caution; it also failed the urn comprehension check.
  - **Every LLM crowd beats a Bayesian crowd at k ≥ 2**, which cascades on the planted votes and drops to 0.
- **Large models on NIM** (12 real claim prompts each, launched together; 5-minute cap per call):
  - DeepSeek-v4.1-flash, GLM-5.3 and GLM-5.3-flash, Kimi-K3 and Gemma-4-31B: 0/12. Llama-3.2-90B: 1/12. The errors were timeouts and 503s, which is free-tier capacity, not our code.
  - **Nemotron-3-Ultra-550B-A55B (thinking off): 30/30 with retries** (22 503s absorbed), median 5 s. It leans TRUE (21/30).
  - **Decision (lead):** finish the current analysis first, then decide on adding Ultra (≈ 53k calls; a 3B → 12B → 55B-active scale ladder within the Nemotron family) and on re-probing GLM, Kimi and DeepSeek.
- **Running:** the S4 private-first crowd (H6, keys 0–11) and the silent-herding judge (H5, keys 12–23). `analyze` now rebuilds the S3 table from every cache, so L2 and L2T enter the H1/H2 table.

### 2026-10-06 · H5 silent-herding judge

- **Run:** M6 (gpt-oss-20b, low effort, T = 0) judged 1,998 S3 flips toward the majority, stratified by model, across all 9 models (L2 and L2T included). 100% parsed. The judge sees only the reply's `reason` field, not the hidden reasoning.
- **Silent-herding rate**, the share of flips whose reason never mentions the other votes:

| L1 | L2 | L2T | M1 | M2 | M3 | M4 | M6 | M6H |
|---|---|---|---|---|---|---|---|---|
| 0.76 | 0.995 | 0.82 | 0.93 | 0.98 | 0.98 | 0.94 | 0.995 | 0.98 |

- **Reading.** H5 (more than 50% of flips are unacknowledged) holds for every model by a wide margin. Flips are justified with claim content ("As of 2011, Wisconsin governors were limited to a $125,000 salary…"), not with the crowd. Thinking variants acknowledge influence somewhat more often (L2T 0.82 vs L2 0.995; M4 0.94 vs M2 0.98).
- **Pending validation.** The judge counts only if Cohen's κ ≥ 0.7 against human labels. `results/judge_validation.csv` holds 200 reasons; the judge said NO to 181 and YES to 19. Pushkar and Teammate 3 fill in `label_pushkar` and `label_teammate3` independently, and κ is computed on all 200.

### 2026-10-06 · Final analysis pass (`analyze all`, 2,000 resamples, 9 models)

- **H1/H2 now include L2 and L2T** (`results/S6_h1_h2.csv`):
  - L2 (Qwen3-8B, thinking off): ρ = 0.42 [0.37, 0.47], q\* = 0.60.
  - L2T (thinking on): ρ = 0.20 [0.14, 0.28], q\* = 0.54.
  - **The reasoning effect replicates in a second family.** Thinking roughly halves the weight given to a shown vote on real claims: Qwen 0.42 → 0.20 and Nemotron 0.43 → 0.15. gpt-oss is already low at both effort levels (0.09 → 0.10).
- **H6 private-first crowd** (`results/S6_private_first.csv`; paired over the same claims, about 100 per crowd):

| Crowd | Independent | Sequential | Private-first | Gain [95% CI] | Share of loss recovered |
|---|---|---|---|---|---|
| M1 | 0.87 | 0.73 | 0.87 | +0.14 [0.07, 0.22] | 100% |
| M2 | 0.875 | 0.795 | 0.875 | +0.08 [0.02, 0.15] | 100% |
| M3 | 0.75 | 0.69 | 0.73 | +0.05 [−0.05, 0.14] | 69% |
| M6 | 0.845 | 0.805 | 0.835 | +0.03 [−0.04, 0.10] | 75% |
| mixed | 0.87 | 0.80 | 0.855 | +0.06 [−0.02, 0.13] | 79% |
| M4 | 0.84 | 0.85 | 0.85 | 0 | no loss to recover |

- **Reading for H6.** Private-first recovers 69–100% of the accuracy that visible votes cost, above the pre-registered 50% in every crowd that lost accuracy. It is significant for M1 and M2. With about 100 claims per crowd the other intervals are wide; extending the F1 crowd to 300 claims would firm them up (about 13k calls, solos cached).
- **Figures 1–5** were regenerated from the final tables. Fig. 1 is now a 2 × 5 grid (7.3 in, ACM full width), and Fig. 2's x-range is −0.1 to 1.3.
- **Hypothesis status:**
  - H1: fails in the opposite direction; models under-weight votes, and Nemotron without thinking is the near-Bayes exception.
  - H2: reframe as framing insensitivity.
  - H3: ✅ for the follower models; nuanced against Bayes.
  - H4: ✅ for Nemotron, mirror image for gpt-oss.
  - H5: ✅ pending κ.
  - H6: ✅, private-first works.

### 2026-10-06 · Runs stopped (lead's decision)

- **No new runs for now;** the team moves to writing and human labeling. Nothing is left running: every NIM job and both Kaggle kernels have finished.
- **Deferred, can resume any time:**
  - Nemotron-3-Ultra-550B as M7 (≈ 53k calls; it works with retries).
  - A re-probe of GLM, Kimi and DeepSeek (0/12 answered on Oct 6).
  - The H6 private-first crowd extended to 300 claims (≈ 13k calls).
  - To resume, add the model line in `config.yaml` and re-run the stage commands. Cached calls are free.
- **Human tasks:**
  1. Pushkar and Teammate 3 label `results/judge_validation.csv` independently; κ ≥ 0.7 is needed to keep the judge.
  2. Reconcile the draft subreddit map, and review the Bayes-rule prompt.
  3. The lead's calls on the roster, the G2 TRUE-share item, and the H1/H2 reporting.

### 2026-10-07 · Ownership, §12 tasks and decisions, blind κ sheet, exhaustive oracle check

- **Ownership (lead's note).** Pushkar does all the work in this plan. §6 and the stage cards now say so, and the two-person checks are replaced:
  - **Oracle.** `python -m src.oracle` now also enumerates every private-signal sequence of Bayesian predecessors, for 4 accuracy settings and histories up to 6 votes, and compares the oracle's vote and P(TRUE) with the exact posterior. The check does not use `informative_lead()`. All 464 cases pass, and `bhw_sequences(n)` equals the set of histories the predecessors can produce.
  - **κ sheet.** The old `judge_validation.csv` showed the judge's answer next to the label columns, which would anchor the labeller and inflate κ.
    - It is replaced by a blind `judge_labels.csv` (reason only, random order, `label_pushkar`, optional `label_second`) and a separate `judge_key.csv`.
    - The sheet has the same 200 rows in the same order, regenerated from the cache with 0 new calls. The judge run's manifest was restored byte for byte.
    - A sequencing slip removed the old sheet before the new one was written. Nothing was lost, because the sample is deterministic and every answer is cached.
    - `python -m src.analyze kappa` computes the gate. On simulated labels it matches statsmodels (0.801 = 0.801).
- **§12 rewritten** as human tasks T1–T3, with step-by-step instructions, and decisions D1–D8, with recommendations and a column for the lead's calls.
- **Gap found while writing it:** the pooled, Holm-corrected tests (S6 row 9) were never computed; `holm()` is defined but unused. Two pooled numbers were computed for the register:
  - H3 wisdom loss: 6.0 pp [4.1, 7.8] over 1,565 crowds.
  - H6 private-first gain: +6.4 pp [3.1, 9.7] over 549 paired claims.
- **Commands.** §8.3 and the README now list the real stage commands, including the S4 `--f1` crowd, M4's crowd, the oracle check, `judge` and `kappa`.

### 2026-10-08 · Lead's decisions; large-model re-probe; Ultra (M7) and the Bayes-rule crowd started

- **Decisions (lead).** D1 roster approved, no paper paragraph on the swap. D2 TRUE share reported as a finding. D3 hypotheses reported in the findings, results and analysis sections. D4 headline later. D5 items 1–3 run, item 4 skipped. D6 the lead labels alone. D7 contamination check not done. D8 anonymized release at the end.
- **Re-probe (D5.1).** 12 real claims per candidate, 108 calls, Oct 8:

| Model | Answered | Parsed | Median latency |
|---|---|---|---|
| Nemotron-3-Ultra-550B, thinking off | 12/12 | 12/12 | 10 s |
| GLM-5.3, thinking on | 12/12 | 8/12 | 22 s |
| GLM-5.3, thinking off | 12/12 | 5/12 | 11 s |
| GLM-5.3-flash, thinking off | 10/12 | 8/12 | 165 s |
| Gemma-4-31B | 12/12 | 12/12 | 103 s |
| Llama-3.2-90B | 5/12 | 5/12 | 127 s |
| Kimi-K3 | 0/12 | 0/12 | BadRequest and timeouts |
| DeepSeek-v4.1-flash, either setting | 0/12 | 0/12 | timeouts |

  The roster is unchanged by this probe. GLM-5.3 answers but fails to parse about a third of its replies, and Gemma-4-31B is slow. Key pool: 23 of 24 keys live (key #23 returns 403, as before).
- **Ultra added as M7** (D5.3). `nvidia/nemotron-3-ultra-550b-a55b`, thinking off, 512-token cap (as probed). It uses the M4 design: 2,500 claims screened, 300-claim pool.
  - **Pilot** (100 claims, 600 calls): 100% parsed, 53% TRUE, solo accuracy 0.82 on those claims, and the recognition probe is valid (70% YES). **It recognises 59% of the claims**, the highest of any model (M4 49%; every other model 0–3%). Its pool keeps only unrecognised claims, about 14% of the screened set.
  - **Cost correction.** On Oct 8 I estimated about 17k calls for this scope. The real count is about 36k: S1 screening 15.3k (2,500 claims × 6 calls), S2 1.2k, S3 about 5.7k, S4 about 13.4k.
- **Bayes-rule crowd** (D5.2). New S4 mode `--br`: the same claims and k as the social crowd, with the Bayes-rule paragraph in the system prompt. Solo calls come from cache, so only the social calls run. M2 and M3 at k = 0 (300 claims) and k = 2 (150 claims): about 10.2k calls, not the 13k I estimated. Paired results go to `results/S6_bayes_rule_crowds.csv` through `analyze all`.
- **Code.** `s4_crowds.paired()` replaces `private_first()` and handles both the F1 and BR comparisons. The `S6_private_first.csv` columns are renamed (`acc_*`); the H6 numbers should not change. Check after regeneration.
- **Running at entry:** Ultra S1 screen on keys 0–11 (about 1 h at 240 calls per minute); Bayes-rule crowd on keys 12–23. Next, once the pool is built: S2, S3 and S4 for M7, S4 F1 for M7, then `analyze all`.
- **Ultra screen interrupted and restarted** (2026-10-08, 15:45 IST). The first run omitted `--refill`, so it screened only the 1,500 eval claims (9,000 calls, not 15,000). At 3,000 of 9,000 it hit a rate-limit storm (5,562 429s, 529 errors, parse rate 82%). I stopped it; the cache kept 2,624 completed calls. The restart uses `--refill 1000` (2,500 screened, as for M4), `NIM_RPM=15` per key and keys 0–11. Failed calls are not cached, so they are retried in the restart.

- **Bayes-rule crowd, M2 and M3, done** (2026-10-08, 15:51 IST). Requests: 10,500 (M2) and 10,480 (M3); 100% of M2 replies and 99.95% of M3 replies parsed. The 5 unparsed M3 replies left 2 M3 chains incomplete at k = 0 and 1 at k = 2; the paired comparison drops them. Paired with the social crowd on the same claims and k (`s4_crowds.paired()`, 2,000 resamples; `analyze all` will regenerate these):

| Crowd | k | Claims | Social accuracy | Bayes-rule accuracy | Gain [95% CI] | Share of the visible-vote loss recovered |
|---|---|---|---|---|---|---|
| M2 | 0 | 300 | 0.773 | 0.787 | +0.013 [−0.017, +0.045] | 19% |
| M2 | 2 | 150 | 0.503 | 0.393 | −0.110 [−0.190, −0.037] | — |
| M3 | 0 | 297 | 0.729 | 0.714 | −0.015 [−0.061, +0.029] | −21% |
| M3 | 2 | 148 | 0.416 | 0.132 | −0.284 [−0.361, −0.203] | — |

- **Reading.** The Bayes-rule prompt does not help crowds. Without bots it changes nothing; under two planted votes it makes the crowd follow the bots, and M3's accuracy falls to 13%. Private-first, at k = 0 on the same models, gains +8 pp (M2) and +4.5 pp (M3). This supports H6 (the cheap fix that works is private-first) and reverses the planned "Bayes-rule baseline" comparison: the baseline does not beat private-first, and it is more manipulable.
- **H6 check.** The refactored `paired()` reproduces every `S6_private_first.csv` gain and CI exactly (M1–M6 and mixed), so the earlier H6 numbers stand.
