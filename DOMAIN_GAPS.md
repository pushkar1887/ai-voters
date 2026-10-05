# Gaps and Problems in the Domain

**LLM agents as participants in crowd judgments on the web** · companion to `EXPERIMENT_PLAN.md` (v2) · Oct 5, 2026

> **Scope.** Settings where an LLM agent sees other participants' votes, ratings or verdicts before giving its own: crowd fact-checking, up/down voting, product ratings, multi-agent deliberation, and LLM-based simulations of such platforms.
>
> **Tags.** **[V]** we opened the source and checked the claim · **[I]** our inference from the literature · **[C]** check before the Oct 7 freeze.

---

## The problem in one paragraph

*Why this matters, before any detail.*

LLM agents are moving from describing crowds to sitting inside them. Since July 2025, X has piloted AI chatbots that write Community Notes, vetted by the same process as human notes [V]. AI shopping agents already respond to ratings, positions and endorsements, and concentrate demand on a few products (Allouah et al., WWW 2026) [V]. A crowd signal (a vote count, an average rating, a verdict tally) is only worth something if participants add information rather than copy it. Today's research measures whether LLMs *change their answers* when others disagree. It does not measure whether they weigh others' votes *correctly*, so it cannot tell rational social learning from harmful herding. That is the central gap.

---

## 1. What already exists

*The field in seven strands. The last column is what each strand cannot answer.*

| Strand | Representative work | What it measures | What it cannot answer |
|---|---|---|---|
| LLM conformity and peer pressure | BenchForm (Weng, Chen & Wang, ICLR 2025 oral) [V]; Hu & Qu 2026 [V]; "LLMs Can't Handle Peer Pressure" (arXiv 2508.18321) [C] | How often a model changes its answer toward a group | Whether that change was right or wrong for the evidence |
| LLM-agent herding and social learning | Jain & Krishnamurthy, arXiv 2411.01271 [V]; Arghal et al., ICLR 2026 [C] | Sequential LLM agents herd on hate-speech and product-quality text; control can delay herding | How much herding is rational, and how it compares with people |
| LLM-agent networks | Liu, Xiong, Zhang & Tang 2026 [V] | Herding versus wisdom of crowds as attention width varies, with Katz–Bonacich theory | Behaviour on real web content and real platform signals |
| Validity of LLM social simulation | de Arruda et al. 2026 [V]; Li & Ji 2026 [V]; *The Illusion of Intervention* 2026 [V] | Whether simulations reproduce aggregate outcomes, treatment effects and a stable simulated population | Whether agents process *social signals* correctly, the mechanism behind cascades |
| Correlated errors across LLMs | Denisov-Blanch et al. 2026 [V] | Errors are strongly correlated across models; polling many models amplifies shared misconceptions | What happens when agents also *see* each other's votes |
| AI participants on real platforms | X AI note writers (TechCrunch, Jul 1, 2025) [V]; Allouah et al., WWW 2026 [V] | Deployment, and agent sensitivity to platform signals | Feedback loops when AI participants read signals that other AIs produced |
| Classic human evidence | Bikhchandani, Hirshleifer & Welch 1992; Anderson & Holt 1997 [V]; Weizsäcker 2010 [V]; Muchnik, Aral & Taylor 2013 [V]; Salganik & Watts 2008 [V]; Enke & Zimmermann 2019 [V] | Rational and actual human herding, manipulation by fake popularity, correlation neglect | Nothing missing on the human side; these references are simply unused for LLM agents so far [I] |

---

## 2. The gaps

*Ten gaps, ordered by how much they limit what the field can conclude. Each states what is missing, the evidence, and what goes wrong if it stays open.*

### G1 · "Conformity" has no normative reference

- **Missing.** Benchmarks count answer changes toward a group. BenchForm reports conformity and independence rates [V].
- **Why that is not enough.** In sequential social learning, following the crowd is often the *correct* move. Under BHW, a rational agent ignores its own signal once the informative lead reaches two. In the lab, people did exactly that in 87 of 122 possible periods, 71% (Anderson & Holt 1997) [V].
- **What goes wrong.** A model that never updates scores as "independent"; a model that updates exactly as Bayes prescribes scores as "conformist". Mitigation then optimizes the wrong target: BenchForm proposes enhanced personas and a reflection mechanism to *reduce* conformity [V], without asking whether accuracy rises or falls.

### G2 · Social influence is confounded with repetition and anchoring

- **Evidence.** When a wrong answer is merely repeated with *no speaker attached*, models make harmful revisions in 66.5% of initially correct cases, against 10.3% under plain re-asking (Hu & Qu 2026) [V].
- **What goes wrong.** Much measured "peer pressure" may not be social at all. Any herding study without a non-social control overstates social influence.

### G3 · LLM herding is shown, but never scored against theory or people

- **Evidence.** Jain & Krishnamurthy show LLM agents form information cascades on real text and propose stochastic control to delay herding. They compare against no BHW benchmark and no human data [V].
- **What goes wrong.** We know LLM agents herd. We do not know whether they herd *too much, too little, or at the wrong moment*, which is the only form of the question a platform designer can act on.

### G4 · Correlated evidence is treated as independent

- **Evidence.** LLM errors are strongly correlated across models, even when models are asked for pseudo-random outputs (Denisov-Blanch et al. 2026) [V]. A rational voter should discount votes from voters who saw each other; people often fail to (*correlation neglect*, Enke & Zimmermann 2019) [V].
- **Missing.** No study we found tests whether LLMs discount sequential (correlated) votes relative to independent ones [I].
- **What goes wrong.** AI crowds can count the same evidence many times over. That is the mechanism that locks a cascade onto a wrong verdict.

### G5 · Micro and macro validity are checked separately, if at all

- **Evidence.** Nine open-weight LLMs reproduce macro cooperation in a networked prisoner's dilemma while underestimating individual heterogeneity (de Arruda et al. 2026) [V]. Across 59,508 participants in 62 countries, how realistic simulated responses look correlates only weakly with how accurately they recover treatment effects (Li & Ji 2026) [V]. Prompt interventions can shift the simulated population itself, "user drift" (*The Illusion of Intervention*) [V].
- **What goes wrong.** A simulated crowd can look right for the wrong reasons. Platform-policy conclusions drawn from such simulations are unreliable unless the decision mechanism is checked too.

### G6 · Nobody has measured how AI crowds respond to manipulated signals

- **Human evidence.** One random upvote raised final ratings by 25%, while random downvotes were corrected by later voters (Muchnik et al. 2013) [V]. Inverted popularity rankings were partly self-fulfilling (Salganik & Watts 2008) [V].
- **AI evidence.** Shopping agents are sensitive to ratings, positions and endorsements (Allouah et al.) [V]. Seeded-vote cascades among AI participants are unmeasured [I].
- **What goes wrong.** Once AI participants join crowd systems, bot seeding could steer outcomes more cheaply than with people, or less. Platforms cannot set policy on an unmeasured risk.

### G7 · Agents may not disclose what moved them

- **Evidence.** Chain-of-thought explanations can omit the features that actually drove an answer (Turpin et al., NeurIPS 2023) [V].
- **Missing.** No measurement of whether LLM raters *say* that earlier votes influenced them [I].
- **What goes wrong.** Where AI raters publish reasons (notes, reviews), undisclosed herding misleads readers and auditors.

### G8 · Fixes are evaluated without a target

- **Evidence.** Mitigations push toward independence, such as BenchForm's personas and reflection [V]. Under BHW, both over- and under-weighting others costs accuracy. People earn only 53% of possible payoffs in situations where following others is optimal, barely above random, because they wait for about 2:1 odds instead of 1:1 (Weizsäcker 2010, 13 experiments) [V].
- **What goes wrong.** A "fix" that halves conformity can lower accuracy. Without an oracle, nobody notices.

### G9 · Claim benchmarks leak their labels

- **Evidence.** LIAR and LIAR2 ship each speaker's rating history (`*_counts` columns), and LIAR2 adds the fact-checker's `justification` [V]. Both datasets are old and widely used, so memorization is likely [I].
- **What goes wrong.** Studies that keep these fields, or skip a memorization check, measure retrieval rather than judgment and inflate the private signal.

### G10 · Precise human reference data sits unused

- **Evidence.** Lab cascade studies give exact numbers: 87 of 122 possible cascades formed, and about 4% of decisions contradicted both Bayes and the subject's own signal (Anderson & Holt 1997) [V]. The 2:1 threshold and the 53% payoff share come from 13 experiments (Weizsäcker 2010) [V].
- **Missing.** We found no LLM study that uses these numbers as a reference [I]. Re-check before the freeze [C].

---

## 3. Why these gaps matter now

*Four pressures that make this a 2026–27 problem rather than a hypothetical one.*

1. **Platforms.** Ranking, moderation and fact-check visibility run on crowd signals, and AI participants are already in pilots [V].
2. **The agentic web.** Agents read the ratings and votes that other agents produced, so errors can compound through the signal itself [I].
3. **Science.** LLM simulations are used to evaluate platform interventions; G5 shows that is unsafe without mechanism checks [V].
4. **Security.** Cheap seeding attacks on AI crowds are untested (G6) [I].

---

## 4. Open problems this project does not take on

*Stated so reviewers can see the boundary was drawn on purpose.*

- **Normative influence.** Deutsch & Gerard (1955) separate informational influence (others' choices as evidence) from normative influence (wanting approval) [V]. Preference-based social comparison, as in reference-dependent "keeping up with the Joneses" utility (Langtry 2023) [V], is a different channel. We study informational influence only.
- **Network topology** beyond sequential chains. Liu et al. (2026) cover part of it [V].
- **Non-English and non-US claims.**
- **Deployment logs** from real AI participants, which are not public.

---

## Sources opened

- Weng, Chen & Wang (ICLR 2025), *Do as We Do, Not as You Think: the Conformity of Large Language Models* — https://arxiv.org/abs/2501.13381
- Hu & Qu (2026), *Most LLM Conformity Needs No Speaker* — https://arxiv.org/abs/2607.05545
- Jain & Krishnamurthy, *Interacting Large Language Model Agents. Interpretable Models and Social Learning* — https://arxiv.org/html/2411.01271v1
- Liu, Xiong, Zhang & Tang (2026), *Social Networks of LLM Agents* — https://arxiv.org/pdf/2607.03695
- de Arruda, Gracia Lázaro, Aleta & Moreno (2026), *Collective cooperation without individual fidelity in LLM agents* — https://arxiv.org/abs/2606.30454
- Li & Ji (2026), *Statistical realism is not evidence that LLMs can estimate treatment effects in social science experiments* — https://arxiv.org/html/2604.02458
- *The Illusion of Intervention: Your LLM-Simulated Experiment is an Observational Study* (2026) — https://arxiv.org/abs/2605.20767
- Denisov-Blanch et al. (2026), *Consensus is Not Verification* — https://arxiv.org/abs/2603.06612
- Allouah, Besbes, Figueroa, Kanoria & Kumar, *What Is Your AI Agent Buying?* — https://arxiv.org/abs/2508.02630 (WWW 2026 proceedings: https://dl.acm.org/doi/proceedings/10.1145/3774904)
- TechCrunch (Jul 1, 2025), X pilots AI chatbots writing Community Notes — https://techcrunch.com/2025/07/01/x-is-piloting-a-program-that-lets-ai-chatbots-generate-community-notes/
- Turpin, Michael, Perez & Bowman (NeurIPS 2023), *Language Models Don't Always Say What They Think* — https://proceedings.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf
- Anderson & Holt (1997), *Information Cascades in the Laboratory* — https://econweb.ucsd.edu/~jandreon/Econ264/papers/Holt%20Anderson%20AER%201997.pdf
- Weizsäcker (2010), *Do We Follow Others When We Should?* — https://ideas.repec.org/p/ehl/lserod/4945.html
- Muchnik, Aral & Taylor (2013), *Social Influence Bias: A Randomized Experiment* — https://snap.stanford.edu/class/cs224w-readings/muchnik13bias.pdf
- Salganik & Watts (2008), *Leading the Herd Astray* (summary) — https://acawiki.org/Leading_the_Herd_Astray:_An_Experimental_Study_of_Self-fulfilling_Prophecies_in_an_Artificial_Cultural_Market
- Enke & Zimmermann (2019), *Correlation Neglect in Belief Formation* — https://academic.oup.com/restud/article/86/1/313/4772809
- Deutsch & Gerard (1955), *A Study of Normative and Informational Social Influences upon Individual Judgment* — https://www.scirp.org/reference/referencespapers?referenceid=3958372
- Langtry (2023), *Keeping Up with "The Joneses": Reference-Dependent Choice with Social Comparisons* — https://ideas.repec.org/a/aea/aejmic/v15y2023i3p474-500.html
- LIAR (Wang, ACL 2017) — https://sites.cs.ucsb.edu/~william/papers/acl2017.pdf · LIAR2 — https://github.com/chengxuphd/liar2
