"""Bayes references: the BHW informative lead, the Bayes vote and posterior, and the urn agents.

Votes, signals and draws are +1 (TRUE / urn A / light ball) or -1 (FALSE / urn B / dark ball).

    python -m src.oracle        runs the hand-worked checks below
"""
from __future__ import annotations

import itertools
import math

URN_P = 2 / 3     # Anderson & Holt signal accuracy


def logodds(p: float) -> float:
    return math.log(p / (1 - p))


def bhw_sequences(n: int) -> list[tuple[int, ...]]:
    """Vote sequences Bayesian predecessors can produce: before a cascade any vote is possible;
    once |lead| >= 2 everyone copies the lead."""
    out = []
    for seq in itertools.product((1, -1), repeat=n):
        lead = 0
        for a in seq:
            if abs(lead) >= 2:
                if a != (1 if lead > 0 else -1):
                    break
            else:
                lead += a
        else:
            out.append(seq)
    return out


def informative_lead(seq) -> int:
    """Before a cascade a vote reveals its voter's signal; after |lead| >= 2 votes add nothing."""
    lead = 0
    for a in seq:
        if abs(lead) < 2:
            lead += a
    return lead


def evidence(seq, framing: str) -> int:
    """Net number of informative votes for TRUE, by how the votes were described to the agent."""
    if framing == "random":
        return 0
    if framing == "independent":
        return sum(seq)
    return informative_lead(seq)             # sequential, cue


def oracle(seq, s: int, p_self: float, q: float, framing: str) -> tuple[int, float]:
    """Bayes vote and P(TRUE) for an agent with own signal s of accuracy p_self; prior 1/2.
    Ties go to the own signal."""
    lam = evidence(seq, framing) * logodds(q) + s * logodds(p_self)
    vote = s if lam == 0 else (1 if lam > 0 else -1)
    return vote, 1 / (1 + math.exp(-lam))


def urn_bayes(draw: int, guesses) -> tuple[int, float]:
    """Bayes guess and P(urn A) for an urn agent who saw earlier guesses (Bayesian predecessors)."""
    return oracle(guesses, draw, URN_P, URN_P, "sequential")


if __name__ == "__main__":
    # Hand-worked cases (two-person rule: Teammate 3 adds 20 more before G2).
    assert len(bhw_sequences(3)) == 6                     # (1,1,-1) and (-1,-1,1) are impossible
    assert informative_lead([1, 1, 1, 1, 1]) == 2         # cascade after two votes
    assert informative_lead([1, -1, 1, 1, -1]) == 2       # lead hits 2 at vote 4; vote 5 adds nothing
    assert informative_lead([1, -1, -1, -1, -1]) == -2
    assert informative_lead([1, -1, 1, -1]) == 0
    q = p = 0.7
    assert oracle([1], -1, p, q, "independent")[0] == -1  # one vote vs own signal: tie -> own signal
    assert oracle([1, 1], -1, p, q, "independent")[0] == 1
    assert oracle([1, 1, 1, 1, 1], -1, p, q, "independent")[1] > oracle([1, 1, 1, 1, 1], -1, p, q, "sequential")[1]
    assert oracle([1, 1, 1, 1, 1], -1, p, q, "sequential")[0] == 1   # informative lead 2 beats one signal
    assert oracle([1, 1, 1, 1, 1], -1, p, q, "random")[0] == -1      # random labels count for nothing
    assert abs(oracle([1, 1], -1, p, q, "independent")[1] - 0.7) < 1e-9
    assert urn_bayes(-1, [1, 1])[0] == 1 and urn_bayes(-1, [1])[0] == -1 and urn_bayes(1, [])[0] == 1
    print("oracle checks pass")
