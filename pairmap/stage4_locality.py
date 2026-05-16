#Stage 4 — locality booster (LogMap-inspired, one-shot).
#LogMap's iterative discovery scores a candidate (C1, C2) by the proportion
#of C1's neighbours that map to C2's neighbours. We implement a single-pass
#variant: count neighbour agreement once, use it as a confidence boost.

#This is simpler version  of LogMap's iterative version 
#but captures the same "principle of locality"
#Citation: Jiménez-Ruiz & Cuenca Grau, ISWC 2011 (locality principle).

from __future__ import annotations

from collections import defaultdict

from .types import CandidateSet, Entity


def locality_boost(
    candidates: CandidateSet,
    src_entities: list[Entity],
    tgt_entities: list[Entity],
    alpha: float = 0.15,
) -> CandidateSet:
    #Add a 'locality' score to each candidate based on neighbour agreement.
    #For pair (s, t), the boost is alpha * agreement, where agreement is the
    #fraction of s's parents/children that pair with one of t's parents/children
    #in the candidate set

    if alpha <= 0.0 or not candidates:
        return candidates

    src_by_iri = {e.iri: e for e in src_entities}
    tgt_by_iri = {e.iri: e for e in tgt_entities}

    # Quick lookup: src_iri -> set of tgt_iris paired with it
    pair_lookup: dict[str, set[str]] = defaultdict(set)
    for (s_iri, t_iri) in candidates:
        pair_lookup[s_iri].add(t_iri)

    for (s_iri, t_iri), scores in candidates.items():
        s = src_by_iri.get(s_iri)
        t = tgt_by_iri.get(t_iri)
        if s is None or t is None:
            continue

        s_nbrs = set(s.parents) | set(s.children)
        t_nbrs = set(t.parents) | set(t.children)
        if not s_nbrs:
            continue

        matched = 0
        for s_nbr in s_nbrs:
            paired_targets = pair_lookup.get(s_nbr, ())
            if any(pt in t_nbrs for pt in paired_targets):
                matched += 1

        agreement = matched / len(s_nbrs)
        scores["locality"] = alpha * agreement

    return candidates
