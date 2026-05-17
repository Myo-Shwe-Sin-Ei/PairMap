#Stage 5 — combine the per-source scores into a single confidence, apply
#the threshold, and resolve 1:1 conflicts.

#The combination formula:
#final = lex_weight * lexical + (1 - lex_weight) * embedding + locality
# where `locality` is already pre-scaled by alpha in stage4 
# The final value is clamped to [0, 1]

from __future__ import annotations

from .config import TrackConfig
from .types import CandidateSet, Entity, Mapping


def combine_scores(
    candidates: CandidateSet, config: TrackConfig
) -> dict[tuple[str, str], float]:
    #Reduce multi-source scores to a single confidence

    #Lexical and embedding scores are weighted, but missing scores don't
    #pull the average down - we renormalise over the sources that are
    #actually present. Locality is a pure additive boost

    out: dict[tuple[str, str], float] = {}
    w_lex = config.lex_weight
    w_emb = config.embed_weight
    for key, scores in candidates.items():
        has_lex = "lexical" in scores
        has_emb = "embedding" in scores
        if has_lex and has_emb:
            base = w_lex * scores["lexical"] + w_emb * scores["embedding"]
        elif has_lex:
            base = scores["lexical"]
        elif has_emb:
            base = scores["embedding"]
        else:
            base = 0.0
        boost = scores.get("locality", 0.0)
        out[key] = max(0.0, min(1.0, base + boost))
    return out


def resolve_conflicts(
    scores: dict[tuple[str, str], float], threshold: float
) -> dict[tuple[str, str], float]:
    #Apply threshold and enforce 1:1 matching (each entity used at most once)

    filtered = [(k, v) for k, v in scores.items() if v >= threshold]
    filtered.sort(key=lambda kv: kv[1], reverse=True)

    used_src: set[str] = set()
    used_tgt: set[str] = set()
    kept: dict[tuple[str, str], float] = {}
    for (s_iri, t_iri), score in filtered:
        if s_iri in used_src or t_iri in used_tgt:
            continue
        used_src.add(s_iri)
        used_tgt.add(t_iri)
        kept[(s_iri, t_iri)] = score
    return kept


def to_mappings(
    final_scores: dict[tuple[str, str], float],
    src_entities: list[Entity],
    tgt_entities: list[Entity],
    candidates: CandidateSet,
) -> list[Mapping]:
    #Materialise the surviving pairs as Mapping objects
    src_by_iri = {e.iri: e for e in src_entities}
    tgt_by_iri = {e.iri: e for e in tgt_entities}

    mappings: list[Mapping] = []
    for (s_iri, t_iri), conf in final_scores.items():
        s = src_by_iri.get(s_iri)
        t = tgt_by_iri.get(t_iri)
        if s is None or t is None:
            continue
        sources = set(candidates.get((s_iri, t_iri), {}).keys())
        mappings.append(
            Mapping(
                e1=s,
                e2=t,
                confidence=conf,
                relation="equivalent",
                sources=sources,
            )
        )
    return mappings
