
# Stage 1 — basic lexical matcher

# Pure-Python implementation: normalised exact match (high-confidence anchors) plus Jaro-Winkler similarity for near-matches. Per entity type only —
# classes match classes, properties match properties, individuals match
# individuals.
# https://www.geeksforgeeks.org/dsa/jaro-and-jaro-winkler-similarity/

from __future__ import annotations

from collections import defaultdict

from .types import CandidateSet, Entity

try:
    from rapidfuzz.distance import JaroWinkler
    JW_normalized = JaroWinkler.normalized_similarity
except ImportError:  # pragma: no cover
    # Fallback if rapidfuzz isn't available — we ship our own naive version
    def JW_normalized(a: str, b: str) -> float:
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        # Crude fallback: longest common prefix ratio. Install rapidfuzz
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i / max(len(a), len(b))


def lexical_match(
    src: list[Entity],
    tgt: list[Entity],
    near_match_threshold: float = 0.9,
) -> CandidateSet:
# Generate (uri1, uri2) candidates with lexical scores.

    candidates: CandidateSet = {}

    # Bucket targets by kind so we only ever compare same-kind entities
    tgt_by_kind: dict[str, list[Entity]] = defaultdict(list)
    for t in tgt:
        tgt_by_kind[t.kind].append(t)

    # Build a label
    exact_index: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    for t in tgt:
        for lbl in t.all_labels():
            exact_index[(t.kind, lbl)].append(t)

    for s in src:
        # Exact matches first — cheap and very high precision
        exact_hits: set[str] = set()
        for lbl in s.all_labels():
            for t in exact_index.get((s.kind, lbl), []):
                key = (s.iri, t.iri)
                # Keep the highest score for the pair
                existing = candidates.get(key, {}).get("lexical", 0.0)
                if 1.0 > existing:
                    candidates.setdefault(key, {})["lexical"] = 1.0
                exact_hits.add(t.iri)

        # Near-matches against the rest of the same-kind pool
        s_labels = s.all_labels()
        if not s_labels:
            continue
        for t in tgt_by_kind.get(s.kind, []):
            if t.iri in exact_hits:
                continue  # already scored 1.0
            t_labels = t.all_labels()
            if not t_labels:
                continue
            # Best label-pair similarity
            best = 0.0
            for sl in s_labels:
                for tl in t_labels:
                    sim = JW_normalized(sl, tl)
                    if sim > best:
                        best = sim
                        if best >= 0.999:
                            break
                if best >= 0.999:
                    break
            if best >= near_match_threshold:
                key = (s.iri, t.iri)
                existing = candidates.get(key, {}).get("lexical", 0.0)
                if best > existing:
                    candidates.setdefault(key, {})["lexical"] = best

    return candidates
