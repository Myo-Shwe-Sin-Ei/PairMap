"""
Evaluation — Precision, Recall, F-score against a reference alignment in turtle format.

We extract the set of (subject, object) pairs from both files and compare.
This is a deliberately minimal implementation that matches the spec's
metric definitions exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDFS


# Predicates that count as a mapping in the reference alignment
MAPPING_PREDICATES = {
    OWL.equivalentClass,
    OWL.equivalentProperty,
    OWL.sameAs,
    RDFS.subClassOf,
    RDFS.subPropertyOf,
}

SYMMETRIC_PREDICATES = {
    OWL.equivalentClass,
    OWL.equivalentProperty,
    OWL.sameAs,
}


@dataclass
class EvalResult:
    precision: float
    recall: float
    fscore: float
    system_count: int
    reference_count: int
    intersection_count: int
    elapsed_sec: float = 0.0

    def __str__(self) -> str:
        return (
            f"Pr={self.precision:.4f}  Re={self.recall:.4f}  F={self.fscore:.4f}  "
            f"|M_S|={self.system_count}  |M_RA|={self.reference_count}  "
            f"|M_S cap M_RA|={self.intersection_count}"
        )


def _load_pairs(turtle_path: str | Path) -> set[tuple[URIRef, URIRef]]:
    # Extract (subject, object) pairs for every mapping triple
    graph = Graph()
    graph.parse(str(turtle_path), format="turtle")
    pairs: set[tuple[URIRef, URIRef]] = set()
    for s, p, o in graph:
        if p in MAPPING_PREDICATES and isinstance(s, URIRef) and isinstance(o, URIRef):
            if p in SYMMETRIC_PREDICATES and str(s) > str(o):
                pairs.add((o, s))
            else:
                pairs.add((s, o))
    return pairs


def evaluate(
    system_path: str | Path,
    reference_path: str | Path,
    elapsed_sec: float = 0.0,
) -> EvalResult:
    # Compute Pr, Re, F against the reference alignment
    
    system_pairs = _load_pairs(system_path)
    reference_pairs = _load_pairs(reference_path)

    intersection = system_pairs & reference_pairs
    intersection_count = len(intersection)

    if not system_pairs:
        precision = 0.0
    else:
        precision = intersection_count / len(system_pairs)

    if not reference_pairs:
        recall = 0.0
    else:
        recall = intersection_count / len(reference_pairs)

    if precision + recall == 0.0:
        fscore = 0.0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return EvalResult(
        precision=precision,
        recall=recall,
        fscore=fscore,
        system_count=len(system_pairs),
        reference_count=len(reference_pairs),
        intersection_count=intersection_count,
        elapsed_sec=elapsed_sec,
    )
