#Stage 2 - scalability via inverted-index blocking.

#Builds a token -> {target entities} index over O2, then for each O1 entity
#returns only the targets sharing at least one token.
#Inspired by LogMap by (Jiménez-Ruiz & Cuenca Grau, ISWC 2011).

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .normalise import tokenise
from .types import Entity


class InvertedIndex:
    #token -> set of (kind, iri) tuples
    def __init__(self) -> None:
        self._index: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self._entity_by_iri: dict[str, Entity] = {}

    def add(self, entity: Entity) -> None:
        self._entity_by_iri[entity.iri] = entity
        for label in entity.all_labels():
            for tok in tokenise(label):
                self._index[tok].add((entity.kind, entity.iri))

    @classmethod
    def build(cls, entities: Iterable[Entity]) -> "InvertedIndex":
        idx = cls()
        for e in entities:
            idx.add(e)
        return idx

    def candidates_for(self, entity: Entity) -> set[str]:
        #Return target IRIs sharing at least one token with this entity
        out: set[str] = set()
        for label in entity.all_labels():
            for tok in tokenise(label):
                for (kind, iri) in self._index.get(tok, ()):
                    if kind == entity.kind:
                        out.add(iri)
        return out

    def stats(self) -> dict[str, int]:
        #Diagnostic info for the report's scalability discussion
        return {
            "tokens": len(self._index),
            "entities": len(self._entity_by_iri),
            "avg_postings_per_token": (
                sum(len(v) for v in self._index.values()) // max(len(self._index), 1)
            ),
        }


def shortlist(
    src: list[Entity], tgt_index: InvertedIndex
) -> dict[str, set[str]]:
    return {s.iri: tgt_index.candidates_for(s) for s in src}
