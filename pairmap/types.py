#Shared data types for PairMap.
#This class was written with the help of AI
#Examined and Reviewed
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EntityKind = Literal["class", "object_property", "data_property", "individual"]
Relation = Literal["equivalent", "subclass", "superclass"]
ScoreSource = Literal["lexical", "embedding", "locality"]


@dataclass(frozen=True)
class Entity:
    """A single thing from an ontology: a class, property, or individual.

    - iri: unique ID
    - label: cleaned-up name, used for matching
    - raw_label: original name, kept for output and debugging
    """

    iri: str
    kind: EntityKind
    label: str
    raw_label: str
    synonyms: tuple[str, ...] = ()
    comment: str = ""
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()

    def all_labels(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for s in (self.label, *self.synonyms):
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return tuple(out)

CandidateSet = dict[tuple[str, str], dict[str, float]]

@dataclass
class Mapping:
    #A scored mapping between two entities
    e1: Entity
    e2: Entity
    confidence: float
    relation: Relation = "equivalent"
    sources: set[ScoreSource] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence out of [0, 1]: {self.confidence}")
        if self.e1.kind != self.e2.kind:
            raise ValueError(
                f"Heterogeneous mapping is not supported: "
                f"{self.e1.iri} ({self.e1.kind}) <-> {self.e2.iri} ({self.e2.kind})"
            )

    def predicate(self) -> str:
        #Return the RDF predicate for this entity kind and relation
        if self.e1.kind == "class":
            return (
                "owl:equivalentClass"
                if self.relation == "equivalent"
                else "rdfs:subClassOf"
            )
        if self.e1.kind in ("object_property", "data_property"):
            return (
                "owl:equivalentProperty"
                if self.relation == "equivalent"
                else "rdfs:subPropertyOf"
            )
        # individual
        return "owl:sameAs"

    def to_turtle_triple(self) -> str:
        #Render as a single turtle line with full IRIs
        return f"<{self.e1.iri}> {self.predicate()} <{self.e2.iri}> ."
