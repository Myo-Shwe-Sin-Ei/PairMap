# This class was written with the help of AI
# All generated code was reviewed and understood

# Load an OWL ontology and extract Entity records

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from .normalise import fragment_from_iri, normalise
from .types import Entity, EntityKind


@contextmanager
def _suppress_owlready_warnings():
    # Strip '* Owlready2 *' lines from stdout during ontology load
    
    real_stdout = sys.stdout

    class _Filter:
        def write(self, msg: str) -> int:
            if "* Owlready2 *" in msg:
                return len(msg)
            return real_stdout.write(msg)

        def flush(self) -> None:
            real_stdout.flush()

    sys.stdout = _Filter()
    try:
        yield
    finally:
        sys.stdout = real_stdout

# Checking labels and synonym annotations

SYNONYM_PREDICATES = (
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
)


def _safe_str(value: object) -> str:
    # Best-effort string conversion that handles locstr and bytes
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _get_annotation_values(entity: object, predicate_iri: str) -> list[str]:
    # Pull annotation values for a given predicate IRI from an Owlready2 entity
    out: list[str] = []
    try:
        # Owlready2 exposes annotations via the IRIS dict
        ns_iri, _, name = predicate_iri.rpartition("#")
        if not name:
            ns_iri, _, name = predicate_iri.rpartition("/")
        from owlready2 import IRIS  # local import: optional dep at import time
        annotation_property = IRIS[predicate_iri]
        if annotation_property is None:
            return out
        values = annotation_property[entity]
        for v in values or []:
            s = _safe_str(v).strip()
            if s:
                out.append(s)
    except Exception:
        # Owlready2 raises for many edge cases; treat as "no synonyms"
        pass
    return out


def _extract_entity(owl_entity: object, kind: EntityKind) -> Entity | None:
    # Convert an Owlready2 entity into our Entity dataclass
    iri = getattr(owl_entity, "iri", None)
    if not iri:
        return None

    # Labels: prefer rdfs:label, then skos:prefLabel (for SKOS thesauri),
    # then fall back to the URI fragment.
    labels: list[str] = []
    try:
        labels = [_safe_str(lbl).strip() for lbl in (owl_entity.label or [])]
        labels = [lbl for lbl in labels if lbl]
    except Exception:
        pass

    if not labels:
        skos_pref = _get_annotation_values(
            owl_entity, "http://www.w3.org/2004/02/skos/core#prefLabel"
        )
        labels = [v for v in (s.strip() for s in skos_pref) if v]

    raw_label = labels[0] if labels else fragment_from_iri(iri)
    primary_label = normalise(raw_label)

    # Synonyms: every alternate label or OBO synonym
    synonyms: list[str] = []
    for pred in SYNONYM_PREDICATES:
        for v in _get_annotation_values(owl_entity, pred):
            n = normalise(v)
            if n and n != primary_label and n not in synonyms:
                synonyms.append(n)
    # Also treat secondary labels as synonyms
    for extra in labels[1:]:
        n = normalise(extra)
        if n and n != primary_label and n not in synonyms:
            synonyms.append(n)

    # Comment: take the first rdfs:comment
    comment = ""
    try:
        comments = list(owl_entity.comment or [])
        if comments:
            comment = _safe_str(comments[0]).strip()
    except Exception:
        pass

    # Parents and children (classes and properties only)
    parents: list[str] = []
    children: list[str] = []
    if kind in ("class", "object_property", "data_property"):
        try:
            for p in getattr(owl_entity, "is_a", []) or []:
                pid = getattr(p, "iri", None)
                if pid:
                    parents.append(pid)
        except Exception:
            pass
        try:
            sub_iter = (
                owl_entity.subclasses()
                if kind == "class"
                else getattr(owl_entity, "subclasses", lambda: [])()
            )
            for c in sub_iter or []:
                cid = getattr(c, "iri", None)
                if cid:
                    children.append(cid)
        except Exception:
            pass

    return Entity(
        iri=iri,
        kind=kind,
        label=primary_label,
        raw_label=raw_label,
        synonyms=tuple(synonyms),
        comment=comment,
        parents=tuple(parents),
        children=tuple(children),
    )


def _load_skos_concepts(path: str | Path) -> list[Entity]:
    # Extract skos:Concept instances via rdflib

    import rdflib
    from rdflib import RDF, URIRef

    SKOS = "http://www.w3.org/2004/02/skos/core#"
    SKOS_CONCEPT = URIRef(SKOS + "Concept")
    SKOS_PREFLABEL = URIRef(SKOS + "prefLabel")
    SKOS_ALTLABEL = URIRef(SKOS + "altLabel")
    SKOS_BROADER = URIRef(SKOS + "broader")
    SKOS_DEFINITION = URIRef(SKOS + "definition")
    RDFS_LABEL = URIRef("http://www.w3.org/2000/01/rdf-schema#label")

    g = rdflib.Graph()
    try:
        g.parse(str(path))
    except Exception:
        return []

    def _sort_by_language(literals: list) -> list[str]:

        en, none, other = [], [], []
        for lit in literals:
            text = str(lit).strip()
            if not text:
                continue
            lang = getattr(lit, "language", None)
            if lang and lang.lower().startswith("en"):
                en.append(text)
            elif not lang:
                none.append(text)
            else:
                other.append(text)
        return en + none + other

    entities: list[Entity] = []
    for concept in g.subjects(RDF.type, SKOS_CONCEPT):
        if not isinstance(concept, URIRef):
            continue
        iri = str(concept)
        # The skos:Concept class itself is sometimes typed as itself in the
        # ontology header — that is a meta-entity, not a real concept.
        if iri == str(SKOS_CONCEPT):
            continue
        pref_vals = _sort_by_language(list(g.objects(concept, SKOS_PREFLABEL)))
        label_vals = _sort_by_language(list(g.objects(concept, RDFS_LABEL)))
        labels = pref_vals + label_vals
        raw_label = labels[0] if labels else fragment_from_iri(iri)
        primary_label = normalise(raw_label)
        if not primary_label:
            continue

        synonyms: list[str] = []
        for v in g.objects(concept, SKOS_ALTLABEL):
            n = normalise(str(v))
            if n and n != primary_label and n not in synonyms:
                synonyms.append(n)
        for extra in labels[1:]:
            n = normalise(extra)
            if n and n != primary_label and n not in synonyms:
                synonyms.append(n)

        comment_vals = [str(v).strip() for v in g.objects(concept, SKOS_DEFINITION)]
        comment = comment_vals[0] if comment_vals else ""

        parents = tuple(str(p) for p in g.objects(concept, SKOS_BROADER) if isinstance(p, URIRef))
        children = tuple(str(c) for c in g.subjects(SKOS_BROADER, concept) if isinstance(c, URIRef))

        entities.append(Entity(
            iri=iri,
            kind="individual",
            label=primary_label,
            raw_label=raw_label,
            synonyms=tuple(synonyms),
            comment=comment,
            parents=parents,
            children=children,
        ))
    return entities


def load_ontology(path: str | Path) -> list[Entity]:
    # Parse an OWL file and return all extracted entities

    from owlready2 import get_ontology, Thing, ObjectProperty, DataProperty

    entities: list[Entity] = []

    try:
        with _suppress_owlready_warnings():
            onto = get_ontology(f"file://{Path(path).resolve()}").load()
    except Exception:
        onto = None

    if onto is not None:
        for cls in onto.classes():
            if cls is Thing:
                continue
            e = _extract_entity(cls, "class")
            if e is not None and e.label:
                entities.append(e)

        for prop in onto.object_properties():
            if prop is ObjectProperty:
                continue
            e = _extract_entity(prop, "object_property")
            if e is not None and e.label:
                entities.append(e)

        for prop in onto.data_properties():
            if prop is DataProperty:
                continue
            e = _extract_entity(prop, "data_property")
            if e is not None and e.label:
                entities.append(e)

        for ind in onto.individuals():
            e = _extract_entity(ind, "individual")
            if e is not None and e.label:
                entities.append(e)

    seen = {e.iri for e in entities}
    for e in _load_skos_concepts(path):
        if e.iri not in seen:
            entities.append(e)
            seen.add(e.iri)

    return entities
