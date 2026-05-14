# Serialise a list of Mapping objects to turtle
# Matches the spec's example format using full IRIs and the required predicates

from __future__ import annotations

from pathlib import Path

from .types import Mapping


HEADER = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# PairMap alignment
# Total mappings: {count}
"""


def write_alignment(mappings: list[Mapping], out_path: str | Path) -> None:
    # Write mappings to a turtle file with the required predicates
    # Matches the OAEI reference-alignment (plain triples, no 
    # confidence annotations) so Pr/Re/F comparison is trivial
    
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER.format(count=len(mappings))]
    for m in mappings:
        lines.append(m.to_turtle_triple())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
