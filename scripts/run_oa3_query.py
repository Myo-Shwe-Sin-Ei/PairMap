#OA.3 - run a SPARQL query over O1 U O2 U M after OWL RL reasoning
#Loads:
#data/wine/student-a.owl (O1, Andrew Nguyen)
#data/wine/student-b.owl (O2, Myo Shwe Sin Ei)
#python scripts/run_oa3_query.py

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

from rdflib import Graph

try:
    import owlready2
except ImportError as exc:
    sys.stderr.write(
        "owlready2 is required to parse OWL/XML ontologies. "
        "Install with `pip install owlready2`.\n"
    )
    raise SystemExit(1) from exc

try:
    import owlrl
except ImportError as exc:
    sys.stderr.write(
        "owlrl is required for reasoning. Install with `pip install owlrl`.\n"
    )
    raise SystemExit(1) from exc


project_root = Path(__file__).resolve().parent.parent
onto_a = project_root / "data" / "wine" / "student-a.owl"
onto_b = project_root / "data" / "wine" / "student-b.owl"
alignment = project_root / "results" / "pairmap-student-a-student-b.ttl"
query_file = project_root / "results" / "oa3-query.rq"
output_csv = project_root / "results" / "oa3-results.csv"


def load_owl_xml_into(graph: Graph, owl_path: Path) -> int:
    #Parse an OWL/XML file via owlready2 and merge it into graph
    #Returns the number of triples added.
    
    world = owlready2.World()
    onto = world.get_ontology(str(owl_path)).load()
    # Save the ontology as RDF/XML
    buffer = io.BytesIO()
    onto.save(file=buffer, format="rdfxml")
    buffer.seek(0)
    before = len(graph)
    graph.parse(buffer, format="application/rdf+xml")
    return len(graph) - before


def main() -> int:
    print(f"[OA.3] Loading ontology A: {onto_a}")
    g = Graph()
    added_a = load_owl_xml_into(g, onto_a)
    print(f"        +{added_a} triples")

    print(f"[OA.3] Loading ontology B: {onto_b}")
    added_b = load_owl_xml_into(g, onto_b)
    print(f"        +{added_b} triples")

    print(f"[OA.3] Loading alignment M: {alignment}")
    before = len(g)
    g.parse(alignment, format="turtle")
    print(f"        +{len(g) - before} triples (total {len(g)})")

    print("[OA.3] Applying OWL RL reasoning")
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    print(f"        graph now has {len(g)} triples after closure")

    print(f"[OA.3] Loading query: {query_file}")
    query_text = query_file.read_text(encoding="utf-8")

    print("[OA.3] Executing query")
    results = list(g.query(query_text))
    print(f"[OA.3] Query returned {len(results)} row(s)")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["taster", "name"])
        for row in results:
            writer.writerow([str(cell) if cell is not None else "" for cell in row])
    print(f"[OA.3] Wrote {output_csv}")

    if not results:
        print("[OA.3] WARNING: query returned zero rows.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
