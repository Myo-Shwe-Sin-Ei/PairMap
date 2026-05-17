#Sweep the Stage 5 confidence threshold for a single ontology pair.

#Runs the pipeline once to compute embeddings + locality, then re-applies
#just Stage 5 (threshold + 1:1 conflict resolve) for each value in thesweep range
#Reports Pr / Re / F per threshold so we can pick the best
#Usage     python scripts/sweep_threshold.py \\
#        --source data/anatomy/ontologies/mouse.owl \\
#        --target data/anatomy/ontologies/human.owl \\
#        --reference data/anatomy/reference/human-mouse.ttl \\
#        --track anatomy

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pairmap import evaluate, run_pipeline, write_alignment 
from pairmap.stage5_combine import combine_scores, resolve_conflicts, to_mappings 

sweep = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--track", default="anatomy")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    print(f"Running pipeline once on {args.source.name} <-> {args.target.name} "
          f"(track={args.track})")
    result = run_pipeline(
        src_path=args.source,
        tgt_path=args.target,
        track=args.track,
    )
    print(f"Pipeline done in {result.timings['total']:.1f}s. "
          f"{result.stats['shortlist_pairs']} candidate pairs scored.")

    from pairmap.config import get_config
    from pairmap.loader import load_ontology
    base_cfg = get_config(args.track)
    combined = combine_scores(result.candidates, base_cfg)
    src_entities = load_ontology(args.source)
    tgt_entities = load_ontology(args.target)

    print(f"\n{'threshold':>10}  {'|M_S|':>6}  {'Pr':>8}  {'Re':>8}  {'F':>8}")
    print("-" * 50)

    rows = []
    for thr in sweep:
        kept = resolve_conflicts(combined, thr)
        mappings = to_mappings(kept, src_entities, tgt_entities, result.candidates)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ttl", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            write_alignment(mappings, tmp_path)
            ev = evaluate(tmp_path, args.reference,
                          elapsed_sec=result.timings["total"])
            rows.append((thr, len(mappings), ev.precision, ev.recall, ev.fscore))
            print(f"{thr:>10.2f}  {len(mappings):>6d}  "
                  f"{ev.precision:>8.4f}  {ev.recall:>8.4f}  {ev.fscore:>8.4f}")
        finally:
            tmp_path.unlink(missing_ok=True)

    best = max(rows, key=lambda r: r[4])
    print(f"BEST: threshold={best[0]:.2f}  |M_S|={best[1]}  "
          f"Pr={best[2]:.4f}  Re={best[3]:.4f}  F={best[4]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
