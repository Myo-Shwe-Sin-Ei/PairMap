# Ablation check - measure per-stage contribution to F-score on a single pair

#Runs the pipeline once, then re-applies Stage 5 with different score-source
#subsets active by filtering the cached `CandidateSet`

#Stage 2 (inverted-index blocker) does not add scores - it narrows the
#search space - so its effectiveness is reported separately as shortlist size
#Usage: python scripts/ablate.py

#This code was written with the help of AI
#All methods have been reviewed and understood


from __future__ import annotations

# Pre-import sentence_transformers BEFORE owlready2/rdflib
import sentence_transformers 

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pairmap import evaluate, run_pipeline, write_alignment
from pairmap.config import get_config
from pairmap.loader import load_ontology
from pairmap.stage5_combine import (
    combine_scores, resolve_conflicts, to_mappings,
)

variants: list[tuple[str, set[str]]] = [
    ("Lex only (Stage 1)",      {"lexical"}),
    ("Embed only (Stage 3)",    {"embedding"}),
    ("No lex (3+4)",            {"embedding", "locality"}),
    ("No embed (1+4)",          {"lexical", "locality"}),
    ("No locality (1+3)",       {"lexical", "embedding"}),
    ("Full (1+3+4)",            {"lexical", "embedding", "locality"}),
]

def filter_candidates(candidates, keep_keys):
    #Return a copy of `candidates` with only the specified score keys retained
    out = {}
    for pair, scores in candidates.items():
        kept = {k: v for k, v in scores.items() if k in keep_keys}
        if kept:
            out[pair] = kept
    return out

def evaluate_variant(candidates, cfg, src_entities, tgt_entities, reference, total_time):
    combined = combine_scores(candidates, cfg)
    resolved = resolve_conflicts(combined, cfg.threshold)
    mappings = to_mappings(resolved, src_entities, tgt_entities, candidates)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ttl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_alignment(mappings, tmp_path)
        ev = evaluate(tmp_path, reference, elapsed_sec=total_time)
        return len(mappings), ev.precision, ev.recall, ev.fscore
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--track", default="conference")
    args = parser.parse_args()

    print(f"Running pipeline on {args.source.name} <-> {args.target.name} "
          f"(track={args.track})")
    result = run_pipeline(args.source, args.target, track=args.track)
    print(f"  done in {result.timings['total']:.1f}s "
          f"({result.stats['shortlist_pairs']} candidate pairs scored)")
    print()

    cfg = get_config(args.track)
    src_entities = load_ontology(args.source)
    tgt_entities = load_ontology(args.target)

    #Stage 2 (blocker) effectiveness
    n_shortlist = result.stats["shortlist_pairs"]
    n_src = result.stats["src_entities"]
    n_tgt = result.stats["tgt_entities"]
    n_naive = n_src * n_tgt
    reduction = (1 - n_shortlist / n_naive) * 100 if n_naive else 0.0
    print("Stage 2 effectiveness (inverted-index blocker)")
    print(f" Source entities : {n_src:>10,}")
    print(f" Target entities : {n_tgt:>10,}")
    print(f" Naive pair count : {n_naive:>10,}")
    print(f" Shortlist pairs : {n_shortlist:>10,}")
    print(f" Reduction : {reduction:>10.1f}%")
    print()

    #Per-stage wall-clock (OA.4 discussion)
    print("Per-stage wall-clock")
    stage_keys = [
        "load", "stage1_lexical", "stage2_index",
        "stage3_embeddings", "stage4_locality", "stage5_combine",
    ]
    total = result.timings["total"] or 1.0
    for key in stage_keys:
        secs = result.timings.get(key, 0.0)
        pct = secs / total * 100
        print(f"  {key:<24} {secs:>7.2f}s  ({pct:5.1f}%)")
    print(f"  {'total':<24} {result.timings['total']:>7.2f}s")
    print()

    #Variant-by-variant Pr/Re/F
    print("Stage contribution - Pr/Re/F at threshold "
          f"{cfg.threshold:.2f} (locked in pairmap/config.py)")
    print(f"  {'variant':<28} {'|M_S|':>6} {'Pr':>10} {'Re':>10} {'F':>10}")

    rows: dict[str, tuple[int, float, float, float]] = {}
    for name, keep in variants:
        ablated = filter_candidates(result.candidates, keep)
        n, pr, re_, f = evaluate_variant(
            ablated, cfg, src_entities, tgt_entities,
            args.reference, result.timings["total"],
        )
        rows[name] = (n, pr, re_, f)
        print(f"  {name:<28} {n:>6d} {pr:>10.4f} {re_:>10.4f} {f:>10.4f}")
    print()

    #F delta: each stage's contribution
    full_n, full_pr, full_re, full_f = rows["Full (1+3+4)"]
    print(f"Each stage's contribution to F (Full - 'No X')   [Full F = {full_f:.4f}]")
    contributions = [
        ("Stage 1 (lexical)", "No lex (3+4)"),
        ("Stage 3 (embedding)", "No embed (1+4)"),
        ("Stage 4 (locality)",  "No locality (1+3)"),
    ]
    for stage, dropped_variant in contributions:
        _, _, _, f_dropped = rows[dropped_variant]
        delta = full_f - f_dropped
        sign = "+" if delta >= 0 else ""
        print(f"  {stage:<28} contributes {sign}{delta:.4f} F "
              f"(F drops to {f_dropped:.4f} without it)")
    print()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
