# Results output was written with the help of AI
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

from .config import TRACK_CONFIG
from .evaluation import evaluate
from .pipeline import run_pipeline
from .serialise import write_alignment


def _detect_ontology_suffix(ontologies_dir: Path) -> str:
    # Pick the most common extension among files in ontologies
    counts: dict[str, int] = {}
    for f in ontologies_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".owl", ".rdf"):
            counts[f.suffix.lower()] = counts.get(f.suffix.lower(), 0) + 1
    if not counts:
        return ".owl"
    return max(counts, key=counts.get)

def _resolve_track_layout(data_dir: Path) -> tuple[Path, list[Path]]:
    """Return (ontologies_dir, reference_files) for any supported layout.

    Three layouts are accepted:
    - data/<track>/ontologies/  +  data/<track>/reference/*.ttl   (OAEI)
    - data/<track>/ontologies/  +  data/<track>/*.ttl             (Bio-ML, KG)
    - data/<track>/*.owl|*.rdf  +  (no reference)                 (OA.2 etc.)
    """
    ontologies_subdir = data_dir / "ontologies"
    if ontologies_subdir.is_dir():
        ontologies_dir = ontologies_subdir
    else:
        ontologies_dir = data_dir

    reference_dir = data_dir / "reference"
    if reference_dir.is_dir():
        ref_files = sorted(reference_dir.glob("*.ttl"))
    elif ontologies_dir is not data_dir:
        ref_files = sorted(data_dir.glob("*.ttl"))
    else:
        ref_files = []

    return ontologies_dir, ref_files

def evaluate_track(track: str, data_root: Path, output_dir: Path) -> dict:
    data_dir = data_root / track
    if not data_dir.is_dir():
        print(f"Missing track folder: {data_dir}", file=sys.stderr)
        return {"track": track, "pairs": [], "macro_f": None, "time_s": 0.0}

    ontologies_dir, ref_files = _resolve_track_layout(data_dir)

    onto_files = sorted(
        f for f in ontologies_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".owl", ".rdf")
    )
    if not onto_files:
        print(f"No ontology files in {ontologies_dir}", file=sys.stderr)
        return {"track": track, "pairs": [], "macro_f": None, "time_s": 0.0}

    suffix = _detect_ontology_suffix(ontologies_dir)

    # Build the list of (stem, src_path, tgt_path, ref_path-or-None).
    # Reference-driven when we have a gold standard; otherwise enumerate
    # every unordered pair of ontology files.
    pairs: list[tuple[str, Path, Path, Path | None]] = []
    if ref_files:
        for ref_file in ref_files:
            stem = ref_file.stem
            try:
                src_name, tgt_name = stem.split("-", 1)
            except ValueError:
                print(f"  skip {ref_file.name}: cannot parse src-tgt", file=sys.stderr)
                continue
            src_path = ontologies_dir / f"{src_name}{suffix}"
            tgt_path = ontologies_dir / f"{tgt_name}{suffix}"
            if not src_path.is_file() or not tgt_path.is_file():
                print(f"  skip {stem}: missing ontology file", file=sys.stderr)
                continue
            pairs.append((stem, src_path, tgt_path, ref_file))
    else:
        for src_path, tgt_path in combinations(onto_files, 2):
            stem = f"{src_path.stem}-{tgt_path.stem}"
            pairs.append((stem, src_path, tgt_path, None))

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    label = f"{len(pairs)} pairs" if ref_files else f"{len(pairs)} pairs (no reference)"
    print(f"\n=== Evaluating {track} ({label}) ===\n")
    track_start = time.perf_counter()

    for stem, src_path, tgt_path, ref_file in pairs:
        out_path = output_dir / f"pairmap-{stem}.ttl"
        try:
            result = run_pipeline(src_path, tgt_path, track=track)
            write_alignment(result.mappings, out_path)
            if ref_file is not None:
                ev = evaluate(
                    out_path, ref_file, elapsed_sec=result.timings["total"]
                )
                rows.append({
                    "task": stem, "ok": True,
                    "M_S": ev.system_count, "M_RA": ev.reference_count,
                    "Pr": ev.precision, "Re": ev.recall, "F": ev.fscore,
                    "time_s": result.timings["total"],
                })
                print(f"  {stem:<40} Pr={ev.precision:.4f} Re={ev.recall:.4f} "
                      f"F={ev.fscore:.4f}  ({result.timings['total']:.1f}s)")
            else:
                rows.append({
                    "task": stem, "ok": True,
                    "M_S": len(result.mappings), "M_RA": None,
                    "Pr": None, "Re": None, "F": None,
                    "time_s": result.timings["total"],
                })
                print(f"  {stem:<40} |M_S|={len(result.mappings):<5}  "
                      f"(no reference; {result.timings['total']:.1f}s)")
        except Exception as exc:
            rows.append({"task": stem, "ok": False, "error": str(exc)})
            print(f"  {stem:<40} FAILED: {exc}", file=sys.stderr)

    track_time = time.perf_counter() - track_start

    # Write the Markdown summary
    summary_path = output_dir / f"{track}-summary.md"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# PairMap results — {track} track\n\n")
        fh.write("| Task | |M_S| | |M_RA| | Pr | Re | F | Time (s) |\n")
        fh.write("|------|------|-------|----|----|---|----------|\n")
        for r in rows:
            if not r.get("ok"):
                fh.write(f"| {r['task']} | - | - | - | - | - | "
                         f"FAILED: {r['error']} |\n")
                continue
            if r["F"] is not None:
                fh.write(
                    f"| {r['task']} | {r['M_S']} | {r['M_RA']} | "
                    f"{r['Pr']:.4f} | {r['Re']:.4f} | {r['F']:.4f} | "
                    f"{r['time_s']:.1f} |\n"
                )
            else:
                fh.write(
                    f"| {r['task']} | {r['M_S']} | - | - | - | - | "
                    f"{r['time_s']:.1f} |\n"
                )

    ok_rows = [r for r in rows if r.get("ok")]
    eval_rows = [r for r in ok_rows if r.get("F") is not None]
    macro_f = (
        sum(r["F"] for r in eval_rows) / len(eval_rows) if eval_rows else None
    )
    # Per-track terminal summary
    print()
    print("=" * 62)
    print(f"  TRACK SUMMARY: {track} ({len(ok_rows)}/{len(rows)} pairs OK)")
    print("=" * 62)
    if eval_rows:
        best = max(eval_rows, key=lambda r: r["F"])
        worst = min(eval_rows, key=lambda r: r["F"])
        print(f"  Macro-mean F   : {macro_f:.4f}")
        print(f"  Best pair      : {best['task']} (F={best['F']:.4f})")
        print(f"  Worst pair     : {worst['task']} (F={worst['F']:.4f})")
    elif ok_rows:
        print(f"  No reference alignment — F-score not computed.")
        print(f"  Mappings produced: "
              f"{sum(r['M_S'] for r in ok_rows)} across {len(ok_rows)} pairs")
    print(f"  Total time     : {track_time:.1f}s")
    print(f"  Summary table  : {summary_path}")
    print("=" * 62)

    return {"track": track, "pairs": rows, "macro_f": macro_f, "time_s": track_time}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pairmap",
        description="Run PairMap on an OAEI track (or all configured tracks).",
    )
    parser.add_argument(
        "track",
        nargs="?",
        help="Track name (e.g. conference, anatomy). "
             "Use 'all' to run every track that has a data folder.",
    )
    parser.add_argument(
        "--data-root", default=Path("data"), type=Path,
        help="Root folder for OAEI datasets (default: data/)",
    )
    parser.add_argument(
        "--output-dir", default=Path("results"), type=Path,
        help="Output folder (default: results/)",
    )
    args = parser.parse_args()

    if args.track is None:
        known = sorted(TRACK_CONFIG)
        print("PairMap - OAEI ontology alignment system")
        print()
        print("Run a single track:")
        print("python -m pairmap <track>")
        print()
        print("Run every track that has a data/<track>/ folder:")
        print("python -m pairmap all")
        print()
        print(f"Known tracks: {', '.join(known)}")
        return 0

    if args.track.lower() == "all":
        results = []
        for name in sorted(TRACK_CONFIG):
            if (args.data_root / name).is_dir():
                results.append(
                    evaluate_track(name, args.data_root, args.output_dir)
                )
        # Final scoreboard
        print()
        print("=" * 62)
        print(" Final Results ")
        print("=" * 62)
        print(f"  {'track':<24} {'pairs':>5} {'macro F':>10} {'time (s)':>10}")
        print("  " + "-" * 56)
        for r in results:
            n = sum(1 for p in r["pairs"] if p.get("ok"))
            mf = f"{r['macro_f']:.4f}" if r["macro_f"] is not None else "n/a"
            print(f"  {r['track']:<24} {n:>5} {mf:>10} {r['time_s']:>10.1f}")
        print("=" * 62)
        return 0

    evaluate_track(args.track, args.data_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
