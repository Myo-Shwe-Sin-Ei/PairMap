# Results output was written with the help of AI
from __future__ import annotations

import argparse
import sys
import time
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


def evaluate_track(track: str, data_root: Path, output_dir: Path) -> dict:
    # Run PairMap on every pair in `data_root/<track>/` and return a summary
    data_dir = data_root / track
    ontologies_dir = data_dir / "ontologies"
    if not ontologies_dir.is_dir():
        print(f"Missing ontologies/ in {data_dir}", file=sys.stderr)
        return {"track": track, "pairs": [], "macro_f": None}

    suffix = _detect_ontology_suffix(ontologies_dir)

    # References live in <data-dir>/reference/
    reference_dir = data_dir / "reference"
    ref_glob = sorted(
        reference_dir.glob("*.ttl") if reference_dir.is_dir()
        else data_dir.glob("*.ttl")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(f"\n Evaluating {track} ({len(ref_glob)} pairs) \n")
    track_start = time.perf_counter()

    for ref_file in ref_glob:
        stem = ref_file.stem
        try:
            src_name, tgt_name = stem.split("-", 1)
        except ValueError:
            print(f" skip {ref_file.name}: cannot parse src-tgt", file=sys.stderr)
            continue

        src_path = ontologies_dir / f"{src_name}{suffix}"
        tgt_path = ontologies_dir / f"{tgt_name}{suffix}"
        if not src_path.is_file() or not tgt_path.is_file():
            print(f" skip {stem}: missing ontology file", file=sys.stderr)
            continue

        out_path = output_dir / f"pairmap-{stem}.ttl"

        try:
            result = run_pipeline(src_path, tgt_path, track=track)
            write_alignment(result.mappings, out_path)
            ev = evaluate(out_path, ref_file, elapsed_sec=result.timings["total"])
            rows.append({
                "task": stem, "ok": True,
                "M_S": ev.system_count, "M_RA": ev.reference_count,
                "Pr": ev.precision, "Re": ev.recall, "F": ev.fscore,
                "time_s": result.timings["total"],
            })
            print(f"  {stem:<40} Pr={ev.precision:.4f} Re={ev.recall:.4f} "
                  f"F={ev.fscore:.4f}  ({result.timings['total']:.1f}s)")
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
            if r.get("ok"):
                fh.write(
                    f"| {r['task']} | {r['M_S']} | {r['M_RA']} | "
                    f"{r['Pr']:.4f} | {r['Re']:.4f} | {r['F']:.4f} | "
                    f"{r['time_s']:.1f} |\n"
                )
            else:
                fh.write(f"| {r['task']} | - | - | - | - | - | "
                         f"FAILED: {r['error']} |\n")

    ok_rows = [r for r in rows if r.get("ok")]
    macro_f = sum(r["F"] for r in ok_rows) / len(ok_rows) if ok_rows else None

    # Per-track terminal summary
    print()
    print("=" * 62)
    print(f"  Track Summary : {track} ({len(ok_rows)}/{len(rows)} pairs OK)")
    print("=" * 62)
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["F"])
        worst = min(ok_rows, key=lambda r: r["F"])
        print(f"  Macro-mean F   : {macro_f:.4f}")
        print(f"  Best pair      : {best['task']} (F={best['F']:.4f})")
        print(f"  Worst pair     : {worst['task']} (F={worst['F']:.4f})")
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
        print("    python -m pairmap <track>")
        print()
        print("Run every track that has a data/<track>/ folder:")
        print("    python -m pairmap all")
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
