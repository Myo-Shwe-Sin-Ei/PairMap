from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pairmap import evaluate, run_pipeline, write_alignment 


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--track", default="conference")
    args = parser.parse_args()

    result = run_pipeline(
        src_path=args.source,
        tgt_path=args.target,
        track=args.track,
    )

    write_alignment(result.mappings, args.output)
    print(f"Wrote {len(result.mappings)} mappings to {args.output} "
          f"({result.timings['total']:.1f}s)")

    if args.reference is not None:
        ev = evaluate(args.output, args.reference,
                      elapsed_sec=result.timings["total"])
        print(f"  Pr={ev.precision:.4f}  Re={ev.recall:.4f}  F={ev.fscore:.4f}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
