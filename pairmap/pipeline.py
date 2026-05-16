#Pipeline orchestrator 
#Single entry point for the end-to-end run

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import TrackConfig, get_config
from .loader import load_ontology
from .stage1_lexical import lexical_match
from .stage2_index import InvertedIndex, shortlist
from .stage3_embeddings import score_with_embeddings
from .stage4_locality import locality_boost
from .stage5_combine import combine_scores, resolve_conflicts, to_mappings
from .types import CandidateSet, Mapping

log = logging.getLogger("pairmap.pipeline")

@dataclass
class PipelineResult:
    mappings: list[Mapping]
    candidates: CandidateSet
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, object] = field(default_factory=dict)


def run_pipeline(
    src_path: str | Path,
    tgt_path: str | Path,
    track: str = "conference",
    config: TrackConfig | None = None,
) -> PipelineResult:
    #Run the PairMap pipeline on a pair of ontologies
    cfg = config if config is not None else get_config(track)
    timings: dict[str, float] = {}
    stats: dict[str, object] = {}

    #Loading both ontologies
    t0 = time.perf_counter()
    src_entities = load_ontology(src_path)
    tgt_entities = load_ontology(tgt_path)
    timings["load"] = time.perf_counter() - t0
    stats["src_entities"] = len(src_entities)
    stats["tgt_entities"] = len(tgt_entities)
    log.info(
        "Loaded %d source and %d target entities",
        len(src_entities), len(tgt_entities),
    )

    #Stage 1: lexical match
    t0 = time.perf_counter()
    candidates = lexical_match(src_entities, tgt_entities)
    timings["stage1_lexical"] = time.perf_counter() - t0
    stats["candidates_after_stage1"] = len(candidates)

    #Stage 2: inverted index for shortlist
    t0 = time.perf_counter()
    tgt_index = InvertedIndex.build(tgt_entities)
    short = shortlist(src_entities, tgt_index)
    timings["stage2_index"] = time.perf_counter() - t0
    stats["index_stats"] = tgt_index.stats()
    n_short = sum(len(v) for v in short.values())
    stats["shortlist_pairs"] = n_short
    naive = len(src_entities) * len(tgt_entities)
    stats["shortlist_vs_naive"] = f"{n_short} / {naive}"
    log.info("Inverted index: %d pairs vs %d naive", n_short, naive)

    #Stage 3: embedding scores over the shortlist
    t0 = time.perf_counter()
    candidates = score_with_embeddings(
        src_entities, tgt_entities, short, candidates, cfg
    )
    timings["stage3_embeddings"] = time.perf_counter() - t0
    stats["candidates_after_stage3"] = len(candidates)

    #Stage 4: locality booster
    t0 = time.perf_counter()
    candidates = locality_boost(
        candidates, src_entities, tgt_entities, alpha=cfg.locality_alpha
    )
    timings["stage4_locality"] = time.perf_counter() - t0

    #Stage 5: combine + threshold + 1:1 resolve
    t0 = time.perf_counter()
    combined = combine_scores(candidates, cfg)
    resolved = resolve_conflicts(combined, cfg.threshold)
    mappings = to_mappings(resolved, src_entities, tgt_entities, candidates)
    timings["stage5_combine"] = time.perf_counter() - t0
    stats["mappings_after_stage5"] = len(mappings)

    timings["total"] = sum(timings.values())
    log.info("Pipeline complete: %d mappings in %.2fs", len(mappings), timings["total"])

    return PipelineResult(
        mappings=mappings,
        candidates=candidates,
        timings=timings,
        stats=stats,
    )
