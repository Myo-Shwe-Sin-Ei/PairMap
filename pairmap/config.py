from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class TrackConfig:
    embedding_model: str


TRACK_CONFIG: dict[str, TrackConfig] = {
    "conference": TrackConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",

    ),
    "anatomy": TrackConfig(
        embedding_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",

    ),
    "bio-ml": TrackConfig(
        embedding_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",

    ),
    "digital-humanities": TrackConfig(
        embedding_model="intfloat/multilingual-e5-base",

    ),
    "circular-economy": TrackConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",

    ),
    "knowledge-graph": TrackConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",

    ),
    "wine": TrackConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",

    ),
}

def get_config(track: str) -> TrackConfig:
    config = TRACK_CONFIG.get(track.lower())
    if config is None:
        raise ValueError(
            f"Unknown track: {track!r}. "
            f"Known tracks: {sorted(TRACK_CONFIG)}"
        )
    return config
