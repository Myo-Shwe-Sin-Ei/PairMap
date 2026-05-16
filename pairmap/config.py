from __future__ import annotations

from dataclasses import dataclass
#Threshold was tuned and hardcoded in here using sweep_threshold.py

@dataclass(frozen=True)
class TrackConfig:
    embedding_model: str
    threshold: float
    lex_weight: float
    locality_alpha: float 
    timeout_sec: int

    @property
    def embed_weight(self) -> float:
        return 1.0 - self.lex_weight


TRACK_CONFIG: dict[str, TrackConfig] = {
    "conference": TrackConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.75,
        lex_weight=0.5,
        locality_alpha=0.15,
        timeout_sec=600,
    ),
    "anatomy": TrackConfig(
        embedding_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        threshold=0.75,
        lex_weight=0.4,
        locality_alpha=0.15,
        timeout_sec=3600,
    ),
    "digital-humanities": TrackConfig(
        embedding_model="intfloat/multilingual-e5-base",
        threshold=0.85,
        lex_weight=0.5,
        locality_alpha=0.15,
        timeout_sec=1200,
    ),
    "circular-economy": TrackConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.75,
        lex_weight=0.5,
        locality_alpha=0.15,
        timeout_sec=600,
    ),
    "wine": TrackConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.70,
        lex_weight=0.5,
        locality_alpha=0.20,
        timeout_sec=300,
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
