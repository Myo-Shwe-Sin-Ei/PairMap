#Stage 3 — advanced technique: sentence-transformer embeddings
#For each entity, build a textual profile then encode it once 
#with a pre-trained sentence-transformer
#then score candidate pairs via cosine similarity. 

#This code was written with the help of AI
#Particularly the shortlist integretions

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrackConfig
from .types import CandidateSet, Entity


@dataclass
class EncodedOntology:
    #Vectors for one ontology, indexed by IRI position
    iris: list[str]
    vectors: np.ndarray # shape (n, dim), L2-normalised
    kind_of: dict[str, str]# iri -> kind


def build_profile(
    entity: Entity, parent_labels: dict[str, str], max_chars: int = 400
) -> str:
    #Concatenate entity context into a single string for encoding
    parts: list[str] = [entity.raw_label]
    if entity.synonyms:
        parts.extend(entity.synonyms[:5]) # cap synonyms to keep input short
    if entity.parents:
        parent_names = [
            parent_labels[p] for p in entity.parents[:3] if p in parent_labels
        ]
        if parent_names:
            parts.append("parent: " + ", ".join(parent_names))
    if entity.comment:
        parts.append(entity.comment[:200])
    profile = " | ".join(p for p in parts if p)
    return profile[:max_chars]


class EmbeddingScorer:
    #Encodes ontologies and scores candidate pairs
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None # lazy

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, entities: list[Entity]) -> EncodedOntology:
        #Encode every entity in the list and return an EncodedOntology
        #First pass: map IRI -> label for parent-label lookup
        label_of = {e.iri: e.raw_label for e in entities}
        profiles = [build_profile(e, label_of) for e in entities]

        vectors = self.model.encode(
            profiles,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return EncodedOntology(
            iris=[e.iri for e in entities],
            vectors=vectors.astype("float32"),
            kind_of={e.iri: e.kind for e in entities},
        )

    @staticmethod
    def score_shortlist(
        src_enc: EncodedOntology,
        tgt_enc: EncodedOntology,
        shortlist: dict[str, set[str]],
        candidates: CandidateSet,
    ) -> CandidateSet:
        #Add 'embedding' scores to the candidate set.
        #Only scores pairs that appear in the shortlist (Stage 2's output) and
        #share the same entity kind. Vectors are already L2-normalised, so the
        #cosine similarity reduces to a dot product.
        
        src_idx = {iri: i for i, iri in enumerate(src_enc.iris)}
        tgt_idx = {iri: i for i, iri in enumerate(tgt_enc.iris)}

        # Expand shortlist into a list of (src_pos, tgt_pos) pairs to score in bulk
        rows: list[int] = []
        cols: list[int] = []
        keys: list[tuple[str, str]] = []
        for s_iri, t_iris in shortlist.items():
            s_pos = src_idx.get(s_iri)
            if s_pos is None:
                continue
            s_kind = src_enc.kind_of.get(s_iri)
            for t_iri in t_iris:
                t_pos = tgt_idx.get(t_iri)
                if t_pos is None:
                    continue
                if tgt_enc.kind_of.get(t_iri) != s_kind:
                    continue
                rows.append(s_pos)
                cols.append(t_pos)
                keys.append((s_iri, t_iri))

        if not keys:
            return candidates

        # Compute all cosine similarities in one batched operation
        src_vecs = src_enc.vectors[rows]# (k, dim)
        tgt_vecs = tgt_enc.vectors[cols] # (k, dim)
        sims = np.einsum("ij,ij->i", src_vecs, tgt_vecs)# (k,)

        for key, sim in zip(keys, sims.tolist()):
            # Clip in case of fp noise just above 1.0
            sim = max(0.0, min(1.0, float(sim)))
            candidates.setdefault(key, {})["embedding"] = sim

        return candidates

def score_with_embeddings(
    src: list[Entity],
    tgt: list[Entity],
    shortlist: dict[str, set[str]],
    candidates: CandidateSet,
    config: TrackConfig,
) -> CandidateSet:
    #Convenience wrapper used by the orchestrator (pipeline.py)
    scorer = EmbeddingScorer(config.embedding_model)
    src_enc = scorer.encode(src)
    tgt_enc = scorer.encode(tgt)
    return scorer.score_shortlist(src_enc, tgt_enc, shortlist, candidates)
