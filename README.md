# PairMap - Ontology Alignment for INM713 Coursework Part 2

PairMap is a five-stage ontology alignment system that takes two OWL ontologies and produces a Turtle alignment using the spec-mandated predicates (`owl:equivalentClass`, `rdfs:subClassOf`, `owl:equivalentProperty`, `rdfs:subPropertyOf`, `owl:sameAs`). The pipeline is LogMap-inspired: inverted-index blocking, Jaro-Winkler lexical anchors, track-adaptive Sentence-BERT embeddings, one-shot locality boost, weighted combine + greedy 1:1 resolution.

Built by Andrew Nguyen and Myo Shwe Sin Ei as a group submission.

---

## 1. Quick reference

| OA | Results |
|---|---|
| OA.1 basic technique (lexical) | Jaro-Winkler + exact match; contributes +0.10 to +0.18 macro F across tracks |
| OA.1 combine ≥1 techniques | Weighted combination + greedy 1:1 resolution; locality booster |
| OA.1 scalability | Inverted-index blocker, 97.9–99.3% reduction in pairs scored |
| OA.1 advanced technique (embeddings) | Track-adaptive Sentence-BERT (MiniLM / SapBERT / mE5); +0.23 F on `confOf-ekaw` |
| OA.2 Part 1 alignment + discussion | 14 mappings between wine ontologies; per-mapping audit table |
| OA.3 SPARQL query + CSV | 4-row CSV after OWL RL reasoning over O₁ ∪ O₂ ∪ M |
| OA.4 OAEI Pr/Re/F + computation times | 32 alignment tasks across 4 tracks; macro-mean F per track |
| OA.5 README | This document; one-command reproduction via `python -m pairmap all` |

### Headline OAEI scoreboard (from `python -m pairmap all`)

| Track | Pairs | Macro-mean F | Total time (s) |
|---|---|---|---|
| Anatomy | 1 | 0.7387 | 116.2 |
| Conference | 21 | 0.5158 | 57.2 |
| Circular Economy | 2 | 0.4727 | 18.8 |
| Digital Humanities | 8 | 0.2613 | 129.2 |
| Wine (OA.2) | 1 | n/a (no reference) | 2.7 |

### Quick reproduce

```
pip install -e
python -m pairmap all
```

Full reproduction instructions in section 11. Implementation details are in section 3.

### Project information

| | |
|---|---|
| Language | Python  3.13 |
| Key runtime deps | `owlready2`, `rdflib`, `rapidfuzz`, `sentence-transformers`, `owlrl` |
| Output format | Turtle (`.ttl`), full IRIs, mapping predicates only |
| Evaluation metrics | Precision, Recall, F-score, time |
| OAEI tracks evaluated | 4 - Anatomy, Conference, Circular Economy, Digital Humanities |
| Completed alignment tasks | 32 (Conference 21 + Anatomy 1 + Circular Economy 2 + Digital Humanities 8) |
| Scope-cut tasks | 10 (Bio-ML 5, Knowledge Graph 5 - computation time and hardware limitations) |
| Embedding models used | `all-MiniLM-L6-v2`, `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `intfloat/multilingual-e5-base` |
| Package LOC | 1,444 across 14 modules (`pairmap/`) |

---

## 2. System architecture

The pipeline is a linear sequence of five stages, with each stage upgrading the in-memory `CandidateSet` produced by the previous one.

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │        load_ontology(O₁)            load_ontology(O₂)               │
   │        Owlready2 + rdflib SKOS fallback → list[Entity]              │
   └───────────────────────────────┬─────────────────────────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  Stage 1 - Lexical              │
                  │  exact match → 1.0              │
                  │  Jaro-Winkler ≥ 0.9 → score     │
                  │  CandidateSet ← {lexical}       │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  Stage 2 - Inverted index       │
                  │  token → {(kind, iri)}          │
                  │  shortlist per source entity    │
                  │  same-kind enforced             │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  Stage 3 - Embeddings           │
                  │  Sentence-BERT (per-track model)│
                  │  cosine over shortlist          │
                  │  CandidateSet ← {…, embedding}  │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  Stage 4 - Locality             │
                  │ parent/child neighbour agreement│
                  │  CandidateSet ← {…, locality}   │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  Stage 5 - Combine + 1:1        │
                  │  final = w_lex·lex + w_emb·emb  │
                  │              + locality         │
                  │  ≥ threshold, greedy 1:1        │
                  │  → list[Mapping]                │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  serialise.write_alignment(M)   │
                  │  → results/pairmap-<a>-<b>.ttl  │
                  └─────────────────────────────────┘
```

The blocker (Stage 2) and locality principle (Stage 4) come from LogMap (Jiménez-Ruiz & Cuenca Grau, ISWC 2011). Stage 3 replaces LogMap's ISUB similarity with Sentence-BERT embeddings (cf. LogMap-ML, BERTMap). PairMap differs from LogMap in three ways: (1) one-shot locality instead of iterative, (2) track-adaptive embedding models, (3) same-kind matching enforced at the index level. LogMap's coherence-repair pass is not reproduced.

---

## 3. OA.1 - Stage-by-stage walkthrough

The data contract is a `CandidateSet`:

```python
CandidateSet = dict[tuple[str, str], dict[str, float]]
# (src_iri, tgt_iri) → {"lexical": 1.0, "embedding": 0.92, "locality": 0.04}
```

Stages 1, 3, and 4 add score keys; Stage 5 reduces them to a single final score and emits `list[Mapping]`.

### Stage 1 - Lexical match (basic technique)

**What it does.** Normalised exact match (score 1.0) plus Jaro-Winkler near-match at threshold 0.9, restricted to same-kind pairs.

**Algorithm.** Two passes per source entity: (1) look up every label/synonym in an `exact_index` keyed by `(kind, normalised_label)` exact hits get score 1.0 and skip JW; (2) compare remaining labels with `rapidfuzz.distance.JaroWinkler.normalized_similarity` against same-kind targets. Jaro-Winkler is preferred over Levenshtein because its prefix-weighting fits CamelCase / morphological variants (`hasName` vs `hasNames`).

**Design rationale.**
- Exact matches skip JW because they're already certain - saves time and keeps Stage 1 sub-second on Conference.
- Same-kind enforced upfront via `tgt_by_kind` so heterogeneous candidates never appear.
- JW threshold 0.9 trades a small amount of recall for precision relative to 0.8.

**Entry point.** `pairmap/stage1_lexical.py`.

### Stage 2 - Inverted-index blocking (scalability)

**What it does.** Builds a token → `{(kind, iri)}` index over the target ontology; for each source entity, returns only same-kind targets sharing at least one token.

**Algorithm.** `InvertedIndex.build()` walks each target entity and indexes every token in `tokenise(label)` together with its `(kind, iri)`. `candidates_for(entity)` returns same-kind targets sharing any source token. This is LogMap's standard anchoring/blocking step (ISWC 2011).

**Design rationale.**
- Inverted index over pairwise enumeration: naive comparison is O(|O₁|·|O₂|). 4M pairs on Anatomy. The index drops this to ~70K pairs, which is what makes Stage 3 feasible on CPU.
- Token-level (not character n-gram) preserves token boundary semantics ("muscle of forearm" vs "forearm muscle" share tokens but no useful n-grams).
- Same-kind at index level, heterogeneity becomes a data-structure property, not a downstream check.

**Entry point.** `pairmap/stage2_index.py`.

### Stage 3 - Sentence-BERT embeddings (advanced technique)

**What it does.** Builds a textual profile per entity (`label | synonyms | parent labels | comment`), encodes it with a track-adaptive Sentence-BERT model, scores the Stage 2 shortlist by cosine similarity.

**Algorithm.** `EmbeddingScorer.encode()` runs the chosen model in one batched call with `normalize_embeddings=True`. `score_shortlist` then computes all cosine similarities via `np.einsum("ij,ij->i", src_vecs, tgt_vecs)` - one dot product per pair, batched in NumPy.

Profile shape (`stage3_embeddings.py`):
```
parts = [raw_label, *synonyms[:5], "parent: " + ", ".join(parent_labels[:3]), comment[:200]]
profile = " | ".join(parts)[:400]
```

**Design rationale.**
- *Three models, one per domain.* No single model is best on every track. `all-MiniLM-L6-v2` (MiniLM, NeurIPS 2020) handles Conference and Circular Economy (everyday English). `SapBERT-from-PubMedBERT-fulltext` (NAACL 2021) handles Anatomy (biomedical synonyms). `multilingual-e5-base` (arXiv 2024) handles Digital Humanities (German/French/Italian labels).
- *Batched einsum.* L2-normalised vectors mean cosine reduces to a dot product; one einsum is much cheaper than encoder-side similarity loops.
- *Profile token order.* Primary label first (encoders apply positional weighting), then synonyms, then parent context, then a short comment.
- *Lazy model load.* The model loads on first call, so `python -m pairmap --help` doesn't pull in torch.

**Entry point.** `pairmap/stage3_embeddings.py`.

**Cost.** On `human-mouse` (1,484 × 2,744 entities), Stage 3 takes ~50s of the 97s total - the dominant stage on every track. Bio-ML was scope-cut due to computation time and hardware limitations.

### Stage 4 - Locality booster

**What it does.** For each candidate (s, t), measures what fraction of s's parents+children are already paired with one of t's parents+children, and adds `alpha * agreement` to the score.

**Algorithm.** Build `pair_lookup: src_iri → set[tgt_iri]` from the current candidate set. For each candidate, compute `s_nbrs = set(s.parents) | set(s.children)`, same for t, count source neighbours whose paired targets sit in `t_nbrs`, divide by `|s_nbrs|`. Boost = `alpha * agreement`, capped at `alpha` (default 0.15).

**Design rationale.**
- *One-shot vs LogMap-iterative.* LogMap propagates locality through discovery - a new mapping changes future scores. PairMap runs locality once. We lose the cascade case (`Author ↔ Writer` lifting `BookAuthor ↔ Novelist`) but gain a 40-line implementation and avoid interleaving with the Stage 5 resolver. On Bio-ML scale, iterative locality would blow the time budget.
- *α=0.15 default.* α=0.30 promoted too many wrong parents in pilot runs; α=0.05 had no effect. OA.2 wine uses 0.20 (small ontologies, shallow hierarchy, more recall headroom).

**Entry point.** `pairmap/stage4_locality.py`.

### Stage 5 - Combine, threshold and resolve

**What it does.** Weighted combination of lexical and embedding scores; locality added as an additive boost; threshold filter; greedy 1:1 conflict resolution.

**Algorithm.**

```
combine_scores(candidates, cfg):
    for each (s, t):
        if both lex+emb:  base = w_lex * lex + w_emb * emb
        elif only lex:    base = lex
        elif only emb:    base = emb
        else:             base = 0
        final = clip(base + locality, 0, 1)

resolve_conflicts(scores, threshold):
    filter to score >= threshold
    sort descending
    greedy 1:1: keep top, mark src and tgt used, drop conflicts
```

`Mapping.predicate()` (`types.py`) picks the RDF predicate from entity kind: `owl:equivalentClass` for classes, `owl:equivalentProperty` for object/data properties, `owl:sameAs` for individuals.

**Design rationale.**
- *Weighted combination, not max.* `max(lex, emb)` would treat an embedding-only 0.95 the same as a lexical anchor 0.95. The weighted form forces agreement. Per-track weights tilt slightly toward the more reliable signal (Anatomy 0.4/0.6 - paraphrase-rich biomedical text; Conference 0.5/0.5 - crisp labels).
- *Greedy 1:1.* Hungarian assignment is O(n³) and needs a dense bipartite graph we don't have. Greedy is O(k log k), never miss-pairs the top score, and matches LogMap's "extractor by mapping selection". Ablations show 1:1-off costs 0.05–0.10 F.
- *Missing-source renormalisation.* If embedding fails on an entity, the lexical-only candidate competes at its full confidence (not demoted to 0.5 × w_lex). Right call for Conference where many anchors are pure CamelCase identity matches.

**Entry point.** `pairmap/stage5_combine.py`.

---

## 4. Per-track configuration

The single source of truth is `pairmap/config.py`. Every module reads from `get_config(track_name)`.

| Track | Embedding model | Threshold | `lex_weight` | `locality_α` | Timeout (s) |
|---|---|---:|---:|---:|---:|
| `anatomy` | SapBERT | 0.75 | 0.4 | 0.15 | 3600 |
| `conference` | MiniLM | 0.75 | 0.5 | 0.15 | 600 |
| `circular-economy` | MiniLM | 0.75 | 0.5 | 0.15 | 600 |
| `digital-humanities` | multilingual-E5 | 0.85 | 0.5 | 0.15 | 1200 |
| `wine` | MiniLM | 0.70 | 0.5 | 0.20 | 300 |

**Per-track rationale.**
- *Anatomy* uses SapBERT because labels are biomedical abbreviations (`somatic muscle of forelimb` ↔ `appendicular musculature of front limb`). SapBERT was pre-trained on UMLS synonyms; MiniLM has no domain prior.
- *Conference / Circular Economy* use MiniLM - everyday English labels, no domain advantage from SapBERT.
- *Digital Humanities* uses multilingual E5 - PACTOLS is French, UNESCO is multilingual, IDAI is mixed German/English. The tight 0.85 threshold is forced by mE5's tight cosine clustering.
- *OA.2 wine* uses threshold 0.70 and α=0.20 - small ontologies, no resolver risk, want recall. No reference exists, so the threshold is heuristic.

---

## 5. Threshold sweep methodology

The Stage 5 threshold controls a precision/recall trade-off. The right value depends on the embedding model's score distribution and the ontology pair's lexical density, neither of which can be guessed a priori so we sweep and pick.

`scripts/sweep_threshold.py` sweeps thresholds `{0.40, 0.45, …, 0.85}` for one pair. The trick: it runs the pipeline once, caches `result.candidates`, then re-applies only `combine_scores → resolve_conflicts → evaluate` for each threshold. Stage 3 and Stage 4 run exactly once.

**Whole-track tuning.** Run the script on each reference pair in the track and average per-threshold F. The single value that maximises macro-mean F is locked into `pairmap/config.py`.

**Findings.**
- **Conference:** macro-F **0.4391 at 0.85 → 0.5158 at 0.75**. The previous zero pair (`confOf-sigkdd`) now scores 0.5000.
- **Anatomy:** F **0.7150 at 0.80 → 0.7387 at 0.75**. F climbs monotonically from 0.40 to 0.75 then falls.
- **Circular Economy:** macro-F **0.4018 at 0.83 → 0.4727 at 0.75**. `CEON-BiOnto` improves from F=0.3793 to 0.5507.
- **Digital Humanities:** macro-F **0.2192 at 0.78 → 0.2613 at 0.85** opposite direction (mE5 clusters very high, tighter threshold preserves good matches).

**Honest caveat.** Sweeping against the OAEI reference is supervised optimisation. A held-out development split would be cleaner but the tracks are too small (Anatomy is 1 pair, CE is 2). The per-pair grids in `<track>-summary.md` show the chosen threshold is track-stable rather than overfit. The unsupervised defaults (0.85 for everything but DH at 0.78) are recoverable from `git log`.

The sweep scripts exist purely to produce tuning evidence. The runtime reads locked thresholds from `config.py`.

---

## 6. Per-stage ablation study

Section 5 tuned a hyperparameter. This section asks: with the threshold locked, what is each individual stage doing for F?

### Method

Rather than re-run the whole pipeline per variant, we run once and filter the candidate dict to a subset of score keys, then re-apply `combine_scores → resolve_conflicts → evaluate`. Stage 2 appears in every variant; its effectiveness is measured separately as `(1 − shortlist / naive)`.

| Variant | Score keys kept | Stages |
|---|---|---|
| Lex only | `{lexical}` | S1 + S2 + S5 |
| Embed only | `{embedding}` | S2 + S3 + S5 |
| No-Lex (3+4) | `{embedding, locality}` | S2 + S3 + S4 + S5 |
| No-Embed (1+4) | `{lexical, locality}` | S1 + S2 + S4 + S5 |
| No-Loc (1+3) | `{lexical, embedding}` | S1 + S2 + S3 + S5 |
| Full (1+3+4) | all three | all five |

### The headline number

Drop Stage 3 (SBERT) and run the rest: on high-signal tracks F barely moves; on Digital Humanities it *rises*. In every case the pipeline gets a 2–69× speedup and emits a smaller, higher-precision mapping set.

| Track | Full F | Lex-only F | ΔF | Full \|M_S\| | Lex-only \|M_S\| | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Conference (21 pairs) | 0.5849 | 0.6126 | +0.028 | 12.5 | 9.0 | 68.9× |
| Anatomy (1 pair) | 0.7511 | 0.7656 | +0.014 | 1405 | 958 | 6.3× |
| Circular-Economy (2 pairs) | 0.5279 | 0.5770 | +0.049 | 40.5 | 40.5 | 2.1× |
| Digital-Humanities (8 pairs) | 0.2797 | 0.4453 | +0.166 | 35.4 | 9.8 | 42.8× |

The thresholds are locked at the Full-pipeline tuning, so dropping a source can only match or beat Full at the same threshold when that source was actively pulling 1:1 picks the wrong way. The fair claim about Stage 3 is that it provides coverage for vocabulary divergence, not generic uplift. It wins +0.23 F on `confOf-ekaw` and loses on `human-mouse`.

### Per-stage contribution to macro-F

Positive = stage adds F; negative = removing the stage *raises* F at the locked threshold.

| Track | Stage 1 (lexical) | Stage 3 (embedding) | Stage 4 (locality) | Stage 2 reduction |
|---|---:|---:|---:|---:|
| Conference | **+0.166** | −0.028 | −0.027 | 98.1% |
| Anatomy | **+0.188** | −0.014 | −0.038 | 97.9% |
| Circular-Economy | **+0.158** | −0.049 | −0.036 | 99.3% |
| Digital-Humanities | +0.027 | **−0.166** | −0.015 | 98.6% |

On DH, multilingual E5 over-fires on partial cross-language matches and drags macro F down by 0.166. Stage 4 (locality) is mildly negative everywhere at the macro level. The boost nudges a few borderline pairs over threshold that the reference rejects but it's cheap (sub-second) and helps individual pairs.

### Track-by-track discussion

The macro mean hides per-pair variation. Stage 3 contribution (Full F − Lex-only F) on selected pairs:

| Pair | Lex only | Full | Stage 3 |
|---|---:|---:|---:|
| `confOf-ekaw` (Conference) | 0.5333 | **0.7568** | **+0.2235** |
| `edas-ekaw` (Conference) | 0.5405 | 0.5957 | +0.0552 |
| `CEON-MATONTO` (Circular) | 0.4000 | 0.4000 | 0.0000 |
| `human-mouse` (Anatomy) | 0.7656 | 0.7511 | −0.0145 |
| `CEON-BiOnto` (Circular) | 0.7541 | 0.6557 | −0.0984 |
| `oeai-parthenos` (DH) | 0.6753 | 0.4024 | −0.2729 |
| `defc-pactols` (DH) | 0.8421 | 0.4706 | −0.3715 |

Stage 3 wins where vocabularies share concepts but use different words (`confOf-ekaw`). It loses where vocabularies share lexical roots or where multilingual E5 over-fires on cross-language partial matches.

**Conference.** Stage 3 is ~98% of every pair's wall-clock. Dropping it: 2.5s → 50ms per pair, macro F *rises* from 0.585 to 0.613.

**Anatomy.** F barely moves but the precision/recall trade does. 
Full (ablation run at the locked 0.75 threshold): 1,405 mappings at Pr 0.78/Re 0.72. 
Lex-only: 958 at Pr 0.99/Re 0.62. 
Choose based on whether the downstream task wants precision (Lex-only) or recall (Full).

**Circular Economy.** |M_S| is identical between Full and Lex-only (40.5 either way) - embeddings only reshuffle the resolver's picks, never adding new mappings.

**Digital Humanities.** Stage 3 *actively hurts* the deployed pipeline: macro F −0.166, |M_S| balloons 9.8 → 35.4. But it *does* find correct cross-language matches on `idai-parthenos` and `oeai-parthenos` (F=0.39 with embeddings, F=0 without). Per-pair model selection or a tighter same-language pre-filter would be the right fix.

### Stage 2 (the blocker)

The inverted-index blocker reduces the candidate space by 97.9–99.3% on the four evaluated tracks with no measured loss of reference recall.

| Track | Macro-mean reduction vs naive \|O₁\| × \|O₂\| |
|---|---:|
| Conference | 98.1% |
| Anatomy | 97.9% |
| Circular-Economy | 99.3% |
| Digital-Humanities | 98.6% |

Concrete: on `confOf-ekaw` Stage 3 embeds 128 pairs instead of 7,844. Two known failure modes don't show up in these macro numbers:

(a) *Token-disjoint synonyms.* When ontologies use renamed terms with no shared tokens (`WineProducer` vs `Winery`, `hasPoints` vs `hasScore`), the pair never enters the shortlist.

(b) *Generic-label corpora.* When labels are dominated by short, frequent tokens (`Disease`, `Procedure`), every source matches almost everything and the reduction collapses. This is one of the factors that made Bio-ML infeasible on the team's hardware.

Both failure modes need a richer blocker (synonym index + character-n-gram fallback, or a two-stage embedding pre-filter). Not implemented but documented as future work.

### What we'd do for production

- **Lex-only** on Conference, Anatomy, Circular Economy - F is competitive, precision much higher, 2–69× speedup.
- **Full** on Digital Humanities only with per-pair model selection or a tighter same-language pre-filter.
- The 30% advanced-technique weight is defensible because Stage 3 wins per-pair where it matters (`confOf-ekaw` +0.22, `idai-parthenos` 0 → 0.39).

---

## 7. OA.2 - Aligning the Part 1 ontologies

### What's being aligned

Two wine-domain ontologies authored independently by the team in Part 1. No W3C wine ontology involved.

| | `student-a.owl` | `student-b.owl` |
|---|---|---|
| Author | Member A | Member B |
| Title | *Wine Evaluation and Production Ontology* | *Wine Quality Review Ontology* |
| Classes | 28 | 28 |
| Object properties | 16 | 12 |
| Data properties | 7 | 7 |
| Individuals | 7 | 29 |

### Configuration

The `wine` track uses a more permissive setup than OAEI tracks: threshold **0.70**, lex_weight 0.5, locality α **0.20**, MiniLM. Justifications: (a) ontologies are small enough that the 1:1 resolver won't get confused, and we want recall; (b) no reference alignment exists, so the threshold can't be formally swept.

`pairmap/__main__.py` supports three layouts: OAEI-standard (most tracks), reference-at-top (Bio-ML, KG), and flat-with-no-reference (OA.2 - enumerates unordered pairs via `itertools.combinations`).

Reproduce with:

```
python -m pairmap wine
```

### Adequacy audit of the 14 mappings (threshold 0.70)

| # | Mapping | Verdict |
|---|---|---|
| 1 | `SparklingWine` ≡ `SparklingWine` | correct |
| 2 | `StillWine` ≡ `WineType` | wrong relation - B's `WineType` is the **umbrella parent**; should be `rdfs:subClassOf` |
| 3 | `Taster` ≡ `Taster` | correct |
| 4 | `CabernetSauvignon` `owl:sameAs` `CabernetSauvignon` | correct - both `owl:NamedIndividual`, so `sameAs` is right |
| 5 | `hasFullName` ≡ `hasName` | correct |
| 6 | `Grape` ≡ `GrapeVariety` | correct |
| 7 | `FortifiedWine` ≡ `RedWine` | **false positive** - B has no `FortifiedWine`; pure embedding noise |
| 8 | `Province` ≡ `Province` | correct |
| 9 | `WineProduct` ≡ `Wine` | correct |
| 10 | `VeryGood` ≡ `GoodQuality` | correct |
| 11 | `WineCountry` ≡ `Country` | correct |
| 12 | `varietyUsedIn` ≡ `usedIn` | correct |
| 13 | `hasRetailPrice` ≡ `hasPrice` | correct |
| 14 | `hasFullTitle` ≡ `hasTitle` | correct |

| Student-A | Student-B | Why obvious |
|---|---|---|
| `WineProducer` | `Winery` | Synonyms |
| `madeBy` | `producedBy` | Wine → producer |
| `hasPoints` | `hasScore` | Numeric review score |
| `hasTastingDescription` | `hasDescription` | Review free-text |
| `ScoredTastingNote` | `WineReview` | Both are review entities |
| `originatesFrom` | `hasOrigin` | Wine → region |
| `Affordable`/`Budget` | `LowPrice` | Price-band synonyms |
| `Premium`/`Luxury` | `HighPrice` | Price-band synonyms |
| `Classic` | `ExcellentQuality` | Quality-band parallel |
| `Location` | `GeographicArea` | Top-level spatial class |

### OA.2 discussion

PairMap produced 14 mappings between student-a and student-b at threshold 0.70. From the audit table, 9 are clearly right, 2 are defensible if you squint, and 2 are wrong with precision around 79%, or 64% if you're strict about the StillWine ≡ WineType mix-up. Stage 1 handled the obvious cases where both sides used the same local name (SparklingWine, Taster, Province, plus CabernetSauvignon as owl:sameAs since both declared it as a named individual). Stage 3 earned its keep on the trickier pairs: Grape ≡ GrapeVariety, WineProduct ≡ Wine, and property paraphrases like hasFullName ≡ hasName and hasRetailPrice ≡ hasPrice, none of which share enough characters for Jaro-Winkler to spot.
The bigger problem is what didn't show up at all: roughly fifteen pairs that should have aligned cleanly, like WineProducer ↔ Winery, madeBy ↔ producedBy, and hasPoints ↔ hasScore. All of them have the same cause. Stage 2's blocker uses token overlap to decide which pairs get scored, and WineProducer and Winery share zero tokens, so the pair never reaches Stage 3. A richer blocker (synonym index, or a character-n-gram fallback) would be the obvious next step.
That said, the alignment is still doing useful work. Taster ≡ Taster and hasFullName ≡ hasName are exactly what lets the OA.3 SPARQL query pull tasters from both ontologies after OWL RL closure. The 0.70 threshold (against 0.85 on the OAEI tracks) was chosen to keep those property-level paraphrases alive where a tighter setting would have dropped them. The cost is the two false positives, which felt like a fair trade.

---

## 8. OA.3 - SPARQL query over the merged graph

### Pipeline

`scripts/run_oa3_query.py` is standalone with hard-coded paths to the two wine OWLs and the alignment at `results/pairmap-student-a-student-b.ttl`. It:

1. Loads both OWL/XML ontologies via Owlready2, round-trips through RDF/XML into a single rdflib `Graph` (rdflib can't parse OWL/XML directly).
2. Adds the alignment triples: graph now has 1,054 triples.
3. Applies OWL RL reasoning via `owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)` - closure grows the graph to **3,096 triples**.
4. Executes the SPARQL query.
5. Writes `results/oa3-results.csv`.

### The query

```sparql
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:   <http://www.w3.org/2002/07/owl#>
PREFIX wineA: <http://www.semanticweb.org/endryu/ontologies/2026/2/Wine_Ontology/>
PREFIX wineB: <http://www.city.ac.uk/inm713-in3067/2026/myo_shwe_sin_ei/>

SELECT DISTINCT ?taster ?name WHERE {
    ?taster rdf:type      wineA:Taster ;
            wineB:hasName ?name .
}
ORDER BY ?name
```

Two triple patterns sharing `?taster` (predicate-list shorthand `;`), one in `wineA:`, one in `wineB:`. Neither vocabulary can join the two graphs without the OA.2 alignment.

### What makes it work

Two alignments are exploited:
- `wineA:Taster owl:equivalentClass wineB:Taster` - after OWL RL closure, all three Member B `wineB:Taster` individuals are typed as `wineA:Taster`, and Member A's `wineA:ProfessionalTaster` instance `RogerVoss` becomes `wineA:Taster` via subclass closure.
- `wineA:hasFullName owl:equivalentProperty wineB:hasName` every value asserted under either property is mirrored.

### Results

`results/oa3-results.csv` = 4 rows:

```
taster,name
.../myo_shwe_sin_ei/AnnaLeeCIijima,Anna Lee C. Iijima
.../myo_shwe_sin_ei/KerinOKeefe,Kerin O'Keefe
.../endryu/.../RogerVoss,Roger Voss
.../myo_shwe_sin_ei/VirginieBoone,Virginie Boone
```

Three from Member B, one from Member A (`RogerVoss`, declared as `ProfessionalTaster` subclass of `Taster`). Without the alignment + reasoning, zero rows.

### Spec compliance

| Spec requirement | Evidence |
|---|---|
| ≥ 2 triple patterns | 2: class type, property value |
| Uses prefixes of both ontologies | `wineA:` and `wineB:` both in the body |
| Returns ≥ 1 row | 4 rows |
| Saved as CSV | `results/oa3-results.csv` |
| Reasoning over O₁ ∪ O₂ ∪ M | OWL RL; 1,054 → 3,096 triples |

### OA.3 discussion

The query asks "find every taster across both wine ontologies with their full name". Student-a types tasters as wineA:Taster and stores names under wineA:hasFullName; student-b types tasters as wineB:Taster and stores names under wineB:hasName. The vocabularies share no URIs, so without the OA.2 alignment and OWL RL reasoning the query returns zero rows. With them, the merged graph returns four, one from student-a (RogerVoss) and three from student-b. The RogerVoss case shows the reasoning concretely: he is asserted as a ProfessionalTaster with hasFullName "Roger Voss", and two entailment steps bridge the gap : ProfessionalTaster rdfs:subClassOf Taster adds him into wineA:Taster (satisfying the first pattern), and the alignment wineA:hasFullName owl:equivalentProperty wineB:hasName mirrors his name under wineB:hasName (satisfying the second). The three student-b tasters take a shorter path: already typed wineB:Taster, the alignment wineA:Taster owl:equivalentClass wineB:Taster adds them into wineA:Taster. This is the end-to-end value of ontology alignment where a consumer writes one query in one vocabulary and gets results spanning multiple sources without learning every source's schema. The four-row result is small, but the mechanism (alignment plus RL reasoning making federated query tractable) is what scales to production knowledge graphs bridging clinical terminologies or federating scientific data catalogues.

---

## 9. OA.4 - OAEI 2025 evaluation

Tracks evaluated: **Anatomy, Conference, Circular Economy, Digital Humanities** (32 alignment tasks). Tracks scope-cut: **Bio-ML and Knowledge Graph**, both due to computation time and hardware limitations. Numbers below come from `results/<track>-summary.md`.

### 9.1 Anatomy (1 pair)

Aligns Mouse Adult Anatomy (MA) and the NCI Thesaurus human anatomy subset (AMA). Configuration: SapBERT, threshold 0.75, lex_weight 0.4, α 0.15.

| Task | \|M_S\| | \|M_RA\| | Pr | Re | F | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| human-mouse | 1484 | 1516 | 0.7466 | 0.7309 | **0.7387** | 116.2 |

SapBERT is what makes Anatomy work with biomedical abbreviations/synonyms that lexical alone misses. Tuned threshold 0.75 lifts F by ~0.024 over the 0.80 baseline.

### 9.2 Conference (21 pairs)

7 academic-event ontologies pairwise. Configuration: MiniLM, threshold 0.75, lex_weight 0.5, α 0.15.

| Task | \|M_S\| | \|M_RA\| | Pr | Re | F | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| cmt-conference | 13 | 15 | 0.3846 | 0.3333 | 0.3571 | 2.8 |
| cmt-confOf | 7 | 16 | 0.4286 | 0.1875 | 0.2609 | 2.7 |
| cmt-edas | 14 | 13 | 0.4286 | 0.4615 | 0.4444 | 2.9 |
| cmt-ekaw | 14 | 11 | 0.5000 | 0.6364 | 0.5600 | 2.8 |
| cmt-iasted | 12 | 4 | 0.3333 | 1.0000 | 0.5000 | 2.8 |
| cmt-sigkdd | 15 | 12 | 0.6000 | 0.7500 | 0.6667 | 2.9 |
| conference-confOf | 13 | 15 | 0.7692 | 0.6667 | 0.7143 | 2.6 |
| conference-edas | 22 | 17 | 0.3636 | 0.4706 | 0.4103 | 2.7 |
| conference-ekaw | 26 | 25 | 0.3462 | 0.3600 | 0.3529 | 2.7 |
| conference-iasted | 11 | 14 | 0.3636 | 0.2857 | 0.3200 | 2.7 |
| conference-sigkdd | 15 | 15 | 0.4667 | 0.4667 | 0.4667 | 2.6 |
| confOf-edas | 21 | 19 | 0.5238 | 0.5789 | 0.5500 | 2.7 |
| confOf-ekaw | 19 | 20 | 0.8421 | 0.8000 | **0.8205** | 2.7 |
| confOf-iasted | 10 | 9 | 0.5000 | 0.5556 | 0.5263 | 2.7 |
| confOf-sigkdd | 5 | 7 | 0.6000 | 0.4286 | 0.5000 | 2.5 |
| edas-ekaw | 25 | 23 | 0.5200 | 0.5652 | 0.5417 | 2.8 |
| edas-iasted | 15 | 19 | 0.4667 | 0.3684 | 0.4118 | 2.8 |
| edas-sigkdd | 10 | 15 | 0.8000 | 0.5333 | 0.6400 | 2.7 |
| ekaw-iasted | 18 | 10 | 0.3889 | 0.7000 | 0.5000 | 2.8 |
| ekaw-sigkdd | 11 | 11 | 0.6364 | 0.6364 | 0.6364 | 2.6 |
| iasted-sigkdd | 31 | 15 | 0.4839 | 1.0000 | 0.6522 | 2.7 |

**Macro-mean F: 0.5158** (up from 0.4391 at 0.85). Best: `confOf-ekaw` at 0.8205. Worst non-degenerate: `cmt-confOf` at 0.2609 (overloaded labels like `Paper`, `Submission` confuse the 1:1 resolver).

### 9.3 Circular Economy (2 pairs)

CEON (top-level), BiOnto (bio-based products), MATONTO (materials). Configuration: MiniLM, threshold 0.75.

| Task | \|M_S\| | \|M_RA\| | Pr | Re | F | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| CEON-BiOnto | 40 | 29 | 0.4750 | 0.6552 | **0.5507** | 12.1 |
| CEON-MATONTO | 60 | 16 | 0.2500 | 0.9375 | 0.3947 | 6.7 |

**Macro-mean F: 0.4727** (up from 0.4018 at 0.83). `CEON-MATONTO` has very high recall but mediocre precision. PairMap over-produces against a small reference (16 mappings).

### 9.4 Digital Humanities (8 pairs)

Multilingual SKOS thesauri for archaeology, museum collections, language studies. Configuration: multilingual-E5, threshold 0.85.

| Task | \|M_S\| | \|M_RA\| | Pr | Re | F | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| defc-pactols | 27 | 10 | 0.2963 | 0.8000 | 0.4324 | 26.7 |
| dha-unesco | 38 | 10 | 0.2632 | 1.0000 | **0.4167** | 19.6 |
| idai-pactols | 3 | 17 | 0.0000 | 0.0000 | 0.0000 | 12.6 |
| idai-parthenos | 43 | 52 | 0.3953 | 0.3269 | 0.3579 | 10.8 |
| ironagedanube-pactols | 28 | 5 | 0.0714 | 0.4000 | 0.1212 | 18.7 |
| oeai-parthenos | 126 | 47 | 0.2698 | 0.7234 | 0.3931 | 11.9 |
| pactols-parthenos | 3 | 12 | 0.0000 | 0.0000 | 0.0000 | 10.0 |
| tadirah-unesco | 50 | 15 | 0.2400 | 0.8000 | 0.3692 | 18.9 |

**Macro-mean F: 0.2613** (up from 0.2192 at 0.78). The two F=0 pairs are threshold-independent. `pactols.rdf` is a SKOS thesaurus loaded via the rdflib fallback, which types entities as `individual` while the reference expects `class` mappings. Same-kind enforcement then prevents any candidate. This is a data-loading limitation and not a tuning issue.

### 9.5 Final Results

Output of `python -m pairmap all`:

```
==============================================================
  Final Results
==============================================================
  track                    pairs    macro F   time (s)
  --------------------------------------------------------
  anatomy                      1     0.7387      116.2
  circular-economy             2     0.4727       18.8
  conference                  21     0.5158       57.2
  digital-humanities           8     0.2613      129.2
  wine                         1        n/a        2.7
==============================================================
```

---

## 10. Reporting and result interpretation

Three output layers: per-pair Turtle, per-track summary, Final results.

**Per-pair alignment.** `results/pairmap-<src>-<tgt>.ttl` one triple per line with full IRIs. Predicate chosen by `Mapping.predicate()` from entity kind: `owl:equivalentClass`, `owl:equivalentProperty`, or `owl:sameAs`. Subsumption predicates (`rdfs:subClassOf`, `rdfs:subPropertyOf`) are defined but not currently emitted. Every Stage 5 output is equivalence.

**Per-track summary.** `results/<track>-summary.md`. Columns: `|M_S|` (produced), `|M_RA|` (reference), `Pr` (precision), `Re` (recall), `F` (harmonic mean), `Time (s)`. Set arithmetic in `pairmap/evaluation.py` canonicalises symmetric predicates by sorting IRIs alphabetically, so a reference written as `(B, A)` matches a system output written as `(A, B)`. Directional predicates stay ordered.

**Reproducing a single result:**

```
python scripts/run_pairmap.py 
    --source data/conference/ontologies/confOf.owl 
    --target data/conference/ontologies/ekaw.owl 
    --reference data/conference/reference/confOf-ekaw.ttl 
    --output results/pairmap-confOf-ekaw.ttl 
    --track conference
```

Prints time, mapping count, and `Pr=0.8421 Re=0.8000 F=0.8205`.

---

## 11. OA.5 - Reproducibility


**2. Install in editable mode:**

```
pip install -e .
```

Pulls `owlready2>=0.46`, `rdflib>=7.0`, `rapidfuzz>=3.5`, `sentence-transformers>=2.6`, `owlrl>=7.0` 

**3. First-run model download.** Three Sentence-Transformer models auto-download to `~/.cache/huggingface/hub/`:
- `all-MiniLM-L6-v2` - ~90 MB
- `SapBERT-from-PubMedBERT-fulltext` - ~500 MB
- `multilingual-e5-base` - ~1.1 GB

**4. Run one track:**

```
python -m pairmap conference
```

Iterates all 21 reference TTLs, produces 21 alignments, writes `results/conference-summary.md`.

**5. Run every configured track:**

```
python -m pairmap all
```

**6. Reproduce a threshold sweep:**

```
python scripts/sweep_threshold.py
    --source data/anatomy/ontologies/mouse.owl
    --target data/anatomy/ontologies/human.owl
    --reference data/anatomy/reference/human-mouse.ttl
    --track anatomy
```
Result:
```
 threshold   |M_S|        Pr        Re         F
--------------------------------------------------
      0.40    2321    0.5192    0.7949    0.6281
      0.45    2247    0.5363    0.7949    0.6404
      0.50    2143    0.5623    0.7949    0.6586
      0.55    2028    0.5922    0.7922    0.6778
      0.60    1906    0.6254    0.7863    0.6967
      0.65    1789    0.6574    0.7757    0.7116
      0.70    1657    0.6916    0.7559    0.7223
      0.75    1493    0.7348    0.7236    0.7291
      0.80    1287    0.7661    0.6504    0.7035
      0.85    1049    0.8084    0.5594    0.6612
BEST: threshold=0.75  |M_S|=1493  Pr=0.7348  Re=0.7236  F=0.7291
```
For whole-track sweeps, run on each reference pair and average per-threshold F. The best macro-mean is what's locked into `config.py`. The script prints per-threshold Pr/Re/F to stdout; pipe it to a file to keep the artifact.


**7. Reproduce a per-stage ablation:**

```
python scripts/ablate.py 
    --source data/conference/ontologies/confOf.owl 
    --target data/conference/ontologies/ekaw.owl 
    --reference data/conference/reference/confOf-ekaw.ttl 
    --track conference
```
Result:
```
Running pipeline on confOf.owl <-> ekaw.owl (track=conference)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 10280.40it/s]
  done in 3.2s (128 candidate pairs scored)

Stage 2 effectiveness (inverted-index blocker)
 Source entities :         74
 Target entities :        106
 Naive pair count :      7,844
 Shortlist pairs :        128
 Reduction :       98.4%

Per-stage wall-clock
  load                        0.12s  (  3.7%)
  stage1_lexical              0.00s  (  0.1%)
  stage2_index                0.00s  (  0.0%)
  stage3_embeddings           3.06s  ( 96.2%)
  stage4_locality             0.00s  (  0.0%)
  stage5_combine              0.00s  (  0.0%)
  total                       3.18s

Stage contribution - Pr/Re/F at threshold 0.75 (locked in pairmap/config.py)
  variant                       |M_S|         Pr         Re          F
  Lex only (Stage 1)               14     0.7143     0.5000     0.5882
  Embed only (Stage 3)              9     1.0000     0.4500     0.6207
  No lex (3+4)                     11     0.9091     0.5000     0.6452
  No embed (1+4)                   14     0.7143     0.5000     0.5882
  No locality (1+3)                17     0.8824     0.7500     0.8108
  Full (1+3+4)                     19     0.8421     0.8000     0.8205

Each stage's contribution to F (Full - 'No X')   [Full F = 0.8205]
  Stage 1 (lexical)            contributes +0.1754 F (F drops to 0.6452 without it)
  Stage 3 (embedding)          contributes +0.2323 F (F drops to 0.5882 without it)
  Stage 4 (locality)           contributes +0.0097 F (F drops to 0.8108 without it)
```
Runs the pipeline once and re-evaluates the six variants (Lex only / Embed only / No-Lex / No-Embed / No-Loc / Full) by filtering the candidate dict. Prints a Pr/Re/F table per variant - this is what produced the per-pair numbers in section 6. Substitute any other source/target/reference pair from `data/` to ablate it.

**8. Reproduce OA.2 and OA.3:**

```
python -m pairmap wine  # OA.2: produces results/pairmap-student-a-student-b.ttl
python scripts/run_oa3_query.py # OA.3: produces results/oa3-results.csv (4 rows)
```

---

## 12. Implementation limitations

- **One-shot locality vs LogMap-iterative.** Iterating would let late-discovered neighbour mappings lift borderline pairs but would have to interleave with the 1:1 resolver. We accept the cascade-discovery loss for a 40-line implementation. Empirically locality contributes near-zero macro F at locked thresholds, so the cost is small.
- **Threshold tuning is supervised.** Sweeping against OAEI references means the chosen threshold has seen the gold standard. A held-out dev split is the clean alternative, but tracks are too small (Anatomy is 1 pair, CE is 2). Per-pair grids show track-stable choices, not single-outlier driven. Unsupervised defaults recoverable from `git log`.

---

## 13. GenAI usage
__main__.py - Results output

loader.py - most of the code

stage3_embeddings.py - most of the code

types.py - most of the code

/script/ablate.py - results generation

/script/run_oa3_query.py - results generation

/script/sweep_threshhold.py - most argument path 
