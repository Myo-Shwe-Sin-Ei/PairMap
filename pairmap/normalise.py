# Label normalisation: lowercase, split CamelCase / snake_case, strip
# punctuation, drop English stopwords


from __future__ import annotations

import re
from functools import lru_cache

# Minimal stopword list — keep small so we don't drop signal on small ontologies.
stopwords = frozenset({
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by",
    "and", "or", "with", "is", "are", "was", "were", "be", "as",
})

# Split CamelCase
camel_replace = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
punct_replace = re.compile(r"[^\w\s]")
whitespace_replace = re.compile(r"\s+")


@lru_cache(maxsize=100_000)
def normalise(text: str) -> str:
    # Return a normalised string: lowercased, split, stop-stripped

    if not text:
        return ""
    # Replace underscores and dashes with spaces, then split CamelCase
    s = text.replace("_", " ").replace("-", " ")
    s = camel_replace.sub(" ", s)
    s = punct_replace.sub(" ", s)
    s = s.lower()
    tokens = [t for t in whitespace_replace.split(s) if t and t not in stopwords]
    return " ".join(tokens)


def tokenise(text: str) -> tuple[str, ...]:
    # Normalise and split into tokens. Used by the inverted index
    norm = normalise(text)
    return tuple(norm.split()) if norm else ()


def fragment_from_iri(iri: str) -> str:
    # Pull the local name from an IRI for fallback labelling
    for sep in ("#", "/"):
        if sep in iri:
            return iri.rsplit(sep, 1)[1]
    return iri
