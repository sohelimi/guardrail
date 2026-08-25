"""Feature extraction for the model bake-off.

Two complementary text views:
  * WORD n-grams  -> capture intent ("ignore previous instructions")
  * CHAR n-grams  -> robust to obfuscation (leetspeak, spacing, casing),
                     because char 3-5 grams still overlap after mutation.

The primary model unions both so it reads intent *and* survives obfuscation.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        strip_accents="unicode", lowercase=True,
    )


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
        lowercase=True,
    )


def union_vectorizer() -> FeatureUnion:
    return FeatureUnion([("word", word_vectorizer()), ("char", char_vectorizer())])
