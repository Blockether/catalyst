"""Vectorizers for term extraction in knowledge processing."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class KnowledgeVectorizers:
    """Encapsulates vectorizers for keyword and acronym extraction.

    This class creates and manages two CountVectorizer instances:
    - One for extracting multi-word keywords (n-grams)
    - One for extracting acronyms (uppercase patterns)
    """

    def __init__(self, keywords_min_df: int, acronyms_min_df: int):
        """Initialize vectorizers with specified minimum document frequencies.

        Args:
            keywords_min_df: Minimum document frequency for keywords
            acronyms_min_df: Minimum document frequency for acronyms
        """
        self._keywords_min_df = keywords_min_df
        self._acronyms_min_df = acronyms_min_df

    def keywords_vectorizer(self) -> CountVectorizer:
        """Get the keywords vectorizer."""
        return CountVectorizer(
            stop_words="english",
            strip_accents="ascii",
            ngram_range=(2, 4),
            min_df=self._keywords_min_df,
            analyzer="word",
            dtype=np.int64,
        )

    def acronyms_vectorizer(self) -> CountVectorizer:
        """Get the acronyms vectorizer."""
        return CountVectorizer(
            stop_words=None,
            strip_accents="ascii",
            ngram_range=(1, 1),
            min_df=self._acronyms_min_df,
            token_pattern=r"\b[A-Z]{2,}([_/-][A-Z]+)?\b",
            lowercase=False,
            dtype=np.int64,
        )
