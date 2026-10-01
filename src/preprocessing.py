"""Text preprocessing helpers for the fake-news detection pipeline.

Steps mirror the project methodology: text cleaning (punctuation/stopword
removal), tokenization, and TF-IDF vectorization. Stopword removal and
tokenization are handled by scikit-learn's TfidfVectorizer (english
stopwords); clean_text() normalizes raw article text first.
"""

import re


def clean_text(text: str) -> str:
    """Lowercase text and strip URLs, numbers, and punctuation.

    Returns a whitespace-normalized string of alphabetic tokens.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters only
    return re.sub(r"\s+", " ", text).strip()


def build_vectorizer(max_features: int = 5000):
    """TF-IDF vectorizer with English stopwords removed.

    Args:
        max_features: cap on vocabulary size (most frequent terms kept).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(stop_words="english", max_features=max_features)


def prepare_features(texts, vectorizer=None, max_features: int = 5000):
    """Clean texts and convert to a TF-IDF matrix.

    Fits a new vectorizer when none is supplied (training); otherwise
    transforms with the fitted one (inference/test).
    """
    cleaned = [clean_text(t) for t in texts]
    if vectorizer is None:
        vectorizer = build_vectorizer(max_features=max_features)
        matrix = vectorizer.fit_transform(cleaned)
    else:
        matrix = vectorizer.transform(cleaned)
    return matrix, vectorizer
