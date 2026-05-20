"""BM25 + NMF hybrid search (TF-IDF → Non-negative Matrix Factorization)."""

from __future__ import annotations

from typing import Any

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from web.common import (
    EXTRACT_DIR, PROJECT_ROOT,
    build_result_list, extract_zip, load_pkl,
    minmax_norm, preprocess_text,
)


KEY = 'nmf'
ZIP_PATH = PROJECT_ROOT / 'nmf_models.zip'
MODEL_DIR = EXTRACT_DIR / 'nmf'
SENTINEL = 'nmf_model.pkl'

META = {
    'key': KEY,
    'title': 'BM25 + NMF',
    'subtitle': 'TF-IDF → Non-negative Matrix Factorization (W,H ≥ 0)',
    'semantic_label': 'NMF',
    'color': '#00897b',
}

_state: dict[str, Any] | None = None
_load_error: str | None = None


def load() -> dict[str, Any] | None:
    global _state, _load_error
    if _state is not None:
        return _state
    if _load_error is not None:
        return None
    try:
        extract_zip(ZIP_PATH, MODEL_DIR, SENTINEL)
        _state = {
            'bm25':           load_pkl(MODEL_DIR, 'bm25_model.pkl'),
            'tfidf':          load_pkl(MODEL_DIR, 'tfidf_vectorizer.pkl'),
            'nmf':            load_pkl(MODEL_DIR, 'nmf_model.pkl'),
            'doc_topic_norm': normalize(load_pkl(MODEL_DIR, 'doc_topic_dist.pkl')),
            'documents':      load_pkl(MODEL_DIR, 'documents.pkl'),
            'titles':         load_pkl(MODEL_DIR, 'titles.pkl'),
            'urls':           load_pkl(MODEL_DIR, 'urls.pkl'),
            'config':         load_pkl(MODEL_DIR, 'config.pkl'),
        }
        return _state
    except Exception as exc:
        _load_error = f'{type(exc).__name__}: {exc}'
        return None


def get_error() -> str | None:
    return _load_error


def get_config() -> dict[str, Any]:
    s = load()
    return (s.get('config') if s else {}) or {}


def search(query: str, alpha: float, top_k: int):
    s = load()
    if s is None:
        return []
    pq = preprocess_text(query)
    tokens = pq.split()
    if not tokens:
        return []

    bm25_raw = s['bm25'].get_scores(tokens)

    q_tfidf = s['tfidf'].transform([pq])
    q_topic = normalize(s['nmf'].transform(q_tfidf))
    sem_raw = cosine_similarity(q_topic, s['doc_topic_norm']).flatten()

    return build_result_list(
        bm_n=minmax_norm(bm25_raw),
        sem_n=minmax_norm(sem_raw),
        alpha=alpha,
        top_k=top_k,
        titles=s['titles'], urls=s['urls'], documents=s['documents'],
        query=query,
    )
