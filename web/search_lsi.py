"""BM25 + LSI hybrid search (log-entropy weighting + TruncatedSVD)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from web.common import (
    EXTRACT_DIR, PROJECT_ROOT,
    build_result_list, extract_zip, load_pkl,
    minmax_norm, preprocess_text,
)


KEY = 'lsi'
ZIP_PATH = PROJECT_ROOT / 'lsi_models.zip'
MODEL_DIR = EXTRACT_DIR / 'lsi'
SENTINEL = 'lsi_model.pkl'

META = {
    'key': KEY,
    'title': 'BM25 + LSI',
    'subtitle': 'Log-entropy weighting → TruncatedSVD (Deerwester 1990)',
    'semantic_label': 'LSI',
    'color': '#388e3c',
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
            'bm25':            load_pkl(MODEL_DIR, 'bm25_model.pkl'),
            'count':           load_pkl(MODEL_DIR, 'count_vectorizer.pkl'),
            'global_weights':  load_pkl(MODEL_DIR, 'global_weights.pkl'),
            'lsi':             load_pkl(MODEL_DIR, 'lsi_model.pkl'),
            'lsi_matrix_norm': normalize(load_pkl(MODEL_DIR, 'lsi_matrix.pkl')),
            'documents':       load_pkl(MODEL_DIR, 'documents.pkl'),
            'titles':          load_pkl(MODEL_DIR, 'titles.pkl'),
            'urls':            load_pkl(MODEL_DIR, 'urls.pkl'),
            'config':          load_pkl(MODEL_DIR, 'config.pkl'),
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

    # Log-entropy weighting cho query: log(1+tf) * G  → SVD.transform
    q_count = s['count'].transform([pq]).astype(np.float64)
    q_count.data = np.log1p(q_count.data)
    q_weighted = q_count @ sp.diags(s['global_weights'])
    q_lsi = normalize(s['lsi'].transform(q_weighted))
    sem_raw = cosine_similarity(q_lsi, s['lsi_matrix_norm']).flatten()

    return build_result_list(
        bm_n=minmax_norm(bm25_raw),
        sem_n=minmax_norm(sem_raw),
        alpha=alpha,
        top_k=top_k,
        titles=s['titles'], urls=s['urls'], documents=s['documents'],
        query=query,
    )
