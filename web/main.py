"""
FastAPI Demo: 1 trang duy nhất, radio chọn 1 trong 3 hybrid models.
Chạy:  uvicorn web.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web import search_lda, search_lsa, search_lsi
from web.common import EXAMPLE_QUERIES, truncate


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title='Vietnamese News Search — Hybrid IR Demo')
app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')
templates = Jinja2Templates(directory=BASE_DIR / 'templates')
templates.env.filters['truncate'] = lambda s, length=80: truncate(s, length)


MODELS = {
    'lsa': search_lsa,
    'lsi': search_lsi,
    'lda': search_lda,
}


def parse_alpha(s: str | None, default: float) -> float:
    if not s:
        return default
    try:
        return max(0.0, min(1.0, float(s)))
    except ValueError:
        return default


def parse_top_k(s: str | None, default: int = 10) -> int:
    if not s:
        return default
    try:
        return max(1, min(50, int(s)))
    except ValueError:
        return default


def parse_model(s: str | None, default: str = 'lsa') -> str:
    return s if s in MODELS else default


@app.get('/', response_class=HTMLResponse)
async def home(
    request: Request,
    q: str = '',
    model: str = 'lsa',
    alpha: str = '',
    k: str = '',
):
    model_key = parse_model(model)
    top_k = parse_top_k(k, default=10)
    mod = MODELS[model_key]

    cfg = mod.get_config()
    default_alpha = float(cfg.get('best_alpha', 0.5)) if cfg else 0.5
    alpha_val = parse_alpha(alpha, default=default_alpha)

    results = mod.search(q, alpha=alpha_val, top_k=top_k) if q else []
    load_error = mod.get_error()

    status = {key: (MODELS[key].get_error() is None and MODELS[key].ZIP_PATH.exists())
              for key in MODELS}

    context = {
        'queries':         EXAMPLE_QUERIES,
        'selected_query':  q,
        'model_key':       model_key,
        'models_meta':     {key: MODELS[key].META for key in MODELS},
        'meta':            mod.META,
        'alpha':           alpha_val,
        'top_k':           top_k,
        'results':         results,
        'config':          cfg,
        'load_error':      load_error,
        'status':          status,
    }
    return templates.TemplateResponse(request, 'index.html', context)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
