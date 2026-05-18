"""Shared utilities: preprocessing, normalization, top-k, paths."""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pyvi import ViTokenizer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EXTRACT_DIR = PROJECT_ROOT / 'saved_models'


# --- Vietnamese preprocessing (phải khớp notebook) -------------------------
STOPWORDS = set([
    'và', 'của', 'một', 'những', 'các', 'được', 'có', 'là', 'cho', 'khi',
    'đã', 'sẽ', 'nếu', 'như', 'tại', 'vào', 'ra', 'mà', 'này', 'kia', 'đó',
    'ấy', 'thì', 'rằng', 'để', 'với', 'không', 'cũng', 'lại', 'rất', 'nên',
    'bị', 'do', 'về', 'theo', 'qua', 'trong', 'ngoài', 'trên', 'dưới', 'bởi',
    'cả', 'cùng', 'đang', 'đây', 'đến', 'đi', 'hơn', 'kể', 'lên', 'mới',
    'ngay', 'nữa', 'rồi', 'sau', 'thôi', 'thưa',
])

_VI_CHARS = (
    'a-z0-9'
    'àáạảãâầấậẩẫăằắặẳẵ'
    'èéẹẻẽêềếệểễ'
    'ìíịỉĩ'
    'òóọỏõôồốộổỗơờớợởỡ'
    'ùúụủũưừứựửữ'
    'ỳýỵỷỹđ'
)
_RE_HTML = re.compile(r'<[^>]+>')
_RE_URL = re.compile(r'http\S+|www\S+|https\S+')
_RE_CHARS = re.compile(rf'[^{_VI_CHARS}\s_]')
_RE_WS = re.compile(r'\s+')


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = _RE_HTML.sub(' ', text)
    text = _RE_URL.sub(' ', text)
    text = ViTokenizer.tokenize(text).lower()
    text = _RE_CHARS.sub(' ', text)
    text = _RE_WS.sub(' ', text).strip()
    return ' '.join(t for t in text.split() if t not in STOPWORDS and len(t) > 1)


def minmax_norm(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


def truncate(s: Any, length: int = 240) -> str:
    s = str(s)
    return s if len(s) <= length else s[:length] + '…'


# --- Keyword highlighting --------------------------------------------------
def query_terms(query: str) -> list[str]:
    """Build danh sách term để highlight: gồm từ raw query + token sau pyvi.

    Vd: 'trí tuệ nhân tạo' → ['trí', 'tuệ', 'nhân', 'tạo', 'trí_tuệ_nhân_tạo']
    """
    if not isinstance(query, str):
        return []
    raw = [w.lower() for w in re.split(r'\s+', query.strip()) if len(w) > 1]
    seg = [t for t in preprocess_text(query).split() if len(t) > 1]
    terms = list(dict.fromkeys(raw + seg))   # dedupe, giữ thứ tự
    # sắp xếp theo độ dài giảm dần để match cụm dài trước
    terms.sort(key=len, reverse=True)
    return terms


def highlight_html(text: str, terms: list[str]) -> str:
    """Escape text → wrap mỗi term bằng <mark> → đổi \\n thành <br>."""
    if not text:
        return ''
    escaped = html.escape(str(text))
    if terms:
        pattern = '|'.join(re.escape(t) for t in terms if t)
        if pattern:
            rx = re.compile(rf'({pattern})', flags=re.IGNORECASE)
            escaped = rx.sub(r'<mark>\1</mark>', escaped)
    return escaped.replace('\n', '<br>')


def truncate_around_match(text: str, terms: list[str], window: int = 240) -> str:
    """Cắt đoạn text quanh vị trí đầu tiên match keyword (nếu có).

    Giúp snippet hiển thị phần liên quan thay vì luôn lấy 240 ký tự đầu.
    """
    if not isinstance(text, str):
        return ''
    if not terms:
        return truncate(text, window)
    rx = re.compile('|'.join(re.escape(t) for t in terms if t), flags=re.IGNORECASE)
    m = rx.search(text)
    if not m:
        return truncate(text, window)
    start = max(0, m.start() - window // 4)
    end = min(len(text), start + window)
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(text) else ''
    return prefix + text[start:end] + suffix


# --- Zip extraction --------------------------------------------------------
def extract_zip(zip_path: Path, target_dir: Path, sentinel: str) -> None:
    """Giải nén zip vào target_dir nếu sentinel chưa tồn tại. Raise nếu zip không tồn tại."""
    if not zip_path.exists():
        raise FileNotFoundError(f'Không tìm thấy {zip_path}')
    target_dir.mkdir(parents=True, exist_ok=True)
    if (target_dir / sentinel).exists():
        return
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)


def load_pkl(model_dir: Path, name: str):
    return joblib.load(model_dir / name)


# --- Hybrid scoring (dùng chung cho cả 3 model) ----------------------------
def build_result_list(
    *,
    bm_n: np.ndarray,
    sem_n: np.ndarray,
    alpha: float,
    top_k: int,
    titles, urls, documents,
    query: str = '',
):
    """Kết hợp tuyến tính BM25 & semantic rồi build list result dicts.

    Mỗi result kèm `title_html` & `snippet_html` đã tô đậm keyword bằng <mark>.
    """
    hybrid = alpha * bm_n + (1.0 - alpha) * sem_n
    idx = top_k_indices(hybrid, top_k)
    terms = query_terms(query)
    out = []
    for rank, i in enumerate(idx, 1):
        i_int = int(i)
        url = urls[i_int] if i_int < len(urls) else ''
        title = (titles[i_int] if i_int < len(titles) else '') or '(no title)'
        doc = documents[i_int] if i_int < len(documents) else ''
        out.append({
            'rank': rank,
            'idx': i_int,
            'title': title,
            'title_html': highlight_html(title, terms),
            'url': url if url and url != 'N/A' else '',
            'snippet': doc,
            'snippet_html': highlight_html(doc, terms),
            'bm25': float(bm_n[i_int]),
            'semantic': float(sem_n[i_int]),
            'hybrid': float(hybrid[i_int]),
        })
    return out


# --- Example queries (giống test queries trong notebook) -------------------
EXAMPLE_QUERIES = [
    'thời tiết mưa bão nhiệt độ dự báo',
    'chứng khoán cổ phiếu thị trường đầu tư',
    'bóng đá việt nam cầu thủ đội tuyển',
    'công nghệ AI trí tuệ nhân tạo máy học',
    'giá vàng hôm nay tăng mạnh',
    'dịch covid vaccine tiêm chủng',
    'xe điện vinfast ô tô sản xuất',
    'ca sĩ nhạc mới album âm nhạc',
    'tội phạm ma túy bắt giữ',
    'tai nạn giao thông xe hỏng',
    'thể thao olympic vàng huy chương',
    'du lịch resort khách sạn nghỉ dưỡng',
    'giáo dục trường học sinh viên',
]
