# Vietnamese News Retrieval — Hybrid BM25 + {LSI, LDA, NMF}

Hệ thống tìm kiếm tin tức tiếng Việt trên 20k bài VietNews, kết hợp **BM25** (lexical) với 3 mô hình semantic khác nhau.

## Cấu trúc

```
ir/
├── notebook/
│   ├── lsi.ipynb           # BM25 + LSI  (Log-Entropy + TruncatedSVD - Deerwester 1990)
│   ├── lda.ipynb           # BM25 + LDA  (CountVectorizer + LatentDirichletAllocation)
│   └── nmf.ipynb           # BM25 + NMF  (TF-IDF + Non-negative Matrix Factorization)
│
├── web/
│   ├── main.py             # FastAPI app, 1 route GET /
│   ├── common.py           # preprocess, scoring, highlight
│   ├── search_lsi.py       # load lsi_models.zip + search()
│   ├── search_lda.py       # load lda_models.zip + search()
│   ├── search_nmf.py       # load nmf_models.zip + search()
│   ├── templates/index.html
│   └── static/css/style.css
│
├── scripts/
│   └── load_data.py        # tải VietNews 20k → data/vietnews_20k.csv
│
├── data/
│   └── vietnews_20k.csv    # 20k tin được cache offline
│
└── requirements.txt
```

## Cài đặt

```bash
git clone https://github.com/anhnt-24/vietnews-retrieval.git
cd vietnews-retrieval
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

### Bước 1 — train models (lần đầu)

Mở `notebook/lsi.ipynb`, `notebook/lda.ipynb`, `notebook/nmf.ipynb` rồi `Run All` từng notebook.
Mỗi notebook sẽ tạo `saved_models/<model>_models.zip` ở thư mục gốc.

Sau đó move 3 zip ra root project:

```bash
mv saved_models/lsi_models.zip lsi_models.zip
mv saved_models/lda_models.zip lda_models.zip
mv saved_models/nmf_models.zip nmf_models.zip
```

### Bước 2 — chạy web demo

```bash
uvicorn web.main:app --reload
```

Mở http://127.0.0.1:8000

Trên trang demo có:
- **Radio** chọn 1 trong 3 model (BM25+LSI / BM25+LDA / BM25+NMF)
- **Slider α**: trọng số BM25 (0 = chỉ semantic, 1 = chỉ BM25), mặc định lấy `best_alpha` đã tìm được trong notebook
- **Top-k**: số kết quả trả về
- Mỗi kết quả hiển thị **3 điểm** (BM25, Semantic, Hybrid) và **tô đậm keyword** trong title + snippet

## Cách hoạt động

### Công thức hybrid

$$\text{score}(d, q) = \alpha \cdot \widetilde{BM25}(d, q) + (1 - \alpha) \cdot \widetilde{\text{Semantic}}(d, q)$$

Trong đó `·̃` là điểm đã min-max chuẩn hóa về `[0, 1]` trên toàn corpus.

### Khác biệt 3 model

| Notebook | Vector hóa | Semantic model | Sweep |
|---|---|---|---|
| `lsi.ipynb` | Log-entropy weighting (Deerwester 1990) | TruncatedSVD | `n_components ∈ {50, 100, 200, 300}` |
| `lda.ipynb` | Count matrix | LatentDirichletAllocation | `n_topics ∈ {10, 20, 50, 100}` |
| `nmf.ipynb` | TF-IDF (sublinear) | Non-negative Matrix Factorization (W,H ≥ 0) | `n_components ∈ {50, 100, 200, 300}` |

### Đánh giá

Cả 3 notebook chia sẻ chung:
- **20.000 tài liệu** VietNews
- **15 test queries** đa lĩnh vực (weather/stock/sports/tech/commodity/health/automotive/entertainment/crime/accident/olympic/travel/education/politics/fashion)
- **Heuristic relevance**: doc liên quan ⇔ chứa ≥ 2 keyword
- **Grid search (α × k)** với 6 metrics: Precision, Recall, F1, MAP, MRR, NDCG @k
- Tối ưu bằng cache scores per-query + `np.argpartition`

Kết quả tự động lưu trong `saved_models/<model>_models.zip` cùng `grid_results.pkl` và `config.pkl` (chứa `best_alpha`, `best_k`).

## Dependencies

- Python ≥ 3.10
- `fastapi`, `uvicorn`, `jinja2`
- `scikit-learn`, `numpy`, `scipy`, `pandas`
- `rank_bm25`, `pyvi`
- `datasets` (HuggingFace, chỉ để tải lần đầu nếu chưa có CSV)
- `matplotlib`, `seaborn` (cho notebook)
- `joblib`

## Lưu ý

- 3 file `*_models.zip` (50–92 MB mỗi cái) được `.gitignore` — phải chạy notebook để tạo
- `data/vietnews_20k.csv` đã được commit, tránh phụ thuộc HuggingFace khi chạy lại
- Web app `auto-extract` zip vào `saved_models/<model>/` lần đầu load
