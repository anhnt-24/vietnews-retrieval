# Slidev — Báo cáo đồ án IR

Deck Slidev cho báo cáo *Vietnamese News Retrieval — Hybrid BM25 + {LSA, LSI, LDA}*.

## Cài đặt

```bash
cd slides
npm install
```

## Chạy

```bash
npm run dev          # mở localhost:3030 (live reload)
npm run build        # export SPA tĩnh ra dist/
npm run export       # export PDF (cần playwright)
```

Khi `npm run export` báo thiếu playwright:

```bash
npm install -D playwright-chromium
npm run export
```

## Cấu trúc

```
slides/
├── slides.md              # toàn bộ nội dung deck (18 slide chính)
├── package.json
└── public/                # ảnh embed (heatmap, bar chart, demo screenshot)
    ├── lsa_1.png ... lsa_3.png
    ├── lsi_1.png ... lsi_3.png
    ├── lda_1.png ... lda_4.png
    └── compare_all.png    # bar chart so sánh 3 model
```

## Cấu trúc 18 slide

1. **Cover** — Tên đề tài
2. **Mục lục** — 4 phần lớn
3–4. **Giới thiệu** — Bài toán & mục tiêu
5–9. **Cơ sở lý thuyết** — BM25, LSA, LSI, LDA, công thức hybrid
10–16. **Thực nghiệm** — Dữ liệu, ground-truth, grid search, 3 kết quả model, web demo
17–18. **Kết luận** — So sánh 3 model, hạn chế & hướng phát triển

(+ 4 section divider — không tính vào 18 slide)

## Sửa nội dung

Toàn bộ deck nằm trong `slides.md`. Mỗi slide cách nhau bởi `---`. Cú pháp:

- Công thức: KaTeX `$$...$$` (display) hoặc `$...$` (inline)
- Layout 2 cột: `<div class="grid grid-cols-2 gap-4">`
- Hiệu ứng từng bước: `<v-clicks>...</v-clicks>`
- Section divider: thêm frontmatter `layout: section`

## Số liệu trong deck

Tất cả số liệu (NDCG, MAP, n_components, perplexity...) đều **trích trực tiếp** từ output của 3 notebook trong `notebook/`. Nếu chạy lại notebook và số đổi, cần update lại trong `slides.md`.
