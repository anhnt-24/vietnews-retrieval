---
theme: seriph
title: Vietnamese News Retrieval - Hybrid BM25 + LSA, LSI, LDA
info: Báo cáo đồ án Truy hồi thông tin
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Truy hồi tin tức tiếng Việt

## Hybrid BM25 + LSA · LSI · LDA

<br>

Báo cáo đồ án môn **Truy hồi thông tin (IR)**

<br>

<div class="text-sm opacity-80">
Sinh viên thực hiện · MSSV · Lớp<br>
Giảng viên hướng dẫn: ...<br>
Tháng 05 / 2026
</div>

<div class="abs-br m-6 text-xs opacity-60">
Dataset: VietNews 20k · 3 notebook + FastAPI demo
</div>

---
transition: fade-out
---

# Mục lục

<div class="flex flex-col gap-6 mt-12 text-2xl">

<div class="flex items-center gap-4">
  <span class="text-4xl font-bold opacity-30">01</span>
  <span class="font-semibold">Giới thiệu</span>
</div>

<div class="flex items-center gap-4">
  <span class="text-4xl font-bold opacity-30">02</span>
  <span class="font-semibold">Cơ sở lý thuyết</span>
</div>

<div class="flex items-center gap-4">
  <span class="text-4xl font-bold opacity-30">03</span>
  <span class="font-semibold">Thực nghiệm</span>
</div>

<div class="flex items-center gap-4">
  <span class="text-4xl font-bold opacity-30">04</span>
  <span class="font-semibold">Kết luận</span>
</div>

</div>

---

# 1. Giới thiệu

<div class="grid grid-cols-2 gap-6">

<div>

### Bối cảnh

- Tin tức tiếng Việt khối lượng lớn → cần **search engine** chính xác
- BM25 là chuẩn vàng của IR lexical (sparse retrieval)
- Nhưng BM25 **chỉ khớp chuỗi ký tự** — không hiểu ngữ nghĩa

### Vấn đề

Query `"ô tô"` **miss** bài viết toàn dùng `"xe hơi"`<br>
Query `"AI"` **miss** bài viết viết `"trí tuệ nhân tạo"`

→ Recall thấp với query có từ đồng nghĩa

</div>

<div>

### Hướng tiếp cận

Kết hợp **lexical + semantic** dưới dạng hybrid tuyến tính:

$$\text{score} = \alpha \cdot \text{BM25} + (1-\alpha) \cdot \text{Semantic}$$

<br>

So sánh **3 mô hình semantic kinh điển**:

| Model | Phương pháp |
|---|---|
| **LSA** | TF-IDF + SVD |
| **LSI** | Log-Entropy + SVD |
| **LDA** | Topic modeling sinh |

</div>

</div>

---
layout: section
---

# 2. Cơ sở lý thuyết

---

# 2.1. BM25

### Giới thiệu

BM25 (Best Match 25) là **mô hình xếp hạng từ vựng kinh điển** trong IR, mở rộng từ TF-IDF bằng cách tính đến độ dài tài liệu và bão hòa tần suất từ.

### Công thức Okapi BM25

<div class="text-sm">

$$\text{BM25}(d,q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf_{t,d}\,(k_1+1)}{tf_{t,d} + k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)}, \quad \text{IDF}(t) = \log\frac{N - n_t + 0.5}{n_t + 0.5}$$

</div>

<div class="text-sm opacity-80 mt-2">

- $tf_{t,d}$ — tần suất từ $t$ trong doc $d$
- $|d|$, $\text{avgdl}$ — độ dài doc / trung bình các doc
- $k_1, b$ — hyperparameter
- $N$ — tổng số doc · $n_t$ — số doc chứa từ $t$

</div>

---

# 2.2. LSA

### Ý tưởng

Giả định những từ **hay xuất hiện cùng nhau** trong các tài liệu thì có nghĩa gần nhau. Bằng cách gom các từ đồng nghĩa thành các "khái niệm ẩn", LSA giúp tìm đúng tài liệu kể cả khi query và tài liệu **dùng từ khác nhau nhưng cùng ý**.

### Bước 1: Vector hóa TF-IDF

<div class="mt-4">

$$w_{t,d} = (1 + \log tf_{t,d}) \cdot \log\frac{N}{df_t}$$

</div>

---

# 2.2. LSA

### Bước 2: Truncated SVD

<div class="mt-2">

$$A_{m \times n} \approx U_k \, \Sigma_k \, V_k^T$$

</div>

<div class="text-sm">

- $A$ — ma trận **term–document** gốc ($m$ từ × $n$ tài liệu)
- $U_k$ — ma trận **từ × khái niệm** ($m \times k$): mỗi dòng cho biết một từ liên quan đến $k$ khái niệm ẩn ra sao
- $\Sigma_k$ — ma trận đường chéo chứa **độ quan trọng** của $k$ khái niệm (giá trị suy biến giảm dần)
- $V_k^T$ — ma trận **khái niệm × tài liệu** ($k \times n$): mỗi cột cho biết một tài liệu phân bố trên $k$ khái niệm ra sao
- $k$ — số chiều giữ lại (rất nhỏ so với $m, n$ → khử nhiễu, giữ ngữ nghĩa chính)

</div>

### Bước 3: Cosine similarity trong không gian ẩn

<div class="mt-2">

$$\text{sim}_{\text{LSA}}(d, q) = \frac{\vec{d}_k \cdot \vec{q}_k}{\|\vec{d}_k\|\,\|\vec{q}_k\|}$$

</div>

---

# 2.3. LSI

### Ý tưởng

Cùng cách làm với LSA (đều dùng SVD để tìm khái niệm ẩn), nhưng **cân trọng số từ tốt hơn**. Ví dụ: từ *"lãi suất"* chỉ xuất hiện trong bài về kinh tế → tăng trọng số (rất phân biệt). Trong khi từ *"người", "ngày", "có"* xuất hiện đều trong mọi bài → giảm trọng số (không phân biệt được gì). Nhờ vậy không gian khái niệm thu được phản ánh đúng những từ **thực sự đặc trưng cho chủ đề**.

### Bước 1a: Local weight (trọng số nội tại)

<div class="mt-2">

$$L_{ij} = \log(1 + tf_{ij})$$

</div>

### Bước 1b: Global weight (trọng số toàn cục)

<div class="mt-2">

$$G_i = 1 + \frac{\sum_j p_{ij} \log p_{ij}}{\log N}, \quad p_{ij} = \frac{tf_{ij}}{\sum_j tf_{ij}}$$

</div>

<div class="text-sm opacity-80 mt-2">

→ Từ phổ biến có entropy cao → $G$ nhỏ → giảm trọng số &nbsp;·&nbsp; Từ đặc trưng có entropy thấp → $G$ lớn → tăng trọng số

</div>

---

# 2.3. LSI

### Bước 2: Weighted matrix

<div class="mt-2">

$$A_{ij} = L_{ij} \cdot G_i$$

</div>

### Bước 3: Truncated SVD

<div class="mt-2">

$$A_{m \times n} \approx U_k \, \Sigma_k \, V_k^T$$

</div>

<div class="text-sm">

- $A$ — ma trận **term–document** đã cân trọng số log-entropy ($m$ từ × $n$ tài liệu)
- $U_k$ — ma trận **từ × khái niệm** ($m \times k$): mỗi dòng cho biết một từ liên quan đến $k$ khái niệm ẩn ra sao
- $\Sigma_k$ — ma trận đường chéo chứa **độ quan trọng** của $k$ khái niệm (giá trị suy biến giảm dần)
- $V_k^T$ — ma trận **khái niệm × tài liệu** ($k \times n$): mỗi cột cho biết một tài liệu phân bố trên $k$ khái niệm ra sao
- $k$ — số chiều giữ lại (rất nhỏ so với $m, n$ → khử nhiễu, giữ ngữ nghĩa chính)

</div>

---

# 2.4. LDA

### Ý tưởng

Giả định mỗi bài viết thường **xoay quanh một vài chủ đề** (vd: thể thao, công nghệ, kinh tế...), và những từ **hay đi cùng nhau** (như *"tiền", "lãi suất", "ngân hàng"*) thường thuộc cùng một chủ đề. LDA tự học cách **phân ra các nhóm từ** đó, rồi mô tả mỗi tài liệu bằng **tỉ lệ pha trộn** các chủ đề.

### Quá trình sinh tài liệu

Với mỗi tài liệu $d$:

1. $\theta_d$ — **tỉ lệ pha trộn** $K$ chủ đề của tài liệu (vd: 60% kinh tế, 30% chính trị, 10% khác), được rút từ phân phối Dirichlet
2. Với mỗi từ thứ $n$ trong tài liệu:
   - Chọn một **chủ đề** $z_n$ theo tỉ lệ $\theta_d$
   - Chọn một **từ** $w_n$ theo phân phối từ vựng của chủ đề $z_n$ (vd: chủ đề "kinh tế" thì từ *tiền, lãi suất, ngân hàng* có xác suất cao)

---

# 2.4. LDA

### Huấn luyện mô hình

Quá trình sinh ở trên là **giả định lý thuyết** — thực tế ta chỉ thấy các từ trong tài liệu, không thấy chủ đề. LDA cần đi ngược lại để **tìm ra $\theta_d$ và $\phi_k$**:

1. **Đầu vào**: corpus gồm $N$ tài liệu + chọn trước số chủ đề $K$
2. **Khởi tạo**: gán ngẫu nhiên mỗi từ trong mỗi tài liệu vào 1 trong $K$ chủ đề
3. **Lặp tinh chỉnh** (bằng *Variational Bayes*):
   - Với mỗi tài liệu: cập nhật lại **tỉ lệ chủ đề** $\theta_d$
   - Với mỗi chủ đề: cập nhật lại **phân phối từ vựng** $\phi_k$
   - Lặp cho đến khi các phân phối ổn định
4. **Đầu ra**: mỗi tài liệu có một vector $\theta_d \in \mathbb{R}^K$ (tổng các thành phần = 1) biểu diễn tỉ lệ pha trộn chủ đề

### Biểu diễn query

Query $q$ được chiếu sang không gian chủ đề thông qua phép `transform` của LDA → vector $\theta_q \in \mathbb{R}^K$.

<div class="mt-2">

$$\text{sim}_{\text{LDA}}(d, q) = \cos\bigl(\theta_q,\, \theta_d\bigr) = \frac{\theta_q \cdot \theta_d}{\|\theta_q\|\,\|\theta_d\|}$$

</div>


---

# 2.5. Công thức Hybrid

### Kết hợp tuyến tính BM25 và Semantic

<div class="text-center text-2xl mt-4 mb-6">

$$\text{score}(d, q) = \alpha \cdot \widetilde{\text{BM25}}(d, q) + (1 - \alpha) \cdot \widetilde{\text{Sem}}(d, q)$$

</div>

### Chuẩn hóa min-max

$$\widetilde{s}(d) = \frac{s(d) - \min_{d'} s(d')}{\max_{d'} s(d') - \min_{d'} s(d')}$$

Đảm bảo BM25 và Semantic cùng thang `[0, 1]` trên toàn corpus → có thể trộn tuyến tính.

### Vai trò của α

| α | Ý nghĩa |
|---|---|
| **1.0** | Pure BM25 (chỉ lexical) |
| **0.0** | Pure semantic |
| **0.5** | Cân bằng 50–50 |

---
layout: section
---

# 3. Thực nghiệm

---

# 3.1. Dữ liệu

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

### Dataset VietNews 20k

- **Nguồn:** `nam194/vietnews` (HuggingFace)
- **Quy mô:** 20,000 bài đầu từ split `train`
- **Trường dùng:** `title`, `article`, `url`
- **Ngôn ngữ:** Tiếng Việt
- **Lĩnh vực:** Tin tức tổng hợp (thời sự, kinh tế, thể thao, giải trí, ...)

</div>

<div>

### Thống kê cơ bản

| Chỉ số | Giá trị |
|---|---|
| Số tài liệu | **20,000** |
| Trung bình tokens/doc | **271.5** |
| Vocab sau tiền xử lý | **87,931 từ** |
| Dung lượng CSV | ~170 MB |

</div>

</div>

---

# 3.2. Test queries & Ground-truth

<div class="grid grid-cols-2 gap-4">

<div>

### 15 query đa lĩnh vực

<div class="text-xs">

| # | Category | Query (rút gọn) |
|---|---|---|
| 1 | weather | thời tiết mưa bão... |
| 2 | stock | chứng khoán cổ phiếu... |
| 3 | sports | bóng đá việt nam... |
| 4 | tech | công nghệ AI trí tuệ... |
| 5 | commodity | giá vàng hôm nay tăng |
| 6 | health | dịch covid vaccine |
| 7 | automotive | xe điện vinfast |
| 8 | entertainment | ca sĩ nhạc mới |
| 9 | crime | tội phạm ma túy |
| 10 | accident | tai nạn giao thông |
| 11 | olympic | thể thao olympic |
| 12 | travel | du lịch resort |
| 13 | education | giáo dục trường học |
| 14 | politics | chính trị quốc tế |
| 15 | fashion | thời trang làm đẹp |

</div>

</div>

<div>

### Heuristic relevance

Doc $d$ **relevant** với query $q$ ⇔ $d$ chứa **≥ 2 keyword** của $q$ (lowercase match)

### Số doc relevant per query

```text
weather:        357    crime:         3408
stock:          0      accident:       607
sports:         0      olympic:        6
tech:           4      travel:         1
commodity:    2602     education:    2591
health:        386     politics:       0
automotive:    0       fashion:        0
entertainment: 149
```

</div>

</div>

---

# 3.3. Grid search & metrics

### Lưới siêu tham số

- **α-grid:** `{0.0, 0.1, 0.2, ..., 1.0}` — 11 giá trị
- **k-grid:** `{1, 3, 5, 10, 20}` — 5 giá trị
- **Tổng cộng:** 11 × 5 = **55 cấu hình** cho mỗi mô hình

### Nhóm 1 — Metrics phân loại (set-based): P, R, F1

<div class="mt-4">

$$\text{P@k} = \frac{|\{\text{relevant}\}\cap\{\text{top-k}\}|}{k}, \qquad \text{R@k} = \frac{|\{\text{relevant}\}\cap\{\text{top-k}\}|}{|\{\text{relevant}\}|}$$

$$\text{F1@k} = \frac{2 \cdot P@k \cdot R@k}{P@k + R@k}$$

</div>

---

# 3.3. Grid search & metrics

### Nhóm 2 — Metrics xếp hạng (ranking-based): MAP, MRR, NDCG

#### MAP — Mean Average Precision

<div class="mt-2">

$$\text{AP} = \frac{1}{|\text{rel}|}\sum_{i=1}^{k} P(i) \cdot \mathbb{1}[\text{rel}_i]$$

</div>

#### MRR — Mean Reciprocal Rank

<div class="mt-2">

$$\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q^{\text{first-relevant}}}$$

</div>

#### NDCG — Normalized Discounted Cumulative Gain

<div class="mt-2">

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \qquad \text{DCG@k} = \sum_{i=1}^{k}\frac{2^{r_i}-1}{\log_2(i+1)}$$

</div>

---

# 3.4. Kết quả BM25 + LSA

<div class="grid grid-cols-2 gap-3">

<div>

### Heatmap 6 metrics

<img src="/lsa_1.png" class="rounded shadow w-full" />

</div>

<div>

### Best config

| Tham số | Giá trị |
|---|---|
| **Best α** | **0.5** |
| **Best k** | **1** |
| **Best NDCG** | **0.6000** |
| n_components | 200 (default) |

### So sánh n_components

| $k$ | best α | NDCG@10 | Explained var |
|---|---|---|---|
| 50  | 0.7 | 0.5592 | 14.31% |
| 100 | 0.9 | 0.5530 | 19.18% |
| 200 | 0.7 | 0.5744 | 25.73% |
| **300** | **0.6** | **0.5818** | 30.74% |

→ NDCG tăng dần theo n_components, nhưng fit time cũng tăng tuyến tính

</div>

</div>

---

# 3.4. Kết quả BM25 + LSA (tt)

<div class="grid grid-cols-2 gap-4">

<div>

### NDCG theo α & sweep n_components

<img src="/lsa_2.png" class="rounded shadow w-full" />

</div>

<div>

### Per-query breakdown

<img src="/lsa_3.png" class="rounded shadow w-full" />

<div class="text-xs opacity-75 mt-2">

**Nhận xét:**
- 6 query (stock, sports, automotive, politics, fashion, travel) có 0 relevant doc → metric = 0
- 9 query còn lại: NDCG/MRR ≈ 1.0 cho commodity, health, entertainment...
- Hybrid với α = 0.5 cân bằng tốt giữa BM25 và LSA

</div>

</div>

</div>

---

# 3.5. Kết quả BM25 + LSI

<div class="grid grid-cols-2 gap-3">

<div>

### Heatmap 6 metrics

<img src="/lsi_1.png" class="rounded shadow w-full" />

</div>

<div>

### Best config

| Tham số | Giá trị |
|---|---|
| **Best α** | **0.4** |
| **Best k** | **1** |
| **Best NDCG** | **0.6000** |
| n_components | 200 (default) |

### So sánh n_components

| $k$ | best α | NDCG@10 | Explained var |
|---|---|---|---|
| 50  | 0.9 | 0.5512 | 13.88% |
| 100 | 0.6 | 0.5722 | 18.68% |
| **200** | **0.4** | **0.5766** | 25.09% |
| 300 | 0.6 | 0.5693 | 29.94% |

→ Khác LSA: best ở **n=200** (không phải 300)<br>
→ α* lệch về **semantic** hơn (0.4 vs 0.5 của LSA)

</div>

</div>

---

# 3.5. Kết quả BM25 + LSI (tt)

<div class="grid grid-cols-2 gap-4">

<div>

### NDCG theo α & sweep n_components

<img src="/lsi_2.png" class="rounded shadow w-full" />

</div>

<div>

### Per-query breakdown

<img src="/lsi_3.png" class="rounded shadow w-full" />

<div class="text-xs opacity-75 mt-2">

**Nhận xét LSI vs LSA:**
- NDCG đỉnh ngang nhau: **0.6000**
- LSI ưa **α nhỏ hơn** (0.4) → log-entropy weighting cho semantic vector chất lượng hơn TF-IDF
- LSI cần ít chiều hơn (200 thay vì 300) → tiết kiệm thời gian

</div>

</div>

</div>

---

# 3.6. Kết quả BM25 + LDA

<div class="grid grid-cols-2 gap-3">

<div>

### Heatmap 6 metrics

<img src="/lda_1.png" class="rounded shadow w-full" />

</div>

<div>

### Best config

| Tham số | Giá trị |
|---|---|
| **Best α** | **1.0** |
| **Best k** | **3** |
| **Best NDCG** | **0.5667** |
| n_topics | 50 (default) |

### So sánh n_topics

| $K$ | best α | NDCG@10 | Perplexity |
|---|---|---|---|
| **10**  | **0.8** | **0.5446** | 1099.5 |
| 20  | 1.0 | 0.5416 | 1033.1 |
| 50  | 1.0 | 0.5416 | 983.1 |
| 100 | 1.0 | 0.5416 | 983.2 |

<div class="text-xs opacity-75">
**α* = 1.0** nghĩa là pure BM25 — LDA không cải thiện được hybrid trên test set này
</div>

</div>

</div>

---

# 3.6. Kết quả BM25 + LDA (tt)

<div class="grid grid-cols-2 gap-3">

<div>

### NDCG theo α & sweep n_topics

<img src="/lda_2.png" class="rounded shadow w-full" />

### Per-query breakdown

<img src="/lda_3.png" class="rounded shadow w-full h-48 object-contain" />

</div>

<div>

### Perplexity vs fit time

<img src="/lda_4.png" class="rounded shadow w-full" />

<div class="text-xs opacity-75 mt-2">

**Nhận xét LDA:**
- LDA topic-distribution **mịn** → kém phân biệt cho query cụ thể
- Phù hợp cho clustering / topic discovery, **không phù hợp** cho retrieval điểm chính xác
- Hybrid kéo α về 1.0 (pure BM25) → LDA đang **gây hại** cho ranking

</div>

</div>

</div>

---
layout: section
---

# 4. Kết luận

---

# 4.1. So sánh 3 mô hình — 6 metrics

<div class="text-sm mt-4">

| Metric @ k=10 | **BM25 + LSA** | **BM25 + LSI** | **BM25 + LDA** |
|---|:---:|:---:|:---:|
| Best **α*** | 0.7 | **0.4** | 1.0 |
| **Precision@10** | 0.3867 | **0.4000** | **0.4000** |
| **Recall@10** | 0.0584 | **0.0750** | 0.0586 |
| **F1@10** | 0.0506 | **0.0600** | 0.0510 |
| **MAP@10** | **0.5427** | 0.5419 | 0.4977 |
| **MRR@10** | **0.6000** | **0.6000** | 0.5556 |
| **NDCG@10** | 0.5744 | **0.5766** | 0.5416 |

</div>


---

# 4.2. Nhận xét

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="border-l-4 border-blue-500 pl-3">

### BM25 + LSA
**Khá tốt**

- Cải thiện rõ so với BM25 thuần
- Cho kết quả tốt ở phần lớn truy vấn
- Tốc độ huấn luyện vừa phải

</div>

<div class="border-l-4 border-teal-500 pl-3">

### BM25 + LSI
**Tốt nhất**

- Đứng đầu ở Precision, Recall, F1, NDCG
- Cân bằng giữa từ vựng và ngữ nghĩa tốt nhất
- Huấn luyện nhanh hơn LSA

</div>

<div class="border-l-4 border-orange-500 pl-3">

### BM25 + LDA
**Kém**

- Không cải thiện được kết quả so với BM25 thuần
- Thua ở hầu hết các tiêu chí
- Huấn luyện chậm nhất (gấp ~10 lần)

</div>

</div>

