# Phương pháp Lưu trữ Data cho Search Engine

## 1. Hiện tại (Joblib - Local Files)

**Cách lưu:** `joblib.dump()` ra file `.pkl` trong thư mục `saved_models/`

**Ưu điểm:**
- Đơn giản, nhanh
- Lưu trực tiếp Python objects
- Không cần cấu hình database

**Nhược điểm:**
- Khó mở rộng (không chia sẻ được giữa các servers)
- Không có query capability
- Backup/_sync khó khăn

---

## 2. PostgreSQL + SQLAlchemy (Khuyến nghị cho Production)

```python
# Ví dụ: lưu documents vào PostgreSQL
from sqlalchemy import Column, Integer, String, Text, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    content = Column(Text)
    url = Column(String(1000))
    model_scores = Column(JSON)  # Lưu scores từ nhiều models
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Ưu điểm:**
- Query mạnh mẽ (WHERE, JOIN, FULL TEXT SEARCH)
- Mở rộng tốt (replicate, shard)
- Concurrent access
- Backup dễ dàng

**Nhược điểm:**
- Cần cài đặt PostgreSQL
- Phức tạp hơn cho vectors lớn

---

## 3. Elasticsearch (Khuyến nghị cho Search)

```json
{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "content": { "type": "text", "analyzer": "vietnamese" },
      "url": { "type": "keyword" },
      "bm25_score": { "type": "float" },
      "lda_topic": { "type": "keyword" },
      "lsi_vector": { "type": "dense_vector", "dims": 100 }
    }
  }
}
```

**Ưu điểm:**
- Search engine chuyên dụng
- Hỗ trợ Vietnamese analyzer
- Scalable
- BM25 đã tích hợp sẵn

**Nhược điểm:**
- Tốn tài nguyên (RAM nhiều)
- Cần maintain cluster

---

## 4. FAISS (cho Vector Similarity)

```python
import faiss
import numpy as np

# Lưu LSI vectors
dimension = 100
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
index.add(lsi_matrix)

# Lưu index
faiss.write_index(index, 'lsi.index')
```

**Ưu điểm:**
- Tìm kiếm vector cực nhanh (hàng triệu vectors)
- GPU support
- Memory efficient

**Nhược điểm:**
- Chỉ lưu vectors, cần PostgreSQL lưu metadata
- Không query được text

---

## 5. MongoDB (NoSQL - Flexible)

```python
from pymongo import MongoClient

client = MongoClient('localhost:27017')
db = client['news_search']

# Lưu document
db.documents.insert_one({
    "title": "Tin tức thể thao",
    "content": "Nội dung...",
    "url": "https://...",
    "keywords": ["thể thao", "bóng đá"],
    "embeddings": {
        "bm25": 0.85,
        "lda": [0.1, 0.2, 0.3],
        "lsi": [0.5, 0.3, 0.2]
    }
})
```

**Ưu điểm:**
- Schema linh hoạt
- JSON-like documents
- Horizontal scaling

**Nhược điểm:**
- Không transaction như SQL
- Search chậm hơn Elasticsearch

---

## 6. Redis (Cache + Fast Retrieval)

```python
import redis

r = redis.Redis(host='localhost', port=6379)

# Cache kết quả search
r.setex(f"search:bm25:{query_hash}", 3600, json.dumps(results))

# Lưu document index
r.hset(f"doc:{doc_id}", mapping={
    "title": title,
    "content": content,
    "url": url
})
```

**Ưu điểm:**
- Cực nhanh
- Cache hiệu quả
- Pub/Sub cho real-time

**Nhược điểm:**
- Không thay thế được database chính
- Memory limited

---

## 7. Cloud Solutions

### a) AWS OpenSearch (Elasticsearch managed)
- Tự động scale
- Serverless options
- Tích hợp Vietnamese analyzer

### b) Pinecone (Vector Database)
- Managed vector search
- Easy LSI/semantic search
- Free tier có sẵn

### c) Weaviate (Open source)
- Graph + Vector search
- Built-in semantic search
- Kubernetes deployment

---

## Khuyến nghị theo quy mô

| Quy mô | Giải pháp |
|--------|-----------|
| Demo/MVP | Joblib files (hiện tại) |
| Nhỏ (< 100K docs) | PostgreSQL + FAISS vectors |
| Trung bình (< 1M) | Elasticsearch hoặc OpenSearch |
| Lớn (> 1M) | Elasticsearch + Faiss + PostgreSQL |

---

## Migration Path (Nâng cấp)

**Bước 1:** Giữ nguyên joblib, thêm PostgreSQL cho metadata
```python
# Lưu metadata vào PostgreSQL
db.documents.insert({
    "id": idx,
    "title": title,
    "url": url,
    # vectors vẫn lưu joblib
})
```

**Bước 2:** Thêm Elasticsearch cho search
```python
# Index documents vào Elasticsearch
es.index(index="news", id=idx, body={
    "title": title,
    "content": content
})
```

**Bước 3:** Thêm FAISS cho vector search
```python
# Build FAISS index từ LSI vectors
faiss_index = build_faiss_index(lsi_matrix)
```
