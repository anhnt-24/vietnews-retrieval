"""
Load 20k rows from VietNews dataset and save to CSV
Chạy: python scripts/load_data.py
"""

from datasets import load_dataset
import pandas as pd

def main():
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("nam194/vietnews", split="train")

    MAX_SAMPLES = 20000
    print(f"Limiting to {MAX_SAMPLES} samples...")

    # Trích xuất dữ liệu
    data = []
    for i, item in enumerate(dataset):
        if i >= MAX_SAMPLES:
            break
        data.append({
            'title': item.get('title', ''),
            'content': item.get('content', ''),
            'url': item.get('url', ''),
            'publish_date': item.get('publish_date', '')
        })

    # Tạo DataFrame
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows")

    # Lưu CSV
    output_path = 'data/vietnews_20k.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved to {output_path}")

    # Thống kê
    print(f"\nThống kê:")
    print(f"- Rows: {len(df)}")
    print(f"- Columns: {list(df.columns)}")
    print(f"- Missing titles: {df['title'].isna().sum()}")
    print(f"- Missing content: {df['content'].isna().sum()}")

if __name__ == "__main__":
    main()
