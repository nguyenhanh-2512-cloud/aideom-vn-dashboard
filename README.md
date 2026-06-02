# VN AIDEOM-VN Streamlit Dashboard

Dashboard này tạo 13 trang điều hướng:
1. Trang chủ tổng quan
2. Bài 1 đến Bài 12 theo yêu cầu môn Mô hình ra quyết định

## Cấu trúc thư mục

```text
aideom_vn_streamlit/
├── app.py
├── requirements.txt
├── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_regions_2024.csv
│   └── vietnam_sectors_2024.csv
├── src/
│   ├── __init__.py
│   └── model_catalog.py
└── .streamlit/
    └── config.toml
```

## Chạy local trên máy

```bash
cd aideom_vn_streamlit
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Nếu dùng macOS/Linux:

```bash
source venv/bin/activate
streamlit run app.py
```

## Deploy lên Streamlit Community Cloud

1. Tạo repository mới trên GitHub, ví dụ: `aideom-vn-dashboard`.
2. Upload toàn bộ nội dung thư mục này lên repository. Cần có `app.py`, `requirements.txt`, thư mục `data`.
3. Vào Streamlit Community Cloud.
4. Chọn **Deploy a public app from GitHub**.
5. Chọn repository, branch `main`, main file path là `app.py`.
6. Bấm **Deploy**.
7. Chờ build xong, mở link dạng `https://ten-app.streamlit.app`.

## Ghi chú học thuật

Dashboard này là bản nguyên mẫu trực quan để trình bày kết quả. Với các bài nâng cao như NSGA-II, CVXPY, Pyomo hoặc Q-learning đầy đủ, nhóm có thể bổ sung notebook hoặc module `.py` riêng để đáp ứng phần mã nguồn chuyên sâu trong báo cáo.
