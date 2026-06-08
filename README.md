# VN AIDEOM-VN

## AI-Driven Decision Optimization Model for Vietnam

VN AIDEOM-VN là dashboard mô hình hóa và hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong bối cảnh chuyển đổi số, trí tuệ nhân tạo và bất định kinh tế. Hệ thống được xây dựng bằng Python và Streamlit, tích hợp các kỹ thuật dự báo, tối ưu hóa, ra quyết định đa tiêu chí, tối ưu đa mục tiêu, quy hoạch ngẫu nhiên và học tăng cường.

Dự án gồm 12 bài toán liên kết với nhau, sử dụng dữ liệu vĩ mô, dữ liệu ngành và dữ liệu vùng của Việt Nam giai đoạn 2020–2025, đồng thời mô phỏng các kịch bản chính sách giai đoạn 2026–2035.

---

## 1. Mục tiêu dự án

Dự án hướng đến bốn mục tiêu chính:

1. Mô hình hóa tác động của vốn, lao động, số hóa, AI và nhân lực số đến tăng trưởng kinh tế Việt Nam.
2. Hỗ trợ phân bổ ngân sách giữa các ngành, vùng, hạng mục đầu tư và chương trình chuyển đổi số.
3. So sánh các kịch bản phát triển kinh tế số và AI theo các KPI định lượng.
4. Xây dựng dashboard tương tác phục vụ phân tích, kiểm định, thảo luận và trình bày chính sách.

---

## 2. Danh mục 12 bài toán

| Bài | Nội dung | Phương pháp chính | Đầu ra chính |
|---|---|---|---|
| Bài 1 | Hàm sản xuất Cobb-Douglas mở rộng với số hóa và AI | Growth accounting, TFP | TFP, MAPE, phân rã tăng trưởng, GDP 2030 |
| Bài 2 | Phân bổ ngân sách số 4 hạng mục | Linear Programming | Phân bổ tối ưu, Z*, shadow price, độ nhạy ngân sách |
| Bài 3 | Xây dựng chỉ số Priority cho 10 ngành | Min-max, weighted scoring | Priority, top ngành, độ nhạy trọng số AI |
| Bài 4 | Phân bổ ngân sách theo vùng và hạng mục | LP, fairness constraints | Ma trận 6×4, Z*, chi phí công bằng vùng |
| Bài 5 | Lựa chọn danh mục 15 dự án chuyển đổi số | Mixed Integer Programming | Dự án chọn, chi phí, NPV, NPV/chi phí |
| Bài 6 | Xếp hạng 6 vùng kinh tế | TOPSIS, entropy weight | Điểm TOPSIS, thứ hạng vùng, phân tích độ nhạy |
| Bài 7 | Tối ưu đa mục tiêu | Pareto, NSGA-II | Tập nghiệm Pareto, nghiệm thỏa hiệp, đánh đổi mục tiêu |
| Bài 8 | Tối ưu động 2026–2035 | Dynamic optimization, SLSQP | Quỹ đạo K, D, AI, H, GDP, tiêu dùng |
| Bài 9 | Tác động AI tới thị trường lao động | LP/CVXPY, NetJob | x_AI, x_H, NetJob, retraining capacity |
| Bài 10 | Quy hoạch ngẫu nhiên hai giai đoạn | Stochastic programming, VSS, EVPI | First-stage, recourse, VSS, EVPI, robust |
| Bài 11 | Học tăng cường cho chính sách thích nghi | MDP 81 trạng thái, Q-learning | Q-table, policy π*, learning curve, rule-based comparison |
| Bài 12 | Đồ án tích hợp AIDEOM-VN | M1–M6 pipeline, scenario analysis | KPI 5 kịch bản, cảnh báo, khuyến nghị, dashboard |

---

## 3. Kiến trúc hệ thống tích hợp Bài 12

Bài 12 tổ chức hệ thống AIDEOM-VN theo sáu mô-đun:

| Mô-đun | File | Chức năng | Đầu ra chính |
|---|---|---|---|
| M1 | `src/m1_forecast.py` | Dự báo kinh tế | GDP, TFP, D, AI, H đến 2030 |
| M2 | `src/m2_readiness.py` | Đánh giá sẵn sàng số | TOPSIS, entropy, xếp hạng 6 vùng |
| M3 | `src/m3_allocation.py` | Tối ưu phân bổ ngân sách | Ma trận phân bổ vùng × hạng mục |
| M4 | `src/m4_labor.py` | Mô phỏng lao động và AI | NetJob, DisplacedJob, RetrainingCapacity |
| M5 | `src/m5_risk.py` | Đánh giá rủi ro và bất định | Cyber risk, emission, stochastic, VSS, EVPI |
| M6 | `src/m6_pipeline.py` | Tích hợp hệ thống | KPI 5 kịch bản, điểm tổng hợp, khuyến nghị |

Dashboard `app.py` sử dụng các mô-đun trong `src/` để hiển thị kết quả, biểu đồ, bảng KPI, cảnh báo rủi ro và nội dung diễn giải chính sách.

---

## 4. Đối chiếu yêu cầu Bài 12

### 12.1. Yêu cầu chức năng

AIDEOM-VN gồm 6 module M1–M6:

- M1: dự báo kinh tế vĩ mô;
- M2: đánh giá sẵn sàng số;
- M3: tối ưu phân bổ ngân sách;
- M4: mô phỏng lao động;
- M5: đánh giá rủi ro;
- M6: dashboard ra quyết định.

Trạng thái: đã tách thành các file Python trong `src/`.

### 12.2. Năm kịch bản chính sách

| Kịch bản | Định hướng | Cơ cấu phân bổ |
|---|---|---|
| S1 – Truyền thống | Ưu tiên vốn vật chất, FDI, hạ tầng truyền thống | 70% K, 10% D, 10% AI, 10% H |
| S2 – Số hóa nhanh | Tăng đầu tư chính phủ số, doanh nghiệp số | 25% K, 45% D, 15% AI, 15% H |
| S3 – AI dẫn dắt | Ưu tiên AI, dữ liệu lớn, bán dẫn | 20% K, 20% D, 45% AI, 15% H |
| S4 – Bao trùm số | Ưu tiên vùng yếu, SME, giáo dục số | 30% K, 20% D, 10% AI, 40% H |
| S5 – Tối ưu cân bằng | Kết quả tổng hợp của mô hình | Kết hợp các tiêu chí GDP, việc làm, bao trùm và rủi ro |

Trạng thái: dashboard và pipeline có đủ S1–S5.

### 12.3. Yêu cầu kỹ thuật

Dự án đáp ứng các yêu cầu kỹ thuật chính:

- M1–M6 được viết thành các file Python độc lập trong `src/`;
- có docstring và hàm tái sử dụng;
- có dashboard Streamlit;
- có dữ liệu trong thư mục `data/`;
- có `requirements.txt`;
- có unit test trong `tests/test_files.py`;
- có kiểm tra ít nhất các kịch bản S1, S3, S5;
- có bảng kết quả KPI 2030 trong pipeline Bài 12.

### 12.4. Sản phẩm bàn giao

Phần code hiện gồm:

- mã nguồn Python;
- dashboard Streamlit;
- dữ liệu CSV;
- module M1–M6;
- unit test;
- README;
- requirements.

Các sản phẩm phi-code như báo cáo PDF/Word, slide và video demo sẽ được hoàn thiện riêng.

### 12.5. Tiêu chí đánh giá

Dự án được thiết kế để đáp ứng các nhóm tiêu chí:

1. Mô hình toán học;
2. Chất lượng mã nguồn;
3. Dữ liệu Việt Nam;
4. Phân tích chính sách;
5. Trực quan hóa và dashboard;
6. Báo cáo và thuyết trình.

### 12.6. Hướng mở rộng

Các hướng mở rộng sau đồ án:

- mở rộng sang CGE hoặc DSGE-AI;
- thêm dữ liệu thời gian thực;
- tích hợp dữ liệu quý/tháng;
- phát triển Multi-Agent RL;
- mở rộng phân tích ngành sâu hơn;
- bổ sung module kiểm định độ nhạy và uncertainty dashboard.

---

## 5. Năm kịch bản phân tích

| Kịch bản | Mô tả |
|---|---|
| S1 – Truyền thống | Tập trung vào vốn vật chất, hạ tầng truyền thống và xuất khẩu |
| S2 – Số hóa nhanh | Tăng đầu tư vào hạ tầng số, doanh nghiệp số và chính phủ số |
| S3 – AI dẫn dắt | Ưu tiên AI, dữ liệu lớn, bán dẫn và trung tâm dữ liệu |
| S4 – Bao trùm số | Ưu tiên nhân lực, vùng yếu, SME và nông nghiệp số |
| S5 – Tối ưu cân bằng | Tối ưu theo điểm tổng hợp giữa GDP, việc làm, bao trùm và rủi ro |

Người dùng có thể thay đổi trọng số của các tiêu chí:

- GDP;
- việc làm;
- bao trùm;
- giảm cyber risk;
- giảm phát thải.

Các trọng số được chuẩn hóa tự động để tổng bằng 1.

---

## 6. Dữ liệu sử dụng

Dự án sử dụng ba nhóm dữ liệu chính trong thư mục `data/`:

```text
data/
├── vietnam_macro_2020_2025.csv
├── vietnam_sectors_2024.csv
└── vietnam_regions_2024.csv
```

### 6.1. Dữ liệu vĩ mô

`vietnam_macro_2020_2025.csv` gồm GDP, tăng trưởng GDP, tỷ trọng kinh tế số, năng suất lao động và các biến kinh tế vĩ mô giai đoạn 2020–2025.

### 6.2. Dữ liệu ngành

`vietnam_sectors_2024.csv` gồm 10 ngành kinh tế với các chỉ tiêu tăng trưởng, tỷ trọng GDP, xuất khẩu, lao động, mức sẵn sàng AI và rủi ro tự động hóa.

### 6.3. Dữ liệu vùng

`vietnam_regions_2024.csv` gồm 6 vùng kinh tế – xã hội với các chỉ tiêu GRDP/người, FDI, Digital Index, AI Readiness, lao động qua đào tạo, R&D, Internet và Gini.

### 6.4. Nguồn tham chiếu

Dữ liệu và chỉ tiêu được tổng hợp, hiệu chỉnh hoặc tham chiếu từ:

- NSO/GSO;
- Bộ Kế hoạch và Đầu tư;
- Bộ Khoa học và Công nghệ;
- Bộ Thông tin và Truyền thông;
- World Bank;
- WIPO/GII;
- các giả định mô phỏng phục vụ bài tập.

Một số hệ số beta, rủi ro, tác động AI, đào tạo lại và chỉ số tổng hợp là tham số mô phỏng dùng cho bài tập, không phải dự báo chính thức.

---

## 7. Cấu trúc thư mục

```text
aideom-vn-dashboard/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_sectors_2024.csv
│   └── vietnam_regions_2024.csv
├── src/
│   ├── __init__.py
│   ├── model_catalog.py
│   ├── m1_forecast.py
│   ├── m2_readiness.py
│   ├── m3_allocation.py
│   ├── m4_labor.py
│   ├── m5_risk.py
│   └── m6_pipeline.py
├── tests/
│   └── test_files.py
└── .devcontainer/
```

---

## 8. Yêu cầu môi trường

- Python 3.11 trở lên;
- pip;
- Git;
- trình duyệt web hiện đại.

Các thư viện chính:

```text
streamlit
pandas
numpy
scipy
plotly
pulp
cvxpy
pymoo
pyomo
gymnasium
stable-baselines3
pytest
openpyxl
highspy
matplotlib
tqdm
```

---

## 9. Cài đặt

### 9.1. Clone repository

```bash
git clone <repository-url>
cd aideom-vn-dashboard
```

### 9.2. Cài thư viện

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 9.3. Kiểm tra phụ thuộc

```bash
python -m pip check
```

Kết quả mong đợi:

```text
No broken requirements found.
```

---

## 10. Chạy dashboard

```bash
python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Trong GitHub Codespaces, mở địa chỉ dạng:

```text
https://<codespace-name>-8501.app.github.dev
```

Nếu chạy trên máy cá nhân, mở:

```text
http://localhost:8501
```

---

## 11. Kiểm tra cú pháp

```bash
python -m py_compile app.py src/*.py tests/test_files.py
```

Nếu terminal không báo lỗi và quay lại dấu `$`, các file Python hợp lệ về cú pháp.

---

## 12. Chạy kiểm thử

```bash
python -m pytest -q tests/test_files.py
```

Các nhóm kiểm tra bao gồm:

- file dự án tồn tại;
- đủ ba file dữ liệu;
- import được M1–M6;
- catalog có đủ 12 bài và 5 kịch bản;
- M1 trả về TFP, MAPE, GDP 2030;
- M2 trả về TOPSIS 6 vùng;
- M3 bảo đảm ngân sách không vượt 50.000 và nhân lực tối thiểu;
- M4 trả về NetJob và retraining;
- M5 có stochastic programming 4 kịch bản, VSS, EVPI;
- M6 có đủ S1–S5 và kiểm tra riêng S1, S3, S5.

Kết quả mong đợi:

```text
9 passed
```

---

## 13. Quy trình Git khuyến nghị

### 13.1. Kiểm tra trước khi commit

```bash
python -m py_compile app.py src/*.py tests/test_files.py
python -m pytest -q tests/test_files.py
git status
```

### 13.2. Commit thay đổi README

```bash
git add README.md
git commit -m "Update README for modular AIDEOM VN project"
git push
```

### 13.3. Commit toàn bộ thay đổi code khi cần

```bash
git add app.py src tests requirements.txt README.md
git commit -m "Finalize AIDEOM VN dashboard content and tests"
git push
```

Không nên dùng `git add .` nếu thư mục có file backup, file tạm hoặc `__pycache__`.

---

## 14. Solver và phương pháp tính toán

Dự án sử dụng:

- SciPy HiGHS cho Linear Programming;
- PuLP/CBC cho Mixed Integer Programming;
- CVXPY cho phân bổ AI và đào tạo lại lao động;
- Pyomo/HiGHS cho quy hoạch ngẫu nhiên và robust optimization;
- pymoo cho NSGA-II;
- Gymnasium cho môi trường học tăng cường;
- Stable-Baselines3 cho DQN mở rộng;
- SLSQP cho tối ưu phi tuyến và tối ưu động.

---

## 15. Kết quả đầu ra

Mỗi trang bài tập có thể bao gồm:

- bối cảnh Việt Nam;
- mô hình toán học;
- dữ liệu đầu vào;
- bảng nghiệm;
- KPI;
- biểu đồ;
- phân tích độ nhạy;
- kiểm tra ràng buộc;
- nhận xét chính sách;
- nút tải CSV.

Bài 12 tổng hợp kết quả của các mô-đun thành hệ thống hỗ trợ ra quyết định theo 5 kịch bản chính sách.

---

## 16. Giới hạn của mô hình

1. Một số hệ số là giả định hoặc biến đại diện.
2. Dữ liệu chưa bao phủ đầy đủ mọi ngành, nghề và địa phương.
3. Mô hình chưa phản ánh toàn bộ chi phí điều chỉnh, nợ công và phản ứng hành vi.
4. Kết quả AI và lao động không phải dự báo việc làm chính thức.
5. Kết quả tối ưu phụ thuộc vào trọng số, ràng buộc và giả định đầu vào.
6. Dashboard là công cụ hỗ trợ phân tích, không thay thế quyết định của cơ quan quản lý.
7. Các kịch bản là mô phỏng học thuật, không phải dự báo chính thức.

---

## 17. Nguyên tắc sử dụng kết quả

Kết quả từ AIDEOM-VN nên được sử dụng để:

- so sánh kịch bản;
- kiểm tra độ nhạy;
- minh họa đánh đổi chính sách;
- hỗ trợ thảo luận;
- xác định nhóm rủi ro;
- xây dựng khuyến nghị có điều kiện;
- chuẩn bị báo cáo và thuyết trình môn học.

Quyết định thực tế cần kết hợp dữ liệu được kiểm định, ý kiến chuyên gia, đánh giá tác động, giới hạn an toàn, giải thích mô hình và phê duyệt của con người.

---

## 18. Khai báo sử dụng công cụ AI

Công cụ AI được sử dụng để hỗ trợ:

- xây dựng cấu trúc mã;
- phát hiện và sửa lỗi;
- diễn giải mô hình;
- xây dựng giao diện;
- chuẩn hóa tài liệu;
- đề xuất kiểm thử;
- hỗ trợ hoàn thiện README và nội dung mô tả kỹ thuật.

Người thực hiện chịu trách nhiệm kiểm tra mã nguồn, dữ liệu, kết quả, trích dẫn và nội dung nộp cuối cùng.

---

## 19. Trạng thái phát triển

### Đã hoàn thành

- 12 trang bài toán;
- dashboard Streamlit;
- dữ liệu vĩ mô, ngành và vùng;
- biểu đồ và bảng kết quả;
- M1–M6 trong thư mục `src/`;
- unit test trong `tests/`;
- Bài 5 đúng cấu trúc MIP P1–P15;
- Bài 9 đúng mô hình NetJob 8 ngành;
- Bài 10 có quy hoạch ngẫu nhiên hai giai đoạn;
- Bài 11 có MDP 81 trạng thái và Q-learning;
- Bài 12 tích hợp năm kịch bản;
- bảng KPI tổng hợp;
- cảnh báo rủi ro;
- tải kết quả CSV.

### Còn tiếp tục hoàn thiện

- tinh chỉnh giao diện trực quan;
- bổ sung thêm biểu đồ chi tiết cho từng kịch bản;
- hoàn thiện báo cáo Word/PDF;
- hoàn thiện slide thuyết trình;
- quay video demo 3–5 phút.

---

## 20. Cách kiểm tra nhanh trước khi nộp

Chạy lần lượt:

```bash
python -m py_compile app.py src/*.py tests/test_files.py
python -m pytest -q tests/test_files.py
python -m streamlit run app.py
```

Nếu cú pháp không lỗi, test passed và dashboard mở được thì phần code có thể đưa vào bản nộp.

---

## 21. License

Dự án được xây dựng phục vụ mục đích học tập và nghiên cứu. Không sử dụng kết quả như khuyến nghị đầu tư, dự báo chính thức hoặc quyết định chính sách nếu chưa có bước kiểm định độc lập.
