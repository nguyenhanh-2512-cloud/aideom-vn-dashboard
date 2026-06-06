# VN AIDEOM-VN

## AI-Driven Decision Optimization Model for Vietnam

VN AIDEOM-VN là dashboard mô hình hóa và hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong bối cảnh chuyển đổi số và trí tuệ nhân tạo. Hệ thống được xây dựng bằng Python và Streamlit, tích hợp các kỹ thuật dự báo, tối ưu hóa, ra quyết định đa tiêu chí, tối ưu đa mục tiêu, quy hoạch ngẫu nhiên và học tăng cường.

Dự án gồm 12 bài toán liên kết với nhau, sử dụng dữ liệu vĩ mô, dữ liệu ngành và dữ liệu vùng của Việt Nam giai đoạn 2020–2025.

---

## 1. Mục tiêu dự án

Dự án hướng đến bốn mục tiêu chính:

1. Mô hình hóa tác động của vốn, lao động, số hóa, AI và nhân lực số đến tăng trưởng kinh tế.
2. Hỗ trợ phân bổ ngân sách giữa các ngành, vùng và chương trình đầu tư.
3. So sánh các kịch bản phát triển kinh tế số và AI.
4. Xây dựng dashboard tương tác phục vụ phân tích, kiểm định và thảo luận chính sách.

---

## 2. Danh mục 12 bài toán

| Bài | Nội dung | Phương pháp chính |
|---|---|---|
| Bài 1 | Hàm sản xuất Cobb-Douglas mở rộng với số hóa và AI | Growth accounting, dự báo GDP |
| Bài 2 | Phân bổ ngân sách số | Linear Programming |
| Bài 3 | Xây dựng chỉ số Priority cho 10 ngành | Min-max normalization, weighted scoring |
| Bài 4 | Phân bổ ngân sách theo ngành và vùng | Linear Programming, fairness constraints |
| Bài 5 | Lựa chọn danh mục 15 dự án | Mixed Integer Programming |
| Bài 6 | Xếp hạng 6 vùng | TOPSIS, entropy weight |
| Bài 7 | Tối ưu đa mục tiêu | NSGA-II, Pareto frontier |
| Bài 8 | Tối ưu động giai đoạn 2026–2035 | Dynamic optimization, SLSQP |
| Bài 9 | Phân bổ AI và đào tạo lại lao động | CVXPY, employment transition |
| Bài 10 | Quy hoạch ngẫu nhiên và robust optimization | Pyomo, VSS, EVPI |
| Bài 11 | Học tăng cường cho chính sách đầu tư | Q-learning, DQN |
| Bài 12 | Hệ thống hỗ trợ quyết định tích hợp AIDEOM-VN | Integrated pipeline, scenario analysis |

---

## 3. Kiến trúc hệ thống tích hợp

Bài 12 tổ chức hệ thống theo sáu mô-đun:

| Mô-đun | Chức năng | Đầu ra chính |
|---|---|---|
| M1 | Dự báo kinh tế | GDP, D, AI, H năm 2030 |
| M2 | Đánh giá mức sẵn sàng vùng | Điểm và thứ hạng vùng |
| M3 | Phân bổ ngân sách | Ma trận phân bổ vùng × hạng mục |
| M4 | Lao động và AI | Việc làm tạo mới, thay thế, đào tạo lại |
| M5 | Rủi ro và bất định | Cyber risk, emission risk, inclusion |
| M6 | Dashboard tích hợp | KPI, xếp hạng, cảnh báo, tải CSV |

Dashboard Bài 12 hiện có:

- sơ đồ luồng dữ liệu M1–M6;
- bảng nguồn và phạm vi dữ liệu;
- năm kịch bản phát triển;
- bộ trọng số tương tác;
- `UserScore` và `UserRank`;
- bảng KPI tổng hợp;
- cảnh báo rủi ro;
- chức năng tải kết quả CSV.

---

## 4. Năm kịch bản phân tích

| Kịch bản | Định hướng |
|---|---|
| S1 – Truyền thống | Ưu tiên vốn vật chất |
| S2 – Số hóa nhanh | Tăng mạnh đầu tư hạ tầng số |
| S3 – AI dẫn dắt | Ưu tiên năng lực AI |
| S4 – Bao trùm số | Ưu tiên nhân lực và thu hẹp khoảng cách số |
| S5 – Tối ưu cân bằng | Kết hợp vốn, số hóa, AI và nhân lực |

Người dùng có thể thay đổi trọng số của các tiêu chí:

- GDP;
- việc làm;
- bao trùm;
- giảm cyber risk;
- giảm phát thải.

Các trọng số được chuẩn hóa tự động để có tổng bằng 1.

---

## 5. Dữ liệu sử dụng

Dự án sử dụng ba nhóm dữ liệu chính trong thư mục `data/`:

```text
data/
├── vietnam_macro_2020_2025.csv
├── vietnam_sectors_2024.csv
└── vietnam_regions_2024.csv
```

### 5.1. Dữ liệu vĩ mô

Tệp `vietnam_macro_2020_2025.csv` gồm GDP, vốn, lao động, số hóa, AI, nhân lực số và các biến kinh tế vĩ mô giai đoạn 2020–2025.

### 5.2. Dữ liệu ngành

Tệp `vietnam_sectors_2024.csv` gồm 10 ngành kinh tế với các chỉ tiêu tăng trưởng, tỷ trọng GDP, xuất khẩu, lao động, mức sẵn sàng AI và rủi ro tự động hóa.

### 5.3. Dữ liệu vùng

Tệp `vietnam_regions_2024.csv` gồm 6 vùng kinh tế – xã hội với các chỉ tiêu phát triển, đầu tư, số hóa và mức độ sẵn sàng công nghệ.

### 5.4. Nguồn tham chiếu

Dữ liệu và chỉ tiêu được tổng hợp, hiệu chỉnh hoặc tham chiếu từ:

- NSO/GSO;
- Bộ Kế hoạch và Đầu tư;
- Bộ Khoa học và Công nghệ;
- Bộ Thông tin và Truyền thông;
- World Bank;
- WIPO/GII;
- các giả định mô phỏng phục vụ bài tập.

Một số hệ số beta, rủi ro, tác động AI, đào tạo lại và chỉ số tổng hợp là tham số mô phỏng, không phải dự báo chính thức.

---

## 6. Cấu trúc thư mục

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
├── tests/
└── .devcontainer/
```

Trong phiên bản hiện tại, phần lớn giao diện và logic được triển khai trong `app.py`. Thư mục `src/` được sử dụng cho quá trình mô-đun hóa tiếp theo.

---

## 7. Yêu cầu môi trường

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
```

---

## 8. Cài đặt

### 8.1. Clone repository

```bash
git clone <repository-url>
cd aideom-vn-dashboard
```

### 8.2. Cài thư viện

```bash
python -m pip install -r requirements.txt
```

### 8.3. Kiểm tra xung đột phụ thuộc

```bash
python -m pip check
```

Kết quả mong đợi:

```text
No broken requirements found.
```

---

## 9. Chạy dashboard

```bash
python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Trong GitHub Codespaces, mở địa chỉ dạng:

```text
https://<codespace-name>-8501.app.github.dev
```

---

## 10. Kiểm tra cú pháp

```bash
python -m py_compile app.py
```

Nếu terminal không báo lỗi và quay lại dấu `$`, file `app.py` hợp lệ về cú pháp.

---

## 11. Chạy kiểm thử

```bash
python -m pytest -q
```

Các nhóm kiểm tra cần bao gồm:

- đủ năm kịch bản;
- tổng tỷ trọng mỗi kịch bản bằng 1;
- GDP dương;
- ngân sách không vượt giới hạn;
- số liệu vùng đầy đủ;
- VSS và EVPI không âm;
- pipeline trả về đúng cấu trúc;
- hành động RL hợp lệ.

---

## 12. Quy trình Git khuyến nghị

### 12.1. Kiểm tra trước khi commit

```bash
python -m py_compile app.py
git status
```

### 12.2. Commit thay đổi

```bash
git add app.py README.md requirements.txt
git commit -m "Cap nhat dashboard AIDEOM-VN"
git push origin main
```

Không nên dùng `git add .` khi thư mục có file backup hoặc file tạm chưa cần đưa lên repository.

### 12.3. Nhánh phát triển

```bash
git switch -c final-web-hardening
```

Sau khi kiểm tra thành công:

```bash
git switch main
git pull origin main
git merge final-web-hardening
git push origin main
```

---

## 13. Solver và phương pháp tính toán

Dự án sử dụng:

- SciPy HiGHS cho Linear Programming;
- PuLP/CBC cho Mixed Integer Programming;
- CVXPY cho bài toán phân bổ AI và đào tạo;
- Pyomo cho quy hoạch ngẫu nhiên và robust optimization;
- pymoo cho NSGA-II;
- Gymnasium cho môi trường học tăng cường;
- Stable-Baselines3 cho DQN;
- SLSQP cho tối ưu phi tuyến và tối ưu động.

---

## 14. Kết quả đầu ra

Mỗi trang bài tập có thể bao gồm:

- bảng dữ liệu đầu vào;
- mô hình toán học;
- bảng nghiệm;
- KPI;
- biểu đồ;
- phân tích độ nhạy;
- kiểm tra ràng buộc;
- nhận xét chính sách;
- nút tải CSV.

Bài 12 tổng hợp kết quả của các mô-đun thành hệ thống hỗ trợ ra quyết định.

---

## 15. Giới hạn của mô hình

1. Một số hệ số là giả định hoặc biến đại diện.
2. Dữ liệu chưa bao phủ đầy đủ mọi ngành, nghề và địa phương.
3. Mô hình chưa phản ánh toàn bộ chi phí điều chỉnh, nợ công và phản ứng hành vi.
4. Các kết quả AI và lao động không phải dự báo việc làm chính thức.
5. Kết quả tối ưu phụ thuộc vào trọng số, ràng buộc và giả định đầu vào.
6. Dashboard là công cụ hỗ trợ phân tích, không thay thế quyết định của cơ quan quản lý.

---

## 16. Nguyên tắc sử dụng kết quả

Kết quả từ AIDEOM-VN nên được sử dụng để:

- so sánh kịch bản;
- kiểm tra độ nhạy;
- minh họa đánh đổi chính sách;
- hỗ trợ thảo luận;
- xác định nhóm rủi ro;
- xây dựng khuyến nghị có điều kiện.

Quyết định thực tế cần kết hợp dữ liệu được kiểm định, ý kiến chuyên gia, đánh giá tác động, giới hạn an toàn, giải thích mô hình và phê duyệt của con người.

---

## 17. Khai báo sử dụng công cụ AI

Công cụ AI được sử dụng để hỗ trợ:

- xây dựng cấu trúc mã;
- phát hiện và sửa lỗi;
- diễn giải mô hình;
- xây dựng giao diện;
- chuẩn hóa tài liệu;
- đề xuất kiểm thử.

Người thực hiện chịu trách nhiệm kiểm tra mã nguồn, dữ liệu, kết quả, trích dẫn và nội dung nộp cuối cùng.

---

## 18. Thông tin dự án

- Dự án: VN AIDEOM-VN
- Loại sản phẩm: Dashboard mô hình ra quyết định
- Nền tảng: Python, Streamlit, GitHub
- Phạm vi: Việt Nam, dữ liệu 2020–2025
- Giai đoạn mô phỏng: 2026–2035

---

## 19. Trạng thái phát triển

### Đã hoàn thành

- 12 trang bài toán;
- dashboard Streamlit;
- dữ liệu vĩ mô, ngành và vùng;
- biểu đồ và bảng kết quả;
- NSGA-II;
- tối ưu động;
- CVXPY;
- Pyomo;
- Q-learning và DQN;
- Bài 12 tích hợp năm kịch bản;
- sơ đồ luồng dữ liệu;
- bảng nguồn dữ liệu;
- trọng số tương tác;
- tải kết quả CSV.

### Tiếp tục hoàn thiện

- mô-đun hóa toàn bộ logic sang `src/`;
- bổ sung unit test đầy đủ;
- tối ưu tự động kịch bản S5;
- tích hợp sâu hơn kết quả Bài 7, Bài 10 và Bài 11 vào pipeline Bài 12;
- tạo tab chi tiết cho từng kịch bản;
- hoàn thiện visual và responsive layout.

---

## 20. License

Dự án được xây dựng phục vụ mục đích học tập và nghiên cứu. Không sử dụng kết quả như khuyến nghị đầu tư hoặc dự báo chính thức nếu chưa có bước kiểm định độc lập.
