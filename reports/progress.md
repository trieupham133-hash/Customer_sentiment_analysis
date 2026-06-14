# Báo Cáo Tiến Độ Dự Án (Project Progress Report)

Dưới đây là chi tiết về tiến độ hiện tại của dự án **Customer Review Sentiment Analysis**, các phần đã hoàn thành, các tệp tin hiện có và định hướng các bước tiếp theo.

---

## 📌 Tổng Quan Trạng Thái Các Pha (Project Phase Status)

| Pha (Phase) | Trạng Thái | Người Phụ Trách | Chi Tiết Tiến Độ |
| :--- | :--- | :--- | :--- |
| **Phase 1: Data Understanding & EDA** | 🟢 **Hoàn thành (Notebook)** | Diệu Thùy | Đã phân tích dữ liệu Shopee & Augmented trong Notebook; đã xuất biểu đồ phân phối nhãn & rating. Chưa chuyển mã nguồn sang tệp tin Python (`src/data/load_data.py`). |
| **Phase 2: Text Preprocessing** | 🟢 **Hoàn thành (Notebook & Src)** | Nhật Tiến, Quang Phong | Đã xây dựng hoàn chỉnh Notebook tiền xử lý và viết các module tiền xử lý văn bản trong `src/preprocessing/`. |
| **Phase 3: Feature Engineering** | 🔴 **Chưa bắt đầu** | Hưng Trương, Quang Phong | Tệp tin Notebook và các tệp mã nguồn trích xuất đặc trưng đều đang trống (0 bytes). |
| **Phase 4: Baseline ML Models** | 🔴 **Chưa bắt đầu** | Quốc Triều | Tệp tin Notebook và các tệp mã nguồn huấn luyện mô hình ML cơ sở đang trống (0 bytes). |
| **Phase 5: Deep Learning Models** | 🔴 **Chưa bắt đầu** | Ngọc Tiến, Hoàng Sang | Tệp tin Notebook và các tệp mã nguồn huấn luyện mô hình DL đang trống (0 bytes). |
| **Phase 6: Evaluation & Error Analysis** | 🔴 **Chưa bắt đầu** | Nhật Tiến | Tệp tin Notebook và các tệp mã nguồn đánh giá, phân tích lỗi đang trống (0 bytes). |

---

## 📂 Kiểm Tra & Đánh Giá Chi Tiết Các Tệp Tin (File Audit)

### 1. Thư mục `notebooks/` (Thử nghiệm)
*   🟢 `01_data_understanding.ipynb` (9.8 KB): **Đã hoàn thành**. Thực hiện tải dữ liệu Shopee (9,599 dòng) và Augmented (1,348 dòng), kiểm tra dữ liệu thiếu, trùng lặp và phân phối rating/nhãn.
*   🟢 `02_eda.ipynb` (4.3 MB - do lưu output biểu đồ): **Đã hoàn thành**. Trực quan hóa phân phối nhãn theo rating và độ dài từ/ký tự của đánh giá.
*   🟢 `03_text_preprocessing.ipynb` (10.1 KB): **Đã hoàn thành**. Demo các bước chuẩn hóa Unicode, loại bỏ ký tự đặc biệt, sửa teencode/slang, tách từ (PyVi & Underthesea) và thử nghiệm TF-IDF & PhoBERT Tokenizer.
*   🔴 Các notebook còn lại (`04_feature_engineering.ipynb` đến `09_error_analysis.ipynb`): **Đều trống (0 bytes)**.

### 2. Thư mục `src/` (Mã nguồn dự án)
*   🟢 `src/preprocessing/`: **Đã hoàn thành mã nguồn cơ bản**.
    *   `clean_text.py` (660 bytes): Hàm làm sạch (lowercase, xóa URL, HTML, dấu câu, ký tự đặc biệt).
    *   `normalize_text.py` (845 bytes): Chuẩn hóa Unicode NFC, sửa slang/teencode cơ bản, loại bỏ ký tự lặp.
    *   `word_segmentation.py` (466 bytes): Hỗ trợ tách từ bằng PyVi hoặc Underthesea.
    *   `tokenizer.py` (443 bytes): Chuyển hóa văn bản sang vector TF-IDF hoặc token PhoBERT.
    *   `preprocess_accented_reviews.py` (3.9 KB): Tệp điều phối chính (pipeline) đọc dữ liệu thô từ Shopee, chạy qua các bước làm sạch -> chuẩn hóa -> tách từ -> lưu vào `data/interim/clean_reviews.csv` và tạo vector TF-IDF/PhoBERT lưu vào `data/processed/`.
*   🔴 Các thư mục khác trong `src/` (`src/data/`, `src/features/`, `src/models/`, `src/evaluation/`, `src/utils/`): **Tất cả các tệp tin Python đều trống (0 bytes)**.

### 3. Thư mục `reports/` (Báo cáo & Kết quả)
*   🟢 `reports/figures/`: Đã lưu trữ các biểu đồ phân tích từ EDA bao gồm: `class_distribution.png`, `rating_distribution.png`, `label_by_rating_distribution.png`, `word_count_distribution.png`, và các biểu đồ biểu diễn độ phân tách đặc trưng t-SNE.
*   🔴 `reports/tables/`: Trống (chỉ có `.gitkeep`).
*   🔴 `final_report.md` & `progress.md`: Trống (0 bytes) trước khi cập nhật báo cáo này.

---

## ⚠️ Các Điểm Bất Thường & Lưu Ý Quan Trọng (Discrepancies & Notes)

1.  **Sự khác biệt về Nhãn phân loại (Sentiment Labels):**
    *   *Trong README.md/Overview:* Đề cập dự án sẽ phân loại đánh giá thành **3 nhóm**: *Positive* (Tích cực), *Neutral* (Trung lập), và *Negative* (Tiêu cực).
    *   *Thực tế dữ liệu thô:* Cả hai tập dữ liệu (`shopee_reviews_dataset.jsonl` và `aug_unaccented_reviews.jsonl`) đều **chỉ chứa 2 nhãn**: **`positive`** và **`negative`**, hoàn toàn không có nhãn trung lập.
2.  **Sự khác biệt về tên tệp tin Notebook giữa README và thực tế:**
    *   Trong `README.md` liệt kê danh sách tệp gồm `06_svm_model.ipynb` và kết thúc ở `10_error_analysis.ipynb`.
    *   Trên thư mục thực tế chỉ có 9 tệp tin, tệp LSTM là `06_lstm_model.ipynb`, PhoBERT là `07_phobert_finetuning.ipynb`, và kết thúc ở `09_error_analysis.ipynb`. Cần đồng bộ lại mục lục trong README để khớp với thực tế.
3.  **Tệp tin mã nguồn trống:**
    *   Mặc dù pha 1 (Data Understanding & EDA) đã hoàn thành trong Notebook, nhưng tệp `src/data/load_data.py` vẫn trống. Nên đóng gói hàm đọc/tải dữ liệu từ Notebook vào đây để các pha sau có thể gọi lại một cách modular.

---

## 🚀 Các Bước Đề Xuất Tiếp Theo (Recommended Next Steps)

1.  **Đồng bộ dữ liệu trung gian (Interim Data):** Chạy script `src/preprocessing/preprocess_accented_reviews.py` để tạo ra tệp dữ liệu đã làm sạch `clean_reviews.csv` nhằm phục vụ cho pha Feature Engineering.
2.  **Triển khai Phase 3 (Feature Engineering):**
    *   Hiện thực hóa các phương pháp trích xuất đặc trưng truyền thống như TF-IDF, Word2Vec, FastText trong `src/features/`.
    *   Nghiên cứu cách chuẩn bị ma trận đặc trưng cho mô hình Deep Learning (padding, tạo DataLoader trong PyTorch).
3.  **Xây dựng Baseline Models (Phase 4):**
    *   Hiện thực hóa việc huấn luyện Logistic Regression, Naive Bayes, và SVM trong `src/models/train_*.py`.
