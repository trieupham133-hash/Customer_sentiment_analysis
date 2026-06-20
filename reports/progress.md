# Báo Cáo Tiến Độ Dự Án (Project Progress Report)

Dưới đây là chi tiết về tiến độ hiện tại của dự án **Customer Review Sentiment Analysis**, các phần đã hoàn thành, các tệp tin hiện có và định hướng các bước tiếp theo.

---

## 📌 Tổng Quan Trạng Thái Các Pha (Project Phase Status)

| Pha (Phase) | Trạng Thái | Người Phụ Trách | Chi Tiết Tiến Độ |
| :--- | :--- | :--- | :--- |
| **Phase 1: Data Understanding & EDA** | 🟢 **Hoàn thành (Notebook)** | Diệu Thùy | Đã phân tích dữ liệu Shopee và Augmented; đã kiểm tra số mẫu, missing/duplicate, phân phối rating/label và xuất các biểu đồ EDA vào `reports/figures/`. Phần mã nguồn `src/data/` chưa có module đóng gói. |
| **Phase 2: Text Preprocessing** | 🟢 **Hoàn thành (Notebook, Src & Data Output)** | Nhật Tiến, Quang Phong | Đã có notebook tiền xử lý, module trong `src/preprocessing/`, pipeline tổng hợp `preprocess_reviews.py`, stopwords tiếng Việt và dữ liệu đầu ra trong `data/processed/` + `data/interim/`. |
| **Phase 3: Feature Engineering** | 🟡 **Đã triển khai mã nguồn, cần hoàn tất artifact** | Hưng Trương, Quang Phong | Đã có notebook kiểm thử `04_feature_engineering.ipynb` và các module TF-IDF, Word2Vec, FastText trong `src/features/`. Hiện `data/processed/` mới có `tokens_ml_tfidf.npz` và `tokens_dl_phobert.pkl`; chưa thấy artifact mặc định `tfidf_features.pkl`, `word2vec_features.pkl`, `fasttext_features.pkl`. |
| **Phase 4: Baseline ML Models** | 🔴 **Chưa bắt đầu trong mã nguồn** | Quốc Triều | `05_baseline_model.ipynb` và các file `train_logistic_regression.py`, `train_naive_bayes.py`, `train_svm.py` vẫn trống. Chưa có model baseline được lưu trong `models/baseline/` hoặc `models/svm/`. |
| **Phase 5: Deep Learning Models** | 🟡 **Đã triển khai Notebook & một phần Src** | Ngọc Tiến, Hoàng Sang | Đã có notebook LSTM và PhoBERT fine-tuning; `src/models/train_lstm.py` định nghĩa BiLSTM, `train_phobert.py` có pipeline fine-tune PhoBERT chuẩn và PhoBERT + metadata. Chưa thấy checkpoint `.pt` được lưu trong `models/lstm/` hoặc `models/phobert/`. |
| **Phase 6: Evaluation & Error Analysis** | 🔴 **Chưa hoàn thiện** | Nhật Tiến | `08_evaluation.ipynb`, `09_error_analysis.ipynb` và các file trong `src/evaluation/` vẫn trống. Notebook PhoBERT có phần đánh giá nội bộ, nhưng chưa có module/tệp báo cáo đánh giá chung. |

---

## 📂 Kiểm Tra & Đánh Giá Chi Tiết Các Tệp Tin (File Audit)

### 1. Thư mục `notebooks/` (Thử nghiệm)
*   🟢 `01_data_understanding.ipynb` (9.8 KB): **Đã hoàn thành**. Thực hiện tải dữ liệu Shopee (9,599 dòng) và Augmented (1,348 dòng), kiểm tra cấu trúc dữ liệu, missing values, duplicate và phân phối rating/label.
*   🟢 `02_eda.ipynb` (4.4 MB - có lưu output biểu đồ): **Đã hoàn thành**. Trực quan hóa phân phối nhãn, rating, label theo rating, độ dài review và biểu đồ t-SNE cho khả năng tách đặc trưng.
*   🟢 `03_text_preprocessing.ipynb` (65.4 KB): **Đã hoàn thành**. Demo/kiểm thử các bước làm sạch, chuẩn hóa, xử lý slang/teencode, tách từ và tạo token/đặc trưng phục vụ ML/DL.
*   🟡 `04_feature_engineering.ipynb` (6.2 KB): **Đã có notebook kiểm thử**. Notebook gọi các pipeline TF-IDF, Word2Vec, FastText và kiểm tra artifact đầu ra, nhưng cần chạy/đồng bộ lại để đảm bảo các file artifact mặc định xuất hiện trong `data/processed/`.
*   🔴 `05_baseline_model.ipynb` (0 bytes): **Chưa triển khai**.
*   🟡 `06_lstm_model.ipynb` (319.6 KB): **Đã triển khai thử nghiệm LSTM**. Notebook chuẩn bị tensor từ `tokens_dl_phobert.pkl`, chia train/test 80/20, tạo DataLoader, định nghĩa train/test step và có log test accuracy khoảng 92.65% ở epoch đầu.
*   🟡 `07_phobert_finetuning.ipynb` (114.2 KB): **Đã triển khai thử nghiệm PhoBERT**. Notebook fine-tune PhoBERT chuẩn và PhoBERT + metadata (`rating_stars`, `has_media`), có log validation accuracy tốt nhất khoảng 96.94% cho model chuẩn và 96.89% cho model hybrid.
*   🔴 `08_evaluation.ipynb` (0 bytes): **Chưa triển khai**.
*   🔴 `09_error_analysis.ipynb` (0 bytes): **Chưa triển khai**.

### 2. Thư mục `src/` (Mã nguồn dự án)
*   🔴 `src/data/`: **Chưa có module**. Thư mục/filenames được mô tả trong README nhưng hiện chưa có `load_data.py` hoặc `make_dataset.py`.
*   🟢 `src/preprocessing/`: **Đã hoàn thành phần lõi và có pipeline mới**.
    *   `clean_text.py` (691 bytes): Hàm làm sạch cơ bản: lowercase, xóa URL, HTML, dấu câu, ký tự đặc biệt và chuẩn hóa khoảng trắng.
    *   `normalize_text.py` (845 bytes): Chuẩn hóa Unicode, slang/teencode cơ bản và ký tự lặp.
    *   `word_segmentation.py` (466 bytes): Hỗ trợ tách từ bằng PyVi hoặc Underthesea.
    *   `tokenizer.py` (443 bytes): Chuẩn bị tokenizer TF-IDF và PhoBERT.
    *   `preprocess_accented_reviews.py` (3.9 KB): Pipeline cũ cho Shopee accented reviews, xuất `clean_reviews.csv`, vector TF-IDF và token PhoBERT.
    *   `preprocess_reviews.py` (16.2 KB): Pipeline tổng hợp mới, đọc 2 file JSONL raw, sửa mojibake, chuẩn hóa emoji/slang/dấu tiếng Việt, tách từ, bỏ stopwords cho baseline và xuất các file processed/interim phục vụ ML/DL.
*   🟡 `src/features/`: **Đã có mã nguồn Feature Engineering**.
    *   `tfidf_features.py` (5.2 KB): Pipeline tạo TF-IDF với unigram/bigram, `max_features=30000`, lưu vectorizer, matrix, ids và labels.
    *   `word2vec_features.py` (5.7 KB): Huấn luyện Word2Vec bằng Gensim, tạo document embedding trung bình và lưu model/artifact.
    *   `fasttext_features.py` (6.8 KB): Huấn luyện FastText bằng Gensim, tạo document embedding trung bình và lưu model/artifact.
*   🔴 `src/models/train_logistic_regression.py`, `train_naive_bayes.py`, `train_svm.py`: **Vẫn trống (0 bytes)**.
*   🟡 `src/models/train_lstm.py` (2.4 KB): **Đã có định nghĩa model BiLSTM** với embedding, bidirectional LSTM, fully connected layer và sigmoid output; chưa có script huấn luyện/lưu checkpoint hoàn chỉnh.
*   🟡 `src/models/train_phobert.py` (9.7 KB): **Đã có pipeline fine-tune PhoBERT** gồm dataset class, PhoBERT chuẩn, PhoBERT + metadata, train/evaluate loop và logic lưu checkpoint tốt nhất vào `models/phobert/`; cần chạy xác nhận để sinh file model.
*   🔴 `src/evaluation/`: **Chưa triển khai**. `metrics.py`, `confusion_matrix.py`, `error_analysis.py` đều đang trống.
*   🟡 `src/utils/`: `config.py` đã có cấu hình đường dẫn dùng chung cho data/features; `helpers.py` vẫn trống.

### 3. Thư mục `reports/` (Báo cáo & Kết quả)
*   🟢 `reports/figures/`: Đã lưu các biểu đồ EDA gồm `class_distribution.png`, `rating_distribution.png`, `label_by_rating_distribution.png`, `word_count_distribution.png`, `tsne_feature_separability.png`, `tsne_feature_separability_aug.png`.
*   🔴 `reports/tables/`: Trống (chỉ có `.gitkeep`), chưa có bảng so sánh model hoặc mẫu lỗi.
*   🔴 `final_report.md`: Trống (0 bytes).
*   🟢 `progress.md`: Đã được cập nhật lại theo trạng thái hiện tại của repository.
*   🟡 `data/interim/` và `data/processed/`: Đã có dữ liệu trung gian/processed quan trọng gồm `df_baseline_segmented.csv`, `df_lstm_segmented.csv`, `reviews_preprocessed_all.csv`, `tokens_ml_tfidf.npz`, `tokens_dl_phobert.pkl`; tuy nhiên một số file mà script mới kỳ vọng như `reviews_baseline_tfidf.csv`, `reviews_lstm_sequence.csv`, `tfidf_features.pkl`, `word2vec_features.pkl`, `fasttext_features.pkl` chưa xuất hiện trong thư mục hiện tại.

---

## ⚠️ Các Điểm Bất Thường & Lưu Ý Quan Trọng (Discrepancies & Notes)

1.  **Sự khác biệt về nhãn phân loại (Sentiment Labels):**
    *   *Trong README.md/Dataset Description:* Vẫn ghi nhãn gồm **Positive, Neutral, Negative**.
    *   *Trong Project Overview và dữ liệu thực tế:* Bài toán hiện đang vận hành như phân loại **2 nhãn**: `positive` và `negative`; không thấy nhãn `neutral` trong dữ liệu raw/processed đã kiểm tra.
2.  **Sự khác biệt về tên notebook giữa README và thực tế:**
    *   README liệt kê `06_svm_model.ipynb`, `07_lstm_model.ipynb`, `08_phobert_finetuning.ipynb`, `09_evaluation.ipynb`, `10_error_analysis.ipynb`.
    *   Thư mục thực tế có `06_lstm_model.ipynb`, `07_phobert_finetuning.ipynb`, `08_evaluation.ipynb`, `09_error_analysis.ipynb` và không có `06_svm_model.ipynb` hoặc `10_error_analysis.ipynb`.
3.  **Một số đường dẫn output chưa đồng bộ giữa các pipeline:**
    *   `preprocess_reviews.py` dự kiến xuất `reviews_baseline_tfidf.csv`, `reviews_lstm_sequence.csv`, `preprocessing_metadata.json`, nhưng hiện `data/processed/` chỉ thấy `reviews_preprocessed_all.csv`, `tokens_ml_tfidf.npz`, `tokens_dl_phobert.pkl`.
    *   `src/features/*.py` dự kiến đọc các CSV processed mặc định và lưu artifact `.pkl`, nhưng các artifact này chưa có trong `data/processed/`.
4.  **Chưa có checkpoint model đã lưu:**
    *   Notebook và script DL đã có logic huấn luyện/lưu model, nhưng thư mục `models/baseline/`, `models/svm/`, `models/lstm/`, `models/phobert/` hiện chỉ có `.gitkeep`.
5.  **Mã hóa tiếng Việt trong một số file có dấu hiệu mojibake:**
    *   Nhiều chuỗi tiếng Việt trong README/script hiển thị dạng lỗi encoding. Nên chuẩn hóa lại file về UTF-8 để tránh lỗi khi trình bày báo cáo, huấn luyện hoặc xử lý text.

---

## 🚀 Các Bước Đề Xuất Tiếp Theo (Recommended Next Steps)

1.  **Đồng bộ README với repository thực tế:**
    *   Cập nhật bài toán thành 2 lớp nếu nhóm tiếp tục dùng `positive/negative`, hoặc bổ sung dữ liệu/guideline nếu muốn giữ 3 lớp có `neutral`.
    *   Sửa danh sách notebook cho khớp: `06_lstm_model.ipynb`, `07_phobert_finetuning.ipynb`, `08_evaluation.ipynb`, `09_error_analysis.ipynb`.
2.  **Chạy lại và chuẩn hóa pipeline tiền xử lý:**
    *   Chạy `src/preprocessing/preprocess_reviews.py` để sinh đầy đủ `reviews_preprocessed_all.csv`, `reviews_baseline_tfidf.csv`, `reviews_lstm_sequence.csv`, `preprocessing_metadata.json`.
    *   Giữ một bộ output chuẩn duy nhất để các phase sau không bị lệch đường dẫn.
3.  **Hoàn tất Phase 3 (Feature Engineering):**
    *   Chạy `04_feature_engineering.ipynb` hoặc các script trong `src/features/` để sinh `tfidf_features.pkl`, `word2vec_features.pkl`, `fasttext_features.pkl` và model embedding tương ứng.
    *   Ghi lại shape/dung lượng artifact vào `reports/tables/` để tiện kiểm tra.
4.  **Triển khai Phase 4 (Baseline ML Models):**
    *   Viết `train_logistic_regression.py`, `train_naive_bayes.py`, `train_svm.py` dựa trên artifact TF-IDF.
    *   Lưu model vào `models/baseline/` hoặc `models/svm/` và xuất metric ban đầu.
5.  **Hoàn thiện Phase 5 (Deep Learning Models):**
    *   Bổ sung script train/evaluate/lưu checkpoint hoàn chỉnh cho LSTM.
    *   Chạy `train_phobert.py` hoặc notebook PhoBERT để sinh checkpoint thật trong `models/phobert/`.
6.  **Triển khai Phase 6 (Evaluation & Error Analysis):**
    *   Viết `src/evaluation/metrics.py`, `confusion_matrix.py`, `error_analysis.py`.
    *   Tạo `reports/tables/model_comparison.csv`, `reports/tables/error_samples.csv` và `reports/figures/confusion_matrix.png`.
    *   Cập nhật `final_report.md` sau khi có kết quả so sánh model đầy đủ.
