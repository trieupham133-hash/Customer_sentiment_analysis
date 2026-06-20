# Customer Review Sentiment Analysis

## Project Overview

Customer reviews contain valuable signals about customer satisfaction, product quality, service issues, and brand perception. However, manually reading and classifying large volumes of reviews is slow, inconsistent, and difficult to scale.

This project builds an end-to-end Natural Language Processing pipeline to classify Vietnamese customer reviews into three sentiment categories: **Positive**, and **Negative**. The repository is organized for reproducible experimentation, model comparison, evaluation, and final reporting.

## Objectives

- Collect and organize Vietnamese customer review data.
- Clean noisy review text and normalize Vietnamese-specific language patterns.
- Apply Vietnamese word segmentation and tokenization.
- Engineer classical and neural text features.
- Train and compare baseline machine learning, deep learning, and transformer models.
- Evaluate models using standard classification metrics.
- Perform error analysis to understand failure cases.
- Produce a final project report with insights and recommendations.

## Dataset Description

| Field | Description |
| ----- | ----------- |
| Source | Placeholder: e-commerce platforms, app reviews, survey responses, or public Vietnamese review datasets |
| Number of Samples | Placeholder: to be updated after data collection |
| Features | Review text, optional metadata such as product category, rating, date, or platform |
| Labels | Positive, Neutral, Negative |

## NLP Pipeline Diagram

```text
Raw Reviews
    ↓
Cleaning
    ↓
Normalization
    ↓
Word Segmentation
    ↓
Tokenization
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Evaluation
    ↓
Error Analysis
```

## Models

- Logistic Regression
- Naive Bayes
- Support Vector Machine
- LSTM
- PhoBERT

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Repository Structure

```text
sentiment-analysis-vietnamese/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_svm_model.ipynb
│   ├── 07_lstm_model.ipynb
│   ├── 08_phobert_finetuning.ipynb
│   ├── 09_evaluation.ipynb
│   └── 10_error_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── load_data.py
│   ├── preprocessing/
│   │   ├── clean_text.py
│   │   ├── normalize_text.py
│   │   ├── word_segmentation.py
│   │   └── tokenizer.py
│   ├── features/
│   │   ├── tfidf_features.py
│   │   ├── word2vec_features.py
│   │   └── fasttext_features.py
│   ├── models/
│   │   ├── train_logistic_regression.py
│   │   ├── train_naive_bayes.py
│   │   ├── train_svm.py
│   │   ├── train_lstm.py
│   │   └── train_phobert.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   └── error_analysis.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md
│
├── models/
│   ├── baseline/
│   ├── svm/
│   ├── lstm/
│   └── phobert/
│
└── references/
    ├── papers/
    └── notes/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run notebooks for experimentation:

```bash
jupyter notebook notebooks/
```

Prepare datasets:

```bash
python src/data/make_dataset.py
```

Train baseline models:

```bash
python src/models/train_logistic_regression.py
python src/models/train_naive_bayes.py
python src/models/train_svm.py
```

Evaluate trained models:

```bash
python src/evaluation/metrics.py
python src/evaluation/error_analysis.py
```

## Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression |          |          |
| SVM                 |          |          |
| LSTM                |          |          |
| PhoBERT             |          |          |

## Future Improvements

- Expand dataset coverage across domains and review platforms.
- Add active labeling guidelines for consistent sentiment annotation.
- Compare additional Vietnamese embeddings and transformer checkpoints.
- Add experiment tracking with MLflow or Weights & Biases.
- Add model explainability for production-facing insights.
- Package the final pipeline for inference once deployment becomes in scope.

---
# 📌 PHASES FOR MEMBERS

---

# PHASE 1 — DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)

## 🎯 Mục tiêu
Hiểu rõ dữ liệu trước khi xây dựng mô hình.

**Thành viên phụ trách:** Diệu Thùy

## 📂 Làm việc trên

```text
notebooks/
├── 01_data_understanding.ipynb
└── 02_eda.ipynb

src/data/
└── load_data.py
```

## 📥 Input

```text
data/raw/reviews.csv
```

## 🛠 Công việc

### 1. Khám phá dữ liệu
- Đọc dữ liệu
- Kiểm tra cấu trúc dữ liệu

### 2. Kiểm tra chất lượng dữ liệu
- Missing Values
- Duplicate Records
- Dữ liệu bất thường (Outliers / Invalid Values)
- Mất cân bằng nhãn (Class Imbalance)

### 3. Thống kê mô tả
- Số lượng review
- Số lượng nhãn
- Tỷ lệ Positive / Negative
- Phân bố Rating

### 4. Phân tích văn bản
- Độ dài review
- Tần suất xuất hiện từ
- Word Cloud

### 5. Trực quan hóa dữ liệu
- Vẽ các biểu đồ mô tả dữ liệu

## 📚 Thư viện
 
 - Có thể xem thêm trong requirements.txt

## 📤 Kết quả đầu ra

```text
reports/figures/
├── class_distribution.png
└── rating_distribution.png
```

## 🧠 Kiến thức cần nắm

- Missing Values
- Duplicate Data
- Class Imbalance
- Data Distribution

---

# PHASE 2 — TEXT PREPROCESSING

## 🎯 Mục tiêu

Làm sạch và chuẩn hóa toàn bộ dữ liệu văn bản.

**Thành viên phụ trách:** Nhật Tiến, Quang Phong

## 📂 Làm việc trên

```text
notebooks/
└── 03_text_preprocessing.ipynb

src/preprocessing/
├── clean_text.py
├── normalize_text.py
├── word_segmentation.py
└── tokenizer.py
```

## 📥 Input

```text
data/raw/reviews.csv
```

## 🛠 Công việc

### clean_text.py
- Chuyển về chữ thường (Lowercase)
- Xóa URL
- Xóa HTML Tags
- Xóa dấu câu
- Xóa ký tự đặc biệt

### normalize_text.py
- Chuẩn hóa từ ngữ
- Chuẩn hóa ký tự đặc biệt

### word_segmentation.py
Thử nghiệm các công cụ:
- Underthesea
- PyVi

### tokenizer.py
Chuẩn bị tokenizer cho:
- TF-IDF
- PhoBERT

## 📚 Thư viện

- regex
- underthesea
- pyvi
- transformers

## 📤 Kết quả đầu ra

```text
data/interim/clean_reviews.csv
```

## 🧠 Kiến thức cần nắm

- Tokenization
- Word Segmentation
- Text Normalization
- Stopwords

---

# PHASE 3 — FEATURE ENGINEERING

## 🎯 Mục tiêu

Biến dữ liệu văn bản thành vector số phục vụ huấn luyện mô hình.

**Thành viên phụ trách:** Hưng Trương

## 📂 Làm việc trên

```text
notebooks/
└── 04_feature_engineering.ipynb

src/features/
├── tfidf_features.py
├── word2vec_features.py
└── fasttext_features.py
```

## 📥 Input

```text
data/interim/clean_reviews.csv
```

## 🛠 Công việc

### 1. Trích xuất đặc trưng văn bản
- TF-IDF
- Word2Vec
- FastText

### 2. Sinh đặc trưng từ PhoBERT
- PhoBERT Embeddings

## 📚 Thư viện

- scikit-learn
- transformers

## 📤 Kết quả đầu ra

```text
data/processed/
├── tfidf_features.pkl
├── phobert_features.pkl
└── metadata.csv
```

## 🧠 Kiến thức cần nắm

- Bag of Words
- TF-IDF
- Embedding
- Dense Vector
- Sparse Vector

---

# PHASE 4 — BASELINE MACHINE LEARNING MODELS

## 🎯 Mục tiêu

Xây dựng các mô hình Machine Learning cơ sở để làm chuẩn so sánh.

**Thành viên phụ trách:** Quốc Triều

## 📂 Làm việc trên

```text
notebooks/
├── 05_baseline_model.ipynb
└── 06_svm_model.ipynb

src/models/
├── train_logistic_regression.py
├── train_naive_bayes.py
└── train_svm.py
```

## 📥 Input

```text
data/processed/tfidf_features.pkl
```

## 🛠 Công việc

Xây dựng và đánh giá các mô hình:

- Logistic Regression
- Naive Bayes
- SVM

Mục tiêu:
- Có mô hình phân loại cảm xúc hoạt động tốt
- Làm baseline cho các mô hình Deep Learning

## 📚 Thư viện

- scikit-learn

## 📤 Kết quả đầu ra

```text
models/baseline/
```

## 🧠 Kiến thức cần nắm

- Classification
- Decision Boundary
- Hyperparameter
- Overfitting
- Underfitting

---

# PHASE 5 — DEEP LEARNING MODELS

## 🎯 Mục tiêu

Xây dựng các mô hình NLP hiện đại dựa trên Deep Learning.

**Thành viên phụ trách:** Ngọc Tiến, Hoàng Sang

## 📂 Làm việc trên

```text
notebooks/
├── 07_lstm_model.ipynb
└── 08_phobert_finetuning.ipynb

src/models/
├── train_lstm.py
└── train_phobert.py
```

## 📥 Input

```text
data/interim/clean_reviews.csv
```

## 🛠 Công việc

### 1. LSTM

```text
Text
 ↓
Embedding
 ↓
LSTM
 ↓
Linear
 ↓
Softmax
```

### 2. PhoBERT Fine-Tuning

```text
Text
 ↓
PhoBERT
 ↓
Classification Head
 ↓
Softmax
```

### 3. PhoBERT + Metadata

```text
PhoBERT Embedding
      +
rating_stars
      +
has_media
      ↓
Dense Layer
      ↓
Prediction
```

Có thể tận dụng pipeline hiện tại hoặc xây dựng pipeline mới nếu cần, miễn đạt được mô hình Deep Learning hiệu quả.

## 📚 Thư viện

- torch
- transformers
- datasets

## 📤 Kết quả đầu ra

```text
models/
├── lstm/
└── phobert/
```

## 🧠 Kiến thức cần nắm

- Neural Network
- Embedding
- Backpropagation
- Fine-Tuning
- Transformer
- Attention Mechanism

---

# PHASE 6 — MODEL EVALUATION & ERROR ANALYSIS

## 🎯 Mục tiêu

Đánh giá, so sánh và phân tích lỗi của các mô hình.

**Thành viên phụ trách:** Nhật Tiến

## 📂 Làm việc trên

```text
notebooks/
├── 09_evaluation.ipynb
└── 10_error_analysis.ipynb

src/evaluation/
├── metrics.py
├── confusion_matrix.py
└── error_analysis.py
```

## 📥 Input

```text
Tất cả các mô hình đã được huấn luyện.
```

## 🛠 Công việc

### 1. Đánh giá mô hình
- Accuracy
- Precision
- Recall
- F1 Score

### 2. Trực quan hóa
- Confusion Matrix

### 3. So sánh mô hình
- So sánh kết quả giữa các mô hình

### 4. Error Analysis

Thu thập các mẫu dự đoán sai và phân loại nguyên nhân:

- Teencode
- Sai chính tả
- Sarcasm
- Review quá ngắn
- Từ ngữ đa nghĩa

## 📚 Thư viện

- scikit-learn
- pandas
- matplotlib
- seaborn

## 📤 Kết quả đầu ra

```text
reports/
├── figures/
│   └── confusion_matrix.png
└── tables/
    ├── model_comparison.csv
    └── error_samples.csv
```

## 🧠 Kiến thức cần nắm

- Precision
- Recall
- F1 Score
- Confusion Matrix
- False Positive
- False Negative

# Bảng tóm tắt việc phân chia công việc:
