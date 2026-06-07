# Customer Review Sentiment Analysis

## Project Overview

Customer reviews contain valuable signals about customer satisfaction, product quality, service issues, and brand perception. However, manually reading and classifying large volumes of reviews is slow, inconsistent, and difficult to scale.

This project builds an end-to-end Natural Language Processing pipeline to classify Vietnamese customer reviews into three sentiment categories: **Positive**, **Neutral**, and **Negative**. The repository is organized for reproducible experimentation, model comparison, evaluation, and final reporting.

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
# PHASES FOR MEMBERS
## PHASE 1 — DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)
#### Mục tiêu:
Hiểu dữ liệu trước khi xây dựng mô hình.
Thành viên phụ trách: Diệu Thùy

#### Làm việc trên:
notebooks/
├── 01_data_understanding.ipynb
└── 02_eda.ipynb
src/data/
├── load_data.py

#### Input:
data/raw/reviews.csv

#### Công việc:
1. Đọc dữ liệu.
2. Kiểm tra:
* Missing values
* Duplicate records
* Dữ liệu bất thường
* Nhãn mất cân bằng

3. Thống kê:
* Số lượng review
* Số lượng nhãn
* Tỷ lệ Positive / Neutral / Negative
* Phân bố rating

4. Phân tích văn bản:
* Độ dài review
* Tần suất từ xuất hiện
* WordCloud

5. Vẽ biểu đồ

#### Thư viện có thể sẽ cần:
* pandas
* numpy
* matplotlib
* seaborn
* wordcloud

Kết quả đầu ra:
reports/figures/

Ví dụ:
reports/figures/class_distribution.png
reports/figures/rating_distribution.png

Yêu cầu kiến thức:
Hiểu: Chung là đang làm gì
* Missing Value
* Duplicate Data
* Class Imbalance
* Data Distribution

---

## PHASE 2 — TEXT PREPROCESSING
#### Mục tiêu:
Làm sạch và chuẩn hóa toàn bộ dữ liệu văn bản.
Thành viên phụ trách: Nhật Tiến


#### Làm việc trên:
notebooks/
└── 03_text_preprocessing.ipynb
src/preprocessing/
├── clean_text.py
├── normalize_text.py
├── word_segmentation.py
└── tokenizer.py

#### Input:
data/raw/reviews.csv

#### Công việc:
clean_text.py:
* lowercase
* remove url
* remove html
* remove punctuation
* remove special characters

normalize_text.py:
- Chuẩn hóa từ ngữ
- Xử lí các ký tự đặt biệt

word_segmentation.py:
Thử nghiệm:
* Underthesea
* PyVi

tokenizer.py
Chuẩn bị tokenizer cho:
* TF-IDF
* PhoBERT

#### Thư viện:
* regex
* underthesea
* pyvi
* transformers

#### Kết quả đầu ra:
data/interim/clean_reviews.csv

#### Yêu cầu kiến thức:
Hiểu: Mình đang làm gì
* Tokenization
* Word Segmentation
* Text Normalization
* Stopwords

---

## PHASE 3 — FEATURE ENGINEERING
#### Mục tiêu:
Biến dữ liệu text thành vector số.
Thành viên phụ trách: Hưng Trương

#### Làm việc trên:
notebooks/
└── 04_feature_engineering.ipynb
src/features/
├── tfidf_features.py
├── word2vec_features.py
└── fasttext_features.py

#### Input:
data/interim/clean_reviews.csv

#### Công việc:
1. Xây dựng các vector số từ chữ
2. PhoBERT Features

Thư viện:
* scikit-learn
* transformers

Kết quả đầu ra:
data/processed/tfidf_features.pkl
data/processed/phobert_features.pkl
data/processed/metadata.csv

Yêu cầu kiến thức:
Hiểu: Mình đang làm gì
* Bag of Words
* TF-IDF
* Embedding
* Dense Vector
* Sparse Vector

---

## PHASE 4 — BASELINE MACHINE LEARNING MODELS
#### Mục tiêu:
Tạo các mô hình cơ sở để làm chuẩn so sánh.
Thành viên phụ trách: Quốc Triều

#### Làm việc trên:
notebooks/
├── 05_baseline_model.ipynb
└── 06_svm_model.ipynb
src/models/
├── train_logistic_regression.py
├── train_naive_bayes.py
└── train_svm.py

#### Input:
data/processed/tfidf_features.pkl

#### Công việc:
Xây dựng được một mô hình baseline phân loại được và phân loại tốt cảm xúc của khách hàng

Thư viện:
* scikit-learn

#### Kết quả đầu ra:
models/baseline/

#### Yêu cầu kiến thức:
Hiểu: Mình đang làm gì
* Classification
* Decision Boundary
* Hyperparameter
* Overfitting
* Underfitting

---

## PHASE 5 — DEEP LEARNING MODELS
#### Mục tiêu:
Xây dựng các mô hình hiện đại cho NLP.
Thành viên phụ trách: Ngọc Tiến, Hoàng Sang

#### Làm việc trên:
notebooks/
├── 07_lstm_model.ipynb
└── 08_phobert_finetuning.ipynb 
src/models/
├── train_lstm.py
└── train_phobert.py

#### Input:
data/interim/clean_reviews.csv

#### Công việc:
Có thể dựa trên các pipeline trên, không thì tạo mới cũng oke thoi nhưng chung quy là tạo được một mô hình DL tốt 
1. LSTM
Text
↓
Embedding
↓
LSTM
↓
Linear
↓
Softmax

3. PhoBERT Fine-Tuning
Text
↓
PhoBERT
↓
Classification Head
↓
Softmax

4. PhoBERT + Metadata
PhoBERT Embedding
*
rating_stars
*
has_media
↓
Dense Layer
↓
Prediction

#### Thư viện:
* torch
* transformers
* datasets

#### Kết quả đầu ra:
models/lstm/
models/phobert/

#### Yêu cầu kiến thức:
Hiểu:
* Neural Network
* Embedding
* Backpropagation
* Fine-Tuning
* Transformer
* Attention

---

## PHASE 6 — MODEL EVALUATION & ERROR ANALYSIS
#### Mục tiêu:
So sánh các mô hình, giải thích các tham số
Thành viên phụ trách: Nhật Tiến

#### Làm việc trên:
notebooks/
├── 09_evaluation.ipynb
└── 10_error_analysis.ipynb
src/evaluation/
├── metrics.py
├── confusion_matrix.py
└── error_analysis.py

#### Input:
Tất cả mô hình đã huấn luyện.

#### Công việc:
1. Đánh giá:
* Accuracy
* Precision
* Recall
* F1 Score

2. Vẽ:
* Confusion Matrix

3. So sánh: Các mô hình với nhau

4. Error Analysis
Lấy các mẫu dự đoán sai.
Phân loại nguyên nhân:
* Teencode
* Sai chính tả
* Sarcasm
* Review quá ngắn
* Đa nghĩa

#### Thư viện:
* scikit-learn
* pandas
* matplotlib
* seaborn

#### Kết quả đầu ra:
reports/figures/
reports/tables/

Ví dụ:
model_comparison.csv
confusion_matrix.png
error_samples.csv

Yêu cầu kiến thức:
Hiểu:
* Precision
* Recall
* F1 Score
* Confusion Matrix
* False Positive
* False Negative



