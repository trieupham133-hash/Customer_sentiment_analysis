import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Tự động xác định PROJECT_ROOT dựa trên vị trí file script này
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TFIDF_PATH = PROJECT_ROOT / "data" / "processed" / "tfidf_features.pkl"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"

def main():
    print("=== Phase 4: Logistic Regression Baseline Pipeline ===")
    
    # 1. Tự load dữ liệu từ artifact của Phase 3
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file đặc trưng tại: {TFIDF_PATH}")
    
    data = joblib.load(TFIDF_PATH)
    X = data["matrix"]
    labels = data["labels"]
    
    # Ép nhãn về dạng số: negative -> 0, positive -> 1
    label_map = {"negative": 0, "positive": 1}
    y = np.array([label_map[lbl] for lbl in labels])
    
    # 2. Tự chia tập dữ liệu (Tỷ lệ 80/20, Stratified để giữ cân bằng nhãn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train size: {X_train.shape[0]} mẫu | Test size: {X_test.shape[0]} mẫu")
    
    # 3. Huấn luyện thuật toán
    print("      Đang train Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Tự tính toán metrics đánh giá hiệu năng
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- KẾT QUẢ LOGISTIC REGRESSION ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (lớp positive)")
    print(f"Recall   : {rec:.4f} (lớp positive)")
    print(f"F1 Score : {f1:.4f} (lớp positive)")
    print("\nConfusion Matrix:")
    print(cm)
    
    # 5. Tự lưu mô hình thành phẩm xuống thư mục models/baseline/
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "logistic_regression_baseline.joblib")
    print(f"\n[THÀNH CÔNG] Đã lưu model tại: {MODEL_DIR / 'logistic_regression_baseline.joblib'}")

if __name__ == "__main__":
    main()