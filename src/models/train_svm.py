import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TFIDF_PATH = PROJECT_ROOT / "data" / "processed" / "tfidf_features.pkl"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"

def main():
    print("=== Phase 4: Support Vector Machine Baseline Pipeline ===")
    
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file đặc trưng tại: {TFIDF_PATH}")
    
    data = joblib.load(TFIDF_PATH)
    X = data["matrix"]
    labels = data["labels"]
    
    label_map = {"negative": 0, "positive": 1}
    y = np.array([label_map[lbl] for lbl in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train size: {X_train.shape[0]} mẫu | Test size: {X_test.shape[0]} mẫu")
    
    print("      Đang train LinearSVC...")
    model = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- KẾT QUẢ SVM (LINEARSVC) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (lớp positive)")
    print(f"Recall   : {rec:.4f} (lớp positive)")
    print(f"F1 Score : {f1:.4f} (lớp positive)")
    print("\nConfusion Matrix:")
    print(cm)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "svm_baseline.joblib")
    print(f"\n[THÀNH CÔNG] Đã lưu model tại: {MODEL_DIR / 'svm_baseline.joblib'}")

if __name__ == "__main__":
    main()