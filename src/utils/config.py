from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_STOPWORDS_FILE = DATA_DIR / "vietnamese-stopwords.txt"
DEFAULT_PREPROCESSED_TFIDF_CSV = PROCESSED_DIR / "reviews_baseline_tfidf.csv"
DEFAULT_PREPROCESSED_LSTM_CSV = PROCESSED_DIR / "reviews_lstm_sequence.csv"

DEFAULT_TFIDF_FEATURE_PATH = PROCESSED_DIR / "tfidf_features.pkl"
DEFAULT_WORD2VEC_FEATURE_PATH = PROCESSED_DIR / "word2vec_features.pkl"
DEFAULT_FASTTEXT_FEATURE_PATH = PROCESSED_DIR / "fasttext_features.pkl"
DEFAULT_WORD2VEC_MODEL_PATH = PROCESSED_DIR / "word2vec.model"
DEFAULT_FASTTEXT_MODEL_PATH = PROCESSED_DIR / "fasttext.model"
