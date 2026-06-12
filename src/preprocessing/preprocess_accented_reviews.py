import pandas as pd
import numpy as np
import re
import string
import unicodedata
from underthesea import word_tokenize
from pyvi.ViTokenizer import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import joblib
import torch
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

raw_dir = project_root / "data" / "raw"
accented_review_path = raw_dir / "shopee_reviews_dataset.jsonl"

interim_dir = project_root / "data" / "interim"
interim_dir.mkdir(parents=True, exist_ok=True)
clean_review_path = interim_dir / "clean_reviews.csv"

processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
tfidf_path = processed_dir / "tfidf_vectors.pkl"
phobert_token_path = processed_dir / "phobert_tokens.pt"



# clean text
def remove_url(text):
    return re.sub(r"http\S+|www\S+", "", text)

def remove_html(text):
    return re.sub(r"<.*?>", "", text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', ' ', text)

def remove_special_characters(text):
    return re.sub(r"[^a-zA-ZÀ-ỹ0-9\s]", "", text)

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text):
    text = text.lower()
    text = remove_url(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    return text

# normalize text
SLANG_DICT = {
    "ko": "không",
    "khum": "không",
    "hok": "không",
    "k": "không",
    "dc": "được",
    "mn": "mọi người",
    "ntn": "như thế nào",
    "sp": "sản phẩm"
}
def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)

def normalize_slang(text):
    words = text.split()
    norm_words = []
    for word in words:
        norm_words.append(SLANG_DICT.get(word, word))
        # replace if it's a slang
    return " ".join(norm_words)

def normalize_repeated_characters(text):
    return re.sub(r"(.)\1+", r"\1", text)

def normalize_text(text):
    text = normalize_unicode(text)
    text = normalize_slang(text)
    text = normalize_repeated_characters(text)

    return text

# segment text
def segment_underthesea(text):
    return word_tokenize(text, format="text")
    # segment and return a string (format="text")

def segment_pyvi(text):
    return tokenize(text)

def segment_text(text, method="underthesea"):
    if method == "underthesea":
        return segment_underthesea(text)
    elif method == "pyvi":
        return segment_pyvi(text)
    else:
        raise ValueError("Invalid segmentation method")
    
# tokenize text
def tokenize_tfidf(tokenizer, texts):
    return tokenizer.fit_transform(texts)

def phobert_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base")

def tokenize_phobert(tokenizer, text):
    return tokenizer(text, padding="max_length", 
                     truncation=True,
                     max_length=128, return_tensor="pt") 


def main():
    accented = pd.read_json(accented_review_path, lines=True)
    accented["review"] = accented["review"].apply(clean_text)
    accented["review"] = accented["review"].apply(normalize_text)
    accented["review"] = accented["review"].apply(segment_underthesea)
    accented.to_csv(clean_review_path,
                    index=False,
                    encoding="utf-8-sig")
    
    tfidf = TfidfVectorizer()
    tokens_tfidf = tokenize_tfidf(tfidf, accented["review"])
    joblib.dump(tokens_tfidf, tfidf_path)

    phobert = phobert_tokenizer()
    tokens_phobert = accented["review"].apply(lambda x: tokenize_phobert(phobert, x))
    torch.save(tokens_phobert, phobert_token_path)


if __name__ == "__main__":
    main()