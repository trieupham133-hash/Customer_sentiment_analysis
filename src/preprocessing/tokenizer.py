from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

def tokenize_tfidf(tokenizer, texts):
    return tokenizer.fit_transform(texts)

def phobert_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base")

def tokenize_phobert(tokenizer, text):
    return tokenizer(text, padding="max_length", truncation=True,
                     max_length=128, return_tensor="pt") 