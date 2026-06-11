import re
import string

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