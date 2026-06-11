import unicodedata
import re

SLANG_DICT = {
    "ko": "không",
    "khum": "không",
    "hok": "không",
    "k": "không",
    "dc": "được",
    "đc": "được",
    "mn": "mọi người",
    "ntn": "như thế nào",
    "sp": "sản phẩm",
    "tks": "thanks"
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