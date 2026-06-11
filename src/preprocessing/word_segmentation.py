from underthesea import word_tokenize
from pyvi.ViTokenizer import tokenize

def segment_underthesea(text):
    return word_tokenize(text, format="text")

def segment_pyvi(text):
    return tokenize(text)

def segment_text(text, method="underthesea"):
    if method == "underthesea":
        return segment_underthesea(text)
    elif method == "pyvi":
        return segment_pyvi(text)
    else:
        raise ValueError("Invalid segmentation method")