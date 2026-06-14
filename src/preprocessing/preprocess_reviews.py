"""Preprocess Vietnamese customer reviews for baseline ML and LSTM models.

The pipeline reads raw JSONL review files, merges them vertically, applies a
shared normalization layer, then creates two model-specific text columns:

- Baseline: aggressive cleaning for TF-IDF + classical ML.
- LSTM: lighter cleaning that preserves more sequence information.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import emoji
except ImportError:  # pragma: no cover - optional fallback for fresh envs
    emoji = None

try:
    from pyvi.ViTokenizer import tokenize as pyvi_tokenize
except ImportError:  # pragma: no cover
    pyvi_tokenize = None

try:
    from underthesea import word_tokenize
except ImportError:  # pragma: no cover
    word_tokenize = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILES = (
    PROJECT_ROOT / "data" / "raw" / "aug_unaccented_reviews.jsonl",
    PROJECT_ROOT / "data" / "raw" / "shopee_reviews_dataset.jsonl",
)
DEFAULT_STOPWORDS_FILE = PROJECT_ROOT / "data" / "vietnamese-stopwords.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TEXT_COLUMN_CANDIDATES = ("review", "text", "comment", "content")
LABEL_COLUMN_CANDIDATES = ("label", "sentiment", "rating")

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
VIETNAMESE_ACCENT_RE = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    flags=re.IGNORECASE,
)
MOJIBAKE_HINT_RE = re.compile(r"[ÃÆÄÐÊËÂ]|áº|á»|Æ°|Æ¡|Ä‘|Ä")


SLANG_MAP = {
    "0k": "ổn",
    "ok": "ổn",
    "oke": "ổn",
    "okey": "ổn",
    "oki": "ổn",
    "okie": "ổn",
    "oks": "ổn",
    "k": "không",
    "kh": "không",
    "ko": "không",
    "khong": "không",
    "hok": "không",
    "hong": "không",
    "hông": "không",
    "khum": "không",
    "hem": "không",
    "dc": "được",
    "đc": "được",
    "duoc": "được",
    "đươc": "được",
    "đk": "được",
    "vs": "với",
    "z": "vậy",
    "v": "vậy",
    "zay": "vậy",
    "zậy": "vậy",
    "vay": "vậy",
    "j": "gì",
    "gi": "gì",
    "g": "gì",
    "mn": "mọi người",
    "mng": "mọi người",
    "mk": "mình",
    "m": "mình",
    "mik": "mình",
    "minh": "mình",
    "sp": "sản phẩm",
    "sanpham": "sản phẩm",
    "shoppe": "shopee",
    "ship": "giao hàng",
    "shiper": "người giao hàng",
    "shipper": "người giao hàng",
    "ib": "nhắn tin",
    "rep": "trả lời",
    "feedback": "phản hồi",
    "fb": "phản hồi",
    "rv": "đánh giá",
    "review": "đánh giá",
    "rcm": "gợi ý",
    "recommend": "gợi ý",
    "sd": "sử dụng",
    "sài": "xài",
    "xai": "xài",
    "sdung": "sử dụng",
    "hsd": "hạn sử dụng",
    "date": "hạn sử dụng",
    "auth": "chính hãng",
    "fake": "giả",
    "real": "thật",
    "sale": "giảm giá",
    "voucher": "mã giảm giá",
    "vc": "mã giảm giá",
    "tmdt": "thương mại điện tử",
    "tm": "thương mại",
    "tks": "cảm ơn",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "camon": "cảm ơn",
    "cámơn": "cảm ơn",
    "kg": "không",
    "k0": "không",
    "kb": "không biết",
    "kbt": "không biết",
    "bt": "bình thường",
    "bth": "bình thường",
    "bthg": "bình thường",
    "ntn": "như thế nào",
    "tn": "thế nào",
    "qá": "quá",
    "qa": "quá",
    "wa": "quá",
    "qua": "quá",
    "lun": "luôn",
    "luon": "luôn",
    "nhma": "nhưng mà",
    "nhg": "nhưng",
    "nhưngmà": "nhưng mà",
    "ms": "mới",
    "moi": "mới",
}


NEGATION_STOPWORDS = {"không", "chưa", "chẳng", "chả", "đừng", "đâu", "kém"}
DEFAULT_EXTRA_STOPWORDS = {
    "ạ",
    "à",
    "á",
    "ơi",
    "nha",
    "nhé",
    "nhá",
    "hihi",
    "haha",
    "hehe",
}


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_json(path, lines=True)


def choose_column(columns: Iterable[str], candidates: Iterable[str], kind: str) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(
        f"Could not find a {kind} column. Tried {list(candidates)} in {list(columns)}."
    )


def load_and_merge(input_files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in input_files:
        frame = read_jsonl(Path(path))
        text_col = choose_column(frame.columns, TEXT_COLUMN_CANDIDATES, "text")
        frame = frame.copy()
        frame["text"] = frame[text_col].astype("string")
        frame["source_file"] = Path(path).name
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged


def fix_mojibake(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text)
    if not MOJIBAKE_HINT_RE.search(value):
        return value
    for source_encoding in ("latin1", "cp1252"):
        try:
            fixed = value.encode(source_encoding, errors="strict").decode("utf-8")
        except UnicodeError:
            continue
        if count_vietnamese_chars(fixed) > count_vietnamese_chars(value):
            return fixed
    return value


def count_vietnamese_chars(text: str) -> int:
    return len(VIETNAMESE_ACCENT_RE.findall(text))


def emoji_to_text(text: str) -> str:
    if emoji is None:
        return text
    try:
        return emoji.demojize(text, language="vi")
    except NotImplementedError:
        return emoji.demojize(text, language="en")


def remove_noise(text: str) -> str:
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def reduce_elongated_word(word: str) -> str:
    """Collapse noisy repeated characters while preserving meaningful doubles."""
    if len(word) <= 2:
        return word
    return re.sub(r"(\w)\1{2,}", r"\1", word, flags=re.UNICODE)


def reduce_elongated_words(text: str) -> str:
    return re.sub(
        r"\w+",
        lambda match: reduce_elongated_word(match.group(0)),
        text,
        flags=re.UNICODE,
    )


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def build_accent_restore_map(texts: Iterable[str]) -> dict[str, str]:
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    for text in texts:
        for token in re.findall(r"\w+", str(text).lower(), flags=re.UNICODE):
            if VIETNAMESE_ACCENT_RE.search(token):
                variants[strip_accents(token)][token] += 1
    return {
        unaccented: counter.most_common(1)[0][0]
        for unaccented, counter in variants.items()
    }


def restore_vietnamese_accents(text: str, accent_map: dict[str, str]) -> str:
    """Restore accents with a corpus dictionary fallback.

    If a dedicated accent restoration model is available in the environment,
    integrate it here and call it before this fallback. The current heuristic is
    deterministic and works well for project-local Shopee vocabulary.
    """
    if VIETNAMESE_ACCENT_RE.search(text) or not accent_map:
        return text

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        restored = accent_map.get(token.lower())
        if restored is None:
            return token
        if token.isupper():
            return restored.upper()
        if token[:1].isupper():
            return restored.capitalize()
        return restored

    return re.sub(r"\w+", replace_token, text, flags=re.UNICODE)


def normalize_slang(text: str, slang_map: dict[str, str] | None = None) -> str:
    slang_map = SLANG_MAP if slang_map is None else slang_map
    tokens = TOKEN_RE.findall(text)
    normalized = []
    for token in tokens:
        replacement = slang_map.get(token.lower())
        normalized.append(replacement if replacement is not None else token)
    return normalize_whitespace(" ".join(normalized))


def common_preprocess(text: object, accent_map: dict[str, str]) -> str:
    text = fix_mojibake(text)
    text = emoji_to_text(text)
    text = remove_noise(text)
    text = normalize_unicode(text)
    text = text.lower()
    text = reduce_elongated_words(text)
    text = restore_vietnamese_accents(text, accent_map)
    text = normalize_slang(text)
    return normalize_whitespace(text)


def load_stopwords(path: Path | None) -> set[str]:
    stopwords = set(DEFAULT_EXTRA_STOPWORDS)
    if path is None or not path.exists():
        return stopwords
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            word = normalize_whitespace(line.lower())
            if word:
                stopwords.add(word)
    return stopwords - NEGATION_STOPWORDS


def segment_text(text: str, method: str = "pyvi") -> str:
    if not text:
        return ""
    if method == "pyvi":
        if pyvi_tokenize is None:
            raise ImportError("pyvi is required for method='pyvi'.")
        return pyvi_tokenize(text)
    if method == "underthesea":
        if word_tokenize is None:
            raise ImportError("underthesea is required for method='underthesea'.")
        return word_tokenize(text, format="text")
    if method == "none":
        return text
    raise ValueError("method must be one of: pyvi, underthesea, none")


def remove_punctuation(text: str) -> str:
    return normalize_whitespace(PUNCT_RE.sub(" ", text))


def remove_stopwords_from_segmented_text(text: str, stopwords: set[str]) -> str:
    tokens = []
    for token in text.split():
        lookup = token.replace("_", " ").lower()
        if lookup not in stopwords and token.lower() not in stopwords:
            tokens.append(token)
    return " ".join(tokens)


def preprocess_for_baseline(
    text: str,
    stopwords: set[str],
    segmentation_method: str,
) -> str:
    text = remove_punctuation(text)
    text = segment_text(text, method=segmentation_method)
    text = remove_stopwords_from_segmented_text(text, stopwords)
    return normalize_whitespace(text)


def preprocess_for_lstm(text: str, segmentation_method: str) -> str:
    text = remove_punctuation(text)
    text = segment_text(text, method=segmentation_method)
    return normalize_whitespace(text)


def run_pipeline(
    input_files: Iterable[Path] = DEFAULT_INPUT_FILES,
    stopwords_file: Path = DEFAULT_STOPWORDS_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    segmentation_method: str = "pyvi",
    sample_size: int | None = None,
) -> pd.DataFrame:
    input_files = tuple(Path(path) for path in input_files)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(input_files)
    if sample_size is not None:
        df = df.head(sample_size).copy()

    df["text_raw"] = df["text"].fillna("").astype(str)
    df["text_fixed_encoding"] = df["text_raw"].map(fix_mojibake)
    accent_map = build_accent_restore_map(df["text_fixed_encoding"])
    stopwords = load_stopwords(Path(stopwords_file) if stopwords_file else None)

    df["text_common"] = df["text_raw"].map(lambda value: common_preprocess(value, accent_map))
    df["text_baseline"] = df["text_common"].map(
        lambda value: preprocess_for_baseline(value, stopwords, segmentation_method)
    )
    df["text_lstm"] = df["text_common"].map(
        lambda value: preprocess_for_lstm(value, segmentation_method)
    )

    keep_columns = [
        column
        for column in [
            "id",
            "source_file",
            "rating",
            "label",
            "text_raw",
            "text_fixed_encoding",
            "text_common",
            "text_baseline",
            "text_lstm",
        ]
        if column in df.columns
    ]
    processed = df[keep_columns].copy()

    processed.to_csv(output_dir / "reviews_preprocessed_all.csv", index=False, encoding="utf-8-sig")
    processed.to_json(
        output_dir / "reviews_preprocessed_all.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    label_columns = [column for column in ("id", "rating", "label", "source_file") if column in processed]
    processed[label_columns + ["text_baseline"]].rename(
        columns={"text_baseline": "text"}
    ).to_csv(output_dir / "reviews_baseline_tfidf.csv", index=False, encoding="utf-8-sig")
    processed[label_columns + ["text_lstm"]].rename(columns={"text_lstm": "text"}).to_csv(
        output_dir / "reviews_lstm_sequence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "num_rows": int(len(processed)),
        "input_files": [str(path) for path in input_files],
        "stopwords_file": str(stopwords_file),
        "segmentation_method": segmentation_method,
        "outputs": [
            "reviews_preprocessed_all.csv",
            "reviews_preprocessed_all.jsonl",
            "reviews_baseline_tfidf.csv",
            "reviews_lstm_sequence.csv",
        ],
    }
    (output_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return processed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        default=list(DEFAULT_INPUT_FILES),
        help="Raw JSONL input files.",
    )
    parser.add_argument(
        "--stopwords-file",
        type=Path,
        default=DEFAULT_STOPWORDS_FILE,
        help="Vietnamese stopwords file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed outputs.",
    )
    parser.add_argument(
        "--segmentation-method",
        choices=("pyvi", "underthesea", "none"),
        default="pyvi",
        help="Vietnamese word segmentation backend.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row limit for quick tests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    processed = run_pipeline(
        input_files=args.input_files,
        stopwords_file=args.stopwords_file,
        output_dir=args.output_dir,
        segmentation_method=args.segmentation_method,
        sample_size=args.sample_size,
    )
    print(f"Processed {len(processed):,} reviews.")
    print(f"Saved outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
