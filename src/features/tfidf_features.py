from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from src.utils.config import DEFAULT_PREPROCESSED_TFIDF_CSV, DEFAULT_TFIDF_FEATURE_PATH
except ImportError:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_PREPROCESSED_TFIDF_CSV = PROJECT_ROOT / "data" / "processed" / "reviews_baseline_tfidf.csv"
    DEFAULT_TFIDF_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "tfidf_features.pkl"


def load_preprocessed_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed TF-IDF input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df.columns:
        raise ValueError(f"Expected a 'text' column in {path}, found: {list(df.columns)}")
    return df


def build_tfidf_features(
    texts: Iterable[str],
    max_features: int = 30_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
) -> tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm="l2",
        smooth_idf=True,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_feature_artifacts(
    output_path: Path,
    vectorizer: TfidfVectorizer,
    matrix: object,
    ids: pd.Series,
    labels: pd.Series | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "ids": ids.tolist(),
        "labels": labels.tolist() if labels is not None else None,
    }
    joblib.dump(artifact, output_path)


def run_tfidf_pipeline(
    input_path: Path = DEFAULT_PREPROCESSED_TFIDF_CSV,
    output_path: Path = DEFAULT_TFIDF_FEATURE_PATH,
    max_features: int = 30_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
) -> Path:
    df = load_preprocessed_data(input_path)
    vectorizer, matrix = build_tfidf_features(
        df["text"].astype(str),
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    save_feature_artifacts(output_path, vectorizer, matrix, df.get("id", pd.Series(dtype="object")), df.get("label"))
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and save TF-IDF features for Vietnamese sentiment analysis.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_PREPROCESSED_TFIDF_CSV,
        help="Path to the preprocessed baseline review CSV containing a text column.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_TFIDF_FEATURE_PATH,
        help="Path to save the TF-IDF feature artifact.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30_000,
        help="Maximum number of TF-IDF features to keep.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Minimum document frequency for TF-IDF vocabulary.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Maximum document frequency proportion for TF-IDF vocabulary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = run_tfidf_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    print(f"Saved TF-IDF features to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
