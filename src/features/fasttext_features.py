from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText

try:
    from src.utils.config import DEFAULT_PREPROCESSED_LSTM_CSV, DEFAULT_FASTTEXT_FEATURE_PATH, DEFAULT_FASTTEXT_MODEL_PATH
except ImportError:  
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_PREPROCESSED_LSTM_CSV = PROJECT_ROOT / "data" / "processed" / "reviews_lstm_sequence.csv"
    DEFAULT_FASTTEXT_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "fasttext_features.pkl"
    DEFAULT_FASTTEXT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "fasttext.model"


def load_sequence_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed LSTM input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df.columns:
        raise ValueError(f"Expected a 'text' column in {path}, found: {list(df.columns)}")
    return df


def make_sentences(texts: Iterable[str]) -> list[list[str]]:
    return [str(text).split() for text in texts]


def train_fasttext(
    sentences: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 8,
    min_n: int = 3,
    max_n: int = 6,
    workers: int | None = None,
) -> FastText:
    workers = workers if workers is not None else max(1, os.cpu_count() - 1)
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        min_n=min_n,
        max_n=max_n,
        workers=workers,
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    return model


def document_embedding(model: FastText, tokens: list[str]) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=float)
    return np.mean(vectors, axis=0)


def build_document_embeddings(model: FastText, sentences: list[list[str]]) -> np.ndarray:
    return np.vstack([document_embedding(model, tokens) for tokens in sentences])


def save_artifact(
    output_path: Path,
    model: FastText,
    doc_embeddings: np.ndarray,
    ids: pd.Series,
    labels: pd.Series | None = None,
    model_path: Path = DEFAULT_FASTTEXT_MODEL_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    artifact = {
        "model_path": str(model_path),
        "doc_embeddings": doc_embeddings,
        "ids": ids.tolist(),
        "labels": labels.tolist() if labels is not None else None,
        "vector_size": model.vector_size,
    }
    joblib.dump(artifact, output_path)


def run_fasttext_pipeline(
    input_path: Path = DEFAULT_PREPROCESSED_LSTM_CSV,
    output_path: Path = DEFAULT_FASTTEXT_FEATURE_PATH,
    model_path: Path = DEFAULT_FASTTEXT_MODEL_PATH,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 8,
    min_n: int = 3,
    max_n: int = 6,
    workers: int | None = None,
) -> Path:
    df = load_sequence_data(input_path)
    sentences = make_sentences(df["text"].astype(str))
    model = train_fasttext(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        min_n=min_n,
        max_n=max_n,
        workers=workers,
    )
    doc_embeddings = build_document_embeddings(model, sentences)
    save_artifact(output_path, model, doc_embeddings, df.get("id", pd.Series(dtype="object")), df.get("label"), model_path=model_path)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FastText embeddings and save document features.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_PREPROCESSED_LSTM_CSV,
        help="Path to the preprocessed LSTM sequence CSV containing segmented text.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_FASTTEXT_FEATURE_PATH,
        help="Path to save FastText document vectors.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_FASTTEXT_MODEL_PATH,
        help="Path to save the trained FastText model.",
    )
    parser.add_argument("--vector-size", type=int, default=100, help="Embedding dimension.")
    parser.add_argument("--window", type=int, default=5, help="Context window size.")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum token count.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--min-n", type=int, default=3, help="Minimum character n-gram length.")
    parser.add_argument("--max-n", type=int, default=6, help="Maximum character n-gram length.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = run_fasttext_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        model_path=args.model_path,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        min_n=args.min_n,
        max_n=args.max_n,
        workers=args.workers,
    )
    print(f"Saved FastText features to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
