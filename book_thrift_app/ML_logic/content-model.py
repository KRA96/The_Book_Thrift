"""
All pipes for preprocessing:
- Books data
- User's data
"""
import argparse
import ast
import re
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler

# Books pipeline

## Config

@dataclass
class CFG:
    english_codes = ["eng", "en-US", "en-GB", "en-CA", "NaN"]
    cols_to_drop = [
        "edition_information",
        "asin",
        "kindle_asin",
        "publication_day",
        "publication_month",
        "format",
        "publisher",
        "isbn",
    ]
    numeric_cols = ["average_rating", "ratings_count", "text_reviews_count"]

    # TF-IDF
    stop_words: str = "english"
    max_features: int = 30000
    min_df: int = 5
    max_df: float = 0.8

    # KMeans
    n_clusters: int = 30
    random_state: int = 42
    batch_size: int = 2000

## Loading data

def load_raw(raw_path: str, nrows=None):
    raw_path = Path(raw_path)
    compression = "gzip" if raw_path.suffix == ".gz" else None
    return pd.read_json(raw_path, lines=True, compression=compression, nrows=nrows)

## Custom Transformers

class GoodreadsCleaner(BaseEstimator, TransformerMixin):
    """
    - filter language_code to english_codes
    - drop rows title+description both missing
    - drop irrelevant columns
    - coerce numeric columns
    """
    def __init__(self, english_codes=None, cols_to_drop=None, numeric_cols=None):
        self.english_codes = english_codes or CFG.english_codes
        self.cols_to_drop = cols_to_drop or CFG.cols_to_drop
        self.numeric_cols = numeric_cols or CFG.numeric_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if "language_code" in df.columns:
            df = df[df["language_code"].isin(self.english_codes)].copy()

        df = df.dropna(subset=["title", "description"], how="all").reset_index(drop=True)

        df = df.drop(columns=self.cols_to_drop, errors="ignore")

        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

class GoodreadsFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Adds combined_text.
    Keeps numeric columns for ColumnTransformer.
    """
    def __init__(self, numeric_cols=None, keep_original=True):
        self.numeric_cols = numeric_cols or CFG.numeric_cols
        self.keep_original = keep_original

    # ----- internal utils -----
    @staticmethod
    def _clean_text(s):
        if not isinstance(s, str):
            return ""
        s = s.lower()
        s = re.sub(r"<.*?>", " ", s)
        s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _safe_literal_eval(x):
        if isinstance(x, (list, dict)):
            return x
        if not isinstance(x, str):
            return []
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    @classmethod
    def _parse_shelves(cls, x):
        lst = cls._safe_literal_eval(x)
        if not isinstance(lst, list):
            return []
        out = []
        for item in lst:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    out.append(str(name))
            elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], dict):
                name = item[0].get("name")
                if name:
                    out.append(str(name))
        return out

    @classmethod
    def _parse_authors(cls, x):
        lst = cls._safe_literal_eval(x)
        if not isinstance(lst, list):
            return []
        out = []
        for item in lst:
            if isinstance(item, dict):
                aid = item.get("author_id")
                if aid is not None:
                    out.append(aid)
        return out

    @classmethod
    def _parse_similar(cls, x):
        lst = cls._safe_literal_eval(x)
        return lst if isinstance(lst, list) else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # title_clean
        if "title_without_series" in df.columns:
            title_clean = (
                df["title_without_series"]
                .fillna(df.get("title"))
                .fillna("")
                .astype(str)
                .str.lower()
            )
        else:
            title_clean = df.get("title", "").fillna("").astype(str).str.lower()

        # description_clean
        desc_clean = df.get("description", "").fillna("").astype(str).map(self._clean_text)

        # shelves/authors/similar
        shelf_names = df.get("popular_shelves", pd.Series([[]] * len(df))).map(self._parse_shelves)

        GENERIC_SHELVES = {
        "to-read", "currently-reading", "read", "owned", "kindle",
        "wishlist", "default", "ebooks", "library", "favorites"
        }

        shelf_names = shelf_names.map(
             lambda lst: [s for s in lst if s not in GENERIC_SHELVES] if isinstance(lst, list) else []
        )

        author_ids = df.get("authors", pd.Series([[]] * len(df))).map(self._parse_authors)
        sim_list = df.get("similar_books", pd.Series([[]] * len(df))).map(self._parse_similar)



        shelf_text = shelf_names.map(lambda lst: " ".join(lst) if isinstance(lst, list) else "")
        author_text = author_ids.map(lambda lst: " ".join(f"author_{a}" for a in lst) if isinstance(lst, list) else "")
        sim_text = sim_list.map(lambda lst: " ".join(f"sim_{b}" for b in lst) if isinstance(lst, list) else "")

        df["combined_text"] = (
            title_clean.fillna("") + " " +
            desc_clean.fillna("") + " " +
            shelf_text.fillna("") + " " +
            author_text.fillna("") + " " +
            sim_text.fillna("")
        ).str.strip()

        # ensure numeric cols exist
        for c in self.numeric_cols:
            if c not in df.columns:
                df[c] = np.nan

        if self.keep_original:
            return df
        else:
            return df[["combined_text"] + self.numeric_cols]

## Pipeline builder

def build_pipeline(
    do_k_means=True,    # to test not doing clustering right now
    numeric_cols=CFG.numeric_cols,
    stop_words=CFG.stop_words,
    max_features=CFG.max_features,
    min_df=CFG.min_df,
    max_df=CFG.max_df,
    n_clusters=CFG.n_clusters,
    random_state=CFG.random_state,
    batch_size=CFG.batch_size
) -> Pipeline:
    text_vec = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MaxAbsScaler())
    ])

    features = ColumnTransformer(
        transformers=[
            ("text", text_vec, "combined_text"),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False
    )

    if do_k_means:
        pipe = Pipeline(steps=[
            ("clean", GoodreadsCleaner()),
            ("build_features", GoodreadsFeatureBuilder(numeric_cols=numeric_cols)),
            ("features", features),
            ("cluster", MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=batch_size
            ))
        ])
    else:
        pipe = Pipeline(steps=[
            ("clean", GoodreadsCleaner()),
            ("build_features", GoodreadsFeatureBuilder(numeric_cols=numeric_cols)),
            ("features", features)
        ])
    return pipe

## Post-fit: automatic labels

def make_cluster_labels(df_cleaned, pipeline, top_n=5):
    ct = pipeline.named_steps["features"]
    vectorizer = ct.named_transformers_["text"]
    kmeans = pipeline.named_steps["cluster"]

    terms = vectorizer.get_feature_names_out()
    text_dim = len(terms)

    centers_text = kmeans.cluster_centers_[:, :text_dim]
    order_centroids = centers_text.argsort()[:, ::-1]

    def cluster_keywords(cluster_id):
        return [terms[i] for i in order_centroids[cluster_id, :top_n]]

    shelves_parse = GoodreadsFeatureBuilder._parse_shelves
    shelf_names = df_cleaned.get(
        "popular_shelves",
        pd.Series([[]] * len(df_cleaned))
    ).map(shelves_parse)

    GENERIC_SHELVES = {
        "to-read", "currently-reading", "read", "owned", "kindle",
        "wishlist", "default", "ebooks", "library", "favorites"
    }
    shelf_names = shelf_names.map(
        lambda lst: [s for s in lst if s not in GENERIC_SHELVES] if isinstance(lst, list) else []
    )

    labels = {}
    for c in range(kmeans.n_clusters):
        shelves_series = shelf_names[kmeans.labels_ == c]
        flat = [s for lst in shelves_series for s in (lst or [])]

        top_shelves = [s for s, _ in Counter(flat).most_common(top_n)]
        top_words = cluster_keywords(c)

        label = ", ".join(top_shelves[:3] + top_words[:3]) or f"Cluster {c}"
        labels[c] = label

    return labels

#TODO: Does it make sense to add this function to class so we don't have to get
    # mapping everytime??
def map_book_id_to_index(df):
    """
    Use pipe named steps method to get a stable mapping from book id to index
    """
    pipe = build_pipeline(do_k_means=False)
    clean_df = pipe.named_steps["clean"].tranform(df)
    feat_df = pipe.named_steps["build_features"].transform(clean_df)

    return {book_id: i for i, book_id in enumerate(feat_df["book_id"].tolist())}

# User pipeline
def get_user_books(user_profile: Path, books_data_path_and_size): #TODO: need to build a bookid map to index
                                            #  during above pipeline to use in this part
    """
    Takes a user's goodreads profile and builds a vector of books they've
    read to get similar books
    """
    read_books = user_profile["Book Id"][
    (user_profile["My Rating"] != 0) |
    (user_profile["Exclusive Shelf"] == "read")
    ]
    # DO NOT RUN THE FN AS THE BELOW WILL RUN ON FULL DATASET
    book_id_to_index = map_book_id_to_index(books_data_path_and_size)

    read_books_index = [book_id_to_index.get(b_id) for b_id in read_books]

    mean_user_vector = np.average[None]     #TODO This code needs normalised feature matrix obtained from
                                            # pipeline above. How do I save this matrix in an efficient way?
