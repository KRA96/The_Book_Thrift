import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# 1. Cleaning


class InteractionsCleaner(BaseEstimator, TransformerMixin):
    """
    - supprime duplicates
    - retire interactions user/book manquants
    - corrige incohérences rating/is_read
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # drop duplicates
        df = df.drop_duplicates()

        # drop rows with missing ids
        df = df.dropna(subset=["user_id", "book_id"])

        # correct logical inconsistencies
        # If is_read=0 but rating>0 -> set is_read=1
        mask = (df["is_read"] == 0) & (df["rating"] > 0)
        df.loc[mask, "is_read"] = 1

        # If is_read=1 and rating==0 -> set rating=NaN (unrated but read)
        mask = (df["is_read"] == 1) & (df["rating"] == 0)
        df.loc[mask, "rating"] = np.nan

        return df.reset_index(drop=True)


# 2. User/books filter (on low use)


class InteractionsFilter(BaseEstimator, TransformerMixin):
    """
    supprime :
    - users avec trop peu d'interactions
    - livres avec trop peu d'interactions
    """
    def __init__(self, min_user_interactions=3, min_book_interactions=3):
        self.min_user_interactions = min_user_interactions
        self.min_book_interactions = min_book_interactions

    def fit(self, X, y=None):
        df = X

        user_counts = df["user_id"].value_counts()
        book_counts = df["book_id"].value_counts()

        self.valid_users_ = set(user_counts[user_counts >= self.min_user_interactions].index)
        self.valid_books_ = set(book_counts[book_counts >= self.min_book_interactions].index)

        return self

    def transform(self, X):
        df = X.copy()
        df = df[df["user_id"].isin(self.valid_users_) &
                df["book_id"].isin(self.valid_books_)]

        return df.reset_index(drop=True)



# 3. IDs encoding

class IDEncoder(BaseEstimator, TransformerMixin):
    """
    Convertit user_id et book_id en entiers continus à partir de 0.
    Stocke les mappings pour usage futur (recommandations).
    """
    def fit(self, X, y=None):
        df = X

        self.user_to_index_ = {u: i for i, u in enumerate(df["user_id"].unique())}
        self.book_to_index_ = {b: i for i, b in enumerate(df["book_id"].unique())}

        self.index_to_user_ = {i: u for u, i in self.user_to_index_.items()}
        self.index_to_book_ = {i: b for b, i in self.book_to_index_.items()}

        return self

    def transform(self, X):
        df = X.copy()

        df["user_index"] = df["user_id"].map(self.user_to_index_)
        df["book_index"] = df["book_id"].map(self.book_to_index_)

        return df.reset_index(drop=True)



# 4. Matrice user x book sparse


class InteractionMatrixBuilder(BaseEstimator, TransformerMixin):
    """
    Construit une matrice sparse CSR user × book basée sur :
    - rating si disponible
    - sinon is_read
    """
    def __init__(self, use_rating_if_available=True):
        self.use_rating_if_available = use_rating_if_available
        self.matrix_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X

        users = df["user_index"].values
        books = df["book_index"].values

        if self.use_rating_if_available and "rating" in df:
            values = df["rating"].fillna(1.0).values
        else:
            values = df["is_read"].astype(float).values

        n_users = df["user_index"].max() + 1
        n_books = df["book_index"].max() + 1

        self.matrix_ = csr_matrix(
            (values, (users, books)),
            shape=(n_users, n_books)
        )

        return df  # Return df but store matrix inside the transformer



# Final pipeline


def build_interactions_pipeline(
    min_user_interactions=3,
    min_book_interactions=3
):
    pipe = Pipeline([
        ("clean", InteractionsCleaner()),
        ("filter", InteractionsFilter(
            min_user_interactions=min_user_interactions,
            min_book_interactions=min_book_interactions
        )),
        ("encode", IDEncoder()),
        ("matrix", InteractionMatrixBuilder(use_rating_if_available=True))
    ])
    return pipe
