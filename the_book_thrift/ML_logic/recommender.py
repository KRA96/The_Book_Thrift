"""
This module takes a user's Goodreads profile download (CSV) and uses it to
recommend a book from the existing books in user interactions
"""

import joblib
import pandas as pd
from scipy import sparse
import os
from the_book_thrift.ML_logic.collab_model import get_score
import numpy as np
from google.cloud import bigquery

# Import environ variables
project = os.environ["PROJECT"]
dataset = os.environ["DATASET"]

class ALSRecommender():
    """
    Wraps the implicit ALS model to:
    - Load pickled model
    - Pass a user's Goodread's library download
    - Get back recommendations
    """

    def __init__(
        self,
        model_path: str | None = None,
        book_id_col: str = "book_id"
    ) -> None:
        # Resolve model_path at runtime to avoid accessing environment variables
        # at module import time (which caused failures inside Docker).
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH")
        if not model_path:
            raise RuntimeError("MODEL_PATH is not set. Set it in the environment or in the constructor.")

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.n_items = artifact["n_items"]

        # import data from bigquery
        client = bigquery.Client()
        query = f"""
        SELECT * from `{project}.{dataset}.book_titles`
        """
        self.book_titles = client.query(query).to_dataframe()


    def _get_user_profile(self,
                          profile_csv="/Users/krahmed96/code/KRA96/The_Book_Thrift/raw_data/goodreads_library_export.csv"
        ) -> sparse.csr_matrix:
        """
        Takes a user's csv upload and creates a user csr matrix to use within
        the ALS model
        """
        user_raw = pd.read_csv(profile_csv)     #TODO: put all this in a pipeline maybe
        user = user_raw[
        ["Book Id",
        "My Rating",
        "Exclusive Shelf",
        "My Review"]
        ]

        # Clean Col Names
        user.columns = ["_".join(col.lower().split()) for col in user.columns]

        # Keep only book ids present in our book titles dataset
        my_books = set(user["book_id"][user["book_id"] <= self.n_items])    # Keep only book_ids less than n_items.
        all_books = set(self.book_titles["book_id"])
        common_books = pd.Series(list(my_books.intersection(all_books)))

        # If no common books, rasie error
        if common_books.empty:
            raise ValueError("No overlapping book_ids between user CSV and catalogue")

        user = user[user["book_id"].isin(common_books)]

        # Assign binary is_read and is_reviewed column
        user["is_read"] = 0
        user.loc[user["exclusive_shelf"] == "read", "is_read"] = 1

        user["is_reviewed"] = 0
        user.loc[user["my_review"].notna(), "is_reviewed"] = 1

        # Drop my review and exclusive shelf
        user = user.drop(columns=["exclusive_shelf", "my_review"])

        # Rename my rating column
        user = user.rename(columns={
            "my_rating": "rating"
        })

        # Get score
        user["score"] = get_score(user)

        # Create CSR Matrix
        cols = common_books.astype(np.int32)
        data = user["score"].astype("float32")
        rows = np.zeros_like(common_books, dtype=np.int32)

        user_mat = sparse.csr_matrix(
            (data, (rows, cols)), shape=(1, self.n_items)
        )
        return user_mat

    def recommend_books(self,
                        user_items,
                        n_recs: int = 20):      #TODO: using Nrecs in recommend causes error where implicit asks for int.

        rec_ids, scores = self.model.recommend(
            userid=0,
            user_items=user_items,
            recalculate_user=True
        )

        rec_ids = rec_ids.astype(int)

        titles = self.book_titles.loc[self.book_titles.index.intersection(rec_ids)].copy()
        titles = titles.reindex(rec_ids)

        res = []
        for bid, score in zip(rec_ids, scores):
            row = titles.loc[bid] if bid in titles.index else {}
            res.append(
                {"Recommendations": row.get("title")}
            )

        return res

if __name__ == "__main__":
    ALS = ALSRecommender()
    user = ALS._get_user_profile()  # Returns sparse matrix with user's scores
    print(ALS.recommend_books(user))
