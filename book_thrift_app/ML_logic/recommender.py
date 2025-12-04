"""
This module takes a user's Goodreads profile download (CSV) and uses it to
recommend a book from the existing books in user interactions
"""

import joblib
from pathlib import Path

import pandas as pd
from scipy import sparse
import numpy as np

from .collab_model import get_score
from book_thrift_app.params import (
    COLLAB_MODEL, BOOK_TITLES, BOOK_ID_PATH, BOOK_MAPPING_PATH
)
class ALSRecommender():
    """
    Wraps the implicit ALS model to:
    - Load pickled model
    - Pass a user's Goodread's library download
    - Get back recommendations
    """
    def __init__(
        self,
        model_path: str = COLLAB_MODEL,
    ) -> None:
        # Resolve model_path at runtime to avoid accessing environment variables
        # at module import time (which caused failures inside Docker).
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.n_items = artifact["n_items"]
        self.book_mapping = np.load(BOOK_MAPPING_PATH)
        # import data from bigquery if doesn't exist locally
        titles_path = BOOK_TITLES
        self.book_titles = pd.read_csv(titles_path)
        all_books = np.load(BOOK_ID_PATH)


    def _get_user_profile(self,
                          profile_csv
                          ) -> sparse.csr_matrix:
        """
        Takes a user's csv upload and creates a user csr matrix to use within
        the ALS model
        """
        user_raw = pd.read_csv(
            profile_csv)  # TODO: put all this in a pipeline maybe
        user = user_raw[
            ["Book Id",
             "My Rating",
             "Exclusive Shelf",
             "My Review"]
        ]

        # Clean Col Names
        user.columns = ["_".join(col.lower().split()) for col in user.columns]

        # Load np of book map array of all bookids in our data
        unique_books = self.book_mapping
        user_books = user[user["book_id"].isin(unique_books)]

        # If no common books, rasie error
        if user_books.empty:
            raise ValueError(
                "None of user's books found in interactions dataset")

        # Assign binary is_read and is_reviewed column
        user_books["is_read"] = 0
        user_books.loc[user_books["exclusive_shelf"] == "read", "is_read"] = 1

        user_books["is_reviewed"] = 0
        user_books.loc[user_books["my_review"].notna(), "is_reviewed"] = 1

        # Drop my review and exclusive shelf
        user_books = user_books.drop(columns=["exclusive_shelf", "my_review"])

        # Rename my rating column
        user_books = user_books.rename(columns={
            "my_rating": "rating"
        })

        # Get score
        user_books["score"] = get_score(user_books)

        # load book map
        book_map = {b_id: i for i, b_id in enumerate(unique_books)}

        # Create CSR Matrix
        cols = user_books["book_id"].map(book_map).astype(np.int32)
        data = user_books["score"].astype("float32")
        rows = np.zeros(len(user_books), dtype=np.int32)

        user_mat = sparse.csr_matrix(
            (data, (rows, cols)), shape=(1, self.n_items)
        )
        return user_mat

    def recommend_books(self,
                        user_items,
                        items,
                        n_recs: int = 20):
        """_summary_

        Args:
            user_items (csr_matrix): CSR representation of user's interactions
            with books
            items (list or ndarray): list of available book options to
            select from/recommend from
            n_recs (int, optional): number of recs to generate. Defaults to 20.

        Returns:
            dictionary of recommended books
        """
        rec_ids, scores = self.model.recommend(
            userid=0,
            user_items=user_items,
            recalculate_user=True,
            items=items
        )

        rec_ids = rec_ids.astype(int)
        unique_books = self.book_mapping
        rec_book_ids = unique_books[rec_ids]
        titles = []
        self.book_titles["book_id"] = pd.to_numeric(self.book_titles["book_id"], errors="coerce")
        titles.append(self.book_titles["title"].loc[self.book_titles["book_id"].isin(
            rec_book_ids)])

        titles_df = pd.concat(titles, ignore_index=True)
        res = []
        for book in titles_df:
            res.append({"Recommendations": book})
        return res

if __name__ == "__main__":
    ALS = ALSRecommender()
    # Returns sparse matrix with user's scores
    user = ALS._get_user_profile(
        "/Users/krahmed96/code/KRA96/The_Book_Thrift/raw_data/goodreads_library_export.csv")
    print(ALS.recommend_books(user))
