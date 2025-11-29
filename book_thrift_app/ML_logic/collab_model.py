"""
Clean up and modify interactions data to create CSR matrix that will
be fed into an Alternating Least Squares model to get recommendations
based on similar users.
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

import os
import gzip
import joblib
from pathlib import Path

HERE = HERE = Path(__file__).resolve().parent

def get_score(interactions_df): #TODO: Should this score be modified?
        """
        generate score from user interactions data.
        """
        return interactions_df["is_read"] \
            + interactions_df["rating"] \
            + interactions_df["is_reviewed"] * 2

def load_data_and_build_csr(interactions_csv_path: str):
    """
    Loads interactions dataset, reduces memory storage by downcasting,
    creates scores based on user ratings, and generates a sparse matrix
    of shape (n_users, n_books).
    """
    def downcast(df):
        """
        Downcast datatypes to reduce memory usage.
        """
        int_8_cols = ["is_read", "rating", "is_reviewed"]
        df[int_8_cols] = df[int_8_cols].astype("int8")
        df[["user_id", "book_id"]] = df[["user_id", "book_id"]].astype("int32")
        return df

    def get_csr_matrix(df_path):    #TODO: create interactions pipeline
        """
        Generate arrays of all users, all books, and implicit + explicit scores
        for each user to build a CSR matrix ready to be fed into an Alternating
        Least Squares model. Returns a sparse matrix.
        """
        user_list = []
        book_list = []
        score_list = []
        for chunk_no, chunk in enumerate(pd.read_csv(df_path, chunksize=200_000)):
            print(f"Processed {chunk_no} chunks")
            chunk = downcast(chunk)
            scores = get_score(chunk)

            # Map to 0-based index
            u_idx = chunk["user_id"]
            b_idx = chunk["book_id"]

            # drop NaN
            mask = u_idx.notna() & b_idx.notna()
            u_idx = u_idx[mask]
            b_idx = b_idx[mask]
            scores = scores[mask]

            user_list.append(u_idx.to_numpy())
            book_list.append(b_idx.to_numpy())
            score_list.append(scores.to_numpy())

            if os.environ["TRAINING_ON"] == "local":
                if chunk_no > 5:
                    break

        # Make 1D arrays:
        user_idx = np.concatenate(user_list)
        book_idx = np.concatenate(book_list)
        scores   = np.concatenate(score_list)

        print(f"Created CSR matrix from {chunk_no} chunks totalling {200000*chunk_no} rows")
        # Get sparse matrix of scores with users as rows and books as columns
        return csr_matrix(
            (scores, (user_idx, book_idx)),
            dtype="float32"
        )

    return get_csr_matrix(interactions_csv_path)


def train_and_save(input_csv: str, output_path: str):
    """
    Instantiate an ALS model and fit it on pre-generated sparse matrix.
    Returns a fitted model.
    """
    interactions_csr = load_data_and_build_csr(input_csv)

    model = AlternatingLeastSquares(
        factors = 128,
        regularization=0.1,
        iterations=30,
        use_gpu=False
    )

    model.fit(interactions_csr)

    artifact = {
        "model": model,
        "n_items": interactions_csr.shape[1]
    }

    joblib.dump(artifact, output_path, compress=3)
    print(f"Saved model to {output_path}")
