"""
Clean up and modify interactions data to create CSR matrix that will
be fed into an Alternating Least Squares model to get recommendations
based on similar users.
"""
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

import os
import gzip
import joblib
from pathlib import Path
from book_thrift_app.params import USER_MAPPING_PATH, BOOK_MAPPING_PATH, INTERACTIONS_CLEAN, ALS_4_DEC

HERE = HERE = Path(__file__).resolve().parent

def downcast(df):
        """
        Downcast datatypes to reduce memory usage.
        """
        # remove unnecessary repeated header rows
        df = df[df["is_read"].astype(str).str.match(r"^[0-1]$")]
        int_8_cols = ["is_read", "rating", "is_reviewed"]
        df[int_8_cols] = df[int_8_cols].astype("int8")
        df[["user_id", "book_id"]] = df[["user_id", "book_id"]].astype("int32")
        return df

def get_score(interactions_df): #TODO: Should this score be modified?
        """
        generate score from user interactions data.
        """
        return interactions_df["is_read"] \
            + interactions_df["rating"] \
            + interactions_df["is_reviewed"] * 2

def build_id_map(df_path, chunksize=200_000):
    """
    Generates a stable mapping of users and books starting from 0.
    Returns a tuple containing user_map and book_map
    """
    start = time.time()
    if not (
        Path("/home/krahmed96/the_book_thrift/ML_logic/unique_books.npy").exists() and
        Path("/home/krahmed96/the_book_thrift/ML_logic/unique_users.npy").exists()
        ):
        users = []
        books = []
        for chunk in pd.read_csv(df_path, chunksize=500_000):
            chunk = downcast(chunk)
            users.append(chunk["user_id"].to_numpy())
            books.append(chunk["book_id"].to_numpy())

        # Stable 0-based mappings...
        user_ids = np.unique(np.concatenate(users, axis=0))
        book_ids = np.unique(np.concatenate(books, axis=0))

        # Save to disk
        np.save("unique_users.npy", user_ids.astype("int32"))
        np.save("unique_books.npy", book_ids.astype("int32"))
    else:
        user_ids = np.load("unique_users.npy")
        book_ids = np.load("unique_books.npy")

    user_map = {user_raw_id: i for i, user_raw_id in enumerate(user_ids)}
    book_map = {book_raw_id: i for i, book_raw_id in enumerate(book_ids)}

    print(f"No. of users: {user_ids.shape[0]}", f"No. of books: {book_ids.shape[0]}", sep="\n")
    duration = time.time() - start
    print(f"Time taken: {duration}")
    return (user_map, book_map)

def get_csr_matrix(df_path=INTERACTIONS_CLEAN):    #TODO: create interactions pipeline
    """
    Generate arrays of all users, all books, and implicit + explicit scores
    for each user to build a CSR matrix ready to be fed into an Alternating
    Least Squares model. Returns a sparse matrix.
    """
    user_list = []
    book_list = []
    score_list = []

    user_map, book_map = build_id_map(INTERACTIONS_CLEAN, chunksize=500_000)

    for chunk_no, chunk in enumerate(pd.read_csv(df_path, chunksize=500_000)):
        print(f"Processed {chunk_no} chunks")
        chunk = downcast(chunk)
        scores = get_score(chunk)

        # Map to 0-based index
        u_idx = chunk["user_id"].map(user_map)
        b_idx = chunk["book_id"].map(book_map)

        # drop NaN
        mask = u_idx.notna() & b_idx.notna()
        u_idx = u_idx[mask]
        b_idx = b_idx[mask]
        scores = scores[mask]

        if len(scores) == 0:
            continue

        user_list.append(u_idx.to_numpy(dtype="int32"))
        book_list.append(b_idx.to_numpy(dtype="int32"))
        score_list.append(scores.to_numpy(dtype="float32"))

        if os.environ["TRAINING_ON"] == "local":
            if chunk_no > 5:
                break

    if not score_list:
        raise ValueError("No interactions after mapping/filtering.")

    # Make 1D arrays:
    user_idx = np.concatenate(user_list, axis=0)
    book_idx = np.concatenate(book_list, axis=0)
    scores   = np.concatenate(score_list, axis=0)

    print(f"Created CSR matrix from {chunk_no} chunks totalling {500000*chunk_no} rows")
    # Get sparse matrix of scores with users as rows and books as columns
    interactions_csr = csr_matrix((scores, (user_idx, book_idx)), dtype="float32")

    return interactions_csr


def train_and_save(input_csv: str = INTERACTIONS_CLEAN, output_path: str = ALS_4_DEC):
    """
    Instantiate an ALS model and fit it on pre-generated sparse matrix.
    Returns a fitted model.
    """
    interactions_csr = get_csr_matrix(input_csv)

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
