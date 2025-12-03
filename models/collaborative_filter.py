"""
Clean up and modify interactions data to create CSR matrix that will
be fed into an Alternating Least Squares model to get recommendations
based on similar users.
"""
import pandas as pd
import numpy as np
import time
import gzip
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Functions to modify data
## Downcasting columns
def downcast(df):
    """
    Downcast datatypes to reduce memory usage.
    """
    int_8_cols = ["is_read", "rating", "is_reviewed"]
    df[int_8_cols] = df[int_8_cols].astype("int8")
    df[["user_id", "book_id"]] = df[["user_id", "book_id"]].astype("int32")
    return df

## get score
def get_score(interactions_df): #TODO: Should this score be modified?
    """
    generate score from user interactions data.
    """
    return interactions_df["is_read"] \
           + interactions_df["rating"] \
           + interactions_df["is_reviewed"] * 2


def build_id_map(df_path="../raw_data/goodreads_interactions.csv"):
    """
    Generates a stable mapping of users and books starting from 0.
    Returns a tuple containing user_map and book_map
    """
    user_ids = set()
    book_ids = set()
    start = time.time()
    for chunk in pd.read_csv(df_path, chunksize=200_000):
        chunk = downcast(chunk)

        user_ids.update(chunk["user_id"].unique().tolist())
        book_ids.update(chunk["book_id"].unique().tolist())

    # Stable 0-based mappings...
    user_ids = list(user_ids)
    book_ids = list(book_ids)

    user_map = {u: i for i, u in enumerate(user_ids)}
    book_map = {b: i for i, b in enumerate(book_ids)}

    print(f"No. of users: {len(user_map)}", f"No. of books: {len(book_map)}", sep="\n")
    end = time.time()
    duration = end - start
    print(f"Time taken: {duration}")
    return (user_map, book_map)


def get_csr_matrix(user_map, book_map, df_path="../raw_data/goodreads_interactions.csv"):
    """
    Generate arrays of all users, all books, and implicit + explicit scores
    for each user to build a CSR matrix ready to be fed into an Alternating
    Least Squares model. Returns a sparse matrix.
    """
    user_list = []
    book_list = []
    score_list = []
    for chunk_no, chunk in enumerate(pd.read_csv(df_path, chunksize=200_000)):
        clear_output(wait=True)
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

        user_list.append(u_idx.to_numpy())
        book_list.append(b_idx.to_numpy())
        score_list.append(scores.to_numpy())
        # Last chunk to be processed is currently: 5
        if chunk_no > 5:
            break

    # Make 1D arrays:
    user_idx = np.concatenate(user_list)
    book_idx = np.concatenate(book_list)
    scores   = np.concatenate(score_list)

    # Get sparse matrix of scores with users as rows and books as columns
    return csr_matrix(
        (scores, (user_idx, book_idx)),
        dtype="float32"
    )


def make_and_run_model(sparse_matrix: csr_matrix):
    """
    Instantiate an ALS model and fit it on pregenerated sparse matrix.
    Returns a fitted model.
    """
    model = AlternatingLeastSquares(
        factors = 128,
        regularization=0.1,
        iterations=30
    )
    model.fit(sparse_matrix)
    return model


def get_recommendation(model, user_id, sparse_matrix, n_recs):
    """
    Generate recommendations for a user from a given model, sparse matrix,
    a user id, and the number of recommendations to generate. Currently will
    only return a book ID.
    """
    #TODO: build a book id and title database for quick lookup on book id.
    recs = model.recommend(userid=user_id,
                    user_items=sparse_matrix[user_id],
                    N=n_recs)
    return recs[0]
