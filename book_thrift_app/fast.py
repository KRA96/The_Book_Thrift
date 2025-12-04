from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from .ML_logic.recommender import ALSRecommender
import pandas as pd
import json
import numpy as np
import os
from book_thrift_app.params import INTERACTIONS_CLEAN, BOOK_MAPPING_PATH
from book_thrift_app.ocr.ocr_api import process_image_to_matches_df
from book_thrift_app.ocr.ml_model import predict_from_matches_df
from book_thrift_app.ocr.main import _replace_nans

# from ocr.ocr_api import process_image_to_matches_df
# from ocr.ml_model import predict_from_matches_df
# from ocr.main import _replace_nans

app = FastAPI(title="Dummy Book Recommender Backend")

@app.get('/')
def root():
    return {"message": "Your Dummy Book Recommendation Online App"}

@app.post("/ocr")
async def ocr(shelf: UploadFile = File(...)):
    """
    Runs the OCR to detect book titles
    """

    image_bytes = await shelf.read()

    # OCR + matching dataset
    try:
        df_matches = process_image_to_matches_df(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    df_matches["book_id"] = pd.to_numeric(df_matches["book_id"], errors="coerce")

    mask = df_matches["book_id"].notna()

    shelf_book_ids = (
    df_matches.loc[mask, "book_id"]
    .astype("int32")                # force to int32
    .to_numpy(dtype=np.int32)       # and make sure numpy side is int32 too
    )

    # load book mapping to handle book id to csr index
    unique_books = np.load(BOOK_MAPPING_PATH)
    book_id_to_idx = {b_id: i for i, b_id in enumerate(unique_books)}

    shelf_book_index = (
        pd.Series(shelf_book_ids)
        .map(book_id_to_idx)
        .dropna()
        .astype("int32")
        .tolist()
    )
    print(f"shelf books: {shelf_book_index}")
    return {"items": shelf_book_index}  # 53102, 73423, 10344, 25714

@app.post("/predict")
async def predict(user: UploadFile = File(...),
                  ocr_result: str = Form(...),
                book_titles: str = INTERACTIONS_CLEAN):
    """
    runs an API call to gemini to get back shelf book
    """
    als = ALSRecommender()
    user = als._get_user_profile(user.file)

    item_idx = None
    if ocr_result is not None:
        item_idx = json.loads(ocr_result)
        print(f"Received item idx: {type(item_idx)}, {len(item_idx)}, {item_idx}")
    return als.recommend_books(user, items=item_idx)
