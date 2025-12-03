import math

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ocr_api import process_image_to_matches_df
from ml_model import predict_from_matches_df


app = FastAPI(
    title="Book Thrift",
    description="API to detect books on image, and make recommandations",
    version="1.0",
)

#Function to replace nans (raising errors with FastAPI)
def _replace_nans(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _replace_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nans(v) for v in obj]
    return obj


@app.get("/")
async def root():
    return {"status": "ok", "message": "Book OCR API is running"}


@app.post("/predict-books")
async def predict_books(file: UploadFile = File(...)):
    """
    Upload an image → OCR → Matching dataset → ML model → recommandations
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Format d'image non supporté.")

    image_bytes = await file.read()

    # OCR + matching dataset
    try:
        df_matches = process_image_to_matches_df(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ML prediction
    df_with_pred = predict_from_matches_df(df_matches)

    # Convert DataFrame → JSON
    records = df_with_pred.to_dict(orient="records")
    records = _replace_nans(records)

    return JSONResponse(content={"books": records})
