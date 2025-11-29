from fastapi import FastAPI, UploadFile, File
from .ML_logic.recommender import ALSRecommender
import os
app = FastAPI(title="Dummy Book Recommender Backend")

@app.get('/')
def root():
    return {"message": "Your Dummy Book Recommendation Online App"}

# return recommendations based on whether cloud or local
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    als = ALSRecommender()
    user = als._get_user_profile(file.file)
    return als.recommend_books(user)
