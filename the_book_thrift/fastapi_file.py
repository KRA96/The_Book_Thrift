from fastapi import FastAPI
from ML_logic.recommender import ALSRecommender
app = FastAPI(title="Dummy Book Recommender Backend")

@app.get('/')
def root():
    return {"message": "Your Dummy Book Recommendation Online App"}

@app.get("/predict")
def predict():
    als = ALSRecommender()
    user = als._get_user_profile()
    return als.recommend_books()
