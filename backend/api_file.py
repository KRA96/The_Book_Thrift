from fastapi import FastAPI
from models.dummy_model import dummy_rec
app = FastAPI(title="Dummy Book Recommender Backend")

@app.get('/')
def root():
    return {"message": "Your Dummy Book Recommendation Online App"}

@app.get("/predict")
def predict():
    return dummy_rec()
