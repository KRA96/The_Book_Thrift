from fastapi import FastAPI
from models.dummy_model import dummy_rec
app = FastAPI()

@app.get('/')
def root():
    return {"message": "Welcomeeeeee to the book thrifting app!"}

@app.get("/predict")
def predict():
    return dummy_rec()
