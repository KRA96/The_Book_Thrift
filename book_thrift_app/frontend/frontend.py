import streamlit as st
import requests
import os
import json

from pathlib import Path
# from book_thrift_app.ML_logic.recommender import ALSRecommender

st.set_page_config(page_title='The Book Thrift', page_icon='📚')

# Custom CSS to match the beige color scheme
st.markdown("""
    <style>
    .stApp {
        background-color: #F5EFE7;
    }
    .stButton>button {
        background-color: #D4A574;
        color: white;
    }
    h1 {
        color: #1E3A5F;
    }
    /* Style all text elements */
    .stMarkdown, p, label {
        color: #3D3D3D ;
    }
    /* Make file uploader background match */
    [data-testid="stFileUploader"] {
        background-color: #F5EFE7;
    }
    </style>
    """, unsafe_allow_html=True)

# Get the directory where this file is located
current_dir = Path(__file__).parent
image_path = current_dir / "book_cover2.png"

# Display image
st.image(str(image_path), use_container_width=True)

# Use markdown with HTML for better color control
st.markdown('<h1 style="color: #1E3A5F;">📚 The Book Thrift</h1>', unsafe_allow_html=True)

# load model target and, depending on it, change frontend behaviour
# model_target = os.environ.get("MODEL_TARGET", None)
# set model target as cloud
model_target = "cloud"

if model_target == "cloud":     # assign cloud or local url based on model target
    API_URL = "https://the-book-thrift-43012920273.europe-west1.run.app"
else:
    API_URL = "http://localhost:8000"

# Save session state so OCR runs only once
if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None

# Get shelf picture upload
shelf = st.file_uploader("Upload a picture of the book shelf here",
                         type=["jpeg", "jpg", "png"])
# Get file upload
user = st.file_uploader("Upload your Goodreads CSV", type=["csv"])

# run OCR
if shelf is not None and st.session_state.ocr_result is None:
    # st.write("Detecting book titles...")
    st.markdown("<span style='color:black'>Detecting book titles...</span>", unsafe_allow_html=True)
    files = {
        "shelf": (shelf.name, shelf.getvalue(), shelf.type),
    }
    res = requests.post(f"{API_URL}/ocr", files=files)
    if not res.ok:
        st.error(f"OCR error: {res.status_code} - {res.text}")
        st.session_state.ocr_result = None
    else:
        st.session_state.ocr_result = res.json()
        st.success("Books detected!")

recs = None      # ChatGPT says to guard res at startup time

if user is not None and st.button("Get recommendations"):
    files = {
        "user": (user.name, user.getvalue(), user.type or "image/jpeg")}
    data = {
        "ocr_result": json.dumps(st.session_state.ocr_result["items"])}
    res = requests.post(f"{API_URL}/predict", data=data, files=files)
    if not res.ok:
        st.error(f"Error: {res.status_code} - {res.text}")
        recs = None
    else:
        recs = res.json()

if recs is not None:
    st.write("Recommendations:")
    st.dataframe(recs)
