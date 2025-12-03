import streamlit as st
import requests
import os
from pathlib import Path
# from book_thrift_app.ML_logic.recommender import ALSRecommender

st.set_page_config(page_title='The Book Thrift', page_icon='📚')

# Custom CSS to match the beige color scheme
st.markdown("""
    <style>
    .stApp {
        background-color: #F5EFE7 !important;
    }
    .stButton>button {
        background-color: #D4A574;
        color: white;
    }
    h1 {
        color: #1E3A5F !important;
    }
    /* Style all text elements */
    .stMarkdown, p, label {
        color: #3D3D3D !important;
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

# Get file upload
uploaded_file = st.file_uploader("Upload your Goodreads CSV", type=["csv"])

# load model target and, depending on it, change frontend behaviour
model_target = os.environ.get("MODEL_TARGET",None)

recs = None      # ChatGPT says to guard res at startup time

if model_target == "cloud":
    # Assign google cloud URL
    API_URL = os.environ.get("API_URL")

else:
    API_URL = "http://localhost:8000"

if uploaded_file is not None and st.button("Get recommendations"):
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }
    res = requests.post(f"{API_URL}/predict", files=files)
    if not res.ok:
        st.error(f"Error: {res.status_code} - {res.text}")
        recs = None
    else:
        recs = res.json()

# else:
#     if uploaded_file is not None and st.button("Get recommendations"):
#         user = recsys._get_user_profile(uploaded_file)
#         res = recsys.recommend_books(user)

if recs is not None:
    st.write("Recommendations:")
    st.dataframe(recs)
