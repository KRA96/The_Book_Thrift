import streamlit as st
import requests
import os
from book_thrift_app.ML_logic.recommender import ALSRecommender

st.set_page_config(page_title='Dummy book Recommender', page_icon=':books:')
st.title('MVP Book Recommender')

# Get file upload
uploaded_file = st.file_uploader("Upload your Goodreads CSV", type=["csv"])

# load model target and, depending on it, change frontend behaviour
model_target = os.environ["MODEL_TARGET"]

recs = None      # ChatGPT says to guard res at startup time

if model_target == "cloud":
    # Assign google cloud URL
    API_URL = "https://the-book-thrift-43012920273.europe-west1.run.app"

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
