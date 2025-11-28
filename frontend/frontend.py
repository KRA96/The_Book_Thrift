import streamlit as st
import requests
# from the_book_thrift.ML_logic.recommender import ALSRecommender

st.set_page_config(page_title='Dummy book Recommender', page_icon=':books:')
st.title('MVP Book Recommender')

# @st.cache_resource
# def load_recommender():
#     return ALSRecommender()

# recsys = load_recommender()

uploaded_file = st.file_uploader("Upload your Goodreads CSV", type=["csv"])
# API_URL = "http://localhost:8000"
GC_URL = "https://the-book-thrift-43012920273.europe-west1.run.app"
# send to fastapi
if uploaded_file is not None and st.button("Get recommendations"):
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }
    resp = requests.post(f"{GC_URL}/predict", files=files)
    if resp.ok:
        st.write("Recommendations:")
        st.dataframe(resp.json())
    else:
        st.error(f"Error: {resp.status_code} - {resp.text}")
