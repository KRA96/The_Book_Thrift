import streamlit as st
import requests
from the_book_thrift.ML_logic.recommender import ALSRecommender

st.set_page_config(page_title='Dummy book Recommender', page_icon=':books:')
st.title('MVP Book Recommender')

@st.cache_resource
def load_recommender():
    return ALSRecommender()

recsys = load_recommender()

uploaded_file = st.file_uploader("Upload your Goodreads CSV", type=["csv"])

if uploaded_file is not None and st.button("Get recommendations"):
    user = recsys._get_user_profile(uploaded_file)
    res = recsys.recommend_books(user)
    st.write("Recommendations:")
    st.dataframe(res)
