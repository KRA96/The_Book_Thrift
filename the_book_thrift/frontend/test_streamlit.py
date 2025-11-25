import streamlit as st
from the_book_thrift.ML_logic.recommender import ALSRecommender

@st.cache_resource
def load_recommender():
    return ALSRecommender()

st.set_page_config(page_title='Test book Recommender', page_icon=':books:')
st.title("ALS Book Recommender (Local Test)")

recsys = load_recommender()

uploaded_file = st.file_uploader("Upload your Goodreads CSV", type=["csv"])

if uploaded_file is not None and st.button("Get recommendations"):
    user_items = recsys._get_user_profile(uploaded_file)
    recs = recsys.recommend_books(user_items)

    st.write(f"Returned {len(recs)} items")
    st.dataframe(recs)
