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

# goodreads_url = st.text_input('Paste your Goodreads profile url here')

if uploaded_file is not None and st.button("Get recommendations"):
    res = requests.get(url='https://the-book-thrift-43012920273.europe-west1.run.app/predict').json()
    st.write(res)


    # if not book_image:
    #     st.warning("Please upload a picture of the shelf you're browing")
    # if not goodreads_url:
    #     st.warning('Please enter a Goodreads profile url')
    # else:
    #     st.write('Your book recommendation:', 'The Lord of the Rings')
