import streamlit as st

st.set_page_config(page_title='Dummy book Recommender', page_icon=':books:')
st.title('Your Dummy Book Recommender')
book_image = st.file_uploader('Upload a picture of available books:',
                 type=['png', 'jpeg', 'jpg'])

goodreads_url = st.text_input('Paste your Goodreads profile url here')

if st.button('Generate recommendation'):
    if not book_image:
        st.warning("Please upload a picture of the shelf you're browing")
    if not goodreads_url:
        st.warning('Please enter a Goodreads profile url')
    else:
        st.write('Your book recommendation:', 'The Lord of the Rings')
