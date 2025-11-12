import streamlit as st

st.title('A Dummy Front-End')
book_image = st.file_uploader('Upload a picture of available books:',
                 type=['png', 'jpeg', 'jpg'])

if book_image:
    st.write('Your book recommendation:', 'The Lord of the Rings')
