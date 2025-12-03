import streamlit as st
import requests
import pandas as pd


API_URL = "http://localhost:8000"

# API_URL_CLOUD = "https://url_cloud///"

st.set_page_config(page_title="Book OCR Recommender", page_icon="📚")
st.title("📚 Book OCR Recommender (API)")

st.markdown(
    """
    Upload a **picture with books** 📷
    The API will :
    1. Read titles with Gemini (OCR)
    2. Match them with a dataset
    3. Apply these id to a model
    4. Return recommandations
    """
)

uploaded_file = st.file_uploader(
    "Upload an image of books (JPG/PNG/WebP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

if uploaded_file is not None and st.button("Get recommendations"):
    with st.spinner("Image analysis and recommandations generation..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/jpeg",
            )
        }

        try:
            resp = requests.post(f"{API_URL}/predict-books", files=files)
        except Exception as e:
            st.error(f"Failed to connect to API : {e}")
        else:
            if resp.ok:
                data = resp.json()
                books = data.get("books", [])

                if not books:
                    st.warning("No books detected")
                else:
                    st.success("Recommandations obtained ✅")

                    # Convert dict into dataframe
                    df = pd.DataFrame(books)

                    col_order = [
                        "detected_title",
                        "book_id",
                        "match_type",
                        "prediction",
                        "score",
                    ]
                    df = df[[c for c in col_order if c in df.columns]]

                    st.dataframe(df, use_container_width=True)
            else:
                st.error(f"API error : {resp.status_code} - {resp.text}")
