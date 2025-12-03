from io import BytesIO
import json
import os

from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import pandas as pd


#0. Setup (book dataset, image_path:image given by the user, API Gemini khey:located in .env)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_PERS")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY_PERS manquante dans l'environnement / .env")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


BOOK_DF = pd.read_csv(
    "../raw_data/book_titles.csv",
    usecols=["book_id", "title"],
    dtype={"book_id": "int32", "title": "string"},
)


BOOK_DF["title_lower"] = BOOK_DF["title"].str.lower()


BOOK_DF_INDEXED = BOOK_DF.set_index("title_lower")



prompt = """
You are a book-cover OCR system.

Look at the image and detect every visible book.
Return ONLY valid JSON (no markdown, no comments) with this schema:

{
  "books": [
    {
      "title": "<best guess of the book title>"
    }
  ]
}

- Use null for unknown fields.
- Include only books where you see at least part of the title.
"""


#1. Gemini Call


def detect_books_in_image(image_path: str) -> dict:
    """Version locale : lit une image sur disque."""
    img = Image.open(image_path)

    response = gemini_model.generate_content(
        [prompt, img],
        generation_config={"response_mime_type": "application/json"},
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse non JSON : {response.text}") from e

    return data

#2. Detection of books title using Gemini OCR

def detect_books_in_image_from_bytes(image_bytes: bytes) -> dict:
    """API Version : read an image sent by HTTP"""
    img = Image.open(BytesIO(image_bytes))

    response = gemini_model.generate_content(
        [prompt, img],
        generation_config={"response_mime_type": "application/json"},
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse non JSON : {response.text}") from e

    return data


#3. Getting book id with detected titles (exact match only)

def get_info_from_dataset_by_title_case_insensitive(title: str, df_indexed: pd.DataFrame):
    """
    EXACT Match (case-insensitive).
    """
    title_key = (title or "").strip().lower()

    try:
        row = df_indexed.loc[title_key]
    except KeyError:
        return None

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    result = row.to_dict()
    result["_match_type"] = "exact_case_insensitive"
    result["_match_score"] = 100
    return result


def get_info_from_dataset_by_title(title: str, df_index, fuzzy_threshold=80):
    return get_info_from_dataset_by_title_case_insensitive(title, df_index)



#5. Function which process inputs with exact or fuzzy match and then create a new table with corresponding book id

def build_matched_books_table(
    detection: dict,
    books_df_indexed: pd.DataFrame,
    matcher,
    fuzzy_threshold: int = 80,
):

    rows = []

    for book in detection.get("books", []):
        detected_title = (book.get("title") or "").strip()
        if not detected_title:
            continue

        match = matcher(detected_title, books_df_indexed, fuzzy_threshold)

        if match is None:
            rows.append({
                "detected_title": detected_title,
                "book_id": None,
                "match_type": None,
            })
            continue

        rows.append({
            "detected_title": detected_title,
            "book_id": match.get("book_id"),
            "match_type": match.get("_match_type"),
        })

    return pd.DataFrame(rows)


#5. Complete pipeline for API call


def process_image_to_matches_df(
    image_bytes: bytes,
    fuzzy_threshold: int = 80,
) -> pd.DataFrame:

    detection = detect_books_in_image_from_bytes(image_bytes)

    df_matches = build_matched_books_table(
        detection=detection,
        books_df_indexed=BOOK_DF_INDEXED,
        matcher=get_info_from_dataset_by_title,  # match exact only
        fuzzy_threshold=fuzzy_threshold,
    )

    return df_matches

#Test
