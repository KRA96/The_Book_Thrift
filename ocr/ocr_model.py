from PIL import Image
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
from rapidfuzz import process, fuzz
import time

BOOK_DF = pd.read_csv(
    "../raw_data/book_titles.csv",
    usecols=["book_id", "title"],
    dtype={"book_id": "int32", "title": "string"},
)

BOOK_DF_INDEXED = BOOK_DF.set_index("title")  #Index the title


#0. Setup (book dataset, image_path:image given by the user, API Gemini khey:located in .env)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_PERS")


#1. API Call, Prompt configuration
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

prompt = """
    You are a book-cover OCR system.

    Look at the image and detect every visible book.
    Return ONLY valid JSON (no markdown, no comments) with this schema:

    {
      "books": [
        {
          "title": "<best guess of the book title>",
        }
      ]
    }

    - Use null for unknown fields.
    - Include only books where you see at least part of the title.
    """


#2. Detection of books title using Gemini OCR
def detect_books_in_image(image_path):

    img = Image.open(image_path)

    response = gemini_model.generate_content(
        [prompt, img],
        generation_config={
            "response_mime_type": "application/json"  # Forcing JSON
        }
    )
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse non JSON : {response.text}") from e

    return data


#3. Getting book id with detected titles (exact match or fuzzy match if the first doesn't work)

def get_info_from_dataset_by_title_exact(title: str, df_indexed: pd.DataFrame):
    """
    df_indexed : DataFrame with index = 'title'
    """
    try:
        row = df_indexed.loc[title] #previous one = df[df['title']==title]
    except KeyError:
        return None

    # If couple of rows have same title, take first one
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    return row.to_dict()



def get_info_from_dataset_by_title_fuzzy(title: str, df: pd.DataFrame, threshold: int = 80):

    if not isinstance(title, str) or not title.strip():
        return None

    titles = df["title"].tolist()

    best = process.extractOne(
        title,
        titles,
        scorer=fuzz.WRatio
    )

    if best is None:
        return None

    best_title, score, index = best

    if score < threshold:
        return None

    row = df.iloc[index].to_dict()
    row["_match_score"] = score
    return row

#4. Test of exact match and fuzzy ones if not working
def get_info_from_dataset_by_title(title: str,
                                   df_index: pd.DataFrame,
                                   fuzzy_threshold: int = 80):
    # 1) Exact match
    exact = get_info_from_dataset_by_title_exact(title, df_index)
    if exact is not None:
        exact["_match_type"] = "exact"
        exact["_match_score"] = 100
        return exact

    # 2) Fuzzy match
    #fuzzy = get_info_from_dataset_by_title_fuzzy(title, df, threshold=fuzzy_threshold)
    #if fuzzy is not None:
    #    fuzzy["_match_type"] = "fuzzy"
    #return fuzzy


#5. Function which process inputs with exact or fuzzy match and then create a new table with corresponding book id

def build_matched_books_table(detection,
                              books_df,
                              matcher,
                              fuzzy_threshold = 80):

    rows = []

    for book in detection.get("books", []):
        detected_title = (book.get("title") or "").strip()
        if not detected_title:
            continue

        match = matcher(detected_title, books_df, fuzzy_threshold)

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
