from PIL import Image
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
from rapidfuzz import process, fuzz



#0. Setup (book dataset, image_path:image given by the user, API Gemini khey:located in .env)
load_dotenv()
image_path = "testbook.jpg"
books_df = pd.read_csv("raw_data/book_titles.csv")

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
          "author_guess": "<best guess of the main author, can be empty>",
          "bounding_box": [x1, y1, x2, y2],
          "confidence": <number between 0 and 1>
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
            "response_mime_type": "application/json"  # force du JSON pur
        }
    )
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse non JSON : {response.text}") from e

    return data


#3. Getting book id with detected titles (exact match or fuzzy match if the first doesn't work)

def get_info_from_dataset_by_title_exact(title: str, df: pd.DataFrame):
    """Retourne la ligne du dataset dont le titre correspond exactement."""
    row = df[df["title"] == title]
    if row.empty:
        return None
    # On renvoie la première correspondance comme dict
    return row.iloc[0].to_dict()



def get_info_from_dataset_by_title_fuzzy(title: str, df: pd.DataFrame, threshold: int = 80):
    """
    Retourne la meilleure ligne du dataset dont le titre ressemble au 'title' donné.
    Utilise un score de similarité 0–100. On filtre sous un certain seuil.
    """
    if not isinstance(title, str) or not title.strip():
        return None

    titles = df["title"].tolist()

    best = process.extractOne(
        title,
        titles,
        scorer=fuzz.WRatio  # bon scoreur général
    )

    if best is None:
        return None

    best_title, score, index = best

    if score < threshold:
        # trop différent, on considère que ce n'est pas fiable
        return None

    row = df.iloc[index].to_dict()
    row["_match_score"] = score
    return row

#4. Test of exact match and fuzzy ones if not working
def get_info_from_dataset_by_title(title: str, df: pd.DataFrame, fuzzy_threshold: int = 80):
    """
    Essaie d'abord un match exact, puis un fuzzy match si rien trouvé.
    """
    # 1) Exact match
    exact = get_info_from_dataset_by_title_exact(title, df)
    if exact is not None:
        exact["_match_type"] = "exact"
        exact["_match_score"] = 100
        return exact

    # 2) Fuzzy match
    fuzzy = get_info_from_dataset_by_title_fuzzy(title, df, threshold=fuzzy_threshold)
    if fuzzy is not None:
        fuzzy["_match_type"] = "fuzzy"
    return fuzzy

#5. Function which process inputs with exact or fuzzy match and then create a new table with corresponding book id

def build_matched_books_table(detection,
                              books_df,
                              matcher,
                              fuzzy_threshold = 80):
    """
    detection: dict returned by Gemini
    books_df:  dataset with 'title' and 'book_id' columns
    matcher:   get_info_from_dataset_by_title function
    """

    rows = []

    for book in detection.get("books", []):
        detected_title = (book.get("title") or "").strip()
        if not detected_title:
            continue

        # Call match function
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
