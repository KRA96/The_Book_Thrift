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
    usecols=["book_id", "title"],   # only what you actually need
    dtype={"book_id": "int32", "title": "string"},
)

BOOK_TITLES = BOOK_DF["title"].astype(str).tolist()

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
            "response_mime_type": "application/json"  # force du JSON pur
        }
    )
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse non JSON : {response.text}") from e

    return data

if __name__ == "__main__":
    print(detect_books_in_image("testbook.jpg"))
