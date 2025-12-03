

from ocr_model import detect_books_in_image, build_matched_books_table, matcher
import pandas as pd


def read_csv(path):
    BOOK_DF = pd.read_csv(
    "../raw_data/book_titles.csv",
    usecols=["book_id", "title"],   # only what you actually need
    dtype={"book_id": "int32", "title": "string"},
)
    return BOOK_DF

import time

def time_block(label, func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    print(f"{label}: {t1 - t0:.2f} seconds")
    return result
# Example structure – adapt to your actual function names
# 1.Load CSV
books_df = time_block("Load CSV", read_csv, "../raw_data/book_titles.csv")
# 2. Gemini
detected_titles = time_block("Gemini OCR", detect_books_in_image, '/home/pique/code/KRA96/The_Book_Thrift/ocr/IMG_6984.jpeg')
# 3. Fuzzy matching sur tous les livres détectés
matches_df = time_block(
    "Fuzzy matching",
    build_matched_books_table,
    detected_titles,      # dict {"books": [...]}
    books_df,        # ton DataFrame
    matcher,        # la fonction matcher ci-dessus
    80              # fuzzy_threshold
)
