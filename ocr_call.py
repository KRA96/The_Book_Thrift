from ocr_model import detect_books_in_image, build_matched_books_table, get_info_from_dataset_by_title
import pandas as pd

def process_image_for_book_ids(
    image_path,
    books_df=pd.read_csv("raw_data/book_titles.csv"),
    fuzzy_threshold = 80
):

    # 1) Detection of book titles with Gemini
    detection = detect_books_in_image(image_path)

    # 2) Return final table with title + book_id
    final_table = build_matched_books_table(
        detection=detection,
        books_df=books_df,
        matcher=get_info_from_dataset_by_title,
        fuzzy_threshold=fuzzy_threshold,
    )

    return final_table

if __name__ == "__main__":
    books_df = pd.read_csv("raw_data/book_titles.csv")
    print(process_image_for_book_ids("textbook2.png", books_df))
