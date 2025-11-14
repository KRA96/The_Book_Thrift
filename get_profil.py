from goodreads import client
from collections import Counter, defaultdict
import statistics as stats

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
USER_ID = "USER_ID_GOODREADS"

# Connection to the client
gc = client.GoodreadsClient(API_KEY, API_SECRET)


# Function to collect books based on user_id
def get_books_from_shelf(user_id, shelf_name="read"):
    user = gc.user(user_id)
    shelf = user.shelf(shelf_name)
    return list(shelf)

books = get_books_from_shelf(USER_ID, "read")  #Create a list with all books read by the user


# Creation of sets to collect features based on books read
author_counter = Counter()
year_counter = Counter()
page_counts = []
avg_rating_given = []

for b in books:
    # Author
    try:
        main_author = b.authors[0].name
        author_counter[main_author] += 1
    except Exception:
        pass

    # Publication date
    try:
        year = int(b.publication_year) if b.publication_year else None
        if year:
            year_counter[year] += 1
    except Exception:
        pass

    # Number of pages
    try:
        pages = int(b.num_pages) if b.num_pages else None
        if pages:
            page_counts.append(pages)
    except Exception:
        pass

    # Mean rating
    try:
        rating = float(b.average_rating)
        avg_rating_given.append(rating)
    except Exception:
        pass



shelf_tags = Counter()

for b in books[:30]:  #Determine the number of books to iter to get popular shelves
    full_book = gc.book(b.gid)  # Get global information on book
    raw = full_book._book_dict

    try:
        shelves = raw['popular_shelves']['shelf'] #Get the popular attributed category to the book
        for s in shelves:
            name = s['@name']
            count = int(s['@count'])
            if count > 50:  # Filter shelf with low count
                shelf_tags[name] += count
    except Exception:
        continue
