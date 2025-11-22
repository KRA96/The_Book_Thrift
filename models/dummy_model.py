from the_book_thrift.hardcover_api import get_random_book
def dummy_rec():
    book = get_random_book()
    return(f"A book in your list is {book}")

if __name__ == '__main__':
    print(dummy_rec())
