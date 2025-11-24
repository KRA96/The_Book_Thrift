from the_book_thrift.hardcover_api import get_random_book
def dummy_rec():
    book = get_random_book()
    return(f"A book in your list is {book}")


def generate_rec(user_data):
    """ 
    This function will take a user's data and generate a recommendation
    based currently on the collaborative filter system.
    """

if __name__ == '__main__':
    print(dummy_rec())
