import requests
import pandas as pd
import os
import numpy as np

# get token
token = os.environ['TOKEN']

def get_random_book():
    url = "https://api.hardcover.app/v1/graphql"

    headers = {
    "content-type": "application/json",
    "authorization": f"Bearer {os.environ['TOKEN']}"
}

    query = """
    query UserBooks {
        user_books(
            where: {
                user_id: {_eq: 55022}
            },
            distinct_on: book_id
            offset: 0
        ) {
        book {
                title
        }
        }
    }
    """
    response = requests.post(url, json={"query": query}, headers=headers)

    size = len(response.json()['data']['user_books'])

    return response.json()['data']['user_books'][np.random.choice(size)]['book']['title']
if __name__ == "__main__":
    print(get_random_book())
