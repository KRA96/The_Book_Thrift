import os


# Model
COLLAB_MODEL = os.environ["ALS_4_DEC"]
TFID_FITTED = os.environ["TFID_FITTED"]

# Paths to models and npy files
BOOK_ID_PATH = os.environ["BOOK_ID_PATH"]
USER_MAPPING_PATH = os.environ["USER_MAPPING_PATH"]
BOOK_MAPPING_PATH = os.environ["BOOK_MAPPING_PATH"]
ALS_4_DEC = os.environ["ALS_4_DEC"]
ALL_BOOKS = os.environ["ALL_BOOKS"]

# Paths to data
BOOK_TITLES = os.environ["BOOK_TITLES"]
INTERACTIONS_CLEAN= os.environ["INTERACTIONS_CLEAN"]

# GCP and GCS vars
GCP_PROJECT  =  os.environ["GCP_PROJECT"]
GCP_REGION  =  os.environ["GCP_REGION"]
ARTIFACTSREPO = os.environ["ARTIFACTSREPO"]
VM_INSTANCE = os.environ["VM_INSTANCE"]
PROJECT = os.environ["PROJECT"]
GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
MEMORY = os.environ["MEMORY"]

# BigQuery vars
BUCKET = os.environ["BUCKET"]
DATASET = os.environ["DATASET"]
BOOKS_GRAPH_TABLE = os.environ["BOOKS_GRAPH_TABLE"]

# Docker vars
IMAGE = os.environ["IMAGE"]

# GraphQL Hardcover API Token
TOKEN = os.environ["TOKEN"]
