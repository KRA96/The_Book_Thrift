import os

# Model
MODEL_PATH = os.environ["CURRENT_MODEL_PATH"]
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
