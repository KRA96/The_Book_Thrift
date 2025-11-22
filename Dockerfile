FROM python:3.12.9-slim
# slim version to reduce image size . buster version is a larger

COPY models models
COPY backend/api_file.py backend/api_file.py
COPY hardcover_api/user_books_api.py hardcover_api/user_books_api.py
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -e .

#Local development
# CMD uvicorn book_thrift_package.api_file:app --reload --host 0.0.0.0

#GCP (cloud) development
CMD uvicorn backend.api_file:app --host 0.0.0.0 --port $PORT
# CMD ["sh", "-c", "uvicorn backend.api_file:app --host 0.0.0.0 --port ${PORT:-8080}"]
