FROM python:3.12.9-slim
# slim version to reduce image size . buster version is a larger

COPY backend/models backend/models
COPY backend/api_file backend/api_file
COPY backend/requirements.txt backend/requirements.txt
COPY backend/setup.py backend/setup.py

RUN pip install --upgrade pip
RUN pip install -e .

#Local development
CMD uvicorn book_thrift_package.api_file:app --reload --host 0.0.0.0

#GCP (cloud) development
# CMD uvicorn mush_package.api_file:app --reload --host 0.0.0.0 --port $PORT
