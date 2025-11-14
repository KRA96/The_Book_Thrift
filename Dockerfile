FROM python:3.12.9-slim
#change python version as needed
#slim version to reduce image size . buster version is a larger

COPY models models
COPY book_thrift_package book_thrift_package
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -e .

#Local development
CMD uvicorn book_thrift_package.api_file:app --reload --host 0.0.0.0

#GCP (cloud) development
# CMD uvicorn mush_package.api_file:app --reload --host 0.0.0.0 --port $PORT
