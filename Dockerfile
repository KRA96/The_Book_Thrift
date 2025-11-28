FROM python:3.12.9

# set workdir where pip will look for setup.py
WORKDIR /the_book_thrift

# copy als pickle, collab model, and recommender from ML logic
COPY the_book_thrift/ML_logic /the_book_thrift/ML_logic

# copy requirements and setup
COPY requirements.txt /the_book_thrift/requirements.txt
COPY the_book_thrift/setup.py /the_book_thrift/setup.py

# get requirements to avoid errors later
RUN pip install -r /the_book_thrift/requirements.txt

# copy fastapi file
COPY the_book_thrift/fastapi_file.py /the_book_thrift/fastapi_file.py

# Copy book titles file
COPY the_book_thrift/book_titles.csv /the_book_thrift/book_titles.csv

# install the package in editable mode (setup.py is at /the_book_thrift/setup.py)
RUN pip install -e .

# Run the the_book_thrift with the repo package directory as the working dir so
# top-level imports like `ML_logic` resolve as they did previously.
WORKDIR /the_book_thrift/

# Local development: run uvicorn from inside the package directory
CMD ["uvicorn", "fastapi_file:app", "--reload", "--host", "0.0.0.0"]

# GCP (cloud) development examples (uncomment & adjust if deploying to cloud)
# CMD ["uvicorn", "the_book_thrift.fastapi_file:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]
