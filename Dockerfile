FROM python:3.12.9

COPY requirements.txt /book_thrift_app/requirements.txt
RUN pip install -r /book_thrift_app/requirements.txt

COPY book_thrift_app /book_thrift_app

# Local development: run uvicorn from inside the package directory
# CMD ["uvicorn", "book_thrift_app.fast:app", "--reload", "--host", "0.0.0.0"]

# GCP (cloud) development examples (uncomment & adjust if deploying to cloud)
CMD uvicorn book_thrift_app.fast:app --host 0.0.0.0 --port $PORT
