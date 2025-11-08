FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY coffeedd coffeedd

CMD uvicorn coffeedd.api.fast:app --host 0.0.0.0 --port $PORT
