PY=python
PIP=pip

install:
	$(PIP) install -r requirements.txt

format:
	black .
	ruff check . --fix

test:
	pytest -q

eda:
	$(PY) coffeed/utilities/image-organizer.py

quality: format test

all: install quality eda
