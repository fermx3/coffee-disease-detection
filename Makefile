PY=python
PIP=pip

#Instala dependencias
install:
	$(PIP) install -r requirements.txt


#Limpia y corrige código
format:
	black .
	ruff check . --fix

#Ejecuta test
test:
	pytest -q

#corre análisis base
eda:
	$(PY) coffeed/utilities/image-organizer.py

#Ejecuta format + test
quality: format test

#Ejecuta todo el flujo
all: install quality eda
