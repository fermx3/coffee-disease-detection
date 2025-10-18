.PHONY: default pylint pytest reinstall_package run_split_dataset organize_RoCoLe_images

.DEFAULT_GOAL := default

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y coffeedd || :
	@pip install -e .

run_split_dataset:
	python -c 'from coffeedd.ml_logic.split import split_dataset; split_dataset()'

run_split_resized_dataset:
	python -c 'from coffeedd.ml_logic.split import split_dataset; split_dataset(src_root="data/raw_data_224")'

#################### DEFAULT ACTIONS ###################
default: pylint pytest

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

#################  UTILITIES  #####################
organize_RoCoLe_images: #Debe existir el archivo data/RoCoLe/Annotations/RoCoLe-classes.xlsx y la carpeta data/RoCoLe/Photos
	python coffeedd/utilities/image_organizer.py --excel data/RoCoLe/Annotations/RoCoLe-classes.xlsx --src data/RoCoLe/Photos --dest data/RoCoLe_organized --sheet Hoja1

map_paths_and_labels:
	python -c 'from coffeedd.utilities.map_paths_and_labels import map_paths_and_labels; map_paths_and_labels("data/processed_data/train")'

preprocess_raw_letterbox_224:
	@python coffeedd/utilities/export_preprocessed_images.py --src data/raw_data --dst data/processed_data/resized_224 --target 224 --policy letterbox --clean-output
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
