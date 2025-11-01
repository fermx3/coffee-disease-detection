.PHONY: default pylint pytest reinstall_package run_split_dataset organize_RoCoLe_images

.DEFAULT_GOAL := default

PY=python
PIP=pip

#################### PACKAGE ACTIONS ###################
#Instala dependencias
install:
	$(PIP) install -r requirements.txt

reinstall_package:
	@pip uninstall -y coffeedd || :
	@pip install -e .

run_split_dataset:
	python -c 'from coffeedd.ml_logic.split import split_dataset; split_dataset()'

run_split_resized_dataset:
	python -c 'from coffeedd.ml_logic.split import split_dataset; split_dataset(src_root="data/raw_data_224")'

run_train:
	python -c 'from coffeedd.interface.main import train; train()'

run_pred:
	python -c 'from coffeedd.interface.main import pred; pred()'

run_evaluate:
	python -c 'from coffeedd.interface.main import evaluate; evaluate()'

run_api:
	uvicorn coffeedd.api.fast:app --reload

#################### DEFAULT ACTIONS ###################
default: pylint pytest

#################  CODE QUALITY  #####################
#Limpia y corrige código
format:
	black .
	ruff check . --fix

#Ejecuta test
test:
	pytest -q

#Ejecuta format + test
quality: format test

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

test_gcp_setup:
	@pytest \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/test_gcp_setup.py::TestGcpSetup::test_code_get_project

#################  UTILITIES  #####################
organize_RoCoLe_images: #Debe existir el archivo data/RoCoLe/Annotations/RoCoLe-classes.xlsx y la carpeta data/RoCoLe/Photos
	python coffeedd/utilities/image_organizer.py --excel data/RoCoLe/Annotations/RoCoLe-classes.xlsx --src data/RoCoLe/Photos --dest data/RoCoLe_organized --sheet Hoja1

map_paths_and_labels:
	python -c 'from coffeedd.utilities.map_paths_and_labels import map_paths_and_labels; map_paths_and_labels("data/processed_data/train")'

preprocess_raw_letterbox_224:
	@python coffeedd/utilities/export_preprocessed_images.py --src data/raw_data --dst data/processed_data/resized_224 --target 224 --policy letterbox --clean-output

#corre análisis base
# eda:
# 	$(PY) coffeed/utilities/image-organizer.py

#Ejecuta todo el flujo
all: install quality eda

#################  DATA SOURCES ACTIONS  #####################

ML_DIR=~/.coffeedd/mlops

show_sources_all:
	-ls -laR ${LOCAL_DATA_PATH}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.coffeedd/mlops/training_outputs
	mkdir ~/.coffeedd/mlops/training_outputs/metrics
	mkdir -p ~/.coffeedd/mlops/training_outputs/models
	mkdir -p ~/.coffeedd/mlops/training_outputs/params
	mkdir -p ~/.coffeedd/mlops/training_outputs/checkpoints
