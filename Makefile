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

run_test_train:
	python -c 'from coffeedd.interface.main import train; train(test_mode=True)'

run_pred:
	python -c 'from coffeedd.interface.main import pred; pred()'

run_evaluate:
	python -c 'from coffeedd.interface.main import evaluate; evaluate()'

run_api:
	uvicorn coffeedd.api.fast:app --reload

upload_model:
	python -c "from coffeedd.interface.main import upload_model_to_gcs; upload_model_to_gcs()"

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

#################  GCS UPLOAD TESTS  #####################
test_gcs:
	python -m pytest tests/test_gcs_upload.py -v -s

test_gcs_real:
	python tests/test_gcs_upload.py

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
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.coffeedd/mlops/training_outputs
	mkdir ~/.coffeedd/mlops/training_outputs/metrics
	mkdir -p ~/.coffeedd/mlops/training_outputs/models
	mkdir -p ~/.coffeedd/mlops/training_outputs/models/vgg16
	mkdir -p ~/.coffeedd/mlops/training_outputs/models/efficientnet
	mkdir -p ~/.coffeedd/mlops/training_outputs/params
	mkdir -p ~/.coffeedd/mlops/training_outputs/checkpoints

reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_gcs_files

#################### DOCKER COMMANDS ###################
docker_build:
	docker build -t coffeedd-api .

docker_run: # you can test the API endpoints at http://localhost:8000/docs
	docker run -p 8000:8000 coffeedd-api

#################### YOLO (ULTRALYTICS) ###################

install_yolo:
	# 1) Clean any previous YOLO + torch installs (no wildcards)
	@pip uninstall -y ultralytics torch torchvision torchaudio || :
	# 2) Install CPU-only PyTorch stack
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	# 3) Install YOLO + API deps
	pip install ultralytics
	pip install -r requirements_yolo.txt

run_yolo_api:
	uvicorn coffeedd.api.api_yolo:app --reload

docker_build_yolo:
	docker build -f Dockerfile.yolo -t coffeedd-yolo-api .

docker_run_yolo:
	docker run -p 8000:8000 coffeedd-yolo-api

deploy_yolo_api:
	gcloud run deploy coffeedd-yolo-api \
	--image europe-southwest1-docker.pkg.dev/coffe-disease-detection-477521/coffee-api-repo/coffeedd-yolo-api \
	--region europe-southwest1 \
	--platform managed \
	--memory 2Gi \
	--allow-unauthenticated \
	--env-vars-file .env.yolo.yaml
