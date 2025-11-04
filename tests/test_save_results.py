#!/usr/bin/env python3

import os
os.environ['MODEL_TARGET'] = 'mlflow'

import mlflow
from coffeedd.ml_logic.registry_ml import save_results
from coffeedd.params import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

print('🧪 Test directo de save_results con MLflow...')

# Configurar MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

# Test 1: Sin contexto activo de MLflow
print('\n📋 Test 1: Sin contexto MLflow activo')
test_params = {
    'context': 'test_without_run',
    'model': 'test_model',
    'sample_size': 'test'
}

test_metrics = {
    'test_accuracy': 0.85,
    'test_loss': 0.45,
    'test_recall': 0.78
}

save_results(params=test_params, metrics=test_metrics)

print('\n' + '='*50)

# Test 2: Con contexto activo de MLflow
print('\n📋 Test 2: Con contexto MLflow activo')
with mlflow.start_run() as run:
    print(f'🆔 MLflow Run ID: {run.info.run_id}')

    test_params_2 = {
        'context': 'test_with_run',
        'model': 'test_model_2',
        'sample_size': 'test_2'
    }

    test_metrics_2 = {
        'test_accuracy': 0.90,
        'test_loss': 0.35,
        'test_recall': 0.88
    }

    save_results(params=test_params_2, metrics=test_metrics_2)

print('\n✅ Tests completados')
