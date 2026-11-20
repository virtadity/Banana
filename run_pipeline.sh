#!/bin/bash
set -e

echo "=== [1/3] Запуск ETL и генерации признаков (PySpark) ==="
python src/etl_pipeline.py

echo "=== [2/3] Обучение модели (CatBoost) ==="
python src/train.py

echo "=== [3/3] Распределенный инференс (PySpark Pandas UDF) ==="
python src/batch_predict.py

echo "=== Пайплайн успешно завершен! Результат в data/submission.csv ==="
