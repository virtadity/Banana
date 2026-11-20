# Distributed Batch ML Pipeline: PySpark + CatBoost + Docker

End-to-end ML-пайплайн для решения задачи бинарной классификации (Kaggle) с использованием распределенной обработки данных на **PySpark**, градиентного бустинга **CatBoost** и контейнеризации в **Docker**.

## Архитектура проекта

Пайплайн разделен на три независимых этапа, связанных через формат хранения **Parquet**:

1. **ETL & Feature Engineering (`src/etl_pipeline.py`)**
   * Чтение сырых данных и генерация признаков на PySpark.
   * Сохранение обработанных витрин в сжатый колоночный формат Parquet.
2. **Model Training (`src/train.py`)**
   * Обучение `CatBoostClassifier` с нативной поддержкой категориальных признаков и early stopping.
   * Сохранение весов модели (`.cbm`) и метаданных признаков (`model_meta.json`) для гарантии консистентности схемы при инференсе.
3. **Distributed Batch Inference (`src/batch_predict.py`)**
   * Распределенный скоринг тестовой выборки с помощью **PySpark Pandas UDF** и **Apache Arrow**.
   * Использование `SparkContext.broadcast()` для эффективной рассылки модели на воркеры без лишней сериализации.
   * Экспорт результатов в Parquet и формирование итогового `submission.csv`.

## Структура репозитория

```text
├── data/                   # Директория для сырых данных, Parquet-витрин и сабмитов (в .gitignore)
├── models/                 # Артефакты обученной модели и метаданные (в .gitignore)
├── src/
│   ├── etl_pipeline.py     # Предобработка и генерация фичей на PySpark
│   ├── train.py            # Обучение CatBoost и сохранение метаданных
│   └── batch_predict.py    # Распределенный инференс через Pandas UDF
├── Dockerfile              # Сборка окружения (Python 3.10 + OpenJDK 17 + PySpark + CatBoost)
├── docker-compose.yml      # Оркестрация запуска и проброс volumes
├── requirements.txt        # Зависимости проекта
└── run_pipeline.sh         # Скрипт последовательного запуска всех этапов
```

## Быстрый запуск

### Предварительные требования
* Установленные **Docker** и **Docker Compose**.
* Исходные файлы `train.csv` и `test.csv`, помещенные в папку `data/raw/`.

### Запуск всего пайплайна одной командой
```bash
docker compose up --build
```
После завершения работы скрипта готовый файл с предсказаниями появится по пути `data/submission.csv`.

### Запуск отдельных этапов (опционально)
```bash
docker compose run --rm ml-pipeline python src/etl_pipeline.py
docker compose run --rm ml-pipeline python src/train.py
docker compose run --rm ml-pipeline python src/batch_predict.py
```

## Технологический стек
* **Data Processing:** PySpark, Apache Arrow, Parquet
* **Machine Learning:** CatBoost, Scikit-learn, Pandas
* **Infrastructure:** Docker, Docker Compose, OpenJDK 21
