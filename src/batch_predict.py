import json
import os
import pandas as pd
from catboost import CatBoostClassifier
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


def run_batch_inference(
    test_parquet_path: str,
    model_path: str,
    meta_path: str,
    output_parquet_path: str,
    submission_csv_path: str,
    id_col: str = "PassengerId",
    target_col: str = "Transported"
):
    spark = (
        SparkSession.builder.appName("CatBoostBatchInference")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    with open(meta_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    features_cols = metadata["features_cols"]
    cat_cols = metadata["cat_cols"]

    model = CatBoostClassifier()
    model.load_model(model_path)

    broadcast_model = spark.sparkContext.broadcast(model)
    @F.pandas_udf(IntegerType())
    def predict_udf(*cols: pd.Series) -> pd.Series:
        dataframe = pd.concat(cols, axis=1)
        dataframe.columns = features_cols

        predictions = broadcast_model.value.predict(dataframe)
        return pd.Series(predictions.astype(int))


    test_dataframe = spark.read.parquet(test_parquet_path)

    dataframe_predictions = test_dataframe.withColumn(
        target_col, predict_udf(*[F.col(c) for c in features_cols])
    ).select(id_col, target_col)


    dataframe_predictions.write.mode("overwrite").parquet(output_parquet_path)

    os.makedirs(os.path.dirname(submission_csv_path), exist_ok=True)
    dataframe_predictions.toPandas().to_csv(submission_csv_path, index=False)


if __name__ == "__main__":
    run_batch_inference(
        test_parquet_path="data/processed/test_features.parquet",
        model_path="models/catboost_model.cbm",
        meta_path="models/model_meta.json",
        output_parquet_path="data/predictions/test_preds.parquet",
        submission_csv_path="data/submission.csv",
        id_col="PassengerId",
        target_col="Transported",
    )