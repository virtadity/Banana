from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.dataframe import DataFrame


def run_etl(
        train_path: str,
        test_path: str,
        train_output_path: str,
        test_output_path: str
):
    spark = create_spark_session()

    train_dataset = spark.read.csv(train_path, header=True, inferSchema=True)
    test_dataset = spark.read.csv(test_path, header=True, inferSchema=True)

    updated_train_dataset = apply_stateless_processing(train_dataset)
    updated_test_dataset = apply_stateless_processing(test_dataset)

    updated_train_dataset.write.mode("overwrite").parquet(train_output_path)
    updated_test_dataset.write.mode("overwrite").parquet(test_output_path)
    spark.stop()


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("KaggleSpaceshipTitanicETF")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def apply_stateless_processing(dataset: DataFrame):
    updated_dataset = (dataset
        .transform(preprocess_home_planet)
        .transform(preprocess_cryo_sleep)
        .transform(preprocess_cabin)
        .transform(preprocess_passenger_name)
        .transform(preprocess_vip)
        .transform(preprocess_destination)
        .transform(preprocess_target))
    return updated_dataset


def preprocess_home_planet(dataset: DataFrame):
    updated_dataset = dataset.fillna("unknown", subset="HomePlanet")
    return updated_dataset


def preprocess_cryo_sleep(dataset: DataFrame):
    updated_dataset = dataset.withColumn("CryoSleep",
                       f.when(f.col("CryoSleep") == True,1)
                        .when(f.col("CryoSleep") == False,0)
                        .otherwise(-1))
    return updated_dataset


def preprocess_cabin(dataset: DataFrame):
    updated_dataset = (
        dataset
            .withColumn("Cabin_indexes", f.split(f.col("Cabin"), "/"))
            .withColumn("Cabin_first_index", f.coalesce(f.col("Cabin_indexes").getItem(0), f.lit("unknown")))
            .withColumn("Cabin_second_index", f.coalesce(f.col("Cabin_indexes").getItem(1), f.lit("unknown")))
            .withColumn("Cabin_third_index", f.coalesce(f.col("Cabin_indexes").getItem(2), f.lit("unknown")))
            .drop("Cabin_indexes")
            .drop("Cabin")
    )
    return updated_dataset


def preprocess_passenger_name(dataset: DataFrame):
    updated_dataset = (
        dataset
            .withColumn("Names", f.split(f.col("Name"), " "))
            .withColumn("First_name", f.coalesce(f.col("Names").getItem(0), f.lit("unknown")))
            .withColumn("Second_name", f.coalesce(f.col("Names").getItem(1), f.lit("unknown")))
            .drop("Names")
            .drop("Name")
    )
    return updated_dataset


def preprocess_vip(dataset: DataFrame):
    updated_dataset = (
        dataset.withColumn("VIP", f
            .when(f.col("VIP") == True, 1)
            .when(f.col("VIP") == False, 0)
            .otherwise(-1)
        )
    )
    return updated_dataset


def preprocess_target(dataset: DataFrame):
    if "Transported" in dataset.columns:
        updated_dataset = dataset.withColumn("Transported", f.col("Transported").astype("double"))
        return updated_dataset
    return dataset


def preprocess_destination(dataset: DataFrame):
    updated_dataset = dataset.fillna("unknown", subset="Destination")
    return updated_dataset


if __name__ == "__main__":
    run_etl(
        train_path="data/raw/train.csv",
        train_output_path="data/processed/train_features.parquet",
        test_path="data/raw/test.csv",
        test_output_path="data/processed/test_features.parquet"
    )