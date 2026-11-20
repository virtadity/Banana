import json
import os
import pandas as pd
from catboost import  CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split


def train_model(
    train_parquet_path: str,
    model_output_path: str,
    meta_output_path: str,
    target_col: str,
    id_col: str,
    cat_cols: list[str],
):
    dataset = pd.read_parquet(train_parquet_path)

    features_cols = [col for col in dataset.columns if col not in [target_col, id_col]]

    X = dataset[features_cols]
    y = dataset[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="Accuracy",
        random_seed=42,
        early_stopping_rounds=50,
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True
    )

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    model.save_model(model_output_path)

    metadata = {
        "features_cols": features_cols,
        "cat_cols": cat_cols,
        "best_score": model.get_best_score(),
        "best_iteration": model.get_best_iteration(),
    }

    with open(meta_output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    train_model(
        train_parquet_path="data/processed/train_features.parquet",
        model_output_path="models/catboost_model.cbm",
        meta_output_path="models/model_meta.json",
        target_col="Transported",
        id_col="PassengerId",
        cat_cols=["CryoSleep", "VIP", "Cabin_first_index", "Cabin_second_index", "Cabin_third_index",
                  "HomePlanet", "Destination", "First_name", "Second_name"]
    )