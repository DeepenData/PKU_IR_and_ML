# TODO: Read the dataset

# TODO: Apply ADASYNT to smote the data

# TODO: Train a model with the synthetic data

# TODO: Evaluate the model using SHAP

# TODO: This are functios to call from optuna

# TODO: Dashboard in Streamlit

# %% ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score

SEED = 42

import yaml

with open("params.yaml", "r") as f:
    ext_params = yaml.load(f, Loader=yaml.FullLoader)


def data_preprocess(
    test_size=0.3,
    data_path: str = "data/data.csv",  # Path to the dataset
    removed_features: list[str] = [],  # Features to ignore
) -> list:
    data = pd.read_csv(data_path)

    # Convierte variables categoricas a numericas
    data["Género"] = data["Género"].replace({"M": 0, "F": 1})
    data["ATPII/AHA/IDF"] = data["ATPII/AHA/IDF"].replace({"no": 0, "si": 1})

    y = data.pop("HOMA-IR alterado").replace({"No": 0, "Si": 1})
    X = data.drop(removed_features + ["aleator"], axis="columns")

    oversample = RandomOverSampler(sampling_strategy="minority")
    X_res, y_res = oversample.fit_resample(X, y)
    return train_test_split(
        X_res, y_res, stratify=y_res, test_size=test_size, random_state=SEED
    )


def ml_train():
    X_train, X_test, y_train, y_test = data_preprocess(
        test_size=ext_params["train"]["testset_split"],
        removed_features=ext_params["feature_engineering"]["removed_features"],
    )
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(model):
    feature_metrics = pd.DataFrame(
        {
            "Weight": model.get_booster().get_score(importance_type="weight"),
            "Coverage": model.get_booster().get_score(importance_type="cover"),
            "Gain": model.get_booster().get_score(importance_type="gain"),
        }
    ).sort_values(by="Gain", ascending=False)
    return feature_metrics


def ml_eval():
    model = ml_train()
    _, X_test, _, y_test = data_preprocess()
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    feature_metrics = compute_model_metrics(model)
    return auc, feature_metrics


# %% ---
ml_eval()
