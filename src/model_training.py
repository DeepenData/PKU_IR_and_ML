# TODO: Dashboard in Streamlit

# %% ---

import numpy as np
import pandas as pd
import xgboost
import yaml
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# import shap


def data_preprocess(
    predictor_col: str,
    test_size=0.3,
    data_path: str = "data/data.csv",  # Path to the dataset
    removed_features: list[str] = [],  # Features to ignore
    seed=None,
) -> list:
    dataset = pd.read_csv(data_path)

    # Variable encoding
    dataset["Género"] = dataset["Género"].replace({"M": 0, "F": 1}).astype("category")
    dataset["ATPII/AHA/IDF"] = (
        dataset["ATPII/AHA/IDF"].replace({"no": 0, "si": 1}).astype("category")
    )
    dataset["aleator"] = (
        dataset["aleator"]
        .replace({"Control": 0, "PKU 1": 1, "PKU 2": 2})
        .astype("category")
    )

    y_df = dataset[predictor_col].replace({"No": 0, "Si": 1}).astype("category")
    X_df = dataset.drop(predictor_col, axis="columns")

    X_df = X_df.drop(removed_features, axis="columns")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, stratify=y_df, test_size=test_size, random_state=seed
    )

    # # Resampling to the original ratio of classes
    # global X_test_res  # So I can call it from SHAP
    # X_test_res = pd.concat(
    #     [
    #         X_test[y_test == 0].sample(80, replace=True),
    #         X_test[y_test == 1].sample(20, replace=True),
    #     ],
    #     ignore_index=True,
    # )

    # global y_test_res  # So I can call it from SHAP
    # y_test_res = [0] * 80 + [1] * 20 # Not a number, a list of categories

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )  # X_df, y_df, X_train, y_train  # , y_test_res, X_test_res,


def xg_train(X_train, y_train, xg_params: dict, kfold_splits=5, seed=None):
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X_train, y_train))

    for train_idx, val_idx in folds:
        # Sub-empaquetado del train-set en formato de XGBoost
        dtrain = xgboost.DMatrix(
            X_train.iloc[train_idx, :], label=y_train.iloc[train_idx]
        )
        dval = xgboost.DMatrix(X_train.iloc[val_idx, :], label=y_train.iloc[val_idx])

        model = xgboost.train(
            dtrain=dtrain,
            params=xg_params,
            evals=[(dtrain, "train"), (dval, "val")],
            num_boost_round=1000,
            verbose_eval=False,
            early_stopping_rounds=10,
        )

    return model


## MOVE FROM HERE
with open("params.yml", "r") as f:
    ext_params = yaml.load(f, Loader=yaml.FullLoader)

processed_data = data_preprocess(
    "HOMA-IR alterado",
    data_path="data/resampled_data_SMOTE.csv",
    removed_features=ext_params["feature_engineering"]["removed_features"],
)

default_xg_params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "eval_metric": "logloss",
}

modelo = xg_train(processed_data[0], processed_data[2], default_xg_params)


# %%
def xg_test(model, X_test, y_test) -> float:
    testset = xgboost.DMatrix(X_test, label=y_test)
    y_preds = model.predict(testset)
    return roc_auc_score(testset.get_label(), y_preds)


# %%


def compute_model_metrics(model, dataset, y_df):
    internal_feature_metrics = pd.DataFrame(
        {
            "Weight": model.get_score(importance_type="weight"),
            "Coverage": model.get_score(importance_type="cover"),
            "Gain": model.get_score(importance_type="gain"),
        }
    ).sort_values(by="Gain", ascending=False)

    explainer = shap.TreeExplainer(model)

    # Extrae la explicacion SHAP en un DF
    EXPLAINATION = explainer(dataset).cohorts(
        y_df.replace({0: "Healty", 1: "Abnormal"}).tolist()
    )
    cohort_exps = list(EXPLAINATION.cohorts.values())

    exp_shap_abnormal = pd.DataFrame(
        cohort_exps[0].values, columns=cohort_exps[0].feature_names
    )  # .abs().mean()

    exp_shap_healty = pd.DataFrame(
        cohort_exps[1].values, columns=cohort_exps[1].feature_names
    )  # .abs().mean()

    SHAP_val = pd.concat(
        [exp_shap_healty, exp_shap_abnormal],
        ignore_index=True,
    )

    return internal_feature_metrics, exp_shap_healty, exp_shap_abnormal


# %% ---


def main(
    data_path="data/resampled_data_SMOTE.csv",
    seed=None,
    kfold_splits=5,
    xg_params={
        "eta": 0.01,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "eval_metric": "logloss",
    },
):
    with open("params.yml", "r") as f:
        ext_params = yaml.load(f, Loader=yaml.FullLoader)

    SEED = ext_params["train"]["seed"]
    KFOLD_SPLITS: int = ext_params["train"]["kfold_splits"]

    predictor_col = "HOMA-IR alterado"

    X_df, y_df, X_train, X_test, y_train, y_test = data_preprocess(
        predictor_col=predictor_col,
        test_size=ext_params["train"]["testset_split"],
        removed_features=ext_params["feature_engineering"]["removed_features"],
        data_path=data_path,
        seed=seed,
    )

    model = xg_train(
        X_train, y_train, kfold_splits=kfold_splits, seed=seed, xg_params=xg_params
    )

    auc = xg_test(model, X_test, y_test)

    (
        internal_feature_metrics,
        exp_shap_healty,
        exp_shap_abnormal,
    ) = compute_model_metrics(model, dataset=X_df, y_df=y_df)

    feature_metrics = pd.concat(
        {
            "SHAP_healty": exp_shap_healty.abs().mean(),
            "SHAP_abnormal": exp_shap_abnormal.abs().mean(),
        },
        axis="columns",
    )

    feature_metrics = feature_metrics.join(internal_feature_metrics).fillna(0)

    return model, feature_metrics, auc


# %% ---
model, feature_metrics, auc = main(
    data_path="data/resampled_data_SMOTE.csv",
    seed=None,
    kfold_splits=5,
)

feature_metrics.sort_values(by="SHAP_abnormal", ascending=False).head()


# %% ---
