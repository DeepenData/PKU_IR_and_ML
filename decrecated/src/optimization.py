"""Runs Optuna as an optimizer for certain features of the data. We tested sequentially all the features, finding Phe_C 
as an adequate marker for altered HOMA levels. 

"""
# %%
# Importamos las cosas de Optuna
import optuna
from src.model_training import objective as model_training_main

# First of all, sorry. This is a mixture of using DVC for ease of tracking in several computes, plus Optuna for ease of
# running a lot of trials. As DVC expects a plain-text file for data, said data is saved in metrics.json.
# This constant opening of said file is absurd, even more considering that Optuna has its own SQLite.db file.


def objective(trial) -> float:
    """
    The function that runs a single model and evaluates it.
    """

    params = {
        "seed": trial.suggest_int("seed", 1, 10_000),
        "kfold_splits": trial.suggest_int("kfold_splits", 3, 5),
    }

    # XGB_hiperparams
    default_xg_params = {
        # "eta": 0.01,
        "eta": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 5),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "lambda": 1,
        "alpha": 1,
        # interaction_constraints
    }

    # Load the YAML file into a Python dictionary
    _, feature_metrics, auc = model_training_main(
        # data_path="data/resampled_data_SMOTE.csv",
        seed=params["seed"],
        default_xg_params=default_xg_params,
    )

    trial.set_user_attr("feature_metrics", feature_metrics.to_dict())

    return auc, feature_metrics["SHAP_abnormal"]["fenilalax"]


study_name: str = "no_oversample_adj_params_2"

study = optuna.create_study(
    study_name=study_name,
    storage="sqlite:///homa_studies.db",
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.NSGAIISampler(),
    load_if_exists=True,
)

study.optimize(objective, n_trials=100, n_jobs=-1)


# %%
optuna.visualization.plot_pareto_front(
    study, target_names=["AUC", "Phenylalanine Contribution"]
)

# study.get_trials()

# %%
