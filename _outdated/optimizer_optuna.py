# %% ---
import json
import subprocess

# Importamos las cosas de Optuna
import optuna
import yaml

# First of all, sorry. This is a mixture of using DVC for ease of tracking in several computes, plus Optuna for ease of
# running a lot of trials. As DVC expects a plain-text file for data, said data is saved in metrics.json.
# This constant opening of said file is absurd, even more considering that Optuna has its own SQLite.db file.


def objective(trial) -> float:
    """
    The function that runs a single model and evaluates it.
    """

    params = {
        "seed": trial.suggest_int("seed", 1, 10_000),
        # XGB_hiperparams
    }

    # Load the YAML file into a Python dictionary
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Modify the "B" value of the dictionary
    config["train"]["seed"] = params["seed"]

    # Write the modified dictionary back to the YAML file
    with open("params.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # Execute the "other.py" script
    subprocess.run(["python", "modelo_xg.py"], stdout=None, stderr=None)

    # Read the "metrics.json" file
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    # TODO: Set this as a function
    # model_metrics, _, _ = train_model(seed, dataset)

    trial.set_user_attr("shap_abnormal", metrics["shap_abnormal"])
    trial.set_user_attr("shap_healty", metrics["shap_healty"])

    print("\nSeed: ", params["seed"], "\t AUC: ", metrics["auc_val"], "\n")

    return metrics["auc_val"], metrics["shap_abnormal"]["Phenylalax"]


study_name: str = "oversample_ADASYNT_2"

study = optuna.create_study(
    study_name=study_name,
    storage="sqlite:///homa_studies.db",
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.TPESampler(),
    load_if_exists=True,
)

# Run it with 5,000 trials. May take a while.
study.optimize(objective, n_trials=300)

# %%
