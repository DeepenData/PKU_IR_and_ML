# %% ---
import json
import yaml
import subprocess

# Importamos las cosas de Optuna
import optuna


def objective(trial) -> float:
    """
    Cosa que devuelve Optuna

    Se llama `objective` para hacerlo ideomatico. Usamos los modulos `trial` para
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

    trial.set_user_attr("shap_abnormal", metrics["shap_abnormal"])
    trial.set_user_attr("shap_healty", metrics["shap_healty"])

    print("\nSeed: ", params["seed"], "\t AUC: ", metrics["auc_val"], "\n")

    # avg_phe_rank: float = (
    #     metrics["shap_abnormal"]["Phenylalax"] + metrics["shap_healty"]["Phenylalax"]
    # ) / 2.0

    # Return the "AOC" value
    return metrics["auc_val"], metrics["shap_abnormal"]["Phenylalax"], 0


# Creamos un estudio, que al final es como este espacio hiperparametrico que usa optuna

study_name: str = "TRIALS_4"

study = optuna.create_study(
    study_name=study_name,
    storage="sqlite:///homa_studies.db",
    directions=["maximize", "maximize", "minimize"],
    sampler=optuna.samplers.TPESampler(),
    load_if_exists=True,
)

# Corremos el estudio/experimento, con 100 intentos
study.optimize(objective, n_trials=20)

# %% ---
# Y sacamos los mejores parametros (podemos guardar esto)

# Load the YAML file into a Python dictionary
# with open("params.yaml", "r") as f:
#     config = yaml.safe_load(f)

# # Modify the "B" value of the dictionary
# config["train"]["seed"] = study.best_params["seed"]
#
# print(f"{config['train']['seed'] = }")
#
# # Write the modified dictionary back to the YAML file
# with open("params.yaml", "w") as f:
#     yaml.safe_dump(config, f)
#
# # Execute the "other.py" script
# subprocess.run(["python", "modelo_xg_oldie.py"])

# TODO: Funcion multi-objetivo, premiando las que tienen el ranking de Phe
#   Premiar Phe y IMC

# TODO: Guardar las metricas internas y externas de los runs en un df.
#   Luego, usar este df para buscar patrones (tSNE?)

# TODO: Runs
#   - Sin BMI
#   - Sin BMI ni Circunferencia de Cintura
#   - Solo Prot_avg_(g)
#   - Sin informacion de dieta (Prot_avg_(g))
#   -

optuna.visualization.plot_pareto_front(
    study, target_names=["AUC", "Abnormal_Phenylalax"]
)
# %% ---
