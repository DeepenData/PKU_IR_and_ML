import statistics
import numpy as np
import pandas as pd
import xgboost
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import shap
from sklearn.model_selection import StratifiedKFold
import optuna

import pandas as pd
from optuna.visualization._pareto_front import (
    _get_pareto_front_info,
    _make_scatter_object,
    _make_marker,
    _make_hovertext,
)
from typing import Sequence
from optuna.trial import FrozenTrial
from optuna.visualization._plotly_imports import go


def make_pareto_plot(study):
    


#study = study_ADASYN_sampling

    info = _get_pareto_front_info(study)

    n_targets: int = info.n_targets
    axis_order: Sequence[int] = info.axis_order
    include_dominated_trials: bool = True
    trials_with_values: Sequence[
        tuple[FrozenTrial, Sequence[float]]
    ] = info.non_best_trials_with_values
    hovertemplate: str = "%{text}<extra>Trial</extra>"
    infeasible: bool = False
    dominated_trials: bool = False


    def trials_df(trials_with_values, class_name):
        x = [values[axis_order[0]] for _, values in trials_with_values]
        y = [values[axis_order[1]] for _, values in trials_with_values]

        df = pd.DataFrame({"x": x, "y": y})
        df["class"] = class_name
        return df


    df_best = trials_df(info.best_trials_with_values, "best")
    df_nonbest = trials_df(info.non_best_trials_with_values, "nonbest")
    # df         = pd.concat([df_best, df_nonbest])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_nonbest["x"],
            y=df_nonbest["y"],
            mode="markers",
            marker=dict(
                size=8, opacity=0.5, symbol="circle", line=dict(width=1, color="black")
            ),
            name="Suboptimal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_best["x"],
            y=df_best["y"],
            mode="markers",
            marker=dict(
                size=8, opacity=0.5, symbol="square", line=dict(width=1, color="black")
            ),
            name="Pareto frontier",
        )
    )
    # Define the width and height of the plot
    plot_width = 900
    plot_height = 400
    axis_label_font_size = 19

    # Customize the plot
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Model performance (AUC)", font=dict(size=axis_label_font_size)
            ),
            range=[-0.05, 1.05],
        ),
        yaxis=dict(
            title=dict(
                text="Phe importance (Shapley value)", font=dict(size=axis_label_font_size)
            ),
            range=[-0.05, 0.85],
        ),
        font=dict(size=18),
        # plot_bgcolor='white',
        legend=dict(x=0.3, y=1.1, bgcolor="rgba(255, 255, 255, 0)", orientation="h"),
        width=plot_width,
        height=plot_height,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    x_max = max(df_nonbest["x"].max(), df_best["x"].max())

    fig.update_layout(
        # ... (previous settings)
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=0.8,
                x1=x_max,
                y0=0.0,
                y1=1,
                fillcolor="darkgreen",
                opacity=0.20,
                layer="below",
                line_width=0,
            )
        ],
        # ... (previous settings)
    )


    import plotly.io as pio

    dpi = 300  # Set the desired resolution (dots per inch)
    output_filename = f"pareto_{'study_name'}.png"
    pio.write_image(fig, output_filename, format="png", scale=dpi / 96)

    # Show the plot
    fig.show()
    



class generate_model:
    def __init__(
        self,
        predictor_col: str,
        data_path: str,
        xg_params: dict,
        kfold_splits: int,
        test_size=0.3,
        removed_features: list[str] = None,
        seed: float = None,
    ) -> None:
        if removed_features is None:
            removed_features = []
        dataset = pd.read_csv(data_path)

        # Variable encoding
        dataset["Género"] = (
            dataset["Género"].replace({"M": 0, "F": 1}).astype("category")
        )
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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_df, y_df, stratify=y_df, test_size=test_size, random_state=seed
        )
        # def xg_train(self, xg_params: dict, kfold_splits=5, seed=None):

        cv = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=seed)
        folds = list(cv.split(self.X_train, self.y_train))

        for train_idx, val_idx in folds:
            # Sub-empaquetado del train-set en formato de XGBoost
            dtrain = xgboost.DMatrix(
                self.X_train.iloc[train_idx, :],
                label=self.y_train.iloc[train_idx],
                enable_categorical=True,
            )
            dval = xgboost.DMatrix(
                self.X_train.iloc[val_idx, :],
                label=self.y_train.iloc[val_idx],
                enable_categorical=True,
            )

            self.model = xgboost.train(
                dtrain=dtrain,
                params=xg_params,
                evals=[(dtrain, "train"), (dval, "val")],
                num_boost_round=1000,
                verbose_eval=False,
                early_stopping_rounds=10,
            )

    def get_AUC_on_test_data(self):
        # def xg_test(model, X_test, y_test) -> float:
        testset = xgboost.DMatrix(
            self.X_test, label=self.y_test, enable_categorical=True
        )
        y_preds = self.model.predict(testset)

        return roc_auc_score(testset.get_label(), y_preds)

    def get_feature_metrics(self):
        # internal_feature_metrics = pd.DataFrame(
        #     {
        #         "Weight": self.model.get_score(importance_type="weight"),
        #         "Coverage": self.model.get_score(importance_type="cover"),
        #         "Gain": self.model.get_score(importance_type="gain"),
        #     }
        # ).sort_values(by="Gain", ascending=False)

        explainer = shap.TreeExplainer(self.model)

        # Extrae la explicacion SHAP en un DF
        EXPLAINATION = explainer(self.X_test).cohorts(
            self.y_test.replace({0: "Healty", 1: "Abnormal"}).tolist()
        )
        cohort_exps = list(EXPLAINATION.cohorts.values())

        exp_shap_abnormal = pd.DataFrame(
            cohort_exps[0].values, columns=cohort_exps[0].feature_names
        )  # .abs().mean()

        exp_shap_healty = pd.DataFrame(
            cohort_exps[1].values, columns=cohort_exps[1].feature_names
        )  # .abs().mean()

        feature_metrics = pd.concat(
            {
                "SHAP_healty": exp_shap_healty.abs().mean(),
                "SHAP_abnormal": exp_shap_abnormal.abs().mean(),
            },
            axis="columns",
        )

        # feature_metrics = (
        #     feature_metrics.join(internal_feature_metrics)
        #     .fillna(0)
        #     .sort_values(by="Gain", ascending=False)
        # )

        return feature_metrics


with open("params.yml", "r") as f:
    ext_params = yaml.load(f, Loader=yaml.FullLoader)

import random
def objective(trial, data ,tuned_params = None, finetunning: bool = False) -> float:
    """
    The function that runs a single model and evaluates it.
    """

    if  finetunning:
        seed = random.randint(1, 10_000)

        params={
                "objective":   "binary:logistic",
                "eval_metric": "logloss",
                'max_depth':   trial.suggest_int("max_depth", 2, 6, ),
                "eta":         trial.suggest_float("eta", 0.01, 0.3),
                "subsample":   trial.suggest_float("subsample", 0.5, 0.9),
                "lambda": trial.suggest_float("lambda", 0, 1),
                "alpha": trial.suggest_float("alpha",0,1),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight",0,2)
            }
    else:
        seed = trial.suggest_int("seed", 1, 10_000)
        params = tuned_params

    model_instance = generate_model(
        "HOMA-IR alterado",
        data, #
        removed_features=ext_params["feature_engineering"]["removed_features"],
        xg_params=params,
        kfold_splits=5,
        seed=seed,
    )

    return (
        model_instance.get_AUC_on_test_data(),
        model_instance.get_feature_metrics()["SHAP_abnormal"]["fenilalax"],
    )

def get_median_dict(my_study):
    
    dict_list = [d.params for d in my_study.best_trials]
    # Initialize an empty dictionary to store the median values
    median_dict = {}

    # Iterate over the keys in the first dictionary
    for key in dict_list[0].keys():
        # Extract the values for the current key from all dictionaries
        values = [d[key] for d in dict_list]

        # Calculate the median of the values
        median_value = statistics.median(values)

        # Add the key and the median value to the new dictionary
        median_dict[key] = median_value
        median_dict.update({
                "objective":   "binary:logistic",
                "eval_metric": "logloss"})
        
        
    key_to_convert = 'max_depth'

    # Check if the key exists in the dictionary
    if key_to_convert in median_dict:
        # Convert the value to an integer and assign it back to the key
        median_dict[key_to_convert] = int(median_dict[key_to_convert])

    return median_dict


def make_ranking_plots(sampling, hyperparams_study, data, CUTOFF_AUC, CUTOFF_PHE):
    


    #CUTOFF_AUC = 0.80
    #CUTOFF_PHE = 0.40

    #study = study_ADASYN_sampling
    bs_trials = pd.DataFrame()
    for t in sampling.get_trials():
        if t.values[0] > CUTOFF_AUC and t.values[1] > CUTOFF_PHE:
        
        
            bs_trials[t.number] = list(t.params.values()) + list(t.values)
        #and saves them to rows in a dataframe

    bs_trials.index = ["seed", "auc", "phe",]
    bs_trials       = bs_trials.T
    paretos = {}

    df_ranks = pd.DataFrame()

    for number in bs_trials['seed']:
        # instance_params = bs_trials.loc[number][
        #     ["max_depth", "eta", "subsample", "lambda", "alpha", "scale_pos_weight"]
        # ].to_dict()
        # instance_params["max_depth"] = int(instance_params["max_depth"])

        # instance_params["objective"] = "binary:logistic"
        # instance_params["eval_metric"] = "logloss"

        #print(instance_params)

        model_instance = generate_model(
            "HOMA-IR alterado",
            data, #"_best_artifact/optuna/resampled_data_ADASYN.csv",
            removed_features=ext_params["feature_engineering"]["removed_features"],
            xg_params=hyperparams_study
    ,
            kfold_splits=5,
            seed=int(number),
        )
        #
        print(model_instance.model.attributes())

        df_ranks[number] = model_instance.get_feature_metrics()["SHAP_abnormal"]

        print(
            "Phe_value : ",
            model_instance.get_feature_metrics()["SHAP_abnormal"]["fenilalax"],
            "\t",
            "Phe_ranking : ",
            model_instance.get_feature_metrics()["SHAP_abnormal"].rank(ascending=False)[
                "fenilalax"
            ],
        )

        print("\n")
        
    labels_relabel = {
        "Género": "gender",
        "aleator": "random",
        "Edad": "Age",
        "Peso": "Weight",
        "Estatura": "Height",
        "IMC": "BMI",
        "Circunferencia de cintura": "Waist circumference",
        "ATPII/AHA/IDF": "ATPII/AHA/IDF",
        "fenilalax": "Phenylalax",
        "glupromx": "Glupromx",
        "glummol": "Glummol",
        "insuprom": "Insuprom",
        "HOMA-IR alterado": "HOMA-IR altered",
        "HOMA2-IR": "HOMA2-IR",
        "HOMA2B(%)": "HOMA2B(%)",
        "HOMA2S%": "HOMA2S%",
        "quickix": "Quickix",
        "ohd3x": "ohd3x",
        "tirosinax": "Tyrosinax",
        "Alanina": "Alanine",
        "Aspartato": "Aspartate",
        "Glutamato": "Glutamate",
        "Leucina": "Leucine",
        "Ornitina": "Ornithine",
        "Prolina": "Proline",
        "Tirosina": "Tyrosine",
        "Carnitina libre": "Free Carnitine",
        "Propionilcarnitina": "Propionylcarnitine",
        "Isovalerilcarnitina": "Isovalerylcarnitine",
        "Tiglilcarnitina": "Tiglilcarnitine",
        "Me-Glutarilcarnitina": "Me-Glutarylcarnitine",
        "Decanoilcarnitina": "Decanoylcarnitine",
        "Tetradecanoilcarnitina": "Tetradecanoylcarnitine",
        "3-OH-Isovalerilcarnitina": "3-OH-Isovalerylcarnitine",
        "3-OH-Palmitoilcarnitina": "3-OH-Palmitoylcarnitine",
        "Linoleoilcarnitina": "Linoleoilcarnitine",
        "Arginina": "Arginine",
        "Citrulina": "Citrulline",
        "Glicina": "Glycine",
        "Metionina": "Methionine",
        "Fenilalanina": "Phenylalanine",
        "Succinilacetona": "Succinylacetone",
        "Valina": "Valine",
        "Acetilcarnitina": "Acetylcarnitine",
        "Butirilcarnitina": "Butyrylcarnitine",
        "Glutarilcarnitina": "Glutarylcarnitine",
        "Hexanoilcarnitina": "Hexanoylcarnitine",
        "Octanoilcarnitina": "Octanoylcarnitine",
        "Dodecanoilcarnitina": "Dodecanoylcarnitine",
        "Tetradecenoilcarnitina": "Tetradecenoylcarnitine",
        "Palmitoilcarnitina": "Palmitoylcarnitine",
        "Estearoilcarnitina": "Stearoylcarnitine",
        "3-OH-Linoleoilcarnitina": "3-OH-Linoleoylcarnitine",
        "PROTEINAProm_(G)": "Protein avg. (g)",
        "Proteina_natural": "Protein natural",
        "%_proteina_natural": "% Natural Protein",
        "Proteina_SP": "SP Protein",
        "SP_gr/kg": "SP (gr/kg)",
        "%_Proteina_SP": "% SP Protein",
        "GRASAProm(G)": "Fat avg. (g)",
        "CARBOHIDRATOProm_(G)": "CARBOHYDRATE avg. (g)",
        "ENERGIAProm_(KCAL)": "ENERGY avg. (KCAL)",
        "COLESTEROLProm_(MG)": "CHOLESTEROL avg. (mg)",
        "FENILALANINAProm_(G)": "PHENYLALANINE_avg (g)",
        "TIROSINAProm_(G)": "TYROSINE avg. (g)",
        "VITAMINA_B12Prom_(MCG)": "Vitamin B12 avg. (MCG)",
        "FOLATOProm_(MCG)": "Folate avg. (MCG)",
        "CALCIOProm_(MG)": "Calcium avg. (mg)",
        "COBREProm_(MG)": "Copper avg. (mg)",
        "HIERROProm_(MG)": "Iron avg. (mg)",
        "ZINCProm_(MG)": "Zinc avg. (mg)",
        "VITAMINA_Dprom": "Vitamin D avg.",
    }

    import plotly.express as px

    df_ranks_long = df_ranks.T
    order = (
        df_ranks_long.rank(axis="columns", ascending=False)
        .median(axis="rows")
        .sort_values()
        .index
    )

    # Create a grouped barplot using Plotly
    fig_barplot = px.bar(
        df_ranks,
        barmode="group",
        template="ggplot2",
    )
    fig_barplot.update_xaxes(
        categoryorder="array", categoryarray=order, labelalias=labels_relabel
    )


    fig_barplot.update_layout(
        xaxis_title="Feature in the dataset",
        yaxis_title="Shapley Value importance",
        title="Shapley values for abnormal samples",
        legend_title="Model",
    )

    # Show the plot
    fig_barplot.show()
    df_ranks_long["auc_score"] = bs_trials["auc"]

    fig = px.parallel_coordinates(
        df_ranks_long,
        # color='fenilalax',
        color="auc_score",
        dimensions=order.tolist()[:10] + ["auc_score"],
        labels=labels_relabel,
        title="Shapley values for abnormal samples - ADASYNT",
    )

    fig.update_layout(
        coloraxis_showscale=False,
        # title="Feature importance",
        xaxis_title="Feature in the dataset",
        yaxis_title="Shapley Value importance",
        # legend_title="Shapley values for abnormal samples",
    )

    fig.show()