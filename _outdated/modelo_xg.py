# %% ---
# TODO: Importar dataset
# Params
import yaml

with open("params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# fmt: off
RANDOM_STATE_SEED: int = params["train"]["seed"]  # Random state para replicabilidad
TESTSET_SPLIT: float = params["train"]["testset_split"]  # Splitting para el procentaje para testing
KFOLD_SPLITS: int = params["train"]["kfold_splits"]  # Random state para replicabilidad

CLASIFICADOR: str = "HOMA" # params["train"]["clasificador"]  # Random state para replicabilidad

# Crea un path para figuras
import os

if not os.path.isdir("fig"):
    os.mkdir("fig")

# Loggins con DVC Live
from dvclive import Live

live = Live("metrics")

import numpy as np
# Importa el dataset 
import pandas as pd

dataset = pd.read_csv("data/multi_label.csv")
# print(f"Datos en set: {dataset.shape}")  # LOGGIN

# dataset = dataset.loc[ dataset["aleator"] != "Control" ]
# %% ---
# Renombra variables para mejores graficos
# (no lo aplica inmediatamente. Esta aqui por como luego lo ejecuto)
zip_rename = dict(zip(
    dataset.columns.tolist(),
    ['gender','random', 'Age', 'Weight', 'Height', 'BMI',
        'Waist circumference', 'ATPII/AHA/IDF', 'Phenylalax', 'Glupromx', 'Insuprom',
        'HOMA-IR altered', 'HOMA2-IR', 'HOMA2B(%)', 'HOMA2S%', 'Quickix',
        'ohd3x', 'Tyrosinax', 'Alanine', 'Aspartate', 'Glutamate', 'Leucine',
        'Ornithine', 'Proline', 'Tyrosine', 'Free Carnitine',
        'Propionylcarnitine', 'Isovalerylcarnitine', 'Tiglilcarnitine',
        'Me-Glutarylcarnitine', 'Decanoylcarnitine', 'Tetradecanoylcarnitine',
        '3-OH-Isovalerylcarnitine', '3-OH-Palmitoylcarnitine',
        'Linoleoilcarnitine', 'Arginine', 'Citrulline', 'Glycine', 'Methionine',
        'Phenylalanine', 'Succinylacetone', 'Valine', 'Acetylcarnitine',
        'Butyrylcarnitine', 'Glutarylcarnitine', 'Hexanoylcarnitine',
        'Octanoylcarnitine', 'Dodecanoylcarnitine', 'Tetradecenoylcarnitine',
        'Palmitoylcarnitine', 'Stearoylcarnitine', '3-OH-Linoleoylcarnitine',
        'PROTEIN_Avg_(G)', 'Protein_natural',
        '%_natural_protein', 'SP_Protein', 'SP_gr/kg', '%_SP_Protein',
        'FAT_avg(G)', 'CARBOHYDRATE_avg_(G)', 'ENERGY_avg_(KCAL)',
        'CHOLESTEROL_avg_(MG)', 'PHENYLALANINE_avg_(G)', 'TYROSINE_avg_(G)',
        'VITAMIN_B12_avg_(MCG)', 'FOLATO_avg_(MCG)', 'CALCIUM_avg_(MG)',
        'COPPER_avg_(MG)', 'IRON_avg_(MG)', 'ZINC_avg_(MG)', 'VITAMIN_D_avg']
))

# Convierte variables categoricas a numericas
dataset["Género"] = dataset["Género"].replace({"M":0,"F":1})
dataset["ATPII/AHA/IDF"] = dataset["ATPII/AHA/IDF"].replace({"no":0,"si":1})

# Remocion de los labels para crear nuevas series
label_pku = dataset.pop("aleator").replace({"Control": 0, "PKU 1": 0, "PKU 2": 1})
label_homa = dataset.pop("HOMA-IR alterado").replace({"No": 0, "Si": 1})

y_labels: pd.Series = label_homa

# Usa una lista de features definidas en params para dropear leakers
leakers: list[str] = params["feature_engineering"]["removed_features"]
if not leakers: leakers = []
dataset.drop(leakers,axis="columns",inplace=True)

# %% Resampling
resample : bool = False

if resample:
    from imblearn.over_sampling import ADASYN, SMOTE

    sampler = ADASYN(random_state=params["train"]["seed"])

    dataset, y_labels = sampler.fit_resample(dataset, y_labels)

# %% -- CONT

# Aplica el rename a EN
dataset.rename( columns=zip_rename, inplace=True )

# create a train/test split
import xgboost
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset, 
    y_labels, 
    stratify=y_labels,
    test_size=TESTSET_SPLIT, 
    random_state=RANDOM_STATE_SEED, 
)

# %% ---
# TODO: Usar XGBoost con Cross-K-Validation y ROC
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=KFOLD_SPLITS, 
    #n_repeats=10, 
    shuffle=True,
    random_state=RANDOM_STATE_SEED
)

folds = list(cv.split(X_train, y_train))
#folds = list(cv.split(X_train, y_train))

# Parametros de XGBoost
params["xgboost"] = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
}

# Recoleccion de metricas de XGBoost
metrics = ['auc', 'fpr', 'tpr', 'thresholds']
results = {
    'train': {m:[] for m in metrics},
    'val'  : {m:[] for m in metrics},
    'test' : {m:[] for m in metrics}
}

# Empacado del test-set en formato de XGBoost
dtest = xgboost.DMatrix(X_test, label=y_test)

# Importa metricas de SKLearn
from sklearn.metrics import roc_auc_score, roc_curve

# Loop K-fold
for train, test in folds:

    # Sub-empaquetado del train-set en formato de XGBoost
    dtrain = xgboost.DMatrix(X_train.iloc[train,:], label=y_train.iloc[train])
    dval   = xgboost.DMatrix(X_train.iloc[test,:], label=y_train.iloc[test])

    model  = xgboost.train(
        dtrain                = dtrain,
        params                = params, 
        evals                 = [(dtrain, 'train'), (dval, 'val')],
        num_boost_round       = 1000,
        verbose_eval          = False,
        early_stopping_rounds = 10,
    )

    # Sets de k-fold train, k-fold eval, y test-set
    sets = [dtrain, dval, dtest]

    for i,ds in enumerate(results.keys()):
        y_preds              = model.predict(sets[i])
        labels               = sets[i].get_label()
        fpr, tpr, thresholds = roc_curve(labels, y_preds)
        results[ds]['fpr'].append(fpr)
        results[ds]['tpr'].append(tpr)
        results[ds]['thresholds'].append(thresholds)
        results[ds]['auc'].append(roc_auc_score(labels, y_preds))

kind = 'val'

fpr_mean    = np.linspace(0, 1, 100)
interp_tprs = []

for i in range(len(results[kind]['fpr'])):
    fpr           = results[kind]['fpr'][i]
    tpr           = results[kind]['tpr'][i]
    interp_tpr    = np.interp(fpr_mean, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)
tpr_mean     = np.mean(interp_tprs, axis=0)
tpr_mean[-1] = 1.0
tpr_std      = 2*np.std(interp_tprs, axis=0)
tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
tpr_lower    = tpr_mean-tpr_std
auc          = np.mean(results[kind]['auc'])


# %% ---
# TODO: Usar SHAP para la explicacion
# Predicciones con SHAP
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(dataset)

# %%
# FIGURAS! 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

N_FEATURES : int = 6

fig = plt.figure(
    constrained_layout=True,
)

gs = GridSpec(2, 6, 
    figure=fig,
    wspace=1.5,
    height_ratios=[1,2]
)

ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:6])

# Explicaciones tradicionales usando parametros de XGBoost

# BOOST tiene un parametro para especificar un axe de la figura, asi
# que puedo usar eso directamente. SHAP no, asi que ahi uso el "current_axe", 
# lo que es menos lejible y posiblemente una mejora a sugerir para SHAP

xgboost.plot_importance(
    model, importance_type="weight",
    ax=ax1,
    max_num_features=N_FEATURES,
    show_values=False,
    title='Weight',
    ylabel="",
    xlabel="Feature apparance",
    )

xgboost.plot_importance(
    model, importance_type="gain",
    ax=ax2,
    max_num_features=N_FEATURES,
    show_values=False,
    title='Gain',
    ylabel="",
    xlabel="F score",
    )

xgboost.plot_importance(
    model, importance_type="cover",
    ax=ax3,
    max_num_features=N_FEATURES,
    show_values=False,
    title='Cover',
    ylabel="",
    xlabel="F score",
    )

# Explicaciones complicadas usando SHAP

# Como SHAP no permite definir en que matplotliv.axe va la cosa, ahi que usar el
# "current_axe" e intentar mantener todo limpio y ordenado. 

ax4 = fig.add_subplot(gs[1, 0:3])
shap.plots.bar(
    (
        explainer(dataset)
        .cohorts(
            y_labels.replace(
                {0 : "Healty", 1:"Abnormal"})
            .tolist()
        )
    ),
    max_display=N_FEATURES,
    show=False,
)
plt.title("Global feature importance")

ax5 = fig.add_subplot(gs[1, 3:6])
shap.plots.beeswarm(
    explainer(
        dataset #.loc[ y_labels == 1 ]
    ),
    max_display=N_FEATURES,
    show=False,
)


# Copiado de Stack-Overflow para remover los labels
ax5.set(yticklabels=[])  # remove the tick labels
ax5.tick_params(left=False)  # remove the ticks

# Labels de figuras
ax1.set_title("A", fontfamily='sans-serif', loc='left', fontsize='x-large', weight='heavy')
ax2.set_title("B", fontfamily='sans-serif', loc='left', fontsize='x-large', weight='heavy')
ax3.set_title("C", fontfamily='sans-serif', loc='left', fontsize='x-large', weight='heavy')
ax4.set_title("D", fontfamily='sans-serif', loc='left', fontsize='x-large', weight='heavy')
ax5.set_title("E", fontfamily='sans-serif', loc='left', fontsize='x-large', weight='heavy')

fig.subplots_adjust(
    top=0.9,
    hspace=0.3,
    left=0.18,
    right=0.85
)

plt.xlabel("SHAP value")
plt.title(f"Local explanation summary")

plt.suptitle(f"{CLASIFICADOR.upper()} - Score: {auc:.3f} - Seed: {params['train']['seed']}")

if False: # This, to avoid saving 5,000 figs
    fig.set_size_inches(20, 12)
    fig.savefig(f"fig/panel_{RANDOM_STATE_SEED}.svg", dpi=100)

# plt.tight_layout()

print(f"AUC: {auc}")

# %%

# Extrae la explicacion SHAP en un DF
EXPLAINATION = explainer(dataset).cohorts(y_labels.replace({0 : "Healty", 1:"Abnormal"}).tolist())
cohort_exps = list(EXPLAINATION.cohorts.values())

exp_shap_abnormal = pd.DataFrame( 
    cohort_exps[0].values ,  
    columns = cohort_exps[0].feature_names
) #.abs().mean()

exp_shap_healty = pd.DataFrame( 
    cohort_exps[1].values ,  
    columns = cohort_exps[1].feature_names
) #.abs().mean()

SHAP_val = pd.concat(
    [exp_shap_healty, exp_shap_abnormal],
    ignore_index=True,
)

dataset2 = dataset
dataset2["LABELS"] = label_homa

MERGED = pd.concat(
    [dataset2, SHAP_val],
    axis = "columns", 
    keys = ["Measurements","SHAP"]
)

MERGED.to_parquet(path="datasett3.pkl")
