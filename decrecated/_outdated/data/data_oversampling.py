# %%
from imblearn.over_sampling import ADASYN, SMOTE

import pandas as pd

dataset = pd.read_csv("data/multi_label.csv")
# print(f"Datos en set: {dataset.shape}")  # LOGGIN

# dataset = dataset.loc[ dataset["aleator"] != "Control" ]
# %% ---
# Renombra variables para mejores graficos
# (no lo aplica inmediatamente. Esta aqui por como luego lo ejecuto)

zip_rename = dict(
    zip(
        dataset.columns.tolist(),
        [
            "gender",
            "random",
            "Age",
            "Weight",
            "Height",
            "BMI",
            "Waist circumference",
            "ATPII/AHA/IDF",
            "Phenylalax",
            "Glupromx",
            "Insuprom",
            "HOMA-IR altered",
            "HOMA2-IR",
            "HOMA2B(%)",
            "HOMA2S%",
            "Quickix",
            "ohd3x",
            "Tyrosinax",
            "Alanine",
            "Aspartate",
            "Glutamate",
            "Leucine",
            "Ornithine",
            "Proline",
            "Tyrosine",
            "Free Carnitine",
            "Propionylcarnitine",
            "Isovalerylcarnitine",
            "Tiglilcarnitine",
            "Me-Glutarylcarnitine",
            "Decanoylcarnitine",
            "Tetradecanoylcarnitine",
            "3-OH-Isovalerylcarnitine",
            "3-OH-Palmitoylcarnitine",
            "Linoleoilcarnitine",
            "Arginine",
            "Citrulline",
            "Glycine",
            "Methionine",
            "Phenylalanine",
            "Succinylacetone",
            "Valine",
            "Acetylcarnitine",
            "Butyrylcarnitine",
            "Glutarylcarnitine",
            "Hexanoylcarnitine",
            "Octanoylcarnitine",
            "Dodecanoylcarnitine",
            "Tetradecenoylcarnitine",
            "Palmitoylcarnitine",
            "Stearoylcarnitine",
            "3-OH-Linoleoylcarnitine",
            "PROTEIN_Avg_(G)",
            "Protein_natural",
            "%_natural_protein",
            "SP_Protein",
            "SP_gr/kg",
            "%_SP_Protein",
            "FAT_avg(G)",
            "CARBOHYDRATE_avg_(G)",
            "ENERGY_avg_(KCAL)",
            "CHOLESTEROL_avg_(MG)",
            "PHENYLALANINE_avg_(G)",
            "TYROSINE_avg_(G)",
            "VITAMIN_B12_avg_(MCG)",
            "FOLATO_avg_(MCG)",
            "CALCIUM_avg_(MG)",
            "COPPER_avg_(MG)",
            "IRON_avg_(MG)",
            "ZINC_avg_(MG)",
            "VITAMIN_D_avg",
        ],
    )
)

# Convierte variables categoricas a numericas
dataset["Género"] = dataset["Género"].replace({"M": 0, "F": 1})
dataset["ATPII/AHA/IDF"] = dataset["ATPII/AHA/IDF"].replace({"no": 0, "si": 1})

# Remocion de los labels para crear nuevas series
label_pku = dataset.pop("aleator").replace({"Control": 0, "PKU 1": 0, "PKU 2": 1})
label_homa = dataset.pop("HOMA-IR alterado").replace({"No": 0, "Si": 1})

y_labels: pd.Series = label_homa

# %%

from imblearn.over_sampling import ADASYN, SMOTE

samplers = [
    SMOTE(random_state=0),
    ADASYN(random_state=0),
]

sampler = ADASYN(random_state=0)

resampled_data, resampled_labels = sampler.fit_resample(dataset, y_labels)
