"""Data balancing via SMOTE

Reads from our dataset and synthesizes an enhanced dataset from the original data. This allows to 
publish the synthetic data without compromising the confidentiality of the clinical data, and to account for the 
disparity of the classes in the original samples. 
"""

# %%
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE

# Importa el dataset
dataset = pd.read_csv("data/data.csv")

# Define the desired number of samples per predictor class
predictor_col = "HOMA-IR alterado"


def resample(sampler_class):
    # Variable encoding
    dataset["Género"] = dataset["Género"].replace({"M": 0, "F": 1})
    dataset["ATPII/AHA/IDF"] = dataset["ATPII/AHA/IDF"].replace({"no": 0, "si": 1})
    dataset["aleator"] = dataset["aleator"].replace(
        {"Control": 0, "PKU 1": 1, "PKU 2": 2}
    )

    y_df = dataset[predictor_col].astype("category")
    X_df = dataset.drop(columns=predictor_col)

    if sampler_class == "SMOTE":
        X_resampled, y_resampled = SMOTE(k_neighbors=7).fit_resample(X_df, y_df)
    elif sampler_class == "ADASYNT":
        X_resampled, y_resampled = ADASYN(n_neighbors=7).fit_resample(X_df, y_df)

    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

    # Combine the original dataset and the synthetic dataset
    resampled_data["Género"] = resampled_data["Género"].replace({0: "M", 1: "F"})
    resampled_data["ATPII/AHA/IDF"] = resampled_data["ATPII/AHA/IDF"].replace(
        {0: "no", 1: "si"}
    )
    resampled_data["aleator"] = resampled_data["aleator"].replace(
        {0: "Control", 1: "PKU 1", 2: "PKU 2"}
    )

    return resampled_data


# Save the synthetic dataset to a CSV file
for sampler in ["SMOTE", "ADASYNT"]:
    resampled_data = resample(sampler)
    resampled_data.to_csv(f"data/resampled_data_{sampler}.csv", index=False)

# %%
