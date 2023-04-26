"""Reads from our dataset and synthesizes an enhanced dataset from the original data. This allows to 
publish the synthetic data without compromising the confidentiality of the clinical data, and to account for the 
disparity of the classes in the original samples. 

Author: 
2023
"""
# %%
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTENC


# %% -- Load data
df = pd.read_csv("data/data.csv")

[df[col].astype("category") for col in df.select_dtypes("object").columns]


# Define the desired number of samples per predictor class
desired_samples = 100
predictor_col = "HOMA-IR alterado"

# Apply ADASYN to each class to generate the desired number of samples
X_df, y_df = df.drop(columns=predictor_col), df[predictor_col]

sampler = ADASYN(
    sampling_strategy="minority",
)

synthetic_samples, synthetic_samples[predictor_col] = sampler.fit_resample(X_df, y_df)
synthetic_samples = synthetic_samples[df.columns]

# Combine the original dataset and the synthetic dataset
synthetic_data.append(df)
df_synthetic = pd.concat(synthetic_data)

# Save the synthetic dataset to a CSV file
df_synthetic.to_csv("data_synthetic.csv", index=False)

# %%
# TODO: Regularization functions
