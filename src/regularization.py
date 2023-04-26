"""Reads from our relativly small dataset and synthesizes an enhanced one from the original data. This allows us to 
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

# Define the desired number of samples per predictor class
desired_samples = 100
predictor_col = "HOMA-IR alterado"

# %%
# Group samples by predictor class and count the number of samples in each class

grouped = df.groupby(predictor_col)
class_counts = grouped.size()

# Calculate the ratio of desired samples to current samples for each class
ratios = {cls: desired_samples / count for cls, count in class_counts.items()}

# Apply ADASYN to each class to generate the desired number of samples
X_df, y_df = df.drop(columns=predictor_col), df[predictor_col]

smotenc = ADASYN(
    sampling_strategy="minority",
    # categorical_features=X_df.select_dtypes("object").index,
)
synthetic_samples, synthetic_samples[predictor_col] = smotenc.fit_resample(X_df, y_df)
synthetic_samples = synthetic_samples[df.columns]

# Combine the original dataset and the synthetic dataset
synthetic_data.append(df)
df_synthetic = pd.concat(synthetic_data)

# Save the synthetic dataset to a CSV file
df_synthetic.to_csv("data_synthetic.csv", index=False)

# %%
