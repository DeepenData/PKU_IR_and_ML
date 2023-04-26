# coding: utf-8
import pandas as pd

csv_old = pd.read_csv("multi_label.csv")

csv_new = pd.read_excel("SLEIMPN_HOMA_PKU.xlsx")

newcol = [(col not in csv_old.columns) for col in csv_new.columns]
csv_new.columns[newcol]  # ViewIt

drop_cols = [
    "nomcom",
    "aleatorX",
    "Codigo",
    "num",
    "fechaex",
    "Estudio molecular",
    "Weight(scaled)",
    "Height(scaled)",
    "BMI(scaled)",
    "Waist circumference(scaled)",
]

csv_new_dropped = csv_new.drop(drop_cols, axis="columns")
csv_new_dropped.to_csv("multi_label.new.csv", index=False)

# Manually rename multi_label.new.csv to multi_label.csv, passing inspection
print(
    """
Created new file multi_label.new.csv

Run `rm multi_label.csv && mv multi_label.new.csv multi_label.csv` to finish
"""
)
