# Risk of Developing Insulin Resistance in Adult Subjects with Phenylketonuria: Machine Learning Model Reveals an Association with Phenylalanine Concentrations in Dried Blood Spots

This repository contains the code used in the research paper titled "Risk of Developing Insulin Resistance in Adult Subjects with Phenylketonuria: Machine Learning Model Reveals an Association with Phenylalanine Concentrations in Dried Blood Spots" by María Jesús Leal-Witt, Eugenia Rojas-Agurto, Manuel Muñoz-González, Felipe Peñaloza, Carolina Arias, Karen Fuenzalida, Daniel Bunout, Verónica Cornejo, and Alejandro Acevedo, published in Metabolites in 2023 ([source](https://www.mdpi.com/2218-1989/13/6/677)).

## Research Overview

Our research aims to investigate the risk of insulin resistance (IR) in adult subjects with Phenylketonuria (PKU), a recessive autosomal disease characterized by the accumulation of phenylalanine (Phe) in the plasma. High concentrations of Phe are highly neurotoxic, leading to irreversible damage to the central nervous system if not diagnosed and treated early. Conventional treatment of PKU involves a strict diet to limit the intake of Phe, which is found in high concentrations in many protein-rich foods.

Poor adherence to this diet can result in numerous health complications, one of which is insulin resistance. In our study, we explore how phenylalanine concentrations (PheCs) relate to insulin resistance using machine learning (ML) and derive potential biomarkers. Our machine learning model was trained to predict abnormal homeostatic measurement assessments (HOMA-IRs) using a panel of metabolites measured from dried blood spots (DBSs). The features’ importance ranked PheCs as the second most important feature after BMI for predicting abnormal HOMA-IRs ([source](https://www.mdpi.com/2218-1989/13/6/677)).

## Repository Structure

The repository is structured as follows:

- `data/`: This directory contains the data files used in the research.