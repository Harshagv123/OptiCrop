# OptiCrop: Smart Agricultural Production Optimization Engine

## Project Overview

OptiCrop is a Machine Learning based Agricultural Recommendation System developed to help farmers identify the most suitable crop based on soil nutrients and environmental conditions.

The system analyzes:

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature
- Humidity
- pH Level
- Rainfall

and recommends the best crop for maximum productivity.

---

## Objectives

- Improve agricultural productivity.
- Provide intelligent crop recommendations.
- Support sustainable farming practices.
- Assist researchers and policymakers in agricultural planning.

---

## Technologies Used

### Programming Language

- Python

### Machine Learning

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- K-Means Clustering

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Flask
- Pickle

### Frontend

- HTML
- CSS
- Bootstrap

---

## Dataset

Crop Recommendation Dataset containing:

- 2200 Records
- 22 Crop Categories

Features:

- Nitrogen
- Phosphorous
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

Target:

- Crop Label

---

## Project Workflow

1. Download Dataset
2. Import Libraries
3. Read Dataset
4. Univariate Analysis
5. Bivariate Analysis
6. Multivariate Analysis
7. Check Null Values
8. Handle Outliers
9. Extract Seasonal Crops
10. Train-Test Split
11. K-Means Clustering
12. Logistic Regression
13. Model Evaluation
14. Save Best Model
15. Crop Prediction
16. Build HTML Pages
17. Build Flask Backend
18. Run Application

---

## Project Structure

```text
OptiCrop/
│
├── data/
│   └── Crop_recommendation.csv
│
├── models/
│   ├── crop_model.pkl
│   ├── label_encoder.pkl
│   ├── kmeans_model.pkl
│   └── model_name.txt
│
├── notebooks/
│   └── OptiCrop_Analysis.ipynb
│
├── static/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore