import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Load Dataset

df = pd.read_csv("data/Crop_recommendation.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Check Null Values

print("\nNull Values")
print(df.isnull().sum())

# Create Folders

os.makedirs("models", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# Univariate Analysis

plt.figure(figsize=(12,6))
df["label"].value_counts().plot(kind="bar")
plt.title("Crop Distribution")
plt.tight_layout()
plt.savefig("static/images/crop_distribution.png")
plt.close()

# Bivariate Analysis

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="temperature",
    y="rainfall"
)
plt.title("Temperature vs Rainfall")
plt.tight_layout()
plt.savefig("static/images/temperature_rainfall.png")
plt.close()

# Multivariate Analysis

plt.figure(figsize=(8,6))
sns.heatmap(
    df.drop("label", axis=1).corr(),
    annot=True,
    cmap="YlGnBu"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/images/correlation_heatmap.png")
plt.close()

print("Analysis Graphs Saved")

# Features and Labels

X = df.drop("label", axis=1)
y = df["label"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# KNN

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn_acc = accuracy_score(
    y_test,
    knn.predict(X_test)
)

# Logistic Regression

lr = LogisticRegression(max_iter=5000)
lr.fit(X_train, y_train)

lr_acc = accuracy_score(
    y_test,
    lr.predict(X_test)
)

# Decision Tree

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_acc = accuracy_score(
    y_test,
    dt.predict(X_test)
)

# Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_acc = accuracy_score(
    y_test,
    rf.predict(X_test)
)

# K-Means Clustering

kmeans = KMeans(
    n_clusters=22,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

print("K-Means Clustering Completed")

# Model Comparison

results = pd.DataFrame({
    "Model": [
        "KNN",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        knn_acc,
        lr_acc,
        dt_acc,
        rf_acc
    ]
})

print("\nModel Comparison")
print(results)

best_model_name = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

models = {
    "KNN": knn,
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf
}

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# Save Models

pickle.dump(
    best_model,
    open("models/crop_model.pkl", "wb")
)

pickle.dump(
    encoder,
    open("models/label_encoder.pkl", "wb")
)

pickle.dump(
    kmeans,
    open("models/kmeans_model.pkl", "wb")
)

with open("models/model_name.txt", "w") as f:
    f.write(best_model_name)

print("Models Saved Successfully")

# Sample Prediction

sample = [[90, 42, 43, 20.8, 82, 6.5, 202]]

prediction = best_model.predict(sample)

crop = encoder.inverse_transform(prediction)

print("\nRecommended Crop:", crop[0])