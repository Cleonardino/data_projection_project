import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================
# 1. Chargement des données
# =========================
dataset_dir = "./dataset/Network datatset/csv/"
df = pd.read_csv(dataset_dir + "attack_1.csv")
df = df.rename(columns=lambda x: x.strip())

# =========================
# 2. Prétraitement
# =========================

# Conversion du champ Time en timestamp numérique
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df["Time"] = df["Time"].astype("int64") // 10**9  # secondes Unix

# Colonnes catégorielles à encoder
categorical_cols = [
    "mac_s", "mac_d",
    "ip_s", "ip_d",
    "proto", "flags",
    "modbus_fn", "modbus_response",
    "label"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# 3. Séparation X / y
# =========================

# label_n est la cible numérique (0 = normal, 1 = attaque)
X = df.drop(columns=["label", "label_n"])
y = df["label_n"]

# =========================
# 4. Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# 5. Entraînement RandomForest
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =========================
# 6. Évaluation
# =========================
y_pred = rf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# 7. Importance des features
# =========================
feature_importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nImportance des variables :")
print(feature_importances)
