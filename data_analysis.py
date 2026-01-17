import pandas as pd
from collections import Counter

dataset_dir = "./dataset/"
files = [f"Network datatset/csv/attack_{i}.csv" for i in range(1, 5)] + ["Network datatset/csv/normal.csv"]
files += [f"Physical dataset/phy_att_{i}.csv" for i in range(1, 5)] + ["Physical dataset/phy_norm.csv"]

files_virgule = (
    [f"Network datatset/csv/attack_{i}.csv" for i in range(1, 5)]
    + ["Network datatset/csv/normal.csv"]
    + ["Physical dataset/phy_att_4.csv"]
)

CHUNK_SIZE = 200_000
labels = ['normal', 'nomal',  'DoS', 'physical', 'physical fault', 'MITM', 'scan', 'anomaly',]
different_labels = set()

for file in files:
    print(f"\n===== {file} =====")

    total_rows = 0
    label_counter = Counter()
    anomaly_ip_na = 0
    ip_na = 0
    label_count = {label: 0 for label in labels}

    # 🔹 Définition propre de usecols
    if "Physical dataset" in file:
        usecols = ["Label"]
    elif file == "Network datatset/csv/attack_4.csv" or file == "Network datatset/csv/normal.csv":
        usecols = ["label", "ip_s"]
    else:
        usecols = [" label", " ip_s"]

    # 🔹 Séparateur
    sep = "," if file in files_virgule else r"\s+"

    # 🔹 ENCODAGE (FIX PRINCIPAL)
    encoding = "latin1" if file == "Physical dataset/phy_att_4.csv" else "utf-16" if file.startswith("Physical dataset") else "utf-8"

    for chunk in pd.read_csv(
        dataset_dir + file,
        usecols=usecols,
        sep=sep,
        encoding=encoding,
        engine="python",
        chunksize=CHUNK_SIZE
    ):
        if "Physical dataset" in file:
            chunk = chunk.rename(columns={"Label": "label"})
        chunk.columns = chunk.columns.str.strip()
        chunk["label"] = chunk["label"].astype("category")

        total_rows += len(chunk)
        different_labels.update(chunk["label"].unique())
        label_counter.update(chunk["label"].value_counts().to_dict())

        for label in labels:
            label_count[label] += (chunk["label"] == label).sum()

        if "ip_s" in chunk.columns:
            anomaly_ip_na += ((chunk["label"] == "anomaly") & (chunk["ip_s"].isna())).sum()
            ip_na += chunk["ip_s"].isna().sum()

    print(f"Rows: {total_rows}")
    print(f"Labels uniques: {list(label_counter.keys())}")

    for label, count in label_count.items():
        if count > 0:
            print(f"Nombre total du label {label}: {count}, Pourcentage: {count / total_rows * 100:.2f}%")

    if "ip_s" in usecols:
        print(f"Anomalies avec ip_s manquant: {anomaly_ip_na}")
        print(f"Lignes avec ip_s manquant: {ip_na}")

print(f"\nTous les labels differents: {different_labels}")
