import pandas as pd
from collections import Counter

dataset_dir = "./dataset/Network datatset/csv/"
files = [f"attack_{i}.csv" for i in range(1, 5)] + ["normal.csv"]

CHUNK_SIZE = 200_000
usecols=[" label", " ip_s"]
labels = ['normal', 'anomaly', 'scan', 'DoS', 'physical fault', 'MITM']
for file in files:
    print(f"\n===== {file} =====")

    total_rows = 0
    label_counter = Counter()
    other_labels = 0
    anomaly_ip_na = 0
    ip_na = 0
    anomaly_count = 0
    if file == "attack_4.csv":
        usecols=["label", "ip_s"]
    for chunk in pd.read_csv(
        dataset_dir + file,
        usecols=usecols,
        chunksize=CHUNK_SIZE
    ):
        # Nettoyage léger
        chunk.columns = chunk.columns.str.strip()
        chunk["label"] = chunk["label"].astype("category")

        total_rows += len(chunk)
        label_counter.update(chunk["label"].value_counts().to_dict())

        other_labels += ((chunk["label"] != "normal") & (chunk["label"] != "anomaly")).sum()
        anomaly_ip_na += ((chunk["label"] == "anomaly") & (chunk["ip_s"].isna())).sum()
        ip_na += chunk["ip_s"].isna().sum()
        anomaly_count += (chunk["label"] == "anomaly").sum()

        del chunk  # libération immédiate

    print(f"Shape: ({total_rows}, 2)")
    print(f"Labels uniques: {list(label_counter.keys())}")
    print(f"Lignes avec label != normal/anomaly: {other_labels}")
    print(f"Anomalies avec ip_s manquant: {anomaly_ip_na}")
    print(f"Lignes avec ip_s manquant: {ip_na}")
    print(f"Nombre total d'anomalies: {anomaly_count}")
