import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import os
import json

dataset_dir = "./dataset/"
output_file = "analysis_results.txt"

# Configuration des fichiers
files_network = [f"Network datatset/csv/attack_{i}.csv" for i in range(1, 5)] + ["Network datatset/csv/normal.csv"]
files_physical = [f"Physical dataset/phy_att_{i}.csv" for i in range(1, 5)] + ["Physical dataset/phy_norm.csv"]
files_virgule = [f"Network datatset/csv/attack_{i}.csv" for i in range(1, 5)] + ["Network datatset/csv/normal.csv"] + ["Physical dataset/phy_att_4.csv"]

CHUNK_SIZE = 200_000
labels = ['normal', 'nomal', 'DoS', 'physical', 'physical fault', 'MITM', 'scan', 'anomaly']

def write_section(f, title, content=""):
    """Écrire une section formatée dans le fichier"""
    f.write(f"\n{'='*80}\n")
    f.write(f"{title.center(80)}\n")
    f.write(f"{'='*80}\n")
    if content:
        f.write(f"{content}\n")

def analyze_file_complete(file, dataset_type):
    """
    Analyse complète d'un fichier en un seul passage
    Optimisé pour minimiser les lectures multiples
    """
    print(f"  → Analyse en cours...")

    results = {
        'file': file,
        'type': dataset_type,
        'total_rows': 0,
        'label_distribution': Counter(),
        'missing_values': {},
        'columns': None,
        # Temporal
        'has_time': False,
        'time_range': None,
        'hourly_distribution': Counter(),
        # Network specific
        'unique_ips_src': set(),
        'unique_ips_dst': set(),
        'protocol_distribution': Counter(),
        'top_talkers': Counter(),
        'protocols_by_label': {},
        # Physical specific
        'active_sensors_by_label': {},
        # Quality
        'missing_ip_count': 0
    }

    # Configuration
    if file in files_virgule:
        sep = ","
        on_bad_lines = 'skip'
    elif "Physical dataset" in file:
        sep = "\t"  # Tabulation pour Physical dataset
        on_bad_lines = 'warn'
    else:
        sep = r"\s+"
        on_bad_lines = 'skip'

    encoding = "latin1" if file == "Physical dataset/phy_att_4.csv" else "utf-16" if file.startswith("Physical dataset") else "utf-8"

    chunk_count = 0
    try:
        for chunk in pd.read_csv(
            dataset_dir + file,
            sep=sep,
            encoding=encoding,
            engine="python",
            chunksize=CHUNK_SIZE,
            on_bad_lines=on_bad_lines
        ):
            chunk_count += 1

            # Normalisation
            if "Physical dataset" in file:
                chunk = chunk.rename(columns={"Label": "label"})
            chunk.columns = chunk.columns.str.strip()

            if results['columns'] is None:
                results['columns'] = chunk.columns.tolist()
                results['missing_values'] = {col: 0 for col in results['columns']}

            results['total_rows'] += len(chunk)

            if chunk_count % 5 == 0:
                print(f"    • Chunk {chunk_count} traité ({results['total_rows']:,} lignes)")

            # Labels
            if 'label' in chunk.columns:
                chunk["label"] = chunk["label"].astype("category")
                results['label_distribution'].update(chunk["label"].value_counts().to_dict())

            # Valeurs manquantes
            for col in chunk.columns:
                results['missing_values'][col] += chunk[col].isna().sum()

            # Analyse temporelle
            for col in ['Time', 'time', 'timestamp']:
                if col in chunk.columns:
                    if not results['has_time']:
                        results['has_time'] = True
                    try:
                        # Spécifier le format pour éviter le warning
                        chunk[col] = pd.to_datetime(chunk[col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                        if chunk[col].isna().all():
                            # Si le format ci-dessus ne fonctionne pas, essayer un format automatique
                            chunk[col] = pd.to_datetime(chunk[col], errors='coerce', dayfirst=True)
                        hours = chunk[col].dt.hour.dropna()
                        results['hourly_distribution'].update(hours.value_counts().to_dict())

                        valid_times = chunk[col].dropna()
                        if len(valid_times) > 0:
                            if results['time_range'] is None:
                                results['time_range'] = [valid_times.min(), valid_times.max()]
                            else:
                                results['time_range'][0] = min(results['time_range'][0], valid_times.min())
                                results['time_range'][1] = max(results['time_range'][1], valid_times.max())
                    except:
                        pass
                    break

            # Network specific
            if dataset_type == 'network':
                if 'ip_s' in chunk.columns:
                    results['unique_ips_src'].update(chunk['ip_s'].dropna().unique())
                    results['top_talkers'].update(chunk['ip_s'].value_counts().to_dict())
                    results['missing_ip_count'] += chunk['ip_s'].isna().sum()

                if 'ip_d' in chunk.columns:
                    results['unique_ips_dst'].update(chunk['ip_d'].dropna().unique())

                if 'proto' in chunk.columns:
                    results['protocol_distribution'].update(chunk['proto'].value_counts().to_dict())

                    # Protocoles par label
                    if 'label' in chunk.columns:
                        top_labels = chunk['label'].value_counts().head(3).index
                        for label in top_labels:
                            if label not in results['protocols_by_label']:
                                results['protocols_by_label'][label] = Counter()
                            label_data = chunk[chunk['label'] == label]
                            results['protocols_by_label'][label].update(label_data['proto'].value_counts().to_dict())

            # Physical specific
            elif dataset_type == 'physical':
                if 'label' in chunk.columns:
                    sensor_cols = [col for col in chunk.columns if col not in ['Time', 'label', 'Label_n']]

                    # Compter capteurs actifs par label
                    for label in chunk['label'].unique():
                        if label not in results['active_sensors_by_label']:
                            results['active_sensors_by_label'][label] = set()

                        label_data = chunk[chunk['label'] == label]
                        for col in sensor_cols[:20]:
                            # Vérifier si le capteur est actif dans ce chunk
                            try:
                                if pd.api.types.is_bool_dtype(label_data[col]):
                                    if label_data[col].any():
                                        results['active_sensors_by_label'][label].add(col)
                                else:
                                    if (label_data[col] != 0).any():
                                        results['active_sensors_by_label'][label].add(col)
                            except:
                                pass

    except Exception as e:
        print(f"Erreur lors de l'analyse: {str(e)}")

    print(f"Analyse terminée : {results['total_rows']:,} lignes analysées")

    # Convertir sets en counts
    if dataset_type == 'network':
        results['unique_ips_src'] = len(results['unique_ips_src'])
        results['unique_ips_dst'] = len(results['unique_ips_dst'])
    elif dataset_type == 'physical':
        # Convertir les sets de capteurs en counts
        for label in results['active_sensors_by_label']:
            results['active_sensors_by_label'][label] = len(results['active_sensors_by_label'][label])

    return results

def write_analysis_to_file():
    """Fonction principale optimisée"""
    print("\n" + "="*80)
    print("DÉBUT DE L'ANALYSE DES DATASETS".center(80))
    print("="*80 + "\n")

    all_results = {
        'network': [],
        'physical': []
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        write_section(f, "ANALYSE COMPLÈTE DES DATASETS - NETWORK & PHYSICAL")
        f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Analyse Network Dataset
        write_section(f, "PARTIE 1: NETWORK DATASET")
        print(f"\nAnalyse des fichiers NETWORK ({len(files_network)} fichiers)")

        for idx, file in enumerate(files_network, 1):
            print(f"\n[{idx}/{len(files_network)}] {file}")
            write_section(f, f"Fichier: {file}")

            results = analyze_file_complete(file, 'network')
            all_results['network'].append(results)

            # Écrire les résultats
            f.write("\n--- STATISTIQUES DE BASE ---\n")
            f.write(f"Nombre total de lignes: {results['total_rows']:,}\n")
            f.write(f"Nombre de colonnes: {len(results['columns'])}\n\n")

            f.write("Distribution des labels:\n")
            for label, count in sorted(results['label_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / results['total_rows']) * 100 if results['total_rows'] > 0 else 0
                f.write(f"  {label}: {count:,} ({percentage:.2f}%)\n")

            f.write("\nValeurs manquantes (principales):\n")
            for col, count in sorted(results['missing_values'].items(), key=lambda x: x[1], reverse=True)[:5]:
                if count > 0:
                    percentage = (count / results['total_rows']) * 100
                    f.write(f"  {col}: {count:,} ({percentage:.2f}%)\n")

            f.write("\n--- ANALYSE TEMPORELLE ---\n")
            if results['has_time'] and results['time_range']:
                f.write(f"Période: {results['time_range'][0]} à {results['time_range'][1]}\n")
                f.write("\nDistribution horaire (top 5):\n")
                for hour, count in sorted(results['hourly_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    f.write(f"  Heure {hour}: {count:,} événements\n")
            else:
                f.write("Pas de données temporelles\n")

            f.write("\n--- ANALYSE TOPOLOGIQUE ---\n")
            f.write(f"IPs sources uniques: {results['unique_ips_src']:,}\n")
            f.write(f"IPs destinations uniques: {results['unique_ips_dst']:,}\n")
            f.write(f"IPs sources manquantes: {results['missing_ip_count']:,}\n")

            f.write("\nDistribution des protocoles:\n")
            for proto, count in sorted(results['protocol_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"  {proto}: {count:,}\n")

            f.write("\nTop 10 des IPs les plus actives:\n")
            for ip, count in results['top_talkers'].most_common(10):
                f.write(f"  {ip}: {count:,} paquets\n")

            f.write("\n--- PROTOCOLES PAR LABEL (top 3 labels) ---\n")
            for label, protos in results['protocols_by_label'].items():
                f.write(f"{label}:\n")
                for proto, count in sorted(protos.items(), key=lambda x: x[1], reverse=True)[:5]:
                    f.write(f"  {proto}: {count:,}\n")

        # Analyse Physical Dataset
        write_section(f, "PARTIE 2: PHYSICAL DATASET")
        print(f"\nAnalyse des fichiers PHYSICAL ({len(files_physical)} fichiers)")

        for idx, file in enumerate(files_physical, 1):
            print(f"\n[{idx}/{len(files_physical)}] {file}")
            write_section(f, f"Fichier: {file}")

            results = analyze_file_complete(file, 'physical')
            all_results['physical'].append(results)

            f.write("\n--- STATISTIQUES DE BASE ---\n")
            f.write(f"Nombre total de lignes: {results['total_rows']:,}\n")
            f.write(f"Nombre de capteurs: {len([c for c in results['columns'] if c not in ['Time', 'label', 'Label_n']])}\n\n")

            f.write("Distribution des labels:\n")
            for label, count in sorted(results['label_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / results['total_rows']) * 100 if results['total_rows'] > 0 else 0
                f.write(f"  {label}: {count:,} ({percentage:.2f}%)\n")

            f.write("\n--- ANALYSE TEMPORELLE ---\n")
            if results['has_time'] and results['time_range']:
                f.write(f"Période: {results['time_range'][0]} à {results['time_range'][1]}\n")
            else:
                f.write("Pas de données temporelles\n")

            f.write("\n--- CAPTEURS ACTIFS PAR LABEL ---\n")
            for label, count in results['active_sensors_by_label'].items():
                f.write(f"{label}: {count} capteurs actifs détectés\n")

        # Synthèse globale
        write_section(f, "SYNTHÈSE GLOBALE")

        total_rows_network = sum(r['total_rows'] for r in all_results['network'])
        total_rows_physical = sum(r['total_rows'] for r in all_results['physical'])

        all_labels = set()
        for r in all_results['network'] + all_results['physical']:
            all_labels.update(r['label_distribution'].keys())

        f.write(f"\nFichiers analysés:\n")
        f.write(f"  Network: {len(files_network)} fichiers, {total_rows_network:,} lignes\n")
        f.write(f"  Physical: {len(files_physical)} fichiers, {total_rows_physical:,} lignes\n")
        f.write(f"\nLabels uniques trouvés: {', '.join(sorted(all_labels))}\n")

    print("\n" + "="*80)
    print(f"ANALYSE TERMINÉE".center(80))
    print(f"Résultats sauvegardés dans: {output_file}".center(80))
    print("="*80 + "\n")

        # --- Conversion pour export JSON ---
    results['label_distribution'] = dict(results['label_distribution'])
    results['hourly_distribution'] = dict(results['hourly_distribution'])
    results['protocol_distribution'] = dict(results['protocol_distribution'])
    results['top_talkers'] = dict(results['top_talkers'])

    results['protocols_by_label'] = {
        k: dict(v) for k, v in results['protocols_by_label'].items()
    }

    if results['time_range'] is not None:
        results['time_range'] = [
            results['time_range'][0].isoformat(),
            results['time_range'][1].isoformat()
        ]

        # Export JSON
    json_output_file = "analysis_results.json"
    with open(json_output_file, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, default=str)



# Exécuter l'analyse
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Démarrage: {start_time.strftime('%H:%M:%S')}")

    write_analysis_to_file()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Durée totale: {duration:.1f} secondes ({duration/60:.1f} minutes)")