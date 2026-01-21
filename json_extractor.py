import json
from typing import Dict, List, Tuple
from collections import Counter

class DatasetAnalyzer:
    """Classe pour extraire et formater les données d'analyse pour Streamlit"""

    def __init__(self, json_file: str = "analysis_results.json"):
        """Initialise l'analyseur avec le fichier JSON"""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def get_global_stats(self) -> Dict:
        """Retourne les statistiques globales"""
        network_rows = sum(item['total_rows'] for item in self.data['network'])
        physical_rows = sum(item['total_rows'] for item in self.data['physical'])

        return {
            'network_files': len(self.data['network']),
            'physical_files': len(self.data['physical']),
            'network_total_rows': network_rows,
            'physical_total_rows': physical_rows,
            'total_rows': network_rows + physical_rows
        }

    def get_label_distribution(self) -> Dict[str, Dict]:
        """Retourne la distribution des labels par type de dataset"""
        network_labels = Counter()
        physical_labels = Counter()

        for item in self.data['network']:
            network_labels.update(item['label_distribution'])

        for item in self.data['physical']:
            physical_labels.update(item['label_distribution'])

        return {
            'network': dict(network_labels),
            'physical': dict(physical_labels),
            'combined': dict(network_labels + physical_labels)
        }

    def get_top_labels(self, n: int = 5) -> List[Tuple[str, int]]:
        """Retourne les n labels les plus fréquents"""
        labels = self.get_label_distribution()['combined']
        return sorted(labels.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_attack_vs_normal_ratio(self) -> Dict:
        """Calcule le ratio attaques vs normal"""
        labels = self.get_label_distribution()['combined']

        normal_count = labels.get('normal', 0) + labels.get('nomal', 0)
        total = sum(labels.values())
        attack_count = total - normal_count

        return {
            'normal': normal_count,
            'attacks': attack_count,
            'total': total,
            'normal_percentage': (normal_count / total * 100) if total > 0 else 0,
            'attacks_percentage': (attack_count / total * 100) if total > 0 else 0
        }

    def get_protocol_distribution(self) -> Dict[str, int]:
        """Retourne la distribution globale des protocoles (network uniquement)"""
        protocols = Counter()

        for item in self.data['network']:
            protocols.update(item['protocol_distribution'])

        return dict(protocols)

    def get_top_protocols(self, n: int = 10) -> List[Tuple[str, int]]:
        """Retourne les n protocoles les plus utilisés"""
        protocols = self.get_protocol_distribution()
        return sorted(protocols.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_ip_statistics(self) -> Dict:
        """Retourne les statistiques sur les IPs"""
        total_src = set()
        total_dst = set()
        total_missing = 0

        for item in self.data['network']:
            total_src.add(item['unique_ips_src'])
            total_dst.add(item['unique_ips_dst'])
            # Convertir la chaîne en int si nécessaire
            missing = item['missing_ip_count']
            if isinstance(missing, str):
                missing = int(missing) if missing.isdigit() else 0
            total_missing += missing

        # Prendre le max car ce sont des ensembles uniques
        max_src = max(total_src) if total_src else 0
        max_dst = max(total_dst) if total_dst else 0

        return {
            'unique_sources': max_src,
            'unique_destinations': max_dst,
            'missing_ips': total_missing
        }

    def get_top_ips(self, n: int = 10) -> List[Tuple[str, int]]:
        """Retourne les n IPs les plus actives"""
        all_ips = Counter()

        for item in self.data['network']:
            all_ips.update(item['top_talkers'])

        return all_ips.most_common(n)

    def get_time_ranges(self) -> Dict:
        """Retourne les plages temporelles par dataset"""
        network_times = []
        physical_times = []

        for item in self.data['network']:
            if item['time_range']:
                network_times.append(item['time_range'])

        for item in self.data['physical']:
            if item['time_range']:
                physical_times.append(item['time_range'])

        return {
            'network': network_times,
            'physical': physical_times
        }

    def get_hourly_distribution(self) -> Dict[str, Counter]:
        """Retourne la distribution horaire des événements"""
        network_hours = Counter()
        physical_hours = Counter()

        for item in self.data['network']:
            network_hours.update(item['hourly_distribution'])

        for item in self.data['physical']:
            physical_hours.update(item['hourly_distribution'])

        return {
            'network': dict(network_hours),
            'physical': dict(physical_hours)
        }

    def get_sensor_statistics(self) -> Dict:
        """Retourne les statistiques sur les capteurs (physical uniquement)"""
        sensor_stats = {}

        for item in self.data['physical']:
            file_name = item['file'].split('/')[-1]
            sensor_stats[file_name] = item['active_sensors_by_label']

        return sensor_stats

    def get_missing_values_summary(self) -> Dict:
        """Retourne un résumé des valeurs manquantes"""
        network_missing = {}
        physical_missing = {}

        for item in self.data['network']:
            for col, count in item['missing_values'].items():
                count_int = int(count) if isinstance(count, str) else count
                if count_int > 0:
                    if col not in network_missing:
                        network_missing[col] = 0
                    network_missing[col] += count_int

        for item in self.data['physical']:
            for col, count in item['missing_values'].items():
                count_int = int(count) if isinstance(count, str) else count
                if count_int > 0:
                    if col not in physical_missing:
                        physical_missing[col] = 0
                    physical_missing[col] += count_int

        return {
            'network': network_missing,
            'physical': physical_missing
        }

    def get_protocols_by_attack_type(self) -> Dict:
        """Retourne les protocoles utilisés pour chaque type d'attaque"""
        protocols_by_label = {}

        for item in self.data['network']:
            for label, protocols in item['protocols_by_label'].items():
                if label not in protocols_by_label:
                    protocols_by_label[label] = Counter()
                protocols_by_label[label].update(protocols)

        # Convertir en dict simple et garder top 5 par label
        result = {}
        for label, counter in protocols_by_label.items():
            result[label] = dict(counter.most_common(5))

        return result

    def get_dataset_summary_table(self) -> List[Dict]:
        """Retourne un tableau résumé de tous les fichiers"""
        summary = []

        for item in self.data['network']:
            summary.append({
                'Dataset': 'Network',
                'File': item['file'].split('/')[-1],
                'Rows': item['total_rows'],
                'Labels': len(item['label_distribution']),
                'Unique_IPs_Src': item['unique_ips_src'],
                'Protocols': len(item['protocol_distribution'])
            })

        for item in self.data['physical']:
            summary.append({
                'Dataset': 'Physical',
                'File': item['file'].split('/')[-1],
                'Rows': item['total_rows'],
                'Labels': len(item['label_distribution']),
                'Sensors': len([c for c in item['columns'] if c not in ['Time', 'label', 'Label_n', 'Lable_n', '\ufeffTime']]),
                'Active_Sensors': sum(item['active_sensors_by_label'].values())
            })

        return summary

def load_analyzer(json_file: str = "analysis_results.json") -> DatasetAnalyzer:
    """Charge l'analyseur de données"""
    return DatasetAnalyzer(json_file)


def get_all_data_for_streamlit(json_file: str = "analysis_results.json") -> Dict:
    """Retourne toutes les données formatées pour Streamlit"""
    analyzer = DatasetAnalyzer(json_file)

    return {
        'global_stats': analyzer.get_global_stats(),
        'label_distribution': analyzer.get_label_distribution(),
        'top_labels': analyzer.get_top_labels(),
        'attack_ratio': analyzer.get_attack_vs_normal_ratio(),
        'protocols': analyzer.get_protocol_distribution(),
        'top_protocols': analyzer.get_top_protocols(),
        'ip_stats': analyzer.get_ip_statistics(),
        'top_ips': analyzer.get_top_ips(),
        'time_ranges': analyzer.get_time_ranges(),
        'hourly_dist': analyzer.get_hourly_distribution(),
        'sensor_stats': analyzer.get_sensor_statistics(),
        'missing_values': analyzer.get_missing_values_summary(),
        'protocols_by_attack': analyzer.get_protocols_by_attack_type(),
        'summary_table': analyzer.get_dataset_summary_table()
    }


# Exemple d'utilisation
if __name__ == "__main__":
    data = get_all_data_for_streamlit("analysis_results.json")
    for key in data.keys():
        print(f"{key}: {data[key]}\n")