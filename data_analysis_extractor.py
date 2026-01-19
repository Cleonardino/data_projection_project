"""
Fonctions d'extraction des résultats d'analyse pour affichage Streamlit
Compatible avec la version optimisée de data_analysis_advanced.py
"""

import re
from collections import defaultdict

def extract_file_stats(file_path="analysis_results.txt"):
    """
    Extrait les statistiques de base de chaque fichier
    
    Returns:
        dict: {
            'network': [
                {
                    'filename': str,
                    'total_rows': int,
                    'num_columns': int,
                    'label_distribution': dict {label: {'count': int, 'percentage': float}},
                    'missing_values': dict {col: {'count': int, 'percentage': float}}
                },
                ...
            ],
            'physical': [...]
        }
    """
    results = {
        'network': [],
        'physical': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Séparer par sections de fichiers
    file_sections = re.split(r'={80}\n\s*Fichier: (.*?)\s*\n={80}', content)
    
    current_type = None
    for i in range(1, len(file_sections), 2):
        if i + 1 >= len(file_sections):
            break
            
        filename = file_sections[i].strip()
        section_content = file_sections[i + 1]
        
        # Déterminer le type (Network ou Physical)
        if 'Network datatset' in filename:
            current_type = 'network'
        elif 'Physical dataset' in filename:
            current_type = 'physical'
        else:
            continue
        
        file_data = {
            'filename': filename,
            'total_rows': 0,
            'num_columns': 0,
            'label_distribution': {},
            'missing_values': {}
        }
        
        # Extraire nombre de lignes
        rows_match = re.search(r'Nombre total de lignes:\s*(\d+)', section_content)
        if rows_match:
            file_data['total_rows'] = int(rows_match.group(1))
        
        # Extraire nombre de colonnes
        cols_match = re.search(r'Nombre de colonnes:\s*(\d+)', section_content)
        if cols_match:
            file_data['num_columns'] = int(cols_match.group(1))
        
        # Extraire nombre de capteurs (Physical)
        sensors_match = re.search(r'Nombre de capteurs:\s*(\d+)', section_content)
        if sensors_match:
            file_data['num_sensors'] = int(sensors_match.group(1))
        
        # Extraire distribution des labels
        label_section = re.search(r'Distribution des labels:\s*\n(.*?)(?:\n\n|\n---)', section_content, re.DOTALL)
        if label_section:
            for line in label_section.group(1).strip().split('\n'):
                match = re.search(r'^\s*(.+?):\s*(\d+)\s*\((\d+\.\d+)%\)', line)
                if match:
                    label, count, percent = match.groups()
                    file_data['label_distribution'][label.strip()] = {
                        'count': int(count),
                        'percentage': float(percent)
                    }
        
        # Extraire valeurs manquantes (principales)
        missing_section = re.search(r'Valeurs manquantes.*?:\s*\n(.*?)(?:\n\n|\n---)', section_content, re.DOTALL)
        if missing_section:
            for line in missing_section.group(1).strip().split('\n'):
                match = re.search(r'^\s*(.+?):\s*(\d+)\s*\((\d+\.\d+)%\)', line)
                if match:
                    col, count, percent = match.groups()
                    file_data['missing_values'][col.strip()] = {
                        'count': int(count),
                        'percentage': float(percent)
                    }
        
        results[current_type].append(file_data)
    
    return results

def extract_temporal_analysis(file_path="analysis_results.txt"):
    """
    Extrait les analyses temporelles
    
    Returns:
        dict: {
            'network': [
                {
                    'filename': str,
                    'period_start': str,
                    'period_end': str,
                    'hourly_distribution': dict {hour: count}
                },
                ...
            ],
            'physical': [...]
        }
    """
    results = {
        'network': [],
        'physical': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    file_sections = re.split(r'={80}\n\s*Fichier: (.*?)\s*\n={80}', content)
    
    for i in range(1, len(file_sections), 2):
        if i + 1 >= len(file_sections):
            break
            
        filename = file_sections[i].strip()
        section_content = file_sections[i + 1]
        
        if 'Network datatset' in filename:
            current_type = 'network'
        elif 'Physical dataset' in filename:
            current_type = 'physical'
        else:
            continue
        
        temporal_data = {
            'filename': filename,
            'has_temporal_data': False,
            'period_start': None,
            'period_end': None,
            'hourly_distribution': {}
        }
        
        # Vérifier si "Pas de données temporelles"
        if 'Pas de données temporelles' in section_content:
            results[current_type].append(temporal_data)
            continue
        
        # Extraire période
        period_match = re.search(r'Période:\s*(.+?)\s*à\s*(.+?)(?:\n|$)', section_content)
        if period_match:
            temporal_data['has_temporal_data'] = True
            temporal_data['period_start'] = period_match.group(1).strip()
            temporal_data['period_end'] = period_match.group(2).strip()
        
        # Extraire distribution horaire
        hourly_section = re.search(r'Distribution horaire.*?:\s*\n(.*?)(?:\n\n|\n---)', section_content, re.DOTALL)
        if hourly_section:
            for line in hourly_section.group(1).strip().split('\n'):
                match = re.search(r'Heure\s+(\d+):\s*(\d+)', line)
                if match:
                    hour, count = match.groups()
                    temporal_data['hourly_distribution'][int(hour)] = int(count)
        
        results[current_type].append(temporal_data)
    
    return results

def extract_network_topology(file_path="analysis_results.txt"):
    """
    Extrait les analyses topologiques (Network uniquement)
    
    Returns:
        list: [
            {
                'filename': str,
                'unique_ips_src': int,
                'unique_ips_dst': int,
                'missing_ips': int,
                'protocol_distribution': dict {proto: count},
                'top_talkers': list of tuples [(ip, count), ...]
            },
            ...
        ]
    """
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    file_sections = re.split(r'={80}\n\s*Fichier: (.*?)\s*\n={80}', content)
    
    for i in range(1, len(file_sections), 2):
        if i + 1 >= len(file_sections):
            break
            
        filename = file_sections[i].strip()
        
        if 'Network datatset' not in filename:
            continue
        
        section_content = file_sections[i + 1]
        
        topo_data = {
            'filename': filename,
            'unique_ips_src': 0,
            'unique_ips_dst': 0,
            'missing_ips': 0,
            'protocol_distribution': {},
            'top_talkers': []
        }
        
        # Extraire statistiques topologiques
        ips_src_match = re.search(r'IPs sources uniques:\s*(\d+)', section_content)
        if ips_src_match:
            topo_data['unique_ips_src'] = int(ips_src_match.group(1))
        
        ips_dst_match = re.search(r'IPs destinations uniques:\s*(\d+)', section_content)
        if ips_dst_match:
            topo_data['unique_ips_dst'] = int(ips_dst_match.group(1))
        
        missing_match = re.search(r'IPs sources manquantes:\s*(\d+)', section_content)
        if missing_match:
            topo_data['missing_ips'] = int(missing_match.group(1))
        
        # Extraire distribution des protocoles
        proto_section = re.search(r'Distribution des protocoles:\s*\n(.*?)(?:\n\n|\nTop 10)', section_content, re.DOTALL)
        if proto_section:
            for line in proto_section.group(1).strip().split('\n'):
                match = re.search(r'^\s*(.+?):\s*(\d+)', line)
                if match:
                    proto, count = match.groups()
                    topo_data['protocol_distribution'][proto.strip()] = int(count)
        
        # Extraire top talkers
        talkers_section = re.search(r'Top 10 des IPs les plus actives:\s*\n(.*?)(?:\n\n|\n---)', section_content, re.DOTALL)
        if talkers_section:
            for line in talkers_section.group(1).strip().split('\n'):
                match = re.search(r'^\s*([\d\.]+):\s*(\d+)', line)
                if match:
                    ip, count = match.groups()
                    topo_data['top_talkers'].append((ip, int(count)))
        
        results.append(topo_data)
    
    return results

def extract_label_correlation(file_path="analysis_results.txt"):
    """
    Extrait les corrélations entre labels et attributs
    
    Returns:
        dict: {
            'network': [
                {
                    'filename': str,
                    'protocols_by_label': dict {label: {proto: count}}
                },
                ...
            ],
            'physical': [
                {
                    'filename': str,
                    'active_sensors_by_label': dict {label: count}
                },
                ...
            ]
        }
    """
    results = {
        'network': [],
        'physical': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    file_sections = re.split(r'={80}\n\s*Fichier: (.*?)\s*\n={80}', content)
    
    for i in range(1, len(file_sections), 2):
        if i + 1 >= len(file_sections):
            break
            
        filename = file_sections[i].strip()
        section_content = file_sections[i + 1]
        
        if 'Network datatset' in filename:
            current_type = 'network'
            corr_data = {
                'filename': filename,
                'protocols_by_label': {}
            }
            
            # Extraire protocoles par label
            proto_section = re.search(r'PROTOCOLES PAR LABEL.*?:\s*\n(.*?)(?:\n\n|$)', section_content, re.DOTALL)
            if proto_section:
                current_label = None
                for line in proto_section.group(1).strip().split('\n'):
                    # Ligne de label (pas d'indentation ou commence par une lettre)
                    label_match = re.match(r'^([a-zA-Z][\w\s]*):\s*$', line.strip())
                    if label_match:
                        current_label = label_match.group(1).strip()
                        corr_data['protocols_by_label'][current_label] = {}
                    # Ligne de protocole (indentation)
                    elif current_label and line.strip():
                        proto_match = re.search(r'^\s*(.+?):\s*(\d+)', line)
                        if proto_match:
                            proto, count = proto_match.groups()
                            corr_data['protocols_by_label'][current_label][proto.strip()] = int(count)
            
            results['network'].append(corr_data)
        
        elif 'Physical dataset' in filename:
            current_type = 'physical'
            corr_data = {
                'filename': filename,
                'active_sensors_by_label': {}
            }
            
            # Extraire capteurs actifs par label
            sensors_section = re.search(r'CAPTEURS ACTIFS PAR LABEL.*?:\s*\n(.*?)(?:\n\n|$)', section_content, re.DOTALL)
            if sensors_section:
                for line in sensors_section.group(1).strip().split('\n'):
                    match = re.search(r'^(.+?):\s*(\d+)\s*capteurs', line)
                    if match:
                        label, count = match.groups()
                        corr_data['active_sensors_by_label'][label.strip()] = int(count)
            
            results['physical'].append(corr_data)
    
    return results

def get_all_labels(file_path="analysis_results.txt"):
    """
    Extrait tous les labels uniques trouvés dans l'analyse
    
    Returns:
        dict: {
            'network': set of labels,
            'physical': set of labels,
            'all': set of all labels
        }
    """
    results = {
        'network': set(),
        'physical': set(),
        'all': set()
    }
    
    stats = extract_file_stats(file_path)
    
    for file_data in stats['network']:
        results['network'].update(file_data['label_distribution'].keys())
    
    for file_data in stats['physical']:
        results['physical'].update(file_data['label_distribution'].keys())
    
    results['all'] = results['network'] | results['physical']
    
    return results

def get_summary_statistics(file_path="analysis_results.txt"):
    """
    Calcule des statistiques résumées sur l'ensemble des datasets
    
    Returns:
        dict: {
            'total_files_network': int,
            'total_files_physical': int,
            'total_rows_network': int,
            'total_rows_physical': int,
            'all_labels': set,
            'label_distribution_global': dict {label: count}
        }
    """
    stats = extract_file_stats(file_path)
    
    summary = {
        'total_files_network': len(stats['network']),
        'total_files_physical': len(stats['physical']),
        'total_rows_network': sum(f['total_rows'] for f in stats['network']),
        'total_rows_physical': sum(f['total_rows'] for f in stats['physical']),
        'all_labels': set(),
        'label_distribution_global': defaultdict(int)
    }
    
    for file_data in stats['network'] + stats['physical']:
        summary['all_labels'].update(file_data['label_distribution'].keys())
        for label, data in file_data['label_distribution'].items():
            summary['label_distribution_global'][label] += data['count']
    
    summary['label_distribution_global'] = dict(summary['label_distribution_global'])
    
    return summary

def get_protocol_summary(file_path="analysis_results.txt"):
    """
    Résumé des protocoles les plus utilisés (Network)
    
    Returns:
        dict: {proto: total_count} trié par count décroissant
    """
    topo = extract_network_topology(file_path)
    protocol_totals = defaultdict(int)
    
    for file_data in topo:
        for proto, count in file_data['protocol_distribution'].items():
            protocol_totals[proto] += count
    
    return dict(sorted(protocol_totals.items(), key=lambda x: x[1], reverse=True))

def get_top_ips_global(file_path="analysis_results.txt", top_n=20):
    """
    Top N des IPs les plus actives sur tous les fichiers Network
    
    Returns:
        list: [(ip, total_count), ...] trié par count décroissant
    """
    topo = extract_network_topology(file_path)
    ip_totals = defaultdict(int)
    
    for file_data in topo:
        for ip, count in file_data['top_talkers']:
            ip_totals[ip] += count
    
    return sorted(ip_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Exemple d'utilisation pour Streamlit
if __name__ == "__main__":
    print("=== TEST DES FONCTIONS D'EXTRACTION ===\n")
    
    # Test 1: Statistiques de base
    print("1. Statistiques de base")
    stats = extract_file_stats()
    print(f"   - Fichiers Network: {len(stats['network'])}")
    print(f"   - Fichiers Physical: {len(stats['physical'])}")
    if stats['network']:
        print(f"   - Exemple: {stats['network'][0]['filename']} -> {stats['network'][0]['total_rows']} lignes")
    
    # Test 2: Analyse temporelle
    print("\n2. Analyse temporelle")
    temporal = extract_temporal_analysis()
    for t in temporal['network'][:2]:
        if t['has_temporal_data']:
            print(f"   - {t['filename']}: {t['period_start']} à {t['period_end']}")
    
    # Test 3: Topologie
    print("\n3. Topologie Network")
    topo = extract_network_topology()
    if topo:
        print(f"   - {topo[0]['filename']}: {topo[0]['unique_ips_src']} IPs sources")
        print(f"   - Protocoles: {list(topo[0]['protocol_distribution'].keys())[:5]}")
    
    # Test 4: Labels
    print("\n4. Labels")
    labels = get_all_labels()
    print(f"   - Network: {labels['network']}")
    print(f"   - Physical: {labels['physical']}")
    
    # Test 5: Résumé global
    print("\n5. Résumé global")
    summary = get_summary_statistics()
    print(f"   - Total lignes Network: {summary['total_rows_network']:,}")
    print(f"   - Total lignes Physical: {summary['total_rows_physical']:,}")
    print(f"   - Labels globaux: {summary['all_labels']}")
    
    # Test 6: Protocoles
    print("\n6. Top 5 protocoles")
    protocols = get_protocol_summary()
    for proto, count in list(protocols.items())[:5]:
        print(f"   - {proto}: {count:,}")
    
    # Test 7: Top IPs
    print("\n7. Top 5 IPs")
    top_ips = get_top_ips_global(top_n=5)
    for ip, count in top_ips:
        print(f"   - {ip}: {count:,} paquets")
    
    print("\nTous les tests passés !")