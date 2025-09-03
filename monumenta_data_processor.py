import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
import jellyfish
import re
from collections import defaultdict, Counter
import networkx as nx
import json
from tqdm import tqdm

class MonumentaNameClustering:
    def __init__(self):
        self.similarity_threshold = 0.8
        
    def preprocess_name(self, name):
        """Normalizuje nazwę do analizy"""
        # Usuń przecinki i dodatkowe spacje
        name = re.sub(r',\s*', ' ', name)
        # Usuń wielokrotne spacje
        name = re.sub(r'\s+', ' ', name)
        # Usuń punkty z skrótów typu "v."
        name = re.sub(r'\bv\.\s*', '', name)
        # Usuń przedimki "de", "del" z końca dla lepszego porównania
        name = re.sub(r'\s+(de|del)$', '', name)
        return name.strip().lower()
    
    def extract_surname_firstname(self, name):
        """Wyodrębnia nazwisko i imię z formatu 'Nazwisko, Imię'"""
        if ',' in name:
            parts = name.split(',', 1)
            surname = parts[0].strip()
            firstname = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Jeśli brak przecinka, zakładamy że pierwsze słowo to nazwisko
            parts = name.split()
            surname = parts[0] if parts else ""
            firstname = " ".join(parts[1:]) if len(parts) > 1 else ""
        
        return surname.lower(), firstname.lower()
    
    def calculate_similarity(self, name1, name2):
        """Oblicza podobieństwo między dwoma nazwami używając różnych metryk"""
        
        # Preprocessing
        clean_name1 = self.preprocess_name(name1)
        clean_name2 = self.preprocess_name(name2)
        
        # Wyodrębnienie nazwisk i imion
        surname1, firstname1 = self.extract_surname_firstname(clean_name1)
        surname2, firstname2 = self.extract_surname_firstname(clean_name2)
        
        similarities = {}
        
        # 1. Podobieństwo całych nazw
        similarities['full_ratio'] = fuzz.ratio(clean_name1, clean_name2) / 100.0
        similarities['full_token_sort'] = fuzz.token_sort_ratio(clean_name1, clean_name2) / 100.0
        
        # POPRAWKA: Użyj token_sort_ratio zamiast token_set_ratio dla bardziej restrykcyjnego porównania
        # token_set_ratio jest zbyt liberalne - traktuje różne zestawy słów jako identyczne
        similarities['full_token_set'] = fuzz.token_sort_ratio(clean_name1, clean_name2) / 100.0
        
        # 2. Podobieństwo nazwisk (najważniejsze)
        similarities['surname_ratio'] = fuzz.ratio(surname1, surname2) / 100.0
        similarities['surname_jaro'] = jellyfish.jaro_winkler_similarity(surname1, surname2)
        
        # 3. Podobieństwo imion - bardziej restrykcyjne
        if firstname1 and firstname2:
            similarities['firstname_ratio'] = fuzz.ratio(firstname1, firstname2) / 100.0
            # POPRAWKA: Użyj ratio zamiast partial_ratio dla dokładniejszego dopasowania
            similarities['firstname_partial'] = fuzz.ratio(firstname1, firstname2) / 100.0
        else:
            # POPRAWKA: Jeśli jedno imię brakuje, nie traktuj jako neutralne - obniż podobieństwo
            similarities['firstname_ratio'] = 0.3  
            similarities['firstname_partial'] = 0.3
        
        # 4. Soundex dla nazwisk (dla wariantów fonetycznych)
        soundex1 = jellyfish.soundex(surname1) if surname1 else ""
        soundex2 = jellyfish.soundex(surname2) if surname2 else ""
        similarities['soundex_match'] = 1.0 if soundex1 == soundex2 and soundex1 else 0.0
        
        # POPRAWKA: Bardziej restrykcyjna formuła - większy nacisk na dokładność
        final_score = (
            similarities['surname_jaro'] * 0.3 +         # Jaro-Winkler dla nazwisk
            similarities['surname_ratio'] * 0.3 +        # Dokładne dopasowanie nazwisk
            similarities['full_token_sort'] * 0.2 +      # Uporządkowane tokeny (nie set!)
            similarities['firstname_partial'] * 0.15 +   # Dokładne dopasowanie imion
            similarities['soundex_match'] * 0.05         # Bonus za soundex
        )
        
        return final_score, similarities
    
    def create_similarity_matrix(self, names):
        """Tworzy macierz podobieństw dla wszystkich nazw"""
        n = len(names)
        similarity_matrix = np.zeros((n, n))
        
        # Oblicz całkowitą liczbę porównań
        total_comparisons = (n * (n - 1)) // 2
        
        with tqdm(total=total_comparisons, desc="Obliczanie podobieństw", unit="par") as pbar:
            for i in range(n):
                for j in range(i+1, n):
                    score, _ = self.calculate_similarity(names[i], names[j])
                    similarity_matrix[i][j] = score
                    similarity_matrix[j][i] = score
                    pbar.update(1)
                
                # Diagonala
                similarity_matrix[i][i] = 1.0
        
        return similarity_matrix
    
    def cluster_names_agglomerative(self, names, threshold=0.8):
        """Klastruje nazwy używając hierarchicznego klastrowania"""
        print("Tworzenie macierzy podobieństw...")
        similarity_matrix = self.create_similarity_matrix(names)
        distance_matrix = 1 - similarity_matrix
        
        print("Uruchamiam klastrowanie hierarchiczne...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            linkage='average',
            metric='precomputed'
        )
        
        clusters = clustering.fit_predict(distance_matrix)
        print(f"Klastrowanie zakończone. Utworzono {len(set(clusters))} klastrów.")
        
        # Organizuj wyniki
        clustered_names = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            clustered_names[cluster_id].append((names[idx], idx))
        
        return dict(clustered_names)
    
    def cluster_names_graph(self, names, threshold=0.8):
        """Alternatywne klastrowanie używając grafów (connected components)"""
        G = nx.Graph()
        
        # Dodaj wszystkie nazwy jako węzły
        for i, name in enumerate(names):
            G.add_node(i, name=name)
        
        # Oblicz całkowitą liczbę porównań
        n = len(names)
        total_comparisons = (n * (n - 1)) // 2
        
        # Dodaj krawędzie dla podobnych nazw
        with tqdm(total=total_comparisons, desc="Tworzenie grafu podobieństw", unit="par") as pbar:
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    similarity, _ = self.calculate_similarity(names[i], names[j])
                    if similarity >= threshold:
                        G.add_edge(i, j, weight=similarity)
                    pbar.update(1)
        
        print(f"Graf utworzony: {len(G.nodes)} węzłów, {len(G.edges)} krawędzi")
        
        # Znajdź connected components
        clusters = list(nx.connected_components(G))
        
        clustered_names = {}
        for cluster_id, nodes in enumerate(clusters):
            clustered_names[cluster_id] = [(names[node], node) for node in nodes]
        
        return clustered_names


class MonumentaDataProcessor:
    def __init__(self, clusterer):
        self.clusterer = clusterer
        
    def load_data_from_csv(self, filepath, name_column, volume_column=None):
        """Ładuje dane z pliku CSV"""
        df = pd.read_csv(filepath)
        
        # Przygotuj dane
        if volume_column:
            names_data = [(row[name_column], row[volume_column]) for _, row in df.iterrows()]
        else:
            names_data = [(name, None) for name in df[name_column].tolist()]
        
        return names_data
    
    def load_data_from_dict(self, data_dict):
        """Ładuje dane ze słownika {indeks: nazwa}"""
        print(f"DEBUG load_data_from_dict - typ wejścia: {type(data_dict)}")
        print(f"DEBUG load_data_from_dict - liczba elementów: {len(data_dict)}")
        
        # Sprawdź pierwsze kilka elementów PRZED przetworzeniem
        print("DEBUG - pierwsze elementy wejściowe:")
        for i, (key, value) in enumerate(list(data_dict.items())[:5]):
            print(f"  klucz: {key} (typ: {type(key)}), wartość: '{value}' (typ: {type(value)}, długość: {len(value) if isinstance(value, str) else 'N/A'})")
        
        names_data = [(name, idx) for idx, name in data_dict.items()]
        
        # Sprawdź wynik DOPO przetworzeniu
        print("DEBUG - pierwsze elementy wyjściowe:")
        for i, (name, idx) in enumerate(names_data[:5]):
            print(f"  nazwa: '{name}' (typ: {type(name)}, długość: {len(name) if isinstance(name, str) else 'N/A'}), indeks: {idx}")
        
        return names_data
    
    def load_data_from_list(self, names_list):
        """Ładuje dane z listy nazw"""
        names_data = [(name, i) for i, name in enumerate(names_list)]
        return names_data
    
    def process_monumenta_data(self, names_data, threshold=0.75, method='graph'):
        """Przetwarza dane z Monumenta Peruana"""
        
        # Sprawdź format danych i dostosuj
        if isinstance(names_data, dict):
            # Jeśli to słownik {indeks: nazwa}
            names_data = self.load_data_from_dict(names_data)
        elif isinstance(names_data, list) and len(names_data) > 0:
            # Sprawdź czy to lista nazw czy lista tupli
            if isinstance(names_data[0], str):
                # Lista nazw
                names_data = self.load_data_from_list(names_data)
            elif not isinstance(names_data[0], (tuple, list)) or len(names_data[0]) != 2:
                # Nieprawidłowy format
                raise ValueError("Dane muszą być w formacie: lista tupli (nazwa, indeks) lub lista nazw lub słownik {indeks: nazwa}")
        
        # Wyodrębnij tylko nazwy do klastrowania
        names = [name for name, _ in names_data]
        
        print(f"Rozpoczynam klastrowanie {len(names)} unikalnych nazw (próg: {threshold}, metoda: {method})...")
        
        # Wybierz metodę klastrowania
        if method == 'agglomerative':
            print("Używam klastrowania hierarchicznego...")
            clusters = self.clusterer.cluster_names_agglomerative(names, threshold)
        else:
            print("Używam klastrowania grafowego...")
            clusters = self.clusterer.cluster_names_graph(names, threshold)
        
        print(f"Klastrowanie zakończone. Utworzono {len(clusters)} klastrów.")
        
        # Przygotuj wyniki z dodatkowymi metadanymi
        results = {}
        for cluster_id, cluster_names in clusters.items():
            cluster_info = []
            for name, name_idx in cluster_names:
                # Znajdź oryginalne dane dla tej nazwy
                original_data = names_data[name_idx]
                cluster_info.append({
                    'name': name,
                    'original_index': original_data[1],
                    'name_idx': name_idx
                })
            results[cluster_id] = cluster_info
        
        return results
    
    def analyze_clusters(self, clusters):
        """Analizuje wyniki klastrowania"""
        
        total_names = sum(len(cluster) for cluster in clusters.values())
        single_clusters = sum(1 for cluster in clusters.values() if len(cluster) == 1)
        multi_clusters = len(clusters) - single_clusters
        largest_cluster = max(len(cluster) for cluster in clusters.values()) if clusters else 0
        
        print("=== ANALIZA KLASTRÓW ===")
        print(f"Całkowita liczba nazw: {total_names}")
        print(f"Liczba klastrów: {len(clusters)}")
        print(f"Klastry jednoelementowe: {single_clusters}")
        print(f"Klastry wieloelementowe: {multi_clusters}")
        print(f"Największy klaster: {largest_cluster} elementów")
        
        # Rozkład wielkości klastrów
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        size_distribution = Counter(cluster_sizes)
        
        print("\nRozkład wielkości klastrów:")
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            print(f"  {size} elementów: {count} klastrów")
        
        return {
            'total_names': total_names,
            'total_clusters': len(clusters),
            'single_clusters': single_clusters,
            'multi_clusters': multi_clusters,
            'largest_cluster': largest_cluster,
            'size_distribution': dict(size_distribution)
        }
    
    def export_clusters_to_csv(self, clusters, filename):
        """Eksportuje klastry do pliku CSV"""
        
        rows = []
        for cluster_id, cluster_names in clusters.items():
            for name_info in cluster_names:
                rows.append({
                    'cluster_id': cluster_id,
                    'name': name_info['name'],
                    'original_index': name_info['original_index'],
                    'cluster_size': len(cluster_names)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Wyniki zapisane do: {filename}")
        
        return df
    
    def export_clusters_to_json(self, clusters, filename):
        """Eksportuje klastry do pliku JSON"""
        
        # Konwertuj do serializowalnego formatu
        serializable_clusters = {}
        for cluster_id, cluster_names in clusters.items():
            serializable_clusters[str(cluster_id)] = cluster_names
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_clusters, f, ensure_ascii=False, indent=2)
        
        print(f"Wyniki zapisane do: {filename}")
    
    def show_sample_clusters(self, clusters, min_size=2, max_clusters=10):
        """Pokazuje przykładowe klastry"""
        
        print(f"\n=== PRZYKŁADOWE KLASTRY (min. {min_size} elementów) ===")
        
        # Filtruj klastry według wielkości
        large_clusters = {k: v for k, v in clusters.items() if len(v) >= min_size}
        
        # Sortuj według wielkości (malejąco)
        sorted_clusters = sorted(large_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (cluster_id, cluster_names) in enumerate(sorted_clusters[:max_clusters]):
            print(f"\nKlaster {cluster_id} ({len(cluster_names)} elementów):")
            for name_info in cluster_names:
                original_idx = name_info['original_index']
                idx_str = f" (idx: {original_idx})" if original_idx is not None else ""
                print(f"  - {name_info['name']}{idx_str}")
    
    def find_potential_errors(self, clusters, threshold=0.6):
        """Znajduje potencjalne błędy klastrowania wymagające ręcznej weryfikacji"""
        
        print("\n=== POTENCJALNE BŁĘDY DO WERYFIKACJI ===")
        
        suspicious_clusters = []
        
        for cluster_id, cluster_names in clusters.items():
            if len(cluster_names) >= 2:
                # Sprawdź wszystkie pary w klastrze
                names = [info['name'] for info in cluster_names]
                min_similarity = float('inf')
                problematic_pair = None
                
                for i in range(len(names)):
                    for j in range(i+1, len(names)):
                        similarity, _ = self.clusterer.calculate_similarity(names[i], names[j])
                        if similarity < min_similarity:
                            min_similarity = similarity
                            problematic_pair = (names[i], names[j])
                
                if min_similarity < threshold:
                    suspicious_clusters.append({
                        'cluster_id': cluster_id,
                        'min_similarity': min_similarity,
                        'problematic_pair': problematic_pair,
                        'all_names': names
                    })
        
        # Sortuj według najmniejszego podobieństwa
        suspicious_clusters.sort(key=lambda x: x['min_similarity'])
        
        if suspicious_clusters:
            for cluster in suspicious_clusters[:10]:  # Pokaż 10 najgorszych
                print(f"\nKlaster {cluster['cluster_id']} (min. podobieństwo: {cluster['min_similarity']:.3f}):")
                print(f"  Problematyczna para: '{cluster['problematic_pair'][0]}' vs '{cluster['problematic_pair'][1]}'")
                print("  Wszystkie nazwy w klastrze:")
                for name in cluster['all_names']:
                    print(f"    - {name}")
        else:
            print("Nie znaleziono podejrzanych klastrów.")


# # Funkcja do testowania podobieństwa konkretnych nazw
# def test_specific_similarity(names_list):
#     """Testuje podobieństwo między konkretnymi nazwami"""
#     clusterer = MonumentaNameClustering()
    
#     print("=== TEST PODOBIEŃSTWA KONKRETNYCH NAZW ===")
#     print("Nazwy do sprawdzenia:")
#     for i, name in enumerate(names_list):
#         print(f"  {i}: '{name}'")
    
#     print("\nMacierz podobieństw:")
#     print("     ", end="")
#     for i in range(len(names_list)):
#         print(f"{i:6d}", end="")
#     print()
    
#     for i in range(len(names_list)):
#         print(f"{i:3d}: ", end="")
#         for j in range(len(names_list)):
#             if i == j:
#                 print(f"{1.000:6.3f}", end="")
#             else:
#                 score, details = clusterer.calculate_similarity(names_list[i], names_list[j])
#                 print(f"{score:6.3f}", end="")
#         print()
    
#     print("\nSzczegółowe porównania:")
#     for i in range(len(names_list)):
#         for j in range(i+1, len(names_list)):
#             score, details = clusterer.calculate_similarity(names_list[i], names_list[j])
#             print(f"\n'{names_list[i]}' vs '{names_list[j]}':")
#             print(f"  Wynik końcowy: {score:.6f}")
#             print(f"  Surname Jaro-Winkler: {details['surname_jaro']:.6f}")
#             print(f"  Surname ratio: {details['surname_ratio']:.6f}")
#             print(f"  Full token set: {details['full_token_set']:.6f}")
#             print(f"  Firstname partial: {details['firstname_partial']:.6f}")
#             print(f"  Soundex match: {details['soundex_match']:.6f}")

# # Dodaj też funkcję do diagnozy klastra
# def diagnose_cluster_problem(names_list, threshold=1.0):
#     """Diagnozuje dlaczego nazwy trafiły do jednego klastra"""
#     clusterer = MonumentaNameClustering()
    
#     print(f"=== DIAGNOZA PROBLEMU KLASTRA (threshold={threshold}) ===")
    
#     # Test metodą grafową
#     print("\n1. TEST METODY GRAFOWEJ:")
#     G = nx.Graph()
#     for i, name in enumerate(names_list):
#         G.add_node(i, name=name)
    
#     edges_added = 0
#     for i in range(len(names_list)):
#         for j in range(i+1, len(names_list)):
#             similarity, _ = clusterer.calculate_similarity(names_list[i], names_list[j])
#             print(f"   '{names_list[i]}' <-> '{names_list[j]}': {similarity:.6f}")
#             if similarity >= threshold:
#                 G.add_edge(i, j, weight=similarity)
#                 edges_added += 1
#                 print(f"     -> KRAWĘDŹ DODANA! (>= {threshold})")
#             else:
#                 print(f"     -> krawędź NIE dodana (< {threshold})")
    
#     print(f"\nGraf: {len(G.nodes)} węzłów, {edges_added} krawędzi")
    
#     # Sprawdź connected components
#     components = list(nx.connected_components(G))
#     print(f"Connected components: {len(components)}")
#     for i, comp in enumerate(components):
#         comp_names = [names_list[node] for node in comp]
#         print(f"  Komponent {i}: {comp_names}")
    
#     return G, components
#     """Diagnozuje format danych i pokazuje potencjalne problemy"""
#     print("=== DIAGNOSTYKA FORMATU DANYCH ===")
#     print(f"Typ danych: {type(data)}")
    
#     if isinstance(data, dict):
#         print(f"Słownik z {len(data)} elementami")
#         print("Przykładowe pary klucz-wartość:")
#         for i, (key, value) in enumerate(list(data.items())[:3]):
#             print(f"  {key} -> '{value}' (typ wartości: {type(value)})")
#     elif isinstance(data, list):
#         print(f"Lista z {len(data)} elementami")
#         print("Przykładowe elementy:")
#         for i, item in enumerate(data[:3]):
#             print(f"  [{i}]: '{item}' (typ: {type(item)})")
#             if isinstance(item, (tuple, list)):
#                 print(f"       Zawartość: {item}")
    
#     # Sprawdź czy są pojedyncze znaki
#     if isinstance(data, list) and data and isinstance(data[0], str) and len(data[0]) == 1:
#         print("⚠️  UWAGA: Wykryto pojedyncze znaki - prawdopodobnie nastąpił split!")
        
#     if isinstance(data, dict):
#         values = list(data.values())
#         if values and isinstance(values[0], str) and len(values[0]) == 1:
#             print("⚠️  UWAGA: Wykryto pojedyncze znaki w wartościach słownika!")
    
#     print("=" * 50)

# Funkcja pomocnicza do szybkiego uruchomienia
def quick_cluster(data, threshold=0.75, method='graph', show_results=True):
    """Szybkie klastrowanie - przyjmuje różne formaty danych"""
    clusterer = MonumentaNameClustering()
    processor = MonumentaDataProcessor(clusterer)
    
    # Przetwórz dane
    clusters = processor.process_monumenta_data(data, threshold=threshold, method=method)
    
    if show_results:
        # Pokaż wyniki
        processor.analyze_clusters(clusters)
        processor.show_sample_clusters(clusters, min_size=2)
        processor.find_potential_errors(clusters)
    
    return clusterer, processor, clusters

# Przykład użycia z rzeczywistymi danymi
def run_example():
    """Przykład użycia z Twoimi danymi"""
    
    # Inicjalizuj
    clusterer = MonumentaNameClustering()
    processor = MonumentaDataProcessor(clusterer)
    
    # Przykładowe dane na podstawie Twoich obrazów
    sample_data = {
        16: "Aguilar, Juan de",
        17: "Aguilar, Juan del", 
        34: "Aguilar, Juan, v. Pérez de Aguilar",
        18: "ANGULO, Francisco",  # Zmieniony indeks
        20: "ANGULO, Francisco de",
        25: "Brinton, Daniel Garrison",  # Zmieniony indeks
        15: "Britton, Daniel",
        26: "Britton, Daniel Garrison"  # Zmieniony indeks
    }
    
    # Załaduj dane
    names_data = processor.load_data_from_dict(sample_data)
    
    # Przetwórz
    print("Testowanie z przykładowymi danymi...")
    clusters = processor.process_monumenta_data(names_data, threshold=0.75)
    
    # Analizuj
    stats = processor.analyze_clusters(clusters)
    processor.show_sample_clusters(clusters, min_size=1)
    processor.find_potential_errors(clusters)
    
    # Zapisz wyniki
    processor.export_clusters_to_csv(clusters, 'monumenta_clusters.csv')
    processor.export_clusters_to_json(clusters, 'monumenta_clusters.json')
    
    return clusterer, processor, clusters

# Test funkcji podobieństwa
def test_similarity():
    """Test funkcji podobieństwa"""
    clusterer = MonumentaNameClustering()
    
    print("=== TEST PODOBIEŃSTWA ===")
    test_pairs = [
        ("Aguilar, Juan de", "Aguilar, Juan del"),
        ("Aguilar, Juan de", "Aguilar, Juan, v. Pérez de Aguilar"),
        ("ANGULO, Francisco", "ANGULO, Francisco de"),
        ("Brinton, Daniel Garrison", "Britton, Daniel Garrison"),
        ("Britton, Daniel", "Britton, Daniel Garrison")
    ]
    
    for name1, name2 in test_pairs:
        score, details = clusterer.calculate_similarity(name1, name2)
        print(f"\n'{name1}' vs '{name2}':")
        print(f"  Podobieństwo końcowe: {score:.3f}")
        print(f"  Nazwisko (Jaro-Winkler): {details['surname_jaro']:.3f}")
        print(f"  Pełna nazwa (token set): {details['full_token_set']:.3f}")

if __name__ == "__main__":
    # Uruchom testy
    test_similarity()
    print("\n" + "="*60)
    
    # Uruchom przykład
    clusterer, processor, clusters = run_example()