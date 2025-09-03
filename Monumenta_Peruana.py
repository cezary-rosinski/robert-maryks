import sys
sys.path.insert(1, 'C:/Users/Cezary/Documents/IBL-PAN-Python')
from my_functions import gsheet_to_df
import regex as re
from tqdm import tqdm
import pandas as pd
from itertools import combinations
import networkx as nx
from ipysigma import Sigma
import pickle

#%%
mp1 = gsheet_to_df('1AHNc7DwJH0_1jhriiIVCCeWdklKMsqK2nDFrSakVPjo', 'Arkusz1')
mp2 = gsheet_to_df('16hX8RYUMFCiBDqlkUWHfZ3FlqG4J2m44KVqb3m13wkA', 'Sheet1')
mp3 = gsheet_to_df('14mMODKwkKppKFImTatg9ipQBgg1UgyYFiW2wAQXevTk', 'Sheet1')
mp4 = gsheet_to_df('1LJbAuu1Yo50-0RroF6Gz7hrEn865yH_3ojlhiVzo67s', 'Sheet1')
mp5 = gsheet_to_df('1mucj6Xe9Mmwhq-9kkunyfpgSxp8Z_YIxcN6EdaVqDpQ', 'Sheet1')
mp6 = gsheet_to_df('1iz37Cuqc2beXSV7byGSXrckflawQ7ocdRgEzMEqj6S4', 'Sheet1')
mp7 = gsheet_to_df('1nrLfYEX2cUml7mD_E17Ui0iuU8h59GYP_0EUfiN6roY', 'Sheet1')
mp8 = gsheet_to_df('1phsf4t8n8EWKofSPtRFrn6nxpRkex-xM1mp7xIDwldw', 'Sheet1')
dfs = {'1': mp1, 
       '2': mp2, 
       '3': mp3, 
       '4': mp4, 
       '5': mp5, 
       '6': mp6, 
       '7': mp7, 
       '8': mp8}

#%%
def expand_ranges(text):
    expanded = []
    # znajdź i rozwiń zakresy, np. 439-442
    for start, end in re.findall(r'\b(\d+)-(\d+)\b', text):
        expanded.extend(str(n) for n in range(int(start), int(end)+1))
    # usuń te zakresy z tekstu
    text = re.sub(r'\b\d+-\d+\b', '', text)
    return expanded, text

def extract_numbers(text):
    # wyciągnij WSZYSTKIE liczby
    all_matches = re.finditer(r'\d+', text)
    valid_numbers = []
    for match in all_matches:
        start = match.start()
        number = match.group()

        # odrzuć, jeśli liczba jest częścią przypisu, np. "n11" lub "124n11"
        if start > 0 and text[start - 1] == 'n':
            continue

        # odrzuć, jeśli to przypis po spacji, np. "391 20"
        # czyli jeśli poprzedzająca liczba była tuż przed spacją
        prev_text = text[max(0, start - 3):start]
        if re.fullmatch(r'\d\s', prev_text):
            continue

        valid_numbers.append(number)
    return valid_numbers

def clean_page_number(raw):
    # Usuwa znaki niebędące cyframi i dzieli liczby zlepione (np. 39120 → [391, 20] jeśli >4 cyfry)
    cleaned = re.sub(r'[^\d]', '', raw)
    if len(cleaned) >= 4:
        return int(cleaned[:-2])
    return int(cleaned)

for df in tqdm(dfs):
    df['pages'] = ''
    for i, row in df.iterrows():
        if pd.notnull(row['Additional Information']):
            ranges_expanded, cleaned_text = expand_ranges(row['Additional Information'])
            regular_numbers = [clean_page_number(e) for e in extract_numbers(cleaned_text)]
            df['pages'][i] = regular_numbers

#%% unique namies
from monumenta_data_processor import MonumentaNameClustering, MonumentaDataProcessor

names = []
for index, df in tqdm(dfs.items()):
    len_df = len(df)
    test_list = [df['Name'].to_list(), [index] * len_df]
    names.append(test_list)
    
unique_names = sorted([el for sub in [e[0] for e in names] for el in sub])
# unique_names = sorted([el for sub in [e[0] for e in names] for el in sub])[:500]
    
# Załaduj Twoje dane do słownika {indeks: nazwa}
data = dict(zip(range(1,len(unique_names)+1),unique_names))

clusterer = MonumentaNameClustering()
processor = MonumentaDataProcessor(clusterer)

# Przetwórz
names_data = processor.load_data_from_dict(data)
clusters = processor.process_monumenta_data(names_data, threshold=0.9)

# #testy
# test_data = [e.get('name') for e in clusters.get(6047)]
# test_data = dict(zip(range(1,len(test_data)+1),test_data))
# names_data = processor.load_data_from_dict(test_data)
# clusters_test = processor.process_monumenta_data(names_data, threshold=0.918)
# processor.analyze_clusters(clusters_test)
# processor.show_sample_clusters(clusters_test, min_size=2)

# Analizuj wyniki
processor.analyze_clusters(clusters)
processor.show_sample_clusters(clusters, min_size=2)

with open('data/monumenta_peruana_index_similarity.pickle', 'wb') as handle:
    pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/monumenta_peruana_index_similarity.pickle', 'rb') as handle:
    b = pickle.load(handle)

#%% ujednolicić nazwy w indeksach źródłowych!!!!












#%%

final_data = []
for index, df in tqdm(dfs.items()):
    # Tworzymy listę unikalnych par indeksów (i, j)
    pairs = list(combinations(df.index, 2))
    
    # Przechowuj wyniki (pary nazw)
    matching_names = []
    
    for i, j in tqdm(pairs):
        pages_i = set(df.loc[i, 'pages'])
        pages_j = set(df.loc[j, 'pages'])
        
        if pages_i & pages_j:  # sprawdzamy przecięcie
            matching_names.append((df.loc[i, 'Name'], df.loc[j, 'Name']))
    print(f"Liczba par: {len(matching_names)}")
    final_data.extend(matching_names)

graph_df = pd.DataFrame(final_data, columns=["Person 1", "Person 2"])

#%%

graph_df["Name_1"], graph_df["Name_2"] = zip(*graph_df.apply(lambda row: sorted([row["Person 1"], row["Person 2"]]), axis=1))

graph_df = graph_df[['Name_1', 'Name_2']]

co_occurrence = graph_df.groupby(['Name_1', 'Name_2']).size().reset_index(name='weight')
co_occurrence.to_excel('data/Monumenta Peruana relations.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['Name_1']
    topic = row['Name_2']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='Name_1', color='#e41a1c', size=10)
    G.add_node(topic, type='Name_2', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, topic, weight=weight)


# Dodajemy stopnie jako metrykę (do rozmiarów i etykiet)
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, name='degree')

# ✅ Interaktywny podgląd w Jupyterze (opcjonalnie)
Sigma(
    G,
    node_color="tag",
    node_label_size="degree",
    node_size="degree"
)

# ✅ Eksport do HTML
Sigma.write_html(
    G,
    'data/Peru1.html',
    fullscreen=True,
    node_metrics=['louvain'],
    node_color='louvain',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)


pagerank = nx.pagerank(G)
nx.set_node_attributes(G, pagerank, name='pagerank')

# ✅ Eksport do HTML z własnymi metrykami
Sigma.write_html(
    G,
    'data/Peru1_pagerank_graph.html',
    fullscreen=True,
    node_color='pagerank',         # ręcznie przypisany
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

nx.write_graphml(G, "data/Peru.graphml")

#%%
# === [1] IMPORTY ===
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# === [2] WCZYTANIE GRAFU ===
graph_path = "data/Peru.graphml"  # <-- podaj odpowiednią ścieżkę do pliku
G = nx.read_graphml(graph_path)

# === [3] PODSTAWOWA ANALIZA STRUKTURY ===
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
G_main = G.subgraph(largest_component).copy()

print("🔍 ANALIZA STRUKTURY")
print(f"• Liczba wszystkich węzłów: {G.number_of_nodes()}")
print(f"• Liczba komponentów spójnych: {len(components)}")
print(f"• Największa komponenta: {G_main.number_of_nodes()} węzłów, {G_main.number_of_edges()} krawędzi")
print(f"• Gęstość: {nx.density(G_main):.4f}")
print(f"• Średnia długość ścieżki: {nx.average_shortest_path_length(G_main):.2f}")
avg_degree = sum(dict(G_main.degree()).values()) / G_main.number_of_nodes()
print(f"• Średni stopień węzła: {avg_degree:.2f}")

# === [4] CENTRALNOŚĆ ===
print("\n📊 OBLICZANIE CENTRALNOŚCI...")

deg_cent = nx.degree_centrality(G_main)
btw_cent = nx.betweenness_centrality(G_main, k=1000, seed=42)
cls_cent = nx.closeness_centrality(G_main)

centrality_df = pd.DataFrame({
    "Name": list(G_main.nodes),
    "Degree Centrality": pd.Series(deg_cent),
    "Betweenness Centrality": pd.Series(btw_cent),
    "Closeness Centrality": pd.Series(cls_cent)
})

# === [5] SPOŁECZNOŚCI (Louvain) ===
print("🧩 DETEKCJA SPOŁECZNOŚCI...")

import community.community_louvain as community_louvain

partition = community_louvain.best_partition(G_main, random_state=42)
nx.set_node_attributes(G_main, partition, "community")
centrality_df["Community"] = centrality_df["Name"].map(partition)

# === [6] WIZUALIZACJA SIECI ===
print("🖼️ RYSOWANIE SIECI...")
pos = nx.spring_layout(G_main, seed=42)
node_colors = [partition[n] for n in G_main.nodes]
node_sizes = [deg_cent[n] * 2000 for n in G_main.nodes]

plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(G_main, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Set3, alpha=0.8)
nx.draw_networkx_edges(G_main, pos, alpha=0.1, width=0.2)
plt.title("Sieć misyjna jezuitów w Peru – kolor: wspólnota, rozmiar: centralność")
plt.axis("off")
plt.show()

# === [7] ZAPIS DO PLIKU ===
output_excel = "data/jezuici_peru_analiza.xlsx"
centrality_df.sort_values("Degree Centrality", ascending=False).to_excel(output_excel, index=False)
print(f"\n💾 Wyniki zapisano do: {output_excel}")

# === [8] GEOGRAFICZNA MAPA (opcjonalnie, jeśli są lokalizacje) ===
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import geopandas as gpd

# Wybierz tylko potencjalnie geograficzne węzły (na podstawie nazw)
possible_cities = ["Arequipa", "Cuzco", "Quito", "La Paz", "Puno", "Juli", "Trujillo", "Huamanga", "Potosí", "Lima", "Callao", "Chuquisaca", "Santa Cruz", "Charcas"]
geo_df = centrality_df[centrality_df["Name"].isin(possible_cities)].copy()

# Geokodowanie
geolocator = Nominatim(user_agent="jesuit-missions")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
geo_df["location"] = geo_df["Name"].apply(geocode)
geo_df["lat"] = geo_df["location"].apply(lambda loc: loc.latitude if loc else None)
geo_df["lon"] = geo_df["location"].apply(lambda loc: loc.longitude if loc else None)
geo_df.dropna(subset=["lat", "lon"], inplace=True)

# Mapa Peru
gdf = gpd.GeoDataFrame(geo_df, geometry=gpd.points_from_xy(geo_df["lon"], geo_df["lat"]), crs="EPSG:4326")
# world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
peru = world[world["NAME"] == "Peru"]

# Rysuj mapę
ax = peru.plot(figsize=(10, 10), color="white", edgecolor="black")
gdf.plot(ax=ax, color="red", markersize=gdf["Degree Centrality"] * 5000, alpha=0.7)

for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf["Name"]):
    ax.text(x + 0.1, y + 0.1, label, fontsize=9)

plt.title("Najważniejsze lokalizacje sieci jezuickiej w Peru")
plt.axis("off")
plt.show()









































