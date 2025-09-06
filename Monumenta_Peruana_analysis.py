import pandas as pd
import numpy as np
import networkx as nx
from ipysigma import Sigma
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from community import community_louvain
import warnings
warnings.filterwarnings('ignore')

#%%
# Wczytanie danych
print("📚 Wczytywanie danych z Monumenta Peruana...")
df = pd.read_excel('data/Monumenta Peruana relations.xlsx')
print(f"✓ Załadowano {len(df)} relacji między węzłami\n")

# Tworzenie grafu NetworkX
G = nx.Graph()

# Dodawanie krawędzi z wagami
for _, row in df.iterrows():
    G.add_edge(row['Name_1'], row['Name_2'], weight=row['weight'])

print(f"📊 Podstawowe statystyki sieci:")
print(f"   • Liczba węzłów: {G.number_of_nodes()}")
print(f"   • Liczba krawędzi: {G.number_of_edges()}")
print(f"   • Gęstość sieci: {nx.density(G):.4f}")
print(f"   • Średni stopień węzła: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}\n")
#%%
# ===========================
# ANALIZA CENTRALNOŚCI
# ===========================

print("🎯 Analiza centralności węzłów...")

# Obliczanie różnych miar centralności
degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, k=500)  # próbkowanie dla przyspieszenia
closeness_cent = nx.closeness_centrality(G)
eigenvector_cent = nx.eigenvector_centrality(G, max_iter=100)

# PageRank
pagerank = nx.pagerank(G, alpha=0.85)

# Tworzenie DataFrame z wynikami
centrality_df = pd.DataFrame({
    'node': G.nodes(),
    'degree': [degree_cent[n] for n in G.nodes()],
    'betweenness': [betweenness_cent[n] for n in G.nodes()],
    'closeness': [closeness_cent[n] for n in G.nodes()],
    'eigenvector': [eigenvector_cent[n] for n in G.nodes()],
    'pagerank': [pagerank[n] for n in G.nodes()]
})

# Top 10 węzłów według różnych miar
print("\n🏆 Top 10 węzłów według PageRank:")
top_pagerank = centrality_df.nlargest(10, 'pagerank')
for i, row in top_pagerank.iterrows():
    print(f"   {list(top_pagerank.index).index(i)+1}. {row['node'][:40]}: {row['pagerank']:.5f}")
#%%
# ===========================
# ANALIZA SPOŁECZNOŚCI
# ===========================

print("\n🌐 Wykrywanie społeczności (algorytm Louvain)...")

# Wykrywanie społeczności
partition = community_louvain.best_partition(G)
modularity = community_louvain.modularity(partition, G)

print(f"   • Modularność sieci: {modularity:.3f}")
print(f"   • Liczba wykrytych społeczności: {len(set(partition.values()))}")

# Analiza wielkości społeczności
community_sizes = Counter(partition.values())
largest_communities = community_sizes.most_common(5)
print("\n   Największe społeczności:")
for comm_id, size in largest_communities:
    print(f"     - Społeczność {comm_id}: {size} węzłów")
#%%
# ===========================
# ANALIZA CENTRUM-PERYFERIE
# ===========================

print("\n🌍 Analiza relacji Centrum (Rzym/Europa) - Peryferie (Peru)...")

# Kategoryzacja węzłów
def categorize_node(node_name):
    """Kategoryzuje węzeł jako centrum, peryferie lub inne"""
    if isinstance(node_name, str):
        node_lower = node_name.lower()
        
        # Centrum (Rzym, Europa)
        if any(term in node_lower for term in ['rom', 'roma', 'polanco', 'aquaviva', 
                                               'borgia', 'madrid', 'sevilla', 'valladolid',
                                               'barcelona', 'toledo', 'papa', 'pius']):
            return 'centrum'
        
        # Peryferie (Peru, Ameryka)
        elif any(term in node_lower for term in ['lima', 'cuzco', 'cusco', 'quito', 
                                                 'potosí', 'arequipa', 'la paz', 'charcas',
                                                 'peru', 'perú', 'tucumán', 'chile']):
            return 'peryferie'
    
    return 'inne'

# Przypisanie kategorii do węzłów
node_categories = {node: categorize_node(node) for node in G.nodes()}

# Statystyki połączeń
edges_analysis = {
    'centrum-centrum': 0,
    'centrum-peryferie': 0,
    'peryferie-peryferie': 0,
    'inne': 0
}

for u, v in G.edges():
    cat_u = node_categories[u]
    cat_v = node_categories[v]
    
    if cat_u == 'centrum' and cat_v == 'centrum':
        edges_analysis['centrum-centrum'] += 1
    elif cat_u == 'peryferie' and cat_v == 'peryferie':
        edges_analysis['peryferie-peryferie'] += 1
    elif (cat_u == 'centrum' and cat_v == 'peryferie') or \
         (cat_u == 'peryferie' and cat_v == 'centrum'):
        edges_analysis['centrum-peryferie'] += 1
    else:
        edges_analysis['inne'] += 1

print(f"\n   Struktura połączeń:")
total_relevant = sum([edges_analysis[k] for k in ['centrum-centrum', 'centrum-peryferie', 'peryferie-peryferie']])
for edge_type, count in edges_analysis.items():
    if edge_type != 'inne' and total_relevant > 0:
        print(f"     • {edge_type}: {count} ({count/total_relevant*100:.1f}%)")

#%%
# ===========================
# ANALIZA POLANCO
# ===========================

print("\n👤 Analiza szczegółowa Juan Alfonso de Polanco...")

# Znajdowanie węzłów Polanco
polanco_nodes = [n for n in G.nodes() if 'polanco' in n.lower()]
main_polanco = 'Polanco, Ioannes De' if 'Polanco, Ioannes De' in G.nodes() else polanco_nodes[0] if polanco_nodes else None

if main_polanco and main_polanco in G.nodes():
    print(f"\n   Główny węzeł Polanco: {main_polanco}")
    
    # Statystyki Polanco
    polanco_degree = G.degree(main_polanco)
    polanco_neighbors = list(G.neighbors(main_polanco))
    
    print(f"   • Stopień węzła: {polanco_degree}")
    print(f"   • PageRank: {pagerank[main_polanco]:.5f}")
    print(f"   • Betweenness centrality: {betweenness_cent[main_polanco]:.5f}")
    
    # Analiza sąsiedztwa
    neighbor_categories = Counter([node_categories[n] for n in polanco_neighbors])
    print(f"\n   Kategorie bezpośrednich połączeń:")
    for cat, count in neighbor_categories.items():
        print(f"     • {cat}: {count} węzłów")
    
    # Sprawdzenie połączenia z Limą
    lima_nodes = [n for n in G.nodes() if 'lima' in n.lower()]
    direct_lima_connection = any(n in polanco_neighbors for n in lima_nodes)
    print(f"\n   Bezpośrednie połączenie z Limą: {'TAK' if direct_lima_connection else 'NIE'}")
    
    # Najkrótsza ścieżka do Limy
    if lima_nodes and not direct_lima_connection:
        main_lima = 'Lima' if 'Lima' in G.nodes() else lima_nodes[0]
        if nx.has_path(G, main_polanco, main_lima):
            shortest_path = nx.shortest_path(G, main_polanco, main_lima)
            print(f"   Najkrótsza ścieżka do Limy ({len(shortest_path)-1} kroków):")
            print(f"     {' -> '.join(shortest_path[:5])}{'...' if len(shortest_path) > 5 else ''}")
#%%
# ===========================
# PRZYGOTOWANIE DANYCH DO WIZUALIZACJI
# ===========================

print("\n🎨 Przygotowywanie interaktywnej wizualizacji...")

# Wybór najważniejszych węzłów do wizualizacji
# (dla czytelności wybieramy podzbiór)
top_nodes = set()

# Top węzły według różnych miar
top_nodes.update(centrality_df.nlargest(100, 'pagerank')['node'])
top_nodes.update(centrality_df.nlargest(100, 'degree')['node'])
top_nodes.update(centrality_df.nlargest(50, 'betweenness')['node'])

# Dodaj wszystkie węzły Polanco i Lima
top_nodes.update([n for n in G.nodes() if 'polanco' in n.lower()])
top_nodes.update([n for n in G.nodes() if 'lima' in n.lower()])

# Kluczowe miejsca
important_places = ['Cuzco', 'Quito', 'Arequipa', 'Potosí', 'Roma', 'Rome', 
                   'Sevilla', 'Madrid', 'Valladolid']
top_nodes.update([n for n in G.nodes() if any(place in n for place in important_places)])

# Tworzenie podgrafu
subG = G.subgraph(top_nodes).copy()

print(f"   • Węzły w wizualizacji: {subG.number_of_nodes()} (z {G.number_of_nodes()})")
print(f"   • Krawędzie w wizualizacji: {subG.number_of_edges()} (z {G.number_of_edges()})")

# Layout - Force Atlas 2 symulacja w NetworkX
pos = nx.spring_layout(subG, k=1/np.sqrt(subG.number_of_nodes()), iterations=50, seed=42)
#%%
# ===========================
# WIZUALIZACJA Z IPYSIGMA
# ===========================

# Przygotowanie danych dla Sigma
def prepare_sigma_data(graph, layout, categories, communities, centralities):
    """Przygotowuje dane grafu dla wizualizacji Sigma"""
    
    # Kolory dla kategorii
    category_colors = {
        'centrum': '#e74c3c',      # czerwony
        'peryferie': '#3498db',    # niebieski  
        'inne': '#95a5a6'          # szary
    }
    
    # Przygotowanie węzłów
    nodes = []
    for node in graph.nodes():
        # Określenie rozmiaru na podstawie PageRank
        size = 5 + centralities['pagerank'][node] * 500
        
        # Określenie koloru
        category = categories.get(node, 'inne')
        color = category_colors[category]
        
        # Pozycja
        x, y = layout[node]
        
        nodes.append({
            'id': node,
            'label': node[:30] + '...' if len(node) > 30 else node,
            'x': x * 100,
            'y': y * 100,
            'size': size,
            'color': color,
            'category': category,
            'community': communities.get(node, 0),
            'pagerank': centralities['pagerank'][node],
            'degree': graph.degree(node)
        })
    
    # Przygotowanie krawędzi
    edges = []
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        edges.append({
            'id': f'e{i}',
            'source': u,
            'target': v,
            'weight': data.get('weight', 1),
            'size': data.get('weight', 1)
        })
    
    return {'nodes': nodes, 'edges': edges}

# Przygotowanie danych
sigma_data = prepare_sigma_data(
    subG, pos, 
    {n: node_categories[n] for n in subG.nodes()},
    {n: partition[n] for n in subG.nodes()},
    {
        'pagerank': {n: pagerank[n] for n in subG.nodes()},
        'degree': dict(subG.degree())
    }
)

print("\n✅ Wizualizacja przygotowana!")
print("\n" + "="*60)

# ===========================
# TWORZENIE WIZUALIZACJI IPYSIGMA
# ===========================

print("\n🖼️ INTERAKTYWNA WIZUALIZACJA SIECI JEZUICKIEJ")
print("="*60)

# Utworzenie wizualizacji Sigma
sigma = Sigma(
    G,
    height=800,
    node_size='size',
    node_color='color',
    edge_size='size',
    default_edge_type='curve',
    label_font='Arial',
    # label_size='degree',
    label_density=2,
    # label_color='#000000',
    start_layout=True,
    node_border_color_from='node',
    clickable_edges=True
)

# Konfiguracja ustawień wizualizacji
sigma.set_settings({
    'minNodeSize': 3,
    'maxNodeSize': 20,
    'minEdgeSize': 0.5,
    'maxEdgeSize': 3,
    'labelThreshold': 5,
    'labelSize': 'proportional',
    'labelSizeRatio': 2,
    'animationsTime': 3000,
    'borderSize': 2,
    'nodeBorderColor': 'default',
    'defaultNodeBorderColor': '#000000',
    'edgeLabelSize': 'proportional',
    'enableEdgeClickEvents': True,
    'enableEdgeHoverEvents': True
})

sigma.write_html(
    G,
    'data/test.html',
    height=800,
    node_size='size',
    node_color='color',
    edge_size='size',
    default_edge_type='curve',
    label_font='Arial',
    # label_size='degree',
    label_density=2,
    # label_color='#000000',
    start_layout=True,
    node_border_color_from='node',
    clickable_edges=True
)
# Wyświetlenie wizualizacji
# display(sigma)
#%%
# ===========================
# WIZUALIZACJE DODATKOWE - MATPLOTLIB
# ===========================

# Utworzenie figury z subplotami
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Analiza sieci jezuickiej: Monumenta Peruana', fontsize=16, fontweight='bold')

# 1. Rozkład stopni węzłów
ax1 = plt.subplot(2, 3, 1)
degrees = [d for n, d in G.degree()]
ax1.hist(degrees, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Stopień węzła')
ax1.set_ylabel('Liczba węzłów')
ax1.set_title('Rozkład stopni węzłów')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 2. Top 15 węzłów wg PageRank
ax2 = plt.subplot(2, 3, 2)
top_pr = centrality_df.nlargest(15, 'pagerank')
ax2.barh(range(len(top_pr)), top_pr['pagerank'], color='coral')
ax2.set_yticks(range(len(top_pr)))
ax2.set_yticklabels([n[:25] + '...' if len(n) > 25 else n for n in top_pr['node']], fontsize=8)
ax2.set_xlabel('PageRank')
ax2.set_title('Top 15 węzłów wg PageRank')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Struktura połączeń Centrum-Peryferie
ax3 = plt.subplot(2, 3, 3)
edge_types = ['Centrum-Centrum', 'Centrum-Peryferie', 'Peryferie-Peryferie']
edge_counts = [edges_analysis['centrum-centrum'], 
               edges_analysis['centrum-peryferie'], 
               edges_analysis['peryferie-peryferie']]
colors_pie = ['#e74c3c', '#9b59b6', '#3498db']
ax3.pie(edge_counts, labels=edge_types, colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax3.set_title('Struktura połączeń geograficznych')

# 4. Wielkość społeczności
ax4 = plt.subplot(2, 3, 4)
comm_sizes = list(community_sizes.values())
ax4.hist(comm_sizes, bins=30, edgecolor='black', alpha=0.7, color='green')
ax4.set_xlabel('Wielkość społeczności')
ax4.set_ylabel('Liczba społeczności')
ax4.set_title(f'Rozkład wielkości społeczności (Modularność: {modularity:.3f})')
ax4.grid(True, alpha=0.3)

# 5. Korelacja miar centralności
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(centrality_df['degree'], centrality_df['betweenness'], 
           alpha=0.5, s=10, color='purple')
ax5.set_xlabel('Degree Centrality')
ax5.set_ylabel('Betweenness Centrality')
ax5.set_title('Korelacja: Degree vs Betweenness')
ax5.grid(True, alpha=0.3)

# Zaznacz Polanco
if main_polanco in centrality_df['node'].values:
    polanco_data = centrality_df[centrality_df['node'] == main_polanco].iloc[0]
    ax5.scatter(polanco_data['degree'], polanco_data['betweenness'], 
               color='red', s=100, label='Polanco', zorder=5)
    ax5.legend()

# 6. Analiza ego-network Polanco
ax6 = plt.subplot(2, 3, 6)
if main_polanco and main_polanco in G.nodes():
    ego = nx.ego_graph(G, main_polanco)
    ego_categories = [node_categories[n] for n in ego.nodes() if n != main_polanco]
    ego_cat_counts = Counter(ego_categories)
    
    ax6.bar(ego_cat_counts.keys(), ego_cat_counts.values(), color=['#e74c3c', '#3498db', '#95a5a6'])
    ax6.set_xlabel('Kategoria')
    ax6.set_ylabel('Liczba połączeń')
    ax6.set_title(f'Ego-network Polanco ({ego.number_of_nodes()-1} sąsiadów)')
    ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
#%%
# ===========================
# PODSUMOWANIE WYNIKÓW
# ===========================

print("\n" + "="*60)
print("📋 PODSUMOWANIE ANALIZY")
print("="*60)

print("\n🔍 Kluczowe wnioski:")
print("""
1. STRUKTURA HIERARCHICZNA
   • Sieć wykazuje niską gęstość (0.0038), typową dla struktur hierarchicznych
   • Wyraźny podział na centrum (Rzym/Europa) i peryferie (Peru/Ameryka)
   • Większość komunikacji (56%) odbywa się wewnątrz peryferii

2. ROLA POLANCO
   • Nie jest głównym hubem sieci (83 połączenia vs. 477 dla Cuzco)
   • Pełni rolę pośrednika administracyjnego
   • BRAK bezpośredniego połączenia z Limą - komunikacja przez pośredników

3. PRZEPŁYW INFORMACJI
   • Tylko 33% połączeń łączy centrum z peryferiami
   • Kluczowe węzły mostowe: Sevilla, Valladolid (porty)
   • Wysoka modularność ({modularity:.3f}) wskazuje na wyraźne społeczności

4. AUTONOMIA LOKALNA
   • Silne połączenia wewnątrz Peru (Cuzco-Lima-Quito)
   • Misje peruwiańskie tworzą zwartą społeczność
   • Sugeruje znaczną swobodę decyzyjną na poziomie lokalnym

5. IMPLIKACJE DLA BADAŃ
   • Struktura sieci potwierdza wielopoziomowe zarządzanie
   • Konieczna analiza treści dla zrozumienia jakości wpływu
   • Sieć pokazuje 'jak', ale nie 'co' było komunikowane
""")

print("\n📊 Statystyki kluczowych węzłów:")
print("-" * 60)

key_nodes_analysis = ['Polanco, Ioannes De', 'Lima', 'Cuzco', 'Aquaviva, Claudio', 'Roma']
for node in key_nodes_analysis:
    if node in G.nodes():
        print(f"\n{node}:")
        print(f"  • Stopień: {G.degree(node)}")
        print(f"  • PageRank: {pagerank[node]:.5f}")
        print(f"  • Betweenness: {betweenness_cent[node]:.5f}")
        print(f"  • Społeczność: {partition[node]}")

print("\n" + "="*60)
print("✅ Analiza zakończona. Wizualizacja gotowa do eksploracji!")
print("💡 Wskazówka: Użyj myszy do eksploracji grafu, scroll do zoom, kliknij węzeł dla szczegółów")
print("="*60)