from Monumenta_Peruana_node_analysis_defs import MonumentaPeruanaNetwork

# Podstawowe użycie
network = MonumentaPeruanaNetwork('data/Monumenta Peruana relations.xlsx')

# Pobierz połączenia dla konkretnego węzła
connections = network.get_node_connections('Cuzco', top_n=10)

for neighbor, weight in connections:
    print(f"{neighbor}: weight {weight}")

# Wyświetl statystyki
network.print_node_statistics('Cuzco')

# Wizualizacja sąsiedztwa
network.visualize_node_neighborhood('Cuzco', max_neighbors=20)

#%%
import pandas as pd
import networkx as nx
from ipysigma import Sigma

df = pd.read_excel('data/Monumenta Peruana relations.xlsx')
G = nx.Graph()

# Dodawanie krawędzi z wagami
for _, row in df.iterrows():
    G.add_edge(row['Name_1'], row['Name_2'], weight=row['weight'])

node_name = "Cuzco"

if node_name not in G:
    raise ValueError(f"Węzeł {node_name} nie istnieje w grafie")
else:
    print("Node in the graph")

connections = []
for neighbor in G.neighbors(node_name):
    weight = G[node_name][neighbor]['weight']
    connections.append((neighbor, weight))

# Sort by weight descending
connections.sort(key=lambda x: x[1], reverse=True)

# Optionally limit to top N
connections[:20]

#subgraph
direct_neighbors = set(G.neighbors(node_name))
nodes_in_subgraph = {node_name}
nodes_in_subgraph.update(direct_neighbors)  

# if include_neighbors_of_neighbors:
#     for neighbor in direct_neighbors:
#         neighbors_of_neighbor = set(graph.neighbors(neighbor))
#         nodes_to_include.update(neighbors_of_neighbor)

# Utwórz podgraf
subgraph = nx.Graph()

# Dodaj węzły
for node in nodes_in_subgraph:
    subgraph.add_node(node)

# Dodaj krawędzie TYLKO między węzłami w podgrafie
for _, row in df.iterrows():
    name1, name2, weight = row['Name_1'], row['Name_2'], row['weight']
    
    # Dodaj krawędź tylko jeśli oba węzły są w naszym podgrafie
    if name1 in nodes_in_subgraph and name2 in nodes_in_subgraph:
        subgraph.add_edge(name1, name2, weight=weight)

print(f"Podgraf utworzony:")
print(f"  - Węzły: {subgraph.number_of_nodes()}")
print(f"  - Krawędzie: {subgraph.number_of_edges()}")
print(f"  - Oryginalny graf: {G.number_of_nodes()} węzłów, {G.number_of_edges()} krawędzi")

degree_dict = dict(subgraph.degree())
nx.set_node_attributes(subgraph, degree_dict, name='degree')

Sigma.write_html(
    subgraph,
    'data/subgraph.html',
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    