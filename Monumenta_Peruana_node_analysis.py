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
import numpy as np

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
    
#%%

#def
def compare_basic_metrics(graph, node1, node2):
    """Compare basic metrics of two nodes"""
    
    metrics = {}
    
    # Node degrees
    degree1 = graph.degree(node1)
    degree2 = graph.degree(node2)
    metrics['degrees'] = {
        node1: degree1,
        node2: degree2,
        'difference': abs(degree1 - degree2),
        'ratio': min(degree1, degree2) / max(degree1, degree2) if max(degree1, degree2) > 0 else 0
    }
    
    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(graph)
        metrics['betweenness'] = {
            node1: betweenness.get(node1, 0),
            node2: betweenness.get(node2, 0),
            'difference': abs(betweenness.get(node1, 0) - betweenness.get(node2, 0))
        }
    except:
        metrics['betweenness'] = None
    
    # Closeness centrality
    try:
        if nx.is_connected(graph):
            closeness = nx.closeness_centrality(graph)
            metrics['closeness'] = {
                node1: closeness.get(node1, 0),
                node2: closeness.get(node2, 0),
                'difference': abs(closeness.get(node1, 0) - closeness.get(node2, 0))
            }
        else:
            metrics['closeness'] = None
    except:
        metrics['closeness'] = None
    
    # Clustering coefficient
    try:
        clustering = nx.clustering(graph)
        metrics['clustering'] = {
            node1: clustering.get(node1, 0),
            node2: clustering.get(node2, 0),
            'difference': abs(clustering.get(node1, 0) - clustering.get(node2, 0))
        }
    except:
        metrics['clustering'] = None
        
    return metrics

def compare_neighbors(graph, node1, node2):
    """Compare neighbors of two nodes"""
    
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    
    # Find common neighbors and unique neighbors for each node
    common_neighbors = neighbors1.intersection(neighbors2)
    unique_to_node1 = neighbors1 - neighbors2
    unique_to_node2 = neighbors2 - neighbors1
    
    # If graph has weights, calculate average connection weights
    weighted_analysis = None
    if nx.is_weighted(graph):
        weights1 = [graph[node1][neighbor].get('weight', 1) for neighbor in neighbors1]
        weights2 = [graph[node2][neighbor].get('weight', 1) for neighbor in neighbors2]
        
        weighted_analysis = {
            'avg_weight_node1': np.mean(weights1) if weights1 else 0,
            'avg_weight_node2': np.mean(weights2) if weights2 else 0,
            'max_weight_node1': max(weights1) if weights1 else 0,
            'max_weight_node2': max(weights2) if weights2 else 0
        }
    
    return {
        'neighbors_node1': neighbors1,
        'neighbors_node2': neighbors2,
        'common_neighbors': common_neighbors,
        'unique_to_node1': unique_to_node1,
        'unique_to_node2': unique_to_node2,
        'jaccard_similarity': len(common_neighbors) / len(neighbors1.union(neighbors2)) if neighbors1.union(neighbors2) else 0,
        'common_neighbors_count': len(common_neighbors),
        'weighted_analysis': weighted_analysis
    }

#
node1, node2 = 'Toledo, Francisco', 'Acosta, José De'
compare_basic_metrics(G, node1, node2)














