import pandas as pd
import requests
from io import StringIO
import numpy as np

network_data = pd.read_excel('data\emlo_jesuit_network_1000.xlsx')

# ---------------------------------------------------------
# 7. Ujednolicenie par dla grafu nieskierowanego
# ---------------------------------------------------------

pairs = np.sort(
    network_data[
        ["source", "target"]
    ].values,
    axis=1
)

network_data["person"] = pairs[:, 0]
network_data["response_person"] = pairs[:, 1]


# ---------------------------------------------------------
# 8. Agregacja liczby listów
# ---------------------------------------------------------

co_occurrence = (
    network_data
    .groupby(
        ["person", "response_person"]
    )
    .size()
    .reset_index(name="weight")
)

co_occurrence.sort_values(
    "weight",
    ascending=False
).head(20)

#%%

import networkx as nx
from tqdm.auto import tqdm

G = nx.Graph()

for _, row in tqdm(
    co_occurrence.iterrows(),
    total=len(co_occurrence)
):
    person = row["person"]
    response_person = row["response_person"]
    weight = row["weight"]

    # Węzły
    G.add_node(
        person,
        type="person"
    )

    G.add_node(
        response_person,
        type="person"
    )

    # Krawędź
    G.add_edge(
        person,
        response_person,
        weight=weight
    )
    
# ---------------------------------------------------------
# 9. Degree
# ---------------------------------------------------------

degree_dict = dict(
    G.degree()
)

nx.set_node_attributes(
    G,
    degree_dict,
    name="degree"
)


# ---------------------------------------------------------
# 10. Weighted degree
# ---------------------------------------------------------

weighted_degree_dict = dict(
    G.degree(weight="weight")
)

nx.set_node_attributes(
    G,
    weighted_degree_dict,
    name="weighted_degree"
)

degree_centrality = nx.degree_centrality(G)

betweenness_centrality = nx.betweenness_centrality(
    G
)

closeness_centrality = nx.closeness_centrality(
    G
)

eigenvector_centrality = nx.eigenvector_centrality(
    G,
    max_iter=2000
)

nx.set_node_attributes(
    G,
    degree_centrality,
    "degree_centrality"
)

nx.set_node_attributes(
    G,
    betweenness_centrality,
    "betweenness"
)

nx.set_node_attributes(
    G,
    closeness_centrality,
    "closeness"
)

nx.set_node_attributes(
    G,
    eigenvector_centrality,
    "eigenvector"
)

metrics = pd.DataFrame(
    [
        {
            "person": node,
            "degree": G.nodes[node]["degree"],
            "weighted_degree": G.nodes[node]["weighted_degree"],
            "degree_centrality": G.nodes[node]["degree_centrality"],
            "betweenness": G.nodes[node]["betweenness"],
            "closeness": G.nodes[node]["closeness"],
            "eigenvector": G.nodes[node]["eigenvector"],
        }
        for node in G.nodes()
    ]
)

metrics.sort_values(
    "degree",
    ascending=False
).head(20)

#%%

from ipysigma import Sigma

Sigma(
    G,
    node_size="degree",
    node_label_size="degree"
)

Sigma.write_html(
    G,
    "data/emlo_jesuit_1000_network.html",
    fullscreen=True,
    node_metrics=["louvain"],
    node_color="louvain",
    node_size="degree",
    node_size_range=(3, 25),
    max_categorical_colors=30,
    default_edge_type="curve",
    default_node_label_size=14,
    node_border_color_from="node"
)














