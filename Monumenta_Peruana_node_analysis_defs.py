import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import numpy as np


class MonumentaPeruanaNetwork:
    """
    Class for network analysis of Monumenta Peruana data
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the class and load data
        
        Args:
            file_path: path to Excel file with data
        """
        self.file_path = file_path
        self.df = None
        self.graph = None
        self.load_data()
        self.create_graph()
    
    def load_data(self):
        """Load data from Excel file"""
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Loaded {len(self.df)} relations")
            print(f"Columns: {self.df.columns.tolist()}")
        except Exception as e:
            print(f"Error loading file: {e}")
            raise
    
    def create_graph(self):
        """Create graph based on loaded data"""
        # Create undirected weighted graph
        self.graph = nx.Graph()
        
        # Add edges with weights
        for _, row in self.df.iterrows():
            # If edge already exists, sum the weights
            if self.graph.has_edge(row['Name_1'], row['Name_2']):
                self.graph[row['Name_1']][row['Name_2']]['weight'] += row['weight']
            else:
                self.graph.add_edge(row['Name_1'], row['Name_2'], weight=row['weight'])
        
        print(f"\nGraph created:")
        print(f"- Number of nodes: {self.graph.number_of_nodes()}")
        print(f"- Number of edges: {self.graph.number_of_edges()}")
    
    def get_node_connections(self, node_name: str, 
                            min_weight: Optional[float] = None,
                            top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get list of nodes connected to given node with connection strength
        
        Args:
            node_name: name of node to analyze
            min_weight: minimum connection weight (optional)
            top_n: return only top N strongest connections (optional)
        
        Returns:
            List of tuples (node_name, connection_strength) sorted by strength descending
        """
        if node_name not in self.graph:
            print(f"Node '{node_name}' does not exist in the graph!")
            return []
        
        # Get all connections with weights
        connections = []
        for neighbor in self.graph.neighbors(node_name):
            weight = self.graph[node_name][neighbor]['weight']
            if min_weight is None or weight >= min_weight:
                connections.append((neighbor, weight))
        
        # Sort by weight descending
        connections.sort(key=lambda x: x[1], reverse=True)
        
        # Optionally limit to top N
        if top_n:
            connections = connections[:top_n]
        
        return connections
    
    def print_node_connections(self, node_name: str, 
                              min_weight: Optional[float] = None,
                              top_n: Optional[int] = None):
        """
        Display connections for given node in readable format
        """
        connections = self.get_node_connections(node_name, min_weight, top_n)
        
        if not connections:
            print(f"No connections found for node '{node_name}'")
            return
        
        print(f"\nConnections for node: '{node_name}'")
        print(f"Number of connections: {len(connections)}")
        print(f"{'='*60}")
        print(f"{'Target Node':<45} {'Strength':>10}")
        print(f"{'-'*60}")
        
        for neighbor, weight in connections:
            print(f"{neighbor:<45} {weight:>10.0f}")
        
        print(f"{'-'*60}")
        print(f"Sum of all connection weights: {sum(w for _, w in connections):.0f}")
    
    def get_node_statistics(self, node_name: str) -> Dict:
        """
        Get detailed statistics for given node
        """
        if node_name not in self.graph:
            return None
        
        connections = self.get_node_connections(node_name)
        weights = [w for _, w in connections]
        
        stats = {
            'node_name': node_name,
            'degree': self.graph.degree(node_name),
            'weighted_degree': sum(weights),
            'connections_count': len(connections),
            'max_weight': max(weights) if weights else 0,
            'min_weight': min(weights) if weights else 0,
            'avg_weight': np.mean(weights) if weights else 0,
            'clustering_coefficient': nx.clustering(self.graph, node_name),
            'betweenness_centrality': nx.betweenness_centrality(self.graph, normalized=True).get(node_name, 0)
        }
        
        return stats
    
    def print_node_statistics(self, node_name: str):
        """
        Display statistics for given node
        """
        stats = self.get_node_statistics(node_name)
        
        if not stats:
            print(f"Node '{node_name}' does not exist in the graph!")
            return
        
        print(f"\nStatistics for node: '{node_name}'")
        print(f"{'='*50}")
        print(f"Node degree: {stats['degree']}")
        print(f"Weighted degree: {stats['weighted_degree']:.0f}")
        print(f"Number of unique connections: {stats['connections_count']}")
        print(f"Maximum connection weight: {stats['max_weight']:.0f}")
        print(f"Minimum connection weight: {stats['min_weight']:.0f}")
        print(f"Average connection weight: {stats['avg_weight']:.2f}")
        print(f"Clustering coefficient: {stats['clustering_coefficient']:.4f}")
        print(f"Betweenness centrality: {stats['betweenness_centrality']:.6f}")
    
    def find_nodes_by_pattern(self, pattern: str) -> List[str]:
        """
        Find nodes matching pattern (case insensitive)
        """
        pattern_lower = pattern.lower()
        matching_nodes = [node for node in self.graph.nodes() 
                         if pattern_lower in node.lower()]
        return sorted(matching_nodes)
    
    def get_top_nodes_by_degree(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top N nodes by degree (number of connections)
        """
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_nodes
    
    def get_top_nodes_by_weighted_degree(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N nodes by weighted degree (sum of connection weights)
        """
        weighted_degrees = {}
        for node in self.graph.nodes():
            total_weight = sum(self.graph[node][neighbor]['weight'] 
                             for neighbor in self.graph.neighbors(node))
            weighted_degrees[node] = total_weight
        
        top_nodes = sorted(weighted_degrees.items(), 
                          key=lambda x: x[1], reverse=True)[:n]
        return top_nodes
    
    def visualize_node_neighborhood(self, node_name: str, 
                                   max_neighbors: int = 20,
                                   figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize neighborhood of given node
        """
        if node_name not in self.graph:
            print(f"Node '{node_name}' does not exist in the graph!")
            return
        
        # Create subgraph with node and its neighbors
        connections = self.get_node_connections(node_name, top_n=max_neighbors)
        neighbors = [node_name] + [n for n, _ in connections]
        subgraph = self.graph.subgraph(neighbors)
        
        # Prepare layout
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Prepare node colors and sizes
        node_colors = ['red' if n == node_name else 'lightblue' 
                      for n in subgraph.nodes()]
        node_sizes = [1000 if n == node_name else 300 
                     for n in subgraph.nodes()]
        
        # Draw graph
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)
        
        nx.draw_networkx_labels(subgraph, pos, 
                               font_size=8,
                               font_weight='bold')
        
        # Draw edges with weights
        edges = subgraph.edges()
        weights = [subgraph[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [3 * w / max_weight for w in weights]
        
        nx.draw_networkx_edges(subgraph, pos,
                              width=edge_widths,
                              alpha=0.5)
        
        # Add edge weights as labels
        edge_labels = nx.get_edge_attributes(subgraph, 'weight')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels,
                                    font_size=6)
        
        plt.title(f"Neighborhood of node: {node_name}\n(Top {max_neighbors} connections)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# Example usage
def main():
    # Initialize network
    network = MonumentaPeruanaNetwork('Monumenta Peruana relations.xlsx')
    
    # Example 1: Analyze specific node
    print("\n" + "="*70)
    print("EXAMPLE 1: Analysis of node 'Cuzco'")
    print("="*70)
    
    network.print_node_connections('Cuzco', top_n=10)
    network.print_node_statistics('Cuzco')
    
    # Example 2: Search for nodes
    print("\n" + "="*70)
    print("EXAMPLE 2: Search for nodes containing 'Toledo'")
    print("="*70)
    
    matching = network.find_nodes_by_pattern('Toledo')
    print(f"Found {len(matching)} nodes:")
    for node in matching[:5]:  # Show first 5
        print(f"  - {node}")
    
    # Example 3: Top nodes in network
    print("\n" + "="*70)
    print("EXAMPLE 3: Top 10 nodes by number of connections")
    print("="*70)
    
    top_nodes = network.get_top_nodes_by_degree(10)
    for i, (node, degree) in enumerate(top_nodes, 1):
        print(f"{i:2}. {node:<40} Connections: {degree}")
    
    # Example 4: Interactive node selection
    print("\n" + "="*70)
    print("EXAMPLE 4: Interactive analysis")
    print("="*70)
    
    while True:
        node_input = input("\nEnter node name to analyze (or 'q' to quit): ")
        
        if node_input.lower() == 'q':
            break
        
        # Check if node exists
        if node_input in network.graph:
            network.print_node_connections(node_input, top_n=15)
            
            # Optional visualization
            viz = input("Do you want to see visualization? (y/n): ")
            if viz.lower() == 'y':
                network.visualize_node_neighborhood(node_input, max_neighbors=15)
        else:
            # Try to find similar nodes
            similar = network.find_nodes_by_pattern(node_input)
            if similar:
                print(f"Node '{node_input}' does not exist. Did you mean:")
                for s in similar[:5]:
                    print(f"  - {s}")
            else:
                print(f"Node '{node_input}' does not exist in the graph.")


if __name__ == "__main__":
    main()