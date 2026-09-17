# %% Electrical grid visualization with networkx and pyvis
"""
Developing ElectricalGrid() graph object visualization.
Vision:
- Different symbols for node types: solar, wind, hydro, nuclear, battery, consumer, corporation
- Edges: transmission lines, transformers
- Color code by microgrids, edges by voltage
- Interactive live simulation would be ideal

Modelling the grid as a network using networkx (nx).
Basic graph algorithms: Dijkstra, BFS, DFS, Max-Flow, etc.
All nx algorithms: https://networkx.org/documentation/stable/reference/algorithms/index.html
"""

import numpy as np
import networkx as nx
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

# %% Node class and basic graph creation
class Node:
    """Electrical grid component represented as a graph vertex."""
    def __init__(self, id, type):
        self.id = id
        self.type = type  # consumer, producer, etc.
        self.voltage = 0.0

    def __str__(self) -> str:
        return f"Node {self.id} of type {self.type}"

    def __repr__(self) -> str:
        return f"id:{self.id}"

    def __hash__(self) -> int:
        return self.id

G = nx.Graph()

node1 = Node(4, 'solar')
nodes = [(1, {'type': 'solar'}), (2, {'type': 'wind'}), (3, {'type': 'hydro'}), (4, {'type': node1})]
edges = [(1, 2, {'voltage': 10}), (2, 4, {'voltage': 20})]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# %% Iterating through edges and nodes
print('=== Edges ===')
for e in list(G.edges):
    print(e)

edge = G.edges[1, 2]
print(f"\nEdge (1, 2) data: {edge}, type: {type(edge)}")
print(f"Voltage: {edge['voltage']}")

print("\n=== Edge iteration with data ===")
for e, datadict in G.edges.items():
    print(f"  {e}: {datadict}")

print("\n=== Node iteration with data ===")
for n, datadict in G.nodes.items():
    print(f"  {n}: {datadict}")
    print(f"    type: {datadict['type']}")

print("\n=== Edge attributes (voltage) ===")
for voltage in G.edges.data('voltage'):
    print(f"  {voltage}")

# %% Visualize with networkx (matplotlib)
nx.draw(G, with_labels=True)
plt.show()

# %% PyVis visualization (requires pyvis to be installed)
"""
PyVis for interactive graph visualization.
Note: pyvis not in environment yet — uncomment when available.
See PyVis tutorial: https://pyvis.readthedocs.io/en/latest/tutorial.html
"""
try:
    from pyvis.network import Network

    nt = Network(notebook=False, cdn_resources='in_line')
    nt.from_nx(G)
    nt.show('nx.html')
    print("PyVis visualization saved to 'nx.html'")
except ImportError:
    print("PyVis not installed; skipping interactive visualization")

# %% Detailed PyVis Network configuration example
"""
This shows full PyVis API usage when available.
"""
try:
    from pyvis.network import Network

    net = Network(
        notebook=False,
        cdn_resources='in_line',
        select_menu=True,
        filter_menu=True,
        neighborhood_highlight=True,
        height='750px',
        width='100%',
        heading='Electrical Grid Network',
        directed=False,
    )

    # Add nodes with properties
    net.add_nodes(
        [1, 2, 3, 4, 5],
        value=[10, 20, 30, 40, 50],
        x=[1, 3, 6, 2, 8],
        y=[0, 3, 1, 0, 2],
        label=['Node #1', 'Node #2', 'Node #3', 'Node #4', 'Node #5'],
        title=['Main node', 'Just node', 'Just node', 'Just node', 'Node with self-loop'],
        color=['#d47415', '#22b512', '#42adf5', '#4a21b0', '#e627a3']
    )

    # Add edges
    net.add_edges([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (5, 5)])

    # Physics simulation
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])

    # Save visualization
    net.show('graph.html')
    print("PyVis network saved to 'graph.html'")
except ImportError:
    print("PyVis not installed; skipping detailed network visualization")
