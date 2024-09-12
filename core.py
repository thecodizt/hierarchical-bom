import networkx as nx
import numpy as np
import colorsys
import json
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt


def synthesize_graph(
    total_nodes: int,
    total_layers: int,
    distribution: str,
    clear_cut_layer: int,
    layer_names: list = None,
) -> nx.DiGraph:
    """
    Synthesize a directed graph based on given parameters.

    :param total_nodes: Total number of nodes in the graph
    :param total_layers: Total number of layers in the graph
    :param distribution: Distribution of nodes across layers ('uniform', 'normal', 'pos_exp', 'neg_exp')
    :param clear_cut_layer: Layer number below which there's only one parent per node
    :param layer_names: Optional list of layer names. If provided, should have at least one entry.
    :return: NetworkX DiGraph object
    """
    G = nx.DiGraph()

    # Distribute nodes across layers
    if distribution == "uniform":
        nodes_per_layer = [total_nodes // total_layers] * total_layers
    elif distribution == "normal":
        nodes_per_layer = np.random.normal(
            total_nodes / total_layers, total_nodes / (4 * total_layers), total_layers
        )
    elif distribution == "pos_exp":
        nodes_per_layer = np.exp(np.linspace(0, 1, total_layers))
    elif distribution == "neg_exp":
        nodes_per_layer = np.exp(np.linspace(1, 0, total_layers))
    else:
        raise ValueError("Invalid distribution type")

    nodes_per_layer = np.round(
        nodes_per_layer / sum(nodes_per_layer) * total_nodes
    ).astype(int)
    nodes_per_layer[0] = 1  # Ensure there's only one root node
    nodes_per_layer[-1] += total_nodes - sum(
        nodes_per_layer
    )  # Adjust last layer to match total_nodes

    # Generate colors for layers
    colors = [
        colorsys.hsv_to_rgb(i / total_layers, 0.7, 0.9) for i in range(total_layers)
    ]
    colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in colors
    ]

    # Handle layer names
    if layer_names is None:
        layer_names = [f"Layer{i+1}" for i in range(total_layers)]
    elif len(layer_names) < total_layers:
        base_name = layer_names[-1].split()[0]
        layer_names.extend(
            [f"{base_name}{i+1}" for i in range(len(layer_names), total_layers)]
        )

    # Create nodes with new naming convention
    node_id = 0
    for layer, num_nodes in enumerate(nodes_per_layer):
        layer_name = layer_names[layer]
        for node_in_layer in range(num_nodes):
            if layer < len(layer_names) - 1:
                node_name = f"{layer_name} {node_in_layer + 1}"
            else:
                node_name = f"{layer_name} {layer + 1} {node_in_layer + 1}"
            G.add_node(
                node_name, layer=layer, color=colors[layer], layer_name=layer_name
            )
            node_id += 1

    # Create edges
    for layer in range(1, total_layers):
        current_layer_nodes = [n for n, d in G.nodes(data=True) if d["layer"] == layer]
        previous_layer_nodes = [
            n for n, d in G.nodes(data=True) if d["layer"] == layer - 1
        ]

        for node in current_layer_nodes:
            if layer >= clear_cut_layer:
                # Above or at clear cut layer: Connect to multiple parents
                num_parents = np.random.randint(
                    1, min(3, len(previous_layer_nodes) + 1)
                )
                parents = np.random.choice(
                    previous_layer_nodes, num_parents, replace=False
                )
            else:
                # Below clear cut layer: Connect to a single parent
                parents = [np.random.choice(previous_layer_nodes)]

            for parent in parents:
                G.add_edge(parent, node)

    return G


def save_graph_to_json(G: nx.DiGraph, filename: str) -> None:
    """
    Save the graph to a JSON file.

    :param G: NetworkX DiGraph object
    :param filename: Name of the file to save the JSON data
    """
    data = nx.node_link_data(G)
    with open(filename, "w") as f:
        json.dump(data, f)


def save_graph_to_csv(G: nx.DiGraph, nodes_filename: str, edges_filename: str) -> None:
    """
    Save the graph to CSV files (one for nodes, one for edges).

    :param G: NetworkX DiGraph object
    :param nodes_filename: Name of the file to save node data
    :param edges_filename: Name of the file to save edge data
    """
    # Save nodes
    nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    nodes_df.to_csv(nodes_filename)

    # Save edges
    edges_df = pd.DataFrame(G.edges(), columns=["source", "target"])
    edges_df.to_csv(edges_filename, index=False)


def visualize_graph(G: nx.DiGraph) -> None:
    """
    Visualize the graph and save it as an image using plotly graph objects without hierarchical layout with colors.

    :param G: NetworkX DiGraph object
    :param filename: Name of the file to save the visualization
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Get node positions using spring layout
    pos = nx.spring_layout(G)

    # Add edges as lines
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Add nodes as markers
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(
            data.get("color", "#1f78b4")
        )  # Use node color from attributes
        node_text.append(f"{node}<br>Layer: {data['layer_name']}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(color=node_colors, size=10, line_width=2),
        text=node_text,
    )

    # Create the layout
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    fig.update_layout(
        title="Graph Visualization",
        titlefont_size=16,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


def minimize_crossings(G, layers):
    """Attempt to minimize edge crossings by reordering nodes within layers."""
    for i in range(1, len(layers)):
        upper_layer = layers[i - 1]
        current_layer = layers[i]

        # Calculate barycenter for each node in the current layer
        barycenters = {}
        for node in current_layer:
            parents = list(G.predecessors(node))
            if parents:
                barycenters[node] = sum(upper_layer.index(p) for p in parents) / len(
                    parents
                )
            else:
                barycenters[node] = (
                    len(upper_layer) / 2
                )  # Place nodes without parents in the middle

        # Sort current layer based on barycenters
        layers[i] = sorted(current_layer, key=lambda n: barycenters[n])

    return layers


def create_improved_hierarchical_layout(G: nx.DiGraph) -> dict:
    """Create an improved hierarchical layout for the graph with minimal edge crossings."""
    layers = defaultdict(list)
    for node, data in G.nodes(data=True):
        layers[data["layer"]].append(node)

    # Sort layers by their key (layer number)
    sorted_layers = [layers[i] for i in sorted(layers.keys())]

    # Minimize crossings
    sorted_layers = minimize_crossings(G, sorted_layers)

    pos = {}
    max_layer = len(sorted_layers) - 1
    for layer, nodes in enumerate(sorted_layers):
        y = max_layer - layer
        width = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (i - width / 2, y)

    return pos


def visualize_graph_hierarchical_plotly(G: nx.DiGraph) -> go.Figure:
    """
    Visualize the graph with an improved hierarchical layout using Plotly.

    :param G: NetworkX DiGraph object
    :return: Plotly Figure object
    """
    pos = create_improved_hierarchical_layout(G)

    # Separate node coordinates
    node_x = [coord[0] for coord in pos.values()]
    node_y = [coord[1] for coord in pos.values()]

    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=False, colorscale="YlGnBu", size=10, color=[], line_width=2
        ),
    )

    # Color node points and add hover text
    node_colors = []
    node_text = []
    for node, data in G.nodes(data=True):
        node_colors.append(data.get("color", "#1f78b4"))
        node_text.append(f"{node}<br>Layer: {data['layer_name']}")

    node_trace.marker.color = node_colors
    node_trace.text = node_text

    # Create the layout
    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    # Add layer lines and labels
    layers = nx.get_node_attributes(G, "layer")
    max_layer = max(layers.values())
    layer_names = nx.get_node_attributes(G, "layer_name")
    unique_layer_names = list(dict.fromkeys(layer_names.values()))

    for layer in range(max_layer + 1):
        y = max_layer - layer
        fig.add_shape(
            type="line",
            x0=min(node_x) - 1,
            y0=y,
            x1=max(node_x) + 1,
            y1=y,
            line=dict(color="gray", width=1, dash="dash"),
        )

        layer_name = (
            unique_layer_names[min(layer, len(unique_layer_names) - 1)]
            if unique_layer_names
            else f"Layer {layer}"
        )
        fig.add_annotation(
            x=min(node_x) - 1, y=y, text=layer_name, showarrow=False, xanchor="right"
        )

    return fig


import networkx as nx
import plotly.graph_objs as go
import streamlit as st


def visualize_graph_path(G: nx.DiGraph, source: str, target: str) -> go.Figure:
    """
    Visualize the graph with a path highlighted from the selected source to target.

    :param G: NetworkX DiGraph object
    :param source: Source node
    :param target: Target node
    :return: Plotly Figure object
    """

    # Check if path exists
    if not nx.has_path(G, source, target):
        st.warning("No path found between the selected source and target.")
        return None

    # Find the shortest path
    path = nx.shortest_path(G, source, target)

    # Create a copy of the graph
    G_path = G.copy()

    # Remove nodes not in the path
    nodes_to_remove = set(G_path.nodes()) - set(path)
    G_path.remove_nodes_from(nodes_to_remove)

    # Create a layout for the path
    pos = nx.spring_layout(G_path)

    # Create edge trace for the path
    edge_x, edge_y = [], []
    for edge in G_path.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="red"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace for the path
    node_x = [pos[node][0] for node in G_path.nodes()]
    node_y = [pos[node][1] for node in G_path.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=20, color="red"),
        text=list(G_path.nodes()),
        textposition="top center",
        hoverinfo="text",
        hovertext=[
            f"{node}<br>Layer: {G.nodes[node]['layer_name']}" for node in G_path.nodes()
        ],
    )

    # Create the layout
    layout = go.Layout(
        title=f"Path from {source} to {target}: {' -> '.join(path)}",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig
