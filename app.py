import streamlit as st
import numpy as np
import csv
import json
import networkx as nx
from core import (
    synthesize_graph,
    save_graph_to_json,
    save_graph_to_csv,
    visualize_graph,
    visualize_graph_hierarchical_plotly,
    visualize_graph_path,
    save_to_graphml,
    get_all_parents,
    get_all_children,
    visualize_subgraph,
    find_critical_path,
    analyze_node_properties,
)
import time
from memory_profiler import memory_usage
import functools
import pandas as pd
import plotly.graph_objects as go


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        mem_before = memory_usage()[0]
        result = func(*args, **kwargs)
        end_time = time.time()
        mem_after = memory_usage()[0]
        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        st.session_state.setdefault("profiling_results", []).append(
            {
                "function": func.__name__,
                "execution_time": execution_time,
                "memory_used": memory_used,
            }
        )
        return result

    return wrapper


st.title("Hierarchical BOM Synthesizer")

st.write(
    "This app allows you to synthesize a hierarchical BOM (Bill of Materials) graph based on the number of nodes and layers you specify. You can choose from different distributions for the number of nodes per layer and specify the clear cut layer, which is the layer below which there's only one parent per node."
)

st.write(
    "You can also specify the layer names, which will be used to name the nodes in the graph. The layer names will be repeated for the number of nodes in that layer."
)

st.write(
    "The app will generate a graph in JSON format, which you can then use to create a BOM in a spreadsheet application."
)


def inputs():
    total_nodes = st.number_input(
        "Total Nodes", min_value=1, value=100, key="total_nodes"
    )
    total_layers = st.number_input(
        "Total Layers", min_value=1, value=12, key="total_layers"
    )
    distribution = "pos_exp"
    clear_cut_layer = st.number_input(
        "Clear Cut Layer", min_value=1, value=5, key="clear_cut_layer"
    )
    layer_names = st.text_input("Layer Names (comma-separated)", key="layer_names")
    return total_nodes, total_layers, distribution, clear_cut_layer, layer_names


@profile
def generate_graph(
    total_nodes,
    total_layers,
    distribution,
    clear_cut_layer,
    layer_names,
    num_properties,
    property_names,
):
    return synthesize_graph(
        total_nodes,
        total_layers,
        distribution,
        clear_cut_layer,
        layer_names.split(",") if layer_names else None,
        num_properties,
        property_names,
    )


@profile
def save_outputs(G):
    save_graph_to_json(G, "synthesized_graph.json")
    save_graph_to_csv(G, "nodes.csv", "edges.csv")
    save_to_graphml(G, "synthesized_graph.graphml")


@profile
def visualize_graphs(G):
    fig1 = visualize_graph(G)
    fig = visualize_graph_hierarchical_plotly(G)
    return fig1, fig


def main():
    total_nodes, total_layers, distribution, clear_cut_layer, layer_names = inputs()

    # Add input for custom node properties
    num_properties = st.sidebar.number_input(
        "Number of node properties", min_value=1, max_value=5, value=1
    )
    property_names = []
    for i in range(num_properties):
        prop_name = st.sidebar.text_input(
            f"Property {i+1} name", value=f"property_{i+1}"
        )
        property_names.append(prop_name)

    # Generate the graph
    G = generate_graph(
        total_nodes,
        total_layers,
        distribution,
        clear_cut_layer,
        layer_names,
        num_properties,
        property_names,
    )

    # Save outputs
    save_outputs(G)

    # Visualize the graph
    fig1, fig = visualize_graphs(G)

    st.success("Graph synthesized successfully!")

    st.subheader("Graph Visualization")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Hierarchical Graph Visualization")
    # st.image("graph_visualization_hierarchical.png")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("JSON Output")
    with open("synthesized_graph.json") as f:
        st.download_button("Download JSON", f)

    st.subheader("CSV Output")
    with open("nodes.csv") as f:
        st.download_button("Download Nodes", f)
    with open("edges.csv") as f:
        st.download_button("Download Edges", f)

    st.subheader("Path Analysis")
    source = st.selectbox("Source Node", list(G.nodes), key="source_node")
    target = st.selectbox("Target Node", list(G.nodes), key="target_node")

    if st.button("Analyze Path"):
        if nx.has_path(G, source, target):
            path = nx.shortest_path(G, source, target)
            st.write(f"Shortest path from {source} to {target}:")
            st.write(" -> ".join(path))
            st.write(f"Path length: {len(path) - 1}")

            # Calculate and display all paths
            all_paths = list(nx.all_simple_paths(G, source, target))
            st.write(f"Total number of paths: {len(all_paths)}")

            # Find the longest path
            longest_path = max(all_paths, key=len)
            st.write(f"Longest path length: {len(longest_path) - 1}")

            # Visualize the shortest path
            fig_path = visualize_graph_path(G, source, target)
            if fig_path is not None:
                st.plotly_chart(fig_path, use_container_width=True)
        else:
            st.warning("No path exists between the selected nodes.")

    # Add property-based analysis
    st.subheader("Node Property Analysis")
    selected_property = st.selectbox("Select a property to analyze", property_names)
    analysis_results = analyze_node_properties(G, selected_property)
    st.write(analysis_results)

    # Add property-based visualization
    st.subheader("Property-based Visualization")
    size_property = st.selectbox(
        "Select a property for node sizing", ["None"] + property_names
    )
    if size_property == "None":
        size_property = None
    fig_property = visualize_graph_hierarchical_plotly(G, size_property)
    st.plotly_chart(fig_property, use_container_width=True)

    st.subheader("Parent Nodes Visualization")
    selected_node_parents = st.selectbox(
        "Select a node to view its parents", list(G.nodes), key="parent_select"
    )
    if st.button("Visualize Parents", key="visualize_parents"):
        parents = get_all_parents(G, selected_node_parents)
        st.write(f"All predecessors of {selected_node_parents}:")
        st.write(parents)
        fig_parents = visualize_subgraph(G, parents, selected_node_parents)
        st.plotly_chart(fig_parents, use_container_width=True)

        # Add property statistics for parents
        st.write("Property statistics for parent nodes:")
        for prop in property_names:
            parent_values = [G.nodes[node][prop] for node in parents]
            st.write(
                f"{prop}: Mean = {np.mean(parent_values):.2f}, Std = {np.std(parent_values):.2f}"
            )

    st.subheader("Child Nodes Visualization")
    selected_node_children = st.selectbox(
        "Select a node to view its children", list(G.nodes), key="child_select"
    )
    if st.button("Visualize Children", key="visualize_children"):
        children = get_all_children(G, selected_node_children)
        st.write(f"All descendants of {selected_node_children}:")
        st.write(children)
        fig_children = visualize_subgraph(G, children, selected_node_children)
        st.plotly_chart(fig_children, use_container_width=True)

        # Add property statistics for children
        st.write("Property statistics for child nodes:")
        for prop in property_names:
            child_values = [G.nodes[node][prop] for node in children]
            st.write(
                f"{prop}: Mean = {np.mean(child_values):.2f}, Std = {np.std(child_values):.2f}"
            )

    # Add this new section for critical path analysis
    st.subheader("Critical Path Analysis")
    if st.button("Find Critical Path"):
        critical_path = find_critical_path(G)
        st.write("Critical Path (longest path from any root to any leaf):")
        st.write(" -> ".join(critical_path))
        st.write(f"Critical Path Length: {len(critical_path) - 1}")

        # Visualize the critical path
        fig_critical_path = visualize_graph_path(
            G, critical_path[0], critical_path[-1], path=critical_path
        )
        if fig_critical_path is not None:
            st.plotly_chart(fig_critical_path, use_container_width=True)

    st.subheader("GraphML Output")
    with open("synthesized_graph.graphml") as f:
        st.download_button("Download GraphML", f)

    # Calculate and display graph statistics
    st.subheader("Graph Statistics")
    st.write(f"Total Nodes: {G.number_of_nodes()}")
    st.write(f"Total Edges: {G.number_of_edges()}")
    st.write(
        f"Average Children per Node: {G.number_of_edges() / G.number_of_nodes():.2f}"
    )
    st.write(f"Maximum Children: {max(dict(G.out_degree()).values())}")
    st.write(f"Minimum Children: {min(dict(G.out_degree()).values())}")
    st.write(f"Graph Depth: {max(nx.get_node_attributes(G, 'layer').values()) + 1}")

    # Add profiling results at the end of the app
    st.subheader("Profiling Results")
    if "profiling_results" in st.session_state:
        results_df = pd.DataFrame(st.session_state.profiling_results)
        st.dataframe(results_df)

        # Create a bar chart for execution times
        fig_time = go.Figure(
            data=[go.Bar(x=results_df["function"], y=results_df["execution_time"])]
        )
        fig_time.update_layout(
            title="Execution Time by Function",
            xaxis_title="Function",
            yaxis_title="Time (seconds)",
        )
        st.plotly_chart(fig_time)

        # Create a bar chart for memory usage
        fig_memory = go.Figure(
            data=[go.Bar(x=results_df["function"], y=results_df["memory_used"])]
        )
        fig_memory.update_layout(
            title="Memory Usage by Function",
            xaxis_title="Function",
            yaxis_title="Memory (MB)",
        )
        st.plotly_chart(fig_memory)

    # Clear profiling results for the next run
    st.session_state.profiling_results = []


if __name__ == "__main__":
    main()
