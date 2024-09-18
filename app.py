import streamlit as st
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
)

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


def main():
    total_nodes, total_layers, distribution, clear_cut_layer, layer_names = inputs()

    # Synthesize the graph
    G = synthesize_graph(
        total_nodes=total_nodes,
        total_layers=total_layers,
        distribution=distribution,
        clear_cut_layer=clear_cut_layer,
        layer_names=layer_names.split(",") if layer_names else None,
    )

    # G = synthesize_graph(
    #     total_nodes=100,
    #     total_layers=12,
    #     distribution="pos_exp",
    #     clear_cut_layer=5,
    #     layer_names=[
    #         "Organization",
    #         "Business Group",
    #         "Product Family",
    #         "Product Offering",
    #         "Modules",
    #         "Parts",
    #     ],
    # )

    # Save to JSON
    save_graph_to_json(G, "synthesized_graph.json")

    # Save to CSV
    save_graph_to_csv(G, "nodes.csv", "edges.csv")

    # Visualize the graph
    fig1 = visualize_graph(G)

    fig = visualize_graph_hierarchical_plotly(G)

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
    save_to_graphml(G, "synthesized_graph.graphml")
    with open("synthesized_graph.graphml") as f:
        st.download_button("Download GraphML", f)

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


if __name__ == "__main__":
    main()
