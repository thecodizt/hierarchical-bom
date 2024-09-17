import streamlit as st
import csv
import json
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

    st.subheader("Path Visualization")
    # use select box to select source and target node
    source = st.selectbox("Source Node", list(G.nodes))
    target = st.selectbox("Target Node", list(G.nodes))
    if st.button("Visualize Path"):
        fig_path = visualize_graph_path(G, source, target)
        if fig_path is not None:
            st.plotly_chart(fig_path, use_container_width=True)

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


if __name__ == "__main__":
    main()
