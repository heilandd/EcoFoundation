## Plot Functions
import matplotlib.pyplot as plt
import networkx as nx
import random
import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc
import tangram as tg
import anndata as ad
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

######################################################################### EcoFoundation Plots  #########################################################################

def plotSpatialGraph(adata, subgraph=None, color=False, filter_dist=False, linewidth_sub=0.5):
    coords = adata.obsm["spatial"]  # Spatial coordinates
    barcode_mapping = adata.uns["barcode_mapping"]  # Mapping of graph IDs to barcodes
    G = adata.uns["Graph"]  # Main graph

    # Reverse barcode mapping for easier lookup of coordinates by barcode
    reverse_barcode_mapping = {v: int(k) for k, v in barcode_mapping.items()}

    # Filter edges based on distance if specified
    if filter_dist is not False:
        filtered_edges = [
            (u, v, data) for u, v, data in G.edges(data=True) 
            if data.get('distance', float('inf')) <= filter_dist
        ]
        filtered_G = nx.DiGraph()  
        filtered_G.add_edges_from(filtered_edges)
        G = filtered_G

    plt.figure(figsize=(12, 10))
    
    # Plot spatial scatter if color is provided
    if color is not False:
        library_id = adata.uns.get("spatial", {}).get("library_id", None)
        if library_id is None:
            print("WARNING: Please specify a valid `library_id` in `adata.uns['spatial']`.")
        else:
            sq.pl.spatial_scatter(adata, shape=None, color=color, wspace=0.4, ax=plt.gca(), library_id=library_id)
    
    # Plot the full graph edges
    for u, v in G.edges():
        # Convert numeric edges to barcodes using the mapping
        u_barcode = barcode_mapping.get(str(u), None)
        v_barcode = barcode_mapping.get(str(v), None)

        # Ensure the barcodes exist in the reverse mapping and coordinates
        if u_barcode in reverse_barcode_mapping and v_barcode in reverse_barcode_mapping:
            coord_u = coords[reverse_barcode_mapping[u_barcode]]
            coord_v = coords[reverse_barcode_mapping[v_barcode]]
            
            # Draw a line between the points
            plt.plot([coord_u[0], coord_v[0]], [coord_u[1], coord_v[1]], color='gray', linewidth=0.5)

    # If a subgraph is provided, highlight its edges in red
    if subgraph is not None:
        for u, v in subgraph.edges():
            # Convert numeric edges to barcodes using the mapping
            u_barcode = barcode_mapping.get(str(u), None)
            v_barcode = barcode_mapping.get(str(v), None)

            # Ensure the barcodes exist in the reverse mapping and coordinates
            if u_barcode in reverse_barcode_mapping and v_barcode in reverse_barcode_mapping:
                coord_u = coords[reverse_barcode_mapping[u_barcode]]
                coord_v = coords[reverse_barcode_mapping[v_barcode]]

                # Draw a line between the points in red
                plt.plot([coord_u[0], coord_v[0]], [coord_u[1], coord_v[1]], color='red', linewidth=linewidth_sub)

    plt.xticks(fontsize=8)  
    plt.yticks(fontsize=8)  
    plt.xlabel("X coordinate", fontsize=10)  
    plt.ylabel("Y coordinate", fontsize=10)  
    plt.title("Spatial Data with Connectivity", fontsize=12)  
    plt.show()
def plotSubgraphs(adata, subgraphs, coords_key="spatial"):
    """
    Plot each subgraph in a different color on top of the spatial coordinates in adata.
    Moves legend outside of the main plot area.

    Parameters:
    adata (AnnData): AnnData object containing spatial data.
    subgraphs (list of nx.Graph): List of subgraphs to plot.
    coords_key (str): Key in adata.obsm where spatial coordinates are stored.
    """
    coords = adata.obsm[coords_key]  # Spatial coordinates
    plt.figure(figsize=(12, 10))
    
    # Plot the main spatial scatter as background
    plt.scatter(coords[:, 0], coords[:, 1], color='lightgray', s=5, label="All Cells", alpha=0.5)
    
    # Assign a color to each subgraph and plot
    for i, subgraph in enumerate(subgraphs):
        color = f"C{i % 10}"  # Cycling through 10 distinct matplotlib colors
        nodes = list(subgraph.nodes)
        
        # Plot the nodes of the subgraph
        plt.scatter(coords[nodes, 0], coords[nodes, 1], color=color, s=10, label=f"Subgraph {i+1}")
        
        # Plot edges of the subgraph
        for u, v in subgraph.edges():
            u_coord, v_coord = coords[u], coords[v]
            plt.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], color=color, linewidth=0.5)
    
    # Position the legend outside the main plot area
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize="small", markerscale=1.5, frameon=False)
    
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Spatial Data with Subgraph Connectivity")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to give room for the legend
    plt.show()



