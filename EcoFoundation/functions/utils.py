import pickle
import torch
import pandas as pd
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
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
#from skmisc.loess import loess

######################################################################### EcoFoundation Object  #########################################################################
# Define the custom class
class EcoFoundationObject:
    
    class_attribute = "This is a EcoFoundation Object with meta data information in the first slot and graphs stored as pytorch geometric objects in the 2nd slot"
    
    def __init__(self, meta, graph):
        self.meta = meta
        self.graph = graph
    
    #Save the object to a file using pickle.
    def save(self, filename):
      """Save EcoFoundationObject from a file."""
      with open(filename, 'wb') as f:
        pickle.dump(self, f)
      print(f'Object stored in {filename}')

    @classmethod
    def load(cls, filename):
      """Load EcoFoundationObject from a file."""
      with open(filename, 'rb') as f:
        obj = pickle.load(f)
      print(f'Object loaded from {filename}')
      return obj
    
    @classmethod
    def get_info(cls):
        return cls.class_attribute
    
    ## Filter function
    def filter(self, condition):
        """
        Filter the DataFrame based on the condition and 
        automatically filter the corresponding tensors.

        :param condition: A boolean mask or a query string to filter the DataFrame.
        :return: A new instance of EcoFoundation with filtered DataFrame and tensors.
        """

        if isinstance(condition, str):
            filtered_df = self.meta.query(condition)
        else:
            filtered_df = self.meta[condition]
        
        filtered_indices = filtered_df.index.tolist()
        
        # Ensure indices are within the valid range
        if max(filtered_indices) >= len(self.graph):
            raise IndexError("Filtered indices exceed the length of the graph list.")

        # Re-index the DataFrame to ensure consistency in sequential filters
        filtered_df = filtered_df.reset_index(drop=True)
        
        # Filter the graph list
        filtered_tensors = [self.graph[i] for i in filtered_indices]

        # Return a new instance with the filtered DataFrame and graph list
        return EcoFoundationObject(filtered_df, filtered_tensors)
    
    def print_summary(self):
      """Print summary information about the EcoFoundationObject."""
      num_features = self.meta.shape[1]  # Number of columns in meta DataFrame
      num_subgraphs = len(self.graph)  # Number of subgraphs in the graph list
      print(f"Number of features in meta: {num_features}")
      print(f"Number of subgraphs in the object: {num_subgraphs}")
    
    def __repr__(self):
        """Return a string representation of the EcoFoundationObject."""
        num_features = self.meta.shape[1]  # Number of columns in meta DataFrame
        num_subgraphs = len(self.graph)  # Number of subgraphs in the graph list
        return (f"<EcoFoundationObject with {num_features} features in meta and "
                f"{num_subgraphs} subgraphs in the object>")

## Functions for the ecoClass Object
# This function is build to take a class variable from the meta data and add this as a class in the PYG list
def add_class_variable(EcoFoundationObject, column_name):
    """
    Extracts the specified column from the meta DataFrame of the EcoFoundationObject,
    converts the values to the appropriate data type (integer or float), and adds 
    these as a `.Class` attribute to the corresponding tensors in the graph list.
    
    :param deep_spata_obj: An instance of EcoFoundationObject.
    :param column_name: The name of the column to be added as the .Class attribute.
    :return: None, modifies the graph tensors in-place.
    """
    # Extract the specified column
    variable = EcoFoundationObject.meta[column_name]
    
    # Check the data type and convert if necessary
    if variable.dtype.kind == 'O':  
        # Convert string to categorical codes (integers)
        variable = variable.astype('category').cat.codes
    elif np.issubdtype(variable.dtype, np.number):
        pass
    else:
        raise ValueError(f"Unsupported data type for column {column_name}")
    
    # Convert to appropriate tensor type based on the dtype
    if np.issubdtype(variable.dtype, np.integer):
        tensor_data = torch.as_tensor(np.asarray(variable, dtype="int8"), dtype=torch.float)
    else:
        tensor_data = torch.as_tensor(np.asarray(variable, dtype="float32"), dtype=torch.float)
    
    # Iterate over the graph tensors and add the .Class attribute
    for i, graph in enumerate(EcoFoundationObject.graph):
        graph.Class = tensor_data[i]
    
    return(EcoFoundationObject)
# This function is build to take a class variable from the meta data and add this as a class in the PYG list
def mergeEco(EcoFoundationObject):
    """
    Merges a list of EcoFoundationObject instances into a single EcoFoundationObject.

    :param eco_objects: List of EcoFoundationObject instances to merge.
    :return: A new EcoFoundationObject with combined meta and graph attributes.
    """
    
    # Check if list is not empty
    if not EcoFoundationObject:
        raise ValueError("The list of eco objects is empty.")
    
    # Concatenate meta DataFrames
    meta_combined = pd.concat([eco.meta for eco in EcoFoundationObject], ignore_index=True)
    
    # Concatenate graph lists
    graph_combined = []
    for eco in EcoFoundationObject:
        graph_combined.extend(eco.graph)
    
    # Create and return a new EcoFoundationObject with merged data
    return EcoFoundationObject(meta=meta_combined, graph=graph_combined)





######################################################################### ConFuns #########################################################################

## To stor adata the G needs to be converted to a DF
def Graph2DF(adata):
    """Converts a networkx graph in adata.uns['Graph'] to a DataFrame and stores it in adata.uns['Graph_DF']."""
    G = adata.uns.get('Graph', None)
    if G is None or not isinstance(G, nx.Graph):
        raise ValueError("adata.uns['Graph'] must be a valid networkx graph.")
    
    # Extract edges and associated data
    edges = [(u, v, data) for u, v, data in G.edges(data=True)]
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'data'])
    
    # Separate 'data' dictionary into individual columns
    adata.uns['Graph_DF'] = split_data_column(edges_df, 'data')
    
    # Remove original graph from adata.uns
    del adata.uns['Graph']
    return adata
def DF2Graph(adata):
    """Converts a DataFrame in adata.uns['Graph_DF'] back to a networkx graph and stores it in adata.uns['Graph']."""
    edges_df = adata.uns.get('Graph_DF', None)
    if edges_df is None or not isinstance(edges_df, pd.DataFrame):
        raise ValueError("adata.uns['Graph_DF'] must be a valid DataFrame.")
    
    # Reconstruct the graph
    G_reconstructed = nx.from_pandas_edgelist(edges_df, 'source', 'target', edge_attr=True, create_using=nx.DiGraph())
    adata.uns['Graph'] = G_reconstructed
    
    # Remove the DataFrame from adata.uns
    del adata.uns['Graph_DF']
    return adata
def split_data_column(df, col_name):
    """Splits a column of dictionaries in a DataFrame into individual columns."""
    if col_name in df:
        data_df = pd.DataFrame(df[col_name].tolist())
        df = pd.concat([df.drop(columns=[col_name]), data_df], axis=1)
    return df
## Add barcode mapping
def barcode_mapping(adata):
  barcodes = adata.obs.index  # Barcodes or indices corresponding to the coordinates
  barcode_mapping = {i: barcode for i, barcode in enumerate(barcodes)}
  barcode_mapping = {str(key): value for key, value in barcode_mapping.items()}
  adata.uns["barcode_mapping"] = barcode_mapping  
  return(adata)
## Preprocess adata (from scanpy)
def RunPreprocessing(adata):
  print("Step1/5: FindVarGenes")
  sc.pp.highly_variable_genes(adata, n_top_genes=300)
  print("Step2/5: LogNorm")
  sc.pp.normalize_total(adata, inplace=True)
  sc.pp.log1p(adata)
  print("Step3/5: PCA")
  sc.pp.pca(adata)
  sc.pp.neighbors(adata,n_neighbors=10, n_pcs=10)
  print("Step4/5: NN-Anaysis")
  sc.tl.umap(adata)
  print("Step5/5: Cluster")
  sc.tl.leiden(adata)
  adata.layers['norm'] = adata.X.copy()
  return(adata)




######################################################################## Spatial Graph tools #########################################################################
## Build Spatial Graph from adata
def BuildGraph(adata, distance_threshold = 20):
  sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
  G = nx.from_numpy_array(adata.obsp["spatial_connectivities"].toarray())
  ## Add distance:
  distance_matrix = adata.obsp["spatial_distances"].toarray()  
  connectivity_matrix = adata.obsp["spatial_connectivities"]
  for i, j in zip(*connectivity_matrix.nonzero()):  
      distance = distance_matrix[i, j]
      G[i][j]['distance'] = distance 
  
  ## Distance filter 
  filtered_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data.get('distance', float('inf')) >= distance_threshold]
  filtered_G = nx.DiGraph()  
  filtered_G.add_edges_from(filtered_edges)
  
  adata.uns["Graph"] = filtered_G
  return(adata)

## Build Subgraph 
def create_hop_subgraph(adata, barcode, hops=6):
    G = adata.uns["Graph"]
    barcode_mapping = adata.uns["barcode_mapping"]
    
    # Verify if the barcode exists in the mapping
    if barcode not in barcode_mapping.values():
        raise ValueError(f"Barcode {barcode} not found in the barcode mapping.")
    
    # Retrieve the graph ID corresponding to the barcode
    graph_id = next(k for k, v in barcode_mapping.items() if v == barcode)
    
    # Get nodes within the specified hop distance
    subnodes = nx.single_source_shortest_path_length(G, int(graph_id), cutoff=hops)
    subgraph = G.subgraph(subnodes.keys()).copy()  # Create a copy to avoid view issues
    
    return subgraph

## Build Non-Overlapping Subgraphs
def getMaxSubgraphs(adata, hop=6, max_overlap=2, min_nodes=20):
    """
    Generate non-overlapping subgraphs of fixed hop neighborhoods for each node in G.

    Parameters:
    G (nx.Graph): The input graph.
    fixed_hop (int): The exact number of hops for each neighborhood.
    max_overlap (int): Maximum allowed overlap of nodes between selected subgraphs.

    Returns:
    list of nx.Graph: A list of non-overlapping subgraphs.
    """
    G=adata.uns["Graph"]
    all_subgraphs = {}
    selected_subgraphs = []
    used_nodes = set()

    # Step 1: Generate all fixed-hop subgraphs for each node
    for node in G.nodes:
        # Get nodes exactly at `fixed_hop` distance from the starting node
        hop_nodes = nx.single_source_shortest_path_length(G, node, cutoff=hop)
        
        if len(hop_nodes) >= min_nodes:
            subgraph = G.subgraph(hop_nodes).copy()
            all_subgraphs[node] = subgraph  # Store by central node
            

    # Step 2: Select non-overlapping subgraphs with greedy approach
    for node, subgraph in all_subgraphs.items():
        # Calculate overlap of this subgraph with already selected nodes
        overlap_count = len(set(subgraph.nodes) & used_nodes)

        # Check if overlap is within the allowed limit
        if overlap_count <= max_overlap:
            selected_subgraphs.append(subgraph)
            # Add nodes of this subgraph to used nodes
            used_nodes.update(subgraph.nodes)

    return selected_subgraphs

## Build Full Subgraphs
def getfullSubgraphs(adata, hop=6, min_nodes=20):
    """
    Generate non-overlapping subgraphs of fixed hop neighborhoods for each node in G.

    Parameters:
    G (nx.Graph): The input graph.
    fixed_hop (int): The exact number of hops for each neighborhood.
    max_overlap (int): Maximum allowed overlap of nodes between selected subgraphs.

    Returns:
    list of nx.Graph: A list of non-overlapping subgraphs.
    """
    G=adata.uns["Graph"]
    all_subgraphs = {}
    selected_subgraphs = []
    nodes_all = []
    used_nodes = set()

    # Step 1: Generate all fixed-hop subgraphs for each node
    for node in G.nodes:
        # Get nodes exactly at `fixed_hop` distance from the starting node
        hop_nodes = nx.single_source_shortest_path_length(G, node, cutoff=hop)
        
        if len(hop_nodes) >= min_nodes:
            subgraph = G.subgraph(hop_nodes).copy()
            all_subgraphs[node] = subgraph  # Store by central node
        selected_subgraphs.append(subgraph)
        nodes_all.append(node)
    
    return selected_subgraphs, nodes_all
  
## Build PYG tensors
def BuildPYG(adata, subgraphs):
    """
    Converts each NetworkX subgraph into a PyTorch Geometric `Data` object and adds expression data as node features.

    Parameters:
    adata (AnnData): AnnData object containing the expression data.
    subgraphs (list of nx.Graph): List of NetworkX subgraphs.

    Returns:
    list of torch_geometric.data.Data: List of PyTorch Geometric Data objects with expression features.
    """
    pyg_data_list = []
    
    # Loop through each subgraph
    for subgraph in subgraphs:
        # Convert NetworkX subgraph to PyTorch Geometric Data object
        ptg_data = from_networkx(subgraph)
        
        # Get node indices for this subgraph
        node_indices = list(subgraph.nodes)

        # Extract expression data for the nodes in this subgraph from adata.X
        if isinstance(adata.X, np.ndarray):
          expression_data = adata.X[node_indices]
        else:
          expression_data = adata.X[node_indices].toarray()  # Converts sparse matrix to dense if needed
        
        # Convert expression data to a torch tensor and add as node features
        ptg_data.x = torch.tensor(expression_data, dtype=torch.float)
        
        ## Add node indices
        ptg_data.node_index = torch.tensor(np.asarray(list(subgraph.nodes)), dtype=torch.int)
        
        # Append to the list
        pyg_data_list.append(ptg_data)
      
    return pyg_data_list

## Build Meta Data file
def BuildMeatData(subgraphs, sample_id, required_pram, metadata_df):
    """
    Create a metadata DataFrame for each subgraph.

    Parameters:
    - metadata_df (pd.DataFrame): The original metadata DataFrame with sample-level clinical data.
    - subgraphs (list): A list of subgraphs (from `getMaxSubgraphs` function).
    - sample_ids (list): List of sample IDs corresponding to each subgraph.

    Returns:
    - pd.DataFrame: Metadata DataFrame with unique subgraph ID, size, sample ID, and clinical features.
    """
    subgraph_metadata = []
    

    for i, subgraph in enumerate(subgraphs):

        subgraph_size = len(subgraph.nodes)  # Size of the current subgraph

        # Retrieve clinical features from metadata for the sample ID
        extracted_info = metadata_df[required_pram]
        sample_meta = extracted_info[extracted_info['ID'] == sample_id].iloc[0]

        # Create a unique ID for the subgraph, e.g., "SampleID_Subgraph#"
        unique_subgraph_id = f"{sample_id}_subgraph_{i+1}"

        # Append a dictionary for this subgraph's metadata
        subgraph_metadata.append({
            'Subgraph_ID': unique_subgraph_id,
            'Size': subgraph_size,
            'Sample_ID': sample_id,
            **sample_meta.to_dict()  # Include all clinical features
        })

    # Convert to DataFrame
    return pd.DataFrame(subgraph_metadata)






## Depricated
## Two functions not used any more
def PYG2DIC(adata):
  adata.uns['ecoPYG_DIC'] = [
      {
          'edge_index': data.edge_index.numpy(),
          'weight': data.weight.numpy(),
          'distance': data.distance.numpy(),
          'num_nodes': data.num_nodes,
          'x': data.x.numpy()
      }
      for data in adata.uns['ecoPYG']
  ]
  del adata.uns['ecoPYG']
  return(adata)
def DIC2PYG(adata):
  adata.uns['ecoPYG'] = [
    Data(
        edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
        weight=torch.tensor(d['weight'], dtype=torch.float),
        distance=torch.tensor(d['distance'], dtype=torch.float),
        num_nodes=d['num_nodes'],
        x=torch.tensor(d['x'], dtype=torch.float)
    )
    for d in adata.uns['ecoPYG_DIC']
  ]
  del adata.uns['ecoPYG_DIC']
  return(adata)
  

