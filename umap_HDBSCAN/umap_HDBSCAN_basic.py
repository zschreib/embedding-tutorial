import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
import hdbscan

def load_embedding(file_path):
    try:
        embedding_data = torch.load(file_path)
        embedding = embedding_data["mean_representations"][36].numpy()
        embedding_id = embedding_data.get("label")
        if embedding_id is None:
            raise ValueError(f"Embedding ID not found in file: {os.path.basename(file_path)}")
        return embedding, embedding_id
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

def load_metadata(metadata_file):
    try:
        return pd.read_csv(metadata_file, sep='\t')
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return None
        
def custom_shape_manual(values):
    shape_table = {
        "RDKF": 's', "RDKL": 'p', "RDKY": 'D', "RDKH": '*'
    }
    return {val: shape_table.get(val, 'X') for val in np.unique(values)}

def load_all_embeddings(root_dir):
    embeddings, embedding_ids = [], []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pt"):
                file_path = os.path.join(dirpath, filename)
                embedding, embedding_id = load_embedding(file_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    embedding_ids.append(embedding_id)
    return np.array(embeddings), embedding_ids

def plot_umap_hdbscan(embeddings, embedding_ids, metadata, output_file):
    
    # Perform UMAP dimensionality reduction
    standard_embedding = umap.UMAP(n_neighbors=3, min_dist=0.2, n_components=2, metric='euclidean', random_state=42).fit_transform(embeddings)

    # Perform clustering with HDBSCAN
    labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=3).fit_predict(embeddings)
    clustered = (labels >= 0)
    
    # Generate shapes based on metadata
    metadata_shapes = custom_shape_manual(metadata["signature"])
    id_to_signature = dict(zip(metadata["Query_ID"], metadata["signature"]))
    
    # Ensure all embedding IDs are in the metadata
    valid_embeddings = []
    valid_markers = []
    for eid in embedding_ids:
        signature = id_to_signature.get(eid, None)
        if signature is not None:
            valid_embeddings.append(eid)
            valid_markers.append(metadata_shapes.get(signature, "X"))
        else:
            valid_markers.append("X")  # Default shape if not in metadata

    markers = np.array(valid_markers)

    # Plotting the UMAP results with HDBSCAN clusters
    plt.figure(figsize=(10, 7))
    plt.scatter(
        standard_embedding[~clustered, 0],
        standard_embedding[~clustered, 1],
        color="gray",
        s=20,
        alpha=0.5,
        label="Noise",
        marker="1" #tri_down for unclustered data
    )
    
    #Shapes applied to clustered data only (can modify to all if needed)
    unique_markers = set(markers)
    for marker in unique_markers:
        #Shape to embedding id
        marker_mask = np.array([(m == marker) and clustered[i] for i, m in enumerate(markers)])
        plt.scatter(
            standard_embedding[marker_mask, 0],
            standard_embedding[marker_mask, 1],
            c=labels[marker_mask],
            s=20,
            cmap="Spectral",
            marker=marker,
            alpha=0.8
        )

    # Format plot
    plt.title("UMAP with HDBSCAN Basic Clustering")
    plt.legend(markerscale=2, fontsize="small", loc="best")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python umap_HDBSCAN_basic.py <input_directory> <metadata.tsv> <output_plot_file>")
        sys.exit(1)

    input_dir, metadata_file, output_path = sys.argv[1:]
    embeddings, embedding_ids = load_all_embeddings(input_dir)
    metadata_df = load_metadata(metadata_file)

    if metadata_df is not None and len(embeddings) > 0:
        plot_umap_hdbscan(embeddings, embedding_ids, metadata_df, output_path)
    else:
        print("No valid embeddings or metadata found.")
