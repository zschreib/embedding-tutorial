import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
from matplotlib.colors import to_hex
import seaborn as sns
from hdbscan import HDBSCAN
from seaborn import color_palette
from matplotlib.lines import Line2D

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

def custom_color_auto(values):
    unique_values = np.unique(values)
    palette = color_palette("tab20", len(unique_values))
    color_map = {val: to_hex(color) for val, color in zip(unique_values, palette)}
    default_color = "#CCCCCC"  # Gray for unmapped or noise points
    return [color_map.get(val, default_color) for val in values]


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
    # Perform clustering with HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean').fit(embeddings)
    cluster_labels = clusterer.labels_

    # Generate colors and shapes
    cluster_colors = custom_color_auto(cluster_labels)
    metadata_shapes = custom_shape_manual(metadata["signature"])
    id_to_signature = dict(zip(metadata["Query_ID"], metadata["signature"]))
    markers = [metadata_shapes.get(id_to_signature.get(eid, ""), "X") for eid in embedding_ids]

    # Perform UMAP dimensionality reduction
    mapper = umap.UMAP(n_neighbors=3, min_dist=0.5, n_components=2, metric='euclidean').fit(embeddings)
    umap_embeddings = mapper.embedding_

    # Plot UMAP with cluster colors and shapes
    plt.figure(figsize=(10, 8))
    for embedding, color, marker in zip(umap_embeddings, cluster_colors, markers):
        plt.scatter(embedding[0], embedding[1], color=color, marker=marker, s=80, alpha=0.7)

    # Create legend elements
    unique_labels = sorted(set(cluster_labels))
    color_map = custom_color_auto(unique_labels)  # Get colors for unique labels

    color_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}' if label != -1 else "Noise",
               markerfacecolor=color_map[i], markersize=10)
        for i, label in enumerate(unique_labels)
    ]
    unique_shapes = sorted(set(metadata["signature"]))
    shape_legend_elements = [
        Line2D([0], [0], marker=custom_shape_manual([sig])[sig], color='k', label=sig, markersize=10)
        for sig in unique_shapes
    ]

    # Combine legends
    legend = plt.legend(
        handles=color_legend_elements + shape_legend_elements,
        loc='upper left',  # Position within the plot
        bbox_to_anchor=(1.05, 1),  # Offset outside the plot
        title='Legend',
        scatterpoints=1,
        fontsize='small'
    )

    # Adjust plot to fit the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Add extra space on the right for the legend

    # Save the plot
    plt.title("UMAP with HDBSCAN Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(output_file, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python umap_HDBSCAN.py <input_directory> <metadata.tsv> <output_plot_file>")
        sys.exit(1)

    input_dir, metadata_file, output_path = sys.argv[1:]
    embeddings, embedding_ids = load_all_embeddings(input_dir)
    metadata_df = load_metadata(metadata_file)

    if metadata_df is not None and len(embeddings) > 0:
        plot_umap_hdbscan(embeddings, embedding_ids, metadata_df, output_path)
    else:
        print("No valid embeddings or metadata found.")
