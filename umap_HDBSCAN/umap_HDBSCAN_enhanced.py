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


def plot_umap_hdbscan(embeddings, embedding_ids, output_file):

    # Perform UMAP dimensionality reduction
    standard_embedding = umap.UMAP(n_neighbors=3, min_dist=0.2, n_components=2, metric='euclidean', random_state=42).fit_transform(embeddings)
    #HDBSCAN enhanced. Adjust according to data
    clusterable_embedding = umap.UMAP(n_neighbors=6, min_dist=0.0, n_components=2, random_state=42).fit_transform(embeddings)
    #Adjust according to data
    labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=3).fit_predict(clusterable_embedding)
    #New cluster labels
    clustered = (labels >= 0)

    # Plotting the UMAP results with HDBSCAN clusters
    plt.figure(figsize=(10, 7))
    plt.scatter(
        standard_embedding[~clustered, 0],
        standard_embedding[~clustered, 1],
        color="gray",
        s=20,
        alpha=0.5,
        label="Noise"
    )
    plt.scatter(
        standard_embedding[clustered, 0],
        standard_embedding[clustered, 1],
        c=labels[clustered],
        s=20,
        cmap="Spectral",
    )

    # Format plot
    plt.title("UMAP with HDBSCAN Enhanced Clustering")
    plt.legend(markerscale=2, fontsize="small", loc="best")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python umap_HDBSCAN_enhanced.py <input_directory> <output_plot_file>")
        sys.exit(1)

    input_dir, output_path = sys.argv[1:]
    embeddings, embedding_ids = load_all_embeddings(input_dir)

    if embeddings.size == 0:
        print("No embeddings found. Please check the input directory.")
        sys.exit(1)

    plot_umap_hdbscan(embeddings, embedding_ids, output_path)

