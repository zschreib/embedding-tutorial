import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot

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
    embeddings = []
    embedding_ids = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pt"):
                file_path = os.path.join(dirpath, filename)
                embedding, embedding_id = load_embedding(file_path)

                if embedding is not None:
                    embeddings.append(embedding)
                    embedding_ids.append(embedding_id)

    return np.array(embeddings), embedding_ids

def plot_umap(embeddings, embedding_ids, output_file):
    # Create and fit the UMAP mapper
    mapper = umap.UMAP(n_neighbors=3, min_dist=0.5, n_components=2, metric='euclidean').fit(embeddings)
    
    # Create the UMAP plot
    fig, ax = plt.subplots(figsize=(10, 8))
    umap.plot.points(mapper, ax=ax)
    
    # Save the plot to the output file
    plt.title("UMAP Projection")
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python umap_visualize.py <input_directory> <output_plot_file>")
        sys.exit(1)

    input_dir, output_path = sys.argv[1:]
    embeddings, embedding_ids = load_all_embeddings(input_dir)

    if len(embeddings) > 0:
        plot_umap(embeddings, embedding_ids, output_path)
    else:
        print("No valid embeddings found.")

