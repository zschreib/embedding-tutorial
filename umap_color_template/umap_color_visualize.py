import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
import pandas as pd
from matplotlib.colors import to_hex
import seaborn as sns

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

def custom_color_auto(metadata_df, column_name):

    #Assign pallete to reflect your unique label count.
    palette='tab10'

    #colors by input column found in metadata (you can change this to match what you want in your file)
    if column_name not in metadata_df.columns:
        raise ValueError(f"'{column_name}' column not found in the metadata file.")

    # Get unique values
    unique_values = metadata_df[column_name].unique()

    # Generate colors using seaborn palette
    num_colors = len(unique_values)
    colors = sns.color_palette(palette, num_colors)

    # Map unique values to hex colors
    auto_colors = {value: to_hex(color) for value, color in zip(unique_values, colors)}

    return auto_colors

def custom_color_manual(metadata_df, column_name):
    # Predefined color table
    color_table = {
        "RDKF": '#ec4400',
        "RDKL": '#8c29b1',
        "RDKY": '#1b33e3',
        "RDKH": '#008856',
        # Add more entries as needed
    }

    # Ensure the column exists
    if column_name not in metadata_df.columns:
        raise ValueError(f"'{column_name}' column not found in the metadata DataFrame.")

    unique_values = metadata_df[column_name].dropna().unique()
    # Map colors using the color table, default to gray for unmapped values
    manual_colors = {val: color_table.get(val, "#808080") for val in unique_values}

    return manual_colors

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

def plot_umap(embeddings, embedding_ids, metadata, output_file):

    #maps your ids to color dictionary.
    id_to_color = dict(zip(metadata['Query_ID'], metadata['manual_color']))
    colors = [id_to_color.get(embedding_id, "#808080") for embedding_id in embedding_ids]

    # Create and fit the UMAP mapper
    mapper = umap.UMAP(n_neighbors=3, min_dist=0.5, n_components=2, metric='euclidean').fit(embeddings)

    # Plot UMAP
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=colors, s=10, alpha=0.7)

    # Save the plot
    plt.title("UMAP Projection with colors")
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python umap_color_visualize.py <input_directory> <metadata.tsv> <column name> <output_plot_file>")
        sys.exit(1)

    input_dir, metadata_file, column_name, output_path = sys.argv[1:]
    embeddings, embedding_ids = load_all_embeddings(input_dir)

    metadata_df = load_metadata(metadata_file)

    if metadata_df is not None:

        auto_colors = custom_color_auto(metadata_df, column_name)
        manual_colors = custom_color_manual(metadata_df, column_name)

        #Save to your metadata file for future use
        metadata_df['auto_color'] = metadata_df[column_name].map(auto_colors)
        metadata_df['manual_color'] = metadata_df[column_name].map(manual_colors)

        print("Auto Generated Signature Colors:", auto_colors)
        print("Manually Generated Signature Colors:", manual_colors)

    if len(embeddings) > 0:
        plot_umap(embeddings, embedding_ids, metadata_df, output_path)
    else:
        print("No valid embeddings found.")

