import os
import numpy as np
import json
from PIL import Image
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def load_images_and_labels(dataset_path, image_size=(64, 64)):
    images = []
    labels = []
    categories = os.listdir(dataset_path)

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        for img_name in tqdm(os.listdir(category_path), desc=f"Loading {category}"):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).resize(image_size).convert("RGB")
            images.append(np.array(img).flatten())  # Flatten the image
            labels.append(category)

    return np.array(images), LabelEncoder().fit_transform(labels), categories


def generate_umap_embedding(images, random_state=42):
    umap_reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_embedding = umap_reducer.fit_transform(images)
    return umap_embedding


def generate_tsne_embedding(images, random_state=42):
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_embedding = tsne.fit_transform(images)
    return tsne_embedding


def save_embeddings(embedding, labels, categories, sprite_paths,
                            output_path="embeddings_with_labels_and_sprites.json"):
    data = {
        "items": [
            {
                "embedding": embedding[i].tolist(),
                "label": int(labels[i]),
                "category": categories[labels[i]],
                "sprite_path": sprite_paths[i]
            }
            for i in range(len(labels))
        ],
        "categories": categories
    }
    return data
    # with open(output_path, "w") as json_file:
    #     json.dump(data, json_file, indent=4)
    #
    # print(f"Data successfully saved to {output_path}")