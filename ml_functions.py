import os
import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from datetime import datetime, timedelta
import json
from tqdm import tqdm


def load_images_and_labels(dataset_path, image_size=(64, 64)):
    images, labels = [], []
    categories = os.listdir(dataset_path)

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for img_name in tqdm(os.listdir(category_path), desc=f"Loading {category}"):
                img_path = os.path.join(category_path, img_name)
                try:
                    img = Image.open(img_path).resize(image_size).convert("RGB")
                    images.append(np.array(img).flatten())
                    labels.append(category)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), LabelEncoder().fit_transform(labels), categories


def generate_umap_embedding(images, random_state=42):
    umap_reducer = umap.UMAP(n_components=2, random_state=random_state)
    return umap_reducer.fit_transform(images)


def generate_tsne_embedding(images, random_state=42):
    """
    Generate TSNE embedding with dynamically adjusted perplexity.
    Args:
        images (np.ndarray): Input image data (flattened).
        random_state (int): Random state for reproducibility.
    Returns:
        np.ndarray: TSNE embeddings.
    """
    n_samples = images.shape[0]
    if n_samples < 3:
        raise ValueError("Not enough samples for TSNE. Minimum 3 samples are required.")
    perplexity = min(30, max(1, n_samples // 3))  # Adjust perplexity dynamically
    print(f"Using perplexity: {perplexity} for {n_samples} samples")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(images)


def create_individual_sprites(image_path, output_folder, sprite_width=64, sprite_height=64):
    start = datetime.now()
    sprite_metadata = {}

    for root, _, files in os.walk(image_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                original_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(original_file_path, image_path)
                sprite_save_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(sprite_save_path), exist_ok=True)

                try:
                    img = Image.open(original_file_path).convert('RGB')
                    img_resized = img.resize((sprite_width, sprite_height))
                    img_resized.save(sprite_save_path)
                    sprite_metadata[relative_path] = {
                        "sprite_path": sprite_save_path,
                        "width": sprite_width,
                        "height": sprite_height
                    }
                except Exception as e:
                    print(f"Error processing image {original_file_path}: {e}")

    print(f"Sprite creation time: {datetime.now() - start}")
    return sprite_metadata


def save_embeddings(embedding, labels, categories, sprite_metadata, output_path="embeddings.json"):
    data = {
        "items": [
            {
                "embedding": embedding[i].tolist(),
                "label": int(labels[i]),
                "category": categories[labels[i]],
                "sprite": sprite_metadata[list(sprite_metadata.keys())[i]]
            }
            for i in range(len(embedding))
        ],
        "categories": categories
    }
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data successfully saved to {output_path}")
    return data


def load_embeddings_from_file(file_path="embeddings.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}


def is_embedding_recent(embedding_data, embedding_type, max_age_hours=24):
    if embedding_type not in embedding_data:
        return False
    timestamp = datetime.fromisoformat(embedding_data[embedding_type]["timestamp"])
    return datetime.now() - timestamp < timedelta(hours=max_age_hours)


def get_or_create_embeddings(images, labels, categories, sprite_metadata, embedding_type):
    embedding_data = load_embeddings_from_file()
    if not is_embedding_recent(embedding_data, embedding_type):
        if embedding_type == "umap":
            embedding = generate_umap_embedding(images)
        elif embedding_type == "tsne":
            embedding = generate_tsne_embedding(images)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
        embedding_data[embedding_type] = save_embeddings(embedding, labels, categories, sprite_metadata)
        with open("embeddings.json", "w") as file:
            json.dump(embedding_data, file)
    return embedding_data[embedding_type]

def create_sprites(image_path, output_folder, sprite_width=64, sprite_height=64):
    """
    Creates individual sprites for each image and saves them in the output folder.
    Args:
        image_path (str): Path to the folder containing images.
        output_folder (str): Path to save the sprites.
        sprite_width (int): Width of each sprite.
        sprite_height (int): Height of each sprite.
    Returns:
        dict: Metadata for each sprite, including path and dimensions.
    """
    start = datetime.now()
    sprite_metadata = {}

    # Gather all image files from the input path
    for root, _, files in os.walk(image_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                original_file_path = os.path.join(root, file)

                # Create corresponding output path
                relative_path = os.path.relpath(original_file_path, image_path)
                sprite_save_path = os.path.join(output_folder, relative_path)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(sprite_save_path), exist_ok=True)

                try:
                    # Process the image
                    img = Image.open(original_file_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_resized = img.resize((sprite_width, sprite_height))
                    img_resized.save(sprite_save_path)

                    # Add metadata for the sprite
                    sprite_metadata[relative_path] = {
                        "sprite_path": sprite_save_path,
                        "width": sprite_width,
                        "height": sprite_height,
                    }
                except Exception as e:
                    print(f"Error processing image {original_file_path}: {e}")

    print(f"Sprite creation completed in {datetime.now() - start}")
    return sprite_metadata


dataset_path = 'static/2'
images, labels, categories = load_images_and_labels(dataset_path)
sprite_output_path = f'static/dump/tsne/2/{os.path.basename(dataset_path)}'
sprites = create_individual_sprites(dataset_path, sprite_output_path, 64, 64)
embedding_type = "tsne"

embeddings_data = get_or_create_embeddings(images, labels, categories, sprites, embedding_type)