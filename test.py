import os
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image
import json
import matplotlib.pyplot as plt


def create_sprite_sheet(image_folder, output_sprite, output_json, materials):
    """
    Create a sprite sheet with t-SNE embeddings and generate a JSON object.
    Args:
        image_folder (str): Path to the folder containing images.
        output_sprite (str): Path to save the sprite sheet image.
        output_json (str): Path to save the JSON metadata file.
        materials (list): List of materials corresponding to images.
    """
    # Load images
    images = []
    metadata = []

    for idx, filename in enumerate(sorted(os.listdir(image_folder))):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB').resize((32, 32))  # Resize for sprite
            images.append(np.asarray(img))
            material = materials[idx % len(materials)]  # Cycle through materials if fewer than images
            metadata.append({'filename': filename, 'category': material})

    images = np.array(images)  # Convert to numpy array
    images_flat = images.reshape(len(images), -1)  # Flatten images for t-SNE

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(images_flat)

    # Normalize embeddings for better visualization
    embeddings_min = embeddings.min(axis=0)
    embeddings_max = embeddings.max(axis=0)
    normalized_embeddings = (embeddings - embeddings_min) / (embeddings_max - embeddings_min)

    # Create sprite sheet
    sprite_dim = int(np.ceil(np.sqrt(len(images))))  # Dimensions of the sprite sheet
    sprite_sheet = np.zeros((sprite_dim * 32, sprite_dim * 32, 3), dtype=np.uint8)
    sprite_json = []

    for idx, (img, embedding, meta) in enumerate(zip(images, normalized_embeddings, metadata)):
        row = idx // sprite_dim
        col = idx % sprite_dim
        sprite_sheet[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = img
        sprite_json.append({
            'embedding': embedding.tolist(),
            'label': idx,  # Arbitrary unique label for each image
            'category': meta['category'],  # Material category
            'spriteX': col,
            'spriteY': row
        })

    # Save the sprite sheet
    sprite_image = Image.fromarray(sprite_sheet)
    sprite_image.save(output_sprite)

    # Save the JSON metadata
    with open(output_json, 'w') as f:
        json.dump(sprite_json, f, indent=4)


# Example usage
materials = ['wood', 'metal', 'plastic', 'glass', 'stone']  # Example material categories
create_sprite_sheet(
    image_folder='static/2',
    output_sprite='sprite_sheet1.png',
    output_json='metadata.json',
    materials=materials
)
