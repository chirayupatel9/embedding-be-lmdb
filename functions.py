import json
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import umap.umap_ as umap
import gridfs
from pymongo import MongoClient
from io import BytesIO
from db_config import *

# # MongoDB Configuration
# MONGO_URI = "mongodb://your_username:your_password@localhost:27017/your_database?authSource=admin"
# client = MongoClient(MONGO_URI)
# db = client["your_database"]
# fs = gridfs.GridFS(db)


def fetch_images_from_mongodb():
    """
    Retrieve images and metadata from MongoDB GridFS.
    Returns:
        images (list): List of numpy arrays of images.
        metadata (list): List of dictionaries containing metadata including image ID.
    """
    images = []
    metadata = []

    for file in fs.find():
        image_id = str(file._id)  # Convert ObjectId to string
        image_bytes = file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((32, 32))
        images.append(np.asarray(img))

        # Retrieve corresponding metadata from MongoDB
        document = db["your_collection"].find_one({"image_id": image_id})
        material = document.get("category", "Unknown") if document else "Unknown"

        metadata.append({"image_id": image_id, "filename": file.filename, "category": material})

    return images, metadata


def create_sprite_sheet_from_mongodb(output_sprite, output_json, reduction_method="tsne"):
    """
    Create a sprite sheet using images from MongoDB with dimensionality reduction.
    Args:
        output_sprite (str): Path to save the sprite sheet image.
        output_json (str): Path to save the JSON metadata file.
        reduction_method (str): "tsne" or "umap" for dimensionality reduction.
    """
    # Fetch images and metadata from MongoDB
    images, metadata = fetch_images_from_mongodb()

    if not images:
        print("No images found in MongoDB.")
        return

    images = np.array(images)  # Convert to numpy array
    images_flat = images.reshape(len(images), -1)  # Flatten images for dimensionality reduction

    # Perform dimensionality reduction
    if reduction_method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif reduction_method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid reduction method. Choose 'tsne' or 'umap'.")

    embeddings = reducer.fit_transform(images_flat)

    # Normalize embeddings for canvas dimensions
    canvas_width, canvas_height = 4000, 1000
    embeddings_min = embeddings.min(axis=0)
    embeddings_max = embeddings.max(axis=0)

    normalized_embeddings = (embeddings - embeddings_min) / (embeddings_max - embeddings_min)
    normalized_embeddings[:, 0] *= canvas_width  # Scale x to canvas width
    normalized_embeddings[:, 1] *= canvas_height  # Scale y to canvas height

    # Create sprite sheet
    sprite_dim = int(np.ceil(np.sqrt(len(images))))  # Dimensions of the sprite sheet
    sprite_sheet = np.zeros((sprite_dim * 32, sprite_dim * 32, 3), dtype=np.uint8)
    sprite_json = []

    for idx, (img, embedding, meta) in enumerate(zip(images, normalized_embeddings, metadata)):
        row = idx // sprite_dim
        col = idx % sprite_dim
        sprite_sheet[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = img
        sprite_json.append({
            "image_id": meta["image_id"],  # Include MongoDB image ID
            "embedding": embedding.tolist(),
            "label": idx,
            "category": meta["category"],
            "spriteX": col,
            "spriteY": row
        })

    # Save the sprite sheet
    sprite_image = Image.fromarray(sprite_sheet)
    sprite_image.save(output_sprite)

    # Save the JSON metadata
    with open(output_json, "w") as f:
        json.dump(sprite_json, f, indent=4)

    print(f"✅ Sprite sheet saved as {output_sprite}")
    print(f"✅ Metadata saved as {output_json}")


# Example usage:
create_sprite_sheet_from_mongodb("sprite_sheet.png", "metadata.json", reduction_method="tsne")
import json
import gridfs
from pymongo import MongoClient
from io import BytesIO
from PIL import Image


# MongoDB Configuration
MONGO_URI = "mongodb://your_username:your_password@localhost:27017/your_database?authSource=admin"
client = MongoClient(MONGO_URI)
db = client["your_database"]
fs = gridfs.GridFS(db)


def fetch_images_from_metadata(metadata_json, output_folder=None):
    """
    Fetch images from MongoDB using `image_id` stored in metadata.json.
    
    Args:
        metadata_json (str): Path to the JSON metadata file.
        output_folder (str, optional): If provided, saves images to this folder.

    Returns:
        images_dict (dict): Dictionary with `image_id` as key and `PIL.Image` as value.
    """
    # Load metadata JSON
    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    images_dict = {}

    for entry in metadata:
        image_id = entry["image_id"]

        try:
            # Fetch image from GridFS
            file = fs.get(ObjectId(image_id))
            image_bytes = file.read()
            img = Image.open(BytesIO(image_bytes))

            # Store in dictionary
            images_dict[image_id] = img

            # Save image locally (if output_folder is provided)
            if output_folder:
                img.save(f"{output_folder}/{entry['label']}_{entry['category']}.png")

            print(f"✅ Fetched image {entry['filename']} (ID: {image_id})")

        except Exception as e:
            print(f"❌ Error fetching image {image_id}: {str(e)}")

    return images_dict


# Example usage:
# images = fetch_images_from_metadata("metadata.json", output_folder="downloaded_images")
