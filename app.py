from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.manifold import TSNE
from umap import UMAP
from PIL import Image
import os
import numpy as np
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://localhost:5173"],  # Allow all origins (change to specific domains for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
from fastapi.staticfiles import StaticFiles

app.mount("/output", StaticFiles(directory="output"), name="output")


def create_sprite_sheet(image_folder, output_sprite, output_json, materials, reduction_method="tsne"):
    """
    Create a sprite sheet with dimensionality reduction (t-SNE or UMAP) embeddings and generate a JSON object.
    Args:
        image_folder (str): Path to the folder containing images.
        output_sprite (str): Path to save the sprite sheet image.
        output_json (str): Path to save the JSON metadata file.
        materials (list): List of materials corresponding to images.
        reduction_method (str): "tsne" or "umap" for dimensionality reduction.
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
    images_flat = images.reshape(len(images), -1)  # Flatten images for dimensionality reduction

    # Perform dimensionality reduction
    if reduction_method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif reduction_method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid reduction method. Choose 'tsne' or 'umap'.")

    embeddings = reducer.fit_transform(images_flat)

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

@app.get("/api/embeddings")
async  def get_embeddings():
    # output_sprite = os.path.join()
    with open("./output/metadata.json", "r") as file:  # Replace "data.json" with your file path
        json_data = json.load(file)
    return JSONResponse({
    "spritePath": "/output/sprite_sheet.png",
    "itemsPath":json_data
})

@app.post("/api/embedding")
async def generate_sprite_sheet(dataset_path: str = Form(...), method: str = Form("tsne")):
    """
    API to generate sprite sheet and metadata using t-SNE or UMAP.
    Args:
        dataset_path (str): Path to the dataset folder.
        method (str): Dimensionality reduction method ("tsne" or "umap").
    Returns:
        JSONResponse with paths to sprite sheet and metadata.
    """
    try:
        if not os.path.exists(dataset_path):
            return JSONResponse({"error": "Dataset path does not exist."}, status_code=400)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_sprite = os.path.join(output_dir, "sprite_sheet.png")
        output_json = os.path.join(output_dir, "metadata.json")

        # Example materials (you can modify this)
        materials = ['wood', 'metal', 'plastic', 'glass', 'stone']

        create_sprite_sheet(
            image_folder=dataset_path,
            output_sprite=output_sprite,
            output_json=output_json,
            materials=materials,
            reduction_method=method
        )

        return JSONResponse({
            "spritePath": output_sprite,
            "itemsPath": output_json
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
