import time
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
from db_functions_lmdb import ( 
    get_all_documents as db_get_all_documents,
    get_document_by_field as db_get_document_by_field,
    delete_document as db_delete_document,
    get_all_images as db_get_all_images,
    get_image_with_details as db_get_image_with_details,
    get_image as db_get_image,
    upload_images_from_folder,
    get_metadata_image_relationships,
    create_document_with_image
)
from io import BytesIO
import numpy as np
import base64
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from tqdm import tqdm
from cuml.manifold import TSNE, UMAP
import cupy as cp
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import List, Literal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "http://0.0.0.0:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- CREATE SPRITE SHEET FROM MONGODB -----------------------------
def create_sprite_sheet(output_sprite, output_json, reduction_method, coordinates, metadata):
    """
    Generates a sprite sheet and JSON metadata from coordinates and metadata list.
    Each thumbnail is 32x32 pixels.
    """
    from PIL import Image
    import math

    thumb_size = (32, 32)
    num_images = len(metadata)
    sprite_dim = int(np.ceil(np.sqrt(num_images)))

    sprite_sheet = Image.new("RGB", (sprite_dim * 32, sprite_dim * 32), (0, 0, 0))
    items = []

    for idx, meta in enumerate(metadata):
        try:
            image_data = db_get_image(meta["image_id"])
            img = Image.open(BytesIO(image_data)).convert("RGB")
            img = img.resize(thumb_size)

            # # Draw image_id or idx as overlay (annotation)
            # draw = ImageDraw.Draw(img)
            # draw.text((2, 2), str(idx), fill=(255, 0, 0))  # Optional: use font
            # Calculate sprite position
            col = idx % sprite_dim
            row = idx // sprite_dim
            x = col * 32
            y = row * 32
            
            sprite_sheet.paste(img, (x, y))

            items.append({
                "embedding": [float(coordinates[idx][0]), float(coordinates[idx][1])],
                "image_id": meta["image_id"],
                "filename": meta.get("filename", "unknown"),
                "category": meta.get("category", "Unknown"),
                "spriteX": col,  # Column position in sprite grid
                "spriteY": row,  # Row position in sprite grid
                "sprite_x": x,   # Pixel x coordinate
                "sprite_y": y,   # Pixel y coordinate
                "width": 32,
                "height": 32
            })
        except Exception as e:
            print(f"Skipping image for sprite: {meta.get('image_id', 'unknown')} due to {e}")

    sprite_sheet.save(output_sprite)
    with open(output_json, "w") as f:
        json.dump(items, f, indent=4)

    return True


# ---------- Model Definition ---------- #
def resnet50_embedding(in_channels=3, n_classes=17, dropout=0.5, weights=None):
    model = models.resnet50(weights=weights)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=dropout),
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=dropout),
        nn.Linear(512, 64, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(64),
        nn.Dropout(p=dropout),
        nn.Linear(64, n_classes, bias=True)
    )

    return model
class ResNet50Embedder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-1])  # Remove FC
        self.output_dim = 2048

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class XCiTEmbedder(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        # Load pretrained timm model
        self.model = timm.create_model(
            'xcit_small_12_p8_224',
            pretrained=False,
            num_classes=0,  # Remove classification head
            in_chans=in_channels
        )

        # Optionally load your custom weights
        if pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")

                # Remove `module.` prefix if trained using DataParallel
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                # Remove classifier head weights if present
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded XCiT weights from {pretrained_path}")
                if missing:
                    print("⚠️ Missing keys:", missing)
                if unexpected:
                    print("⚠️ Unexpected keys:", unexpected)
            except Exception as e:
                print(f"❌ Failed to load XCiT weights: {e}")

    def forward(self, x):
        return self.model(x) 
# ----------------------------- INITIALIZE MODEL -----------------------------
def initialize_model(device, model_type: Literal["resnet50", "xcit"] = "resnet50"):
    if model_type == "resnet50":
        model = ResNet50Embedder()
    elif model_type == "xcit":
        model = XCiTEmbedder()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    print(f"✅ {model_type.upper()} model initialized and ready.")
    return model

# def initialize_model(device):
#     model = resnet50_embedding()
#     model = model.to(device)
#     model.eval()
#     model.fc = nn.Sequential(*list(model.fc.children())[:6])  # Take up to 512 dimension
#     print("✅ Model initialized and ready.")
#     return model

# ----------------------------- PROCESS LMDB DOCUMENTS -----------------------------
def process_lmdb_documents(batch_docs, transform):
    image_tensors = []
    valid_metadata = []

    def load_image(doc):
        try:
            img_data = db_get_image(doc["image_id"])
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img_tensor = transform(img)
            return img_tensor, {
                "image_id": doc["image_id"],
                "filename": doc.get("filename", "unknown"),
                "category": doc.get("category", "Unknown")
            }
        except Exception as e:
            print(f"Error loading image {doc.get('image_id', 'unknown')}: {str(e)}")
            return None, None

    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(load_image, batch_docs))

    for img_tensor, meta in results:
        if img_tensor is not None:
            image_tensors.append(img_tensor)
            valid_metadata.append(meta)

    return image_tensors, valid_metadata

# ----------------------------- EXTRACT EMBEDDINGS FROM LMDB -----------------------------
def extract_embeddings_from_lmdb(model, device, batch_size, lmdb_batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    start_time = time.time()
    documents = db_get_all_documents()
    print(f"✅ Retrieved {len(documents)} documents from LMDB in {time.time() - start_time:.2f} seconds.")

    all_embeddings = []
    all_metadata = []

    for batch_start in tqdm(range(0, len(documents), lmdb_batch_size), desc="Processing LMDB batches"):
        batch_docs = documents[batch_start: batch_start + lmdb_batch_size]
        image_tensors, metadata = process_lmdb_documents(batch_docs, transform)

        if image_tensors:
            batch_tensor = torch.stack(image_tensors)
            batch_tensor = batch_tensor.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    feats = model(batch_tensor)
            batch_embeddings = feats.cpu().numpy()
            for emb, meta in zip(batch_embeddings, metadata):
                all_embeddings.append(emb)
                all_metadata.append(meta)
            # all_embeddings.append(feats.cpu().numpy())
            # all_metadata.extend(metadata)

            torch.cuda.empty_cache()

    if not all_embeddings:
        raise Exception("No valid images found in LMDB.")

    embeddings = np.vstack(all_embeddings)
    print(f"✅ Extracted embeddings shape: {embeddings.shape}")
    return embeddings, all_metadata

# ----------------------------- COMPUTE T-SNE -----------------------------
def compute_tsne(embeddings, output_dim=2, perplexity=30, n_iter=1000):
    print("🚀 Running cuML t-SNE...")
    start_time = time.time()
    embeddings_gpu = cp.asarray(embeddings)

    tsne = TSNE(
        n_components=output_dim,
        perplexity=perplexity,
        n_iter=n_iter,
        verbose=1,
        method="barnes_hut",
        num_workers=16
    )
    tsne_result_gpu = tsne.fit_transform(embeddings_gpu)

    tsne_result = cp.asnumpy(tsne_result_gpu)
    print(f"✅ t-SNE completed in {time.time() - start_time:.2f} seconds.")
    return tsne_result

# ----------------------------- GENERATE T-SNE FROM LMDB -----------------------------
def generate_tsne_from_lmdb(batch_size=512, output_dim=2, perplexity=30, device_str="cuda", lmdb_batch_size=2000, model_type: Literal["resnet50", "xcit"] = "resnet50"):
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model = initialize_model(device, model_type)
    embeddings, metadata = extract_embeddings_from_lmdb(model, device, batch_size, lmdb_batch_size)
    tsne_result = compute_tsne(embeddings, output_dim=output_dim, perplexity=perplexity)

    return tsne_result, metadata
# ----------------------------- COMPUTE UMAP -----------------------------

def compute_umap(embeddings, output_dim=2, n_neighbors=15, min_dist=0.1):
    print("🚀 Running cuML UMAP...")
    start_time = time.time()
    embeddings_gpu = cp.asarray(embeddings)

    reducer = UMAP(
        n_components=output_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        verbose=True
    )
    umap_result_gpu = reducer.fit_transform(embeddings_gpu)
    umap_result = cp.asnumpy(umap_result_gpu)

    print(f"✅ UMAP completed in {time.time() - start_time:.2f} seconds.")
    return umap_result

# ----------------------------- GENERATE UMAP FROM LMDB -----------------------------
def generate_umap_from_lmdb(batch_size=512, output_dim=2, device_str="cuda", lmdb_batch_size=2000, model_type: Literal["resnet50", "xcit"] = "resnet50"):
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model = initialize_model(device, model_type)
    embeddings, metadata = extract_embeddings_from_lmdb(model, device, batch_size, lmdb_batch_size)
    umap_result = compute_umap(embeddings, output_dim=output_dim)

    return umap_result, metadata


# ----------------------------- STATIC FILES -----------------------------
app.mount("/api/output", StaticFiles(directory="output"), name="output")

# ----------------------------- ROOT -----------------------------
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

# ----------------------------- GET ALL DOCUMENTS -----------------------------
@app.get("/api/read")
async def get_all_documents():
    try:
        documents = db_get_all_documents()
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- GET SINGLE DOCUMENT -----------------------------
@app.get("/api/read/{field}/{value}")
async def get_document_by_field(field: str, value: str):
    try:
        document = db_get_document_by_field(field, value)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- DELETE DOCUMENT -----------------------------
@app.delete("/api/delete/{field}/{value}")
async def delete_document(field: str, value: str):
    try:
        response = db_delete_document(field, value)
        if not response:
            raise HTTPException(status_code=404, detail="Document not found")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- GET ALL IMAGES -----------------------------
@app.get("/api/get-all-images")
async def get_all_images():
    try:
        images = db_get_all_images()
        if not images:
            raise HTTPException(status_code=404, detail="No images found")
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- GET IMAGE BY ID -----------------------------
@app.get("/api/get-image/{image_id}")
async def get_image(image_id: str):
    try:
        file_data = db_get_image(image_id)
        if not file_data:
            raise HTTPException(status_code=404, detail="Image not found")
        return StreamingResponse(BytesIO(file_data), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- GET IMAGE WITH DETAILS -----------------------------
@app.get("/api/get-image-details/{image_id}")
async def get_image_with_details_api(image_id: str):
    try:
        document_data = db_get_image_with_details(image_id)
        if not document_data:
            raise HTTPException(status_code=404, detail="Image or document not found")
        return {
            "image_id": image_id,
            "document_details": document_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- UPLOAD IMAGES -----------------------------
@app.post("/api/upload-images")
async def upload_images(
    folder_path: str = Form(...),
    metadata_folder: str = Form(None),
    exact_match: bool = Form(False)
):
    try:
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail="Folder not found")

        if metadata_folder and not os.path.exists(metadata_folder):
            raise HTTPException(status_code=400, detail="Metadata folder not found")
        
        response = upload_images_from_folder(folder_path, metadata_folder, exact_match)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- RELATIONSHIPS -----------------------------
@app.get("/api/relationships")
async def get_relationships(field: str = None, value: str = None):
    try:
        filter_by = {field: value} if field and value else None
        relationships = get_metadata_image_relationships(filter_by)

        if not relationships:
            raise HTTPException(status_code=404, detail="No relationships found")
        return {
            "count": len(relationships),
            "relationships": relationships
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- BASE64 Image with Metadata -----------------------------
@app.get("/api/image-with-metadata/{image_id}") 
async def get_image_with_metadata(image_id: str):
    try:
        file_data = db_get_image(image_id)
        if not file_data:
            raise HTTPException(status_code=404, detail="Image not found")

        document = db_get_image_with_details(image_id)

        image_base64 = base64.b64encode(file_data).decode('utf-8')

        return {
            "image_id": image_id,
            "filename": document.get("filename", "unknown") if document else "unknown",
            "image_data": image_base64,
            "content_type": "image/jpeg",
            "has_metadata": bool(document),
            "metadata": document if document else {}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- SAVE ALL EMBEDDINGS -----------------------------
@app.get("/api/save_all_embeddings")
async def save_all_embeddings():
    output_embeddings_path="./output/all_embeddings.npy"
    output_metadata_path="./output/all_metadata.json"
    device = torch.device("cuda")
    model = initialize_model(device)
    embeddings, metadata = extract_embeddings_from_lmdb(model, device, batch_size=512, lmdb_batch_size=2000)

    np.save(output_embeddings_path, embeddings)
    with open(output_metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Saved embeddings to {output_embeddings_path} and metadata to {output_metadata_path}")

# ----------------------------- DIMENSIONALITY REDUCTION -----------------------------
@app.get("/api/dimensionality-reduction/{method}")
async def dimensionality_reduction(method: str, model_type: Literal["resnet50", "xcit"] = "resnet50"):
    """
    Generate dimensionality reduction visualization using specified method and model.
    """
    try:
        method = method.lower()
        if method not in ["tsne", "umap"]:
            raise HTTPException(status_code=400, detail="Method must be either 'tsne' or 'umap'")
        
        # Perform dimensionality reduction
        if method == "tsne":
            result_coords, metadata = generate_tsne_from_lmdb(model_type=model_type)
        else:  # umap
            result_coords, metadata = generate_umap_from_lmdb(model_type=model_type)
        
        # Create output directory if it doesn't exist
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sprite sheet and metadata
        sprite_path = f"{output_dir}/sprite_sheet.png"
        metadata_path = f"{output_dir}/{method}_metadata.json"
        
        result = create_sprite_sheet(
            output_sprite=sprite_path,
            output_json=metadata_path,
            reduction_method=method,
            coordinates=result_coords,
            metadata=metadata
        )
        
        if not result:
            raise HTTPException(status_code=500, detail=f"Failed to generate {method.upper()} sprite sheet")
        
        # Read the generated metadata
        with open(metadata_path, "r") as f:
            json_data = json.load(f)
        
        # Calculate sprite sheet dimensions
        num_images = len(json_data)
        sprite_dim = int(np.ceil(np.sqrt(num_images)))
        sprite_width = 32
        sprite_height = 32
        
        return JSONResponse({
            "spritePath": {
                "columns": sprite_dim,
                "rows": sprite_dim,
                "width": sprite_dim * sprite_width,
                "height": sprite_dim * sprite_height,
                "sprite_width": sprite_width,
                "sprite_height": sprite_height,
                "url": "/output/sprite_sheet.png"
            },
            "itemsPath": json_data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- MAKE T-SNE -----------------------------
@app.get("/api/make_reduction/{method}")
async def make_reduction(method: str, model_type: Literal["resnet50", "xcit"] = "resnet50"):
    """
    Generate dimensionality reduction visualization using specified method and model.
    """
    try:
        method = method.lower()
        if method not in ["tsne", "umap"]:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'tsne' or 'umap'.")

        output_dir = "./output"
        sprite_path = f"{output_dir}/sprite_sheet.png"
        metadata_path = f"{output_dir}/{method}_metadata.json"

        if os.path.exists(sprite_path) and os.path.exists(metadata_path) and os.path.getsize(sprite_path) > 0:
            with open(metadata_path, "r") as f:
                json_data = json.load(f)

            sprite_dim = int(np.ceil(np.sqrt(len(json_data))))
            sprite_width = 32
            sprite_height = 32

            return JSONResponse({
                "spritePath": {
                    "columns": sprite_dim,
                    "rows": sprite_dim,
                    "width": sprite_dim * sprite_width,
                    "height": sprite_dim * sprite_height,
                    "sprite_width": sprite_width,
                    "sprite_height": sprite_height,
                    "url": "/output/sprite_sheet.png"
                },
                "itemsPath": json_data
            })

        # Generate embeddings and reduction
        if method == "tsne":
            result_coords, metadata = generate_tsne_from_lmdb(model_type=model_type)
        else:
            result_coords, metadata = generate_umap_from_lmdb(model_type=model_type)

        os.makedirs(output_dir, exist_ok=True)

        result = create_sprite_sheet(
            output_sprite=sprite_path,
            output_json=metadata_path,
            reduction_method=method,
            coordinates=result_coords,
            metadata=metadata
        )

        if not result:
            raise HTTPException(status_code=500, detail=f"Failed to generate {method.upper()} sprite sheet")

        with open(metadata_path, "r") as f:
            json_data = json.load(f)

        sprite_dim = int(np.ceil(np.sqrt(len(json_data))))
        sprite_width = 32
        sprite_height = 32

        return JSONResponse({
            "spritePath": {
                "columns": sprite_dim,
                "rows": sprite_dim,
                "width": sprite_dim * sprite_width,
                "height": sprite_dim * sprite_height,
                "sprite_width": sprite_width,
                "sprite_height": sprite_height,
                "url": f"/output/sprite_sheet.png"
            },
            "itemsPath": json_data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- MAKE T-SNE SUBSET -----------------------------
class TSNESubsetRequest(BaseModel):
    image_ids: List[str]

# ----------------------------- MAKE T-SNE SUBSET -----------------------------
@app.post("/api/make_reduction_subset/{method}")
async def make_reduction_subset(method: str, payload: TSNESubsetRequest, model_type: Literal["resnet50", "xcit"] = "resnet50"):
    """
    Generate dimensionality reduction visualization for a subset of images using specified method and model.
    """
    try:
        method = method.lower()
        if method not in ["tsne", "umap"]:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'tsne' or 'umap'.")

        image_ids = payload.image_ids
        if not image_ids:
            raise HTTPException(status_code=400, detail="No image_ids provided.")

        metadata_path = f"./output/{method}_metadata.json"
        sprite_path = f"/output/sprite_sheet.png"

        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail=f"{method.upper()} metadata not found.")

        with open(metadata_path, "r") as file:
            full_metadata = json.load(file)

        filtered_metadata = [item for item in full_metadata if item["image_id"] in image_ids]

        if not filtered_metadata:
            raise HTTPException(status_code=404, detail=f"No matching image_ids found in {method.upper()} metadata.")

        sprite_dim = int(np.ceil(np.sqrt(len(full_metadata))))
        sprite_width = 32
        sprite_height = 32

        return JSONResponse({
            "spritePath": {
                "columns": sprite_dim,
                "rows": sprite_dim,
                "width": sprite_dim * sprite_width,
                "height": sprite_dim * sprite_height,
                "sprite_width": sprite_width,
                "sprite_height": sprite_height,
                "url": sprite_path
            },
            "itemsPath": filtered_metadata
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
