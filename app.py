import time
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import io
import json
from urllib.parse import unquote
from functions import *
from db_functions_lmdb import (  # <-- IMPORTANT
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
from collections import OrderedDict
from tqdm import tqdm
from cuml.manifold import TSNE
import cupy as cp
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "0.0.0.0:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/output", StaticFiles(directory="api/output"), name="/api/output")

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

def resnet50_embedding(in_channels=3, n_classes=17, dropout=0.5):
    model = models.resnet50(weights=None)
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
def generate_tsne_from_lmdb(
    batch_size=1024,
    output_dim=2,
    perplexity=30,
    device_str="cuda:0",
    lmdb_batch_size=4000
):
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Initialize model
    model = resnet50_embedding()
    model.to(device)
    model.eval()
    model.fc = nn.Sequential(*list(model.fc.children())[:6])

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load all documents from LMDB
    documents = db_get_all_documents()
    print(f"Processing {len(documents)} images from LMDB...")

    all_embeddings = []
    all_metadata = []

    # Process documents in batches
    for batch_start in tqdm(range(0, len(documents), lmdb_batch_size), desc="Processing LMDB batches"):
        batch_docs = documents[batch_start:batch_start + lmdb_batch_size]
        
        image_tensors = []
        valid_metadata = []

        # Parallel load images
        with ThreadPoolExecutor(max_workers=64) as executor:
            def process_image(doc):
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
                    print(f"Error processing image {doc.get('image_id', 'unknown')}: {str(e)}")
                    return None, None

            results = list(executor.map(process_image, batch_docs))
            
            valid_results = [(img, meta) for img, meta in results if img is not None]
            if valid_results:
                chunk_tensors, chunk_metadata = zip(*valid_results)
                image_tensors.extend(chunk_tensors)
                valid_metadata.extend(chunk_metadata)

            if len(image_tensors) >= batch_size:
                batch_tensor = torch.stack(image_tensors[:batch_size])
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        batch_tensor = batch_tensor.to(device, non_blocking=True)
                        feats = model(batch_tensor)
                        all_embeddings.append(feats.cpu().numpy())
                        all_metadata.extend(valid_metadata[:batch_size])
                
                # Keep leftover
                image_tensors = image_tensors[batch_size:]
                valid_metadata = valid_metadata[batch_size:]

                torch.cuda.empty_cache()

        # Remaining images
        if image_tensors:
            batch_tensor = torch.stack(image_tensors)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    batch_tensor = batch_tensor.to(device, non_blocking=True)
                    feats = model(batch_tensor)
                    all_embeddings.append(feats.cpu().numpy())
                    all_metadata.extend(valid_metadata)

    if not all_embeddings:
        raise Exception("No valid images found in LMDB")

    embeddings = np.vstack(all_embeddings)

    print("Running cuML t-SNE...")
    embeddings_gpu = cp.asarray(embeddings)
    tsne = TSNE(
        n_components=output_dim,
        perplexity=perplexity,
        n_iter=1000,
        verbose=1,
        method='barnes_hut'
    )
    tsne_result_gpu = tsne.fit_transform(embeddings_gpu)
    tsne_result = cp.asnumpy(tsne_result_gpu)

    return tsne_result, all_metadata

@app.get("/api/make_tsne")
async def make_tsne():
    """
    Generate t-SNE embeddings from images stored in LMDB
    """
    try:
        start_time = time.time()
        tsne_result, metadata = generate_tsne_from_lmdb()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        response_data = {
            "time_taken": end_time - start_time,
            "coordinates": tsne_result.tolist(),
            "metadata": metadata
        }
        return JSONResponse(response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
