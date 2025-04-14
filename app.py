from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import io
import json
from urllib.parse import unquote
from functions import *
from db_functions import get_all_documents as db_get_all_documents, get_document_by_field as db_get_document_by_field, delete_document as db_delete_document, get_all_images as db_get_all_images, get_image_with_details as db_get_image_with_details, upload_images_from_folder
from bson import ObjectId
import gridfs
from io import BytesIO

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/api/embeddings")
async def get_embeddings():
    try:
        with open("./output/metadata.json", "r") as file:
            json_data = json.load(file)
        return JSONResponse({
            "spritePath": "/output/sprite_sheet.png",
            "itemsPath": json_data
        })
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metadata file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid metadata file format")

@app.post("/api/embedding")
async def generate_sprite_sheet(dataset_path: str = Form(...), method: str = Form("tsne")):
    dataset_path = unquote(dataset_path)
    try:
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=400, detail="Dataset path does not exist")

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_sprite = os.path.join(output_dir, "sprite_sheet.png")
        output_json = os.path.join(output_dir, "metadata.json")

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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-image/{image_id}")
async def get_image(image_id: str):
    try:
        file = fs.get(ObjectId(image_id))
        image_bytes = file.read()
        return StreamingResponse(BytesIO(image_bytes), media_type="image/jpeg")
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/read")
async def get_all_documents():
    try:
        documents = db_get_all_documents()
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/read/{field}/{value}")
async def get_document_by_field(field: str, value: str):
    try:
        document = db_get_document_by_field(field, value)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete/{field}/{value}")
async def delete_document(field: str, value: str):
    try:
        response = db_delete_document(field, value)
        if not response:
            raise HTTPException(status_code=404, detail="Document not found")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-all-images")
async def get_all_images():
    try:
        images = db_get_all_images()
        if not images:
            raise HTTPException(status_code=404, detail="No images found")
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-image-details/{image_id}")
async def get_image_with_details(image_id: str):
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

@app.post("/api/upload-images")
async def upload_images(folder_path: str = Form(...)):
    try:
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail="Folder not found")
        response = upload_images_from_folder(folder_path)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generate-sprite-sheet")
async def generate_sprite_sheet():
    try:
        images, metadata = fetch_images_from_mongodb()
        if not images:
            raise HTTPException(status_code=404, detail="No images found in database")
        
        return JSONResponse({
            "images": len(images),
            "metadata": metadata
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
