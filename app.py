from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import json
from urllib.parse import unquote
from functions import *
from db_functions import *
from fastapi.responses import StreamingResponse

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

@app.get("/api/embeddings")
async  def get_embeddings():
    # output_sprite = os.path.join()
    with open("./output/metadata.json", "r") as file:  # Replace "data.json" with your file path
        json_data = json.load(file)
    return JSONResponse({
    "spritePath": "/output/sprite_sheet.png",
    "itemsPath": json_data
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
    dataset_path = unquote(dataset_path)
    print(f'dataset_path:{dataset_path}')
    try:
        if not os.path.exists(dataset_path):
            print(f'dataset_pathexsits:{dataset_path}')

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

@app.get("/api/get_image/{image_id}")
async def get_image(image_id: str):
    """Retrieve an image from MongoDB"""
    image_data = await get_image(image_id)
    if image_data:
        return StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="❌ Image Not Found")

@app.get("/api/read")
async def get_all_documents():
    """Fetch all documents with image URLs"""
    documents = get_all_documents()
    if not documents:
        raise HTTPException(status_code=404, detail="❌ No documents found")
    return documents



@app.get("/api/read/{field}/{value}", response_model=dict)
async def get_document_by_field(field: str, value: str):
    """Find a document dynamically by any field, including linked images"""
    document = get_document_by_field(field, value)
    if document:
        return document
    raise HTTPException(status_code=404, detail="❌ Document Not Found")


@app.delete("/api/delete/{field}/{value}", response_model=dict)
async def delete_document(field: str, value: str):
    """Delete a document and its linked image"""
    response = delete_document(field, value)
    if response:
        return response
    raise HTTPException(status_code=404, detail="❌ Document Not Found")


# 🚀 API Endpoints for Image Handling

@app.get("/api/get-image/{image_id}")
async def get_image(image_id: str):
    """Retrieve an image from MongoDB"""
    image_data = await get_image(image_id)
    if image_data:
        return StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="❌ Image Not Found")


@app.get("/api/get-all-images")
async def get_all_images():
    """Fetch metadata of all images stored"""
    images = get_all_images()
    if not images:
        raise HTTPException(status_code=404, detail="❌ No images found")
    return images


@app.get("/api/get-image-details/{image_id}")
async def get_image_with_details(image_id: str):
    """Retrieve an image and its associated document details"""
    document_data = await get_image_with_details(image_id)
    
    if document_data:
        return {
            "image_id": image_id,
            "document_details": document_data,
            # "image": StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")
        }
    
    raise HTTPException(status_code=404, detail="❌ Image or Document Not Found")

@app.post("/api/upload-images")
async def upload_images(folder_path: str = Form(...)):
    """Upload all images in a folder to MongoDB"""
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="❌ Folder not found")

    response = upload_images_from_folder(folder_path)
    return response

@app.get("/api/fetch-image/{image_id}")
async def fetch_image(image_id: str):
    """
    Retrieve an image from MongoDB GridFS using its `image_id`.
    
    Args:
        image_id (str): MongoDB GridFS ObjectId of the image.

    Returns:
        StreamingResponse: Returns the image as a binary stream.
    """
    try:
        file = fs.get(ObjectId(image_id))  # Fetch image from GridFS
        image_bytes = file.read()
        return StreamingResponse(BytesIO(image_bytes), media_type="image/jpeg")
    
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="❌ Image Not Found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Server Error: {str(e)}")
    
@app.get("/api/generate-sprite-sheet")
async def generate_sprite_sheet():
    """
    Generate a sprite sheet and metadata using t-SNE or UMAP.
    """
    return fetch_images_from_mongodb()
