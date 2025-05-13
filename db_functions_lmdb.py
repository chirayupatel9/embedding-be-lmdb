import os
import lmdb
import json
from db_config import env, image_db, metadata_db
from uuid import uuid4

# Helper functions
def encode_key(key):
    return str(key).encode('utf-8')

def decode_key(key_bytes):
    return key_bytes.decode('utf-8')

# CRUD Operations

def create_document_with_image(document_data, filename, file_data):
    """Insert a new document with an image"""
    with env.begin(write=True) as txn:
        image_id = str(uuid4())
        
        # Store image
        txn.put(encode_key(image_id), file_data, db=image_db)
        
        # Store metadata
        document_data.update({
            "filename": filename,
            "image_id": image_id
        })
        doc_id = str(uuid4())
        txn.put(encode_key(doc_id), json.dumps(document_data).encode('utf-8'), db=metadata_db)

    return {
        "message": "✅ Document inserted with image",
        "document_id": doc_id,
        "image_id": image_id
    }

def get_all_documents():
    """Fetch all documents"""
    documents = []
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value in cursor:
            document = json.loads(value.decode('utf-8'))
            document["_id"] = decode_key(key)
            document["image_url"] = f"/get-image/{document.get('image_id')}"
            documents.append(document)
    return documents

def get_document_by_field(field, value):
    """Find document by field"""
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            document = json.loads(value_bytes.decode('utf-8'))
            if document.get(field) == value:
                document["_id"] = decode_key(key)
                if "image_id" in document:
                    document["image_url"] = f"/get-image/{document['image_id']}"
                return document
    return None

def delete_document(field, value):
    """Delete document and associated image"""
    with env.begin(write=True) as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            document = json.loads(value_bytes.decode('utf-8'))
            if document.get(field) == value:
                image_id = document.get("image_id")
                if image_id:
                    txn.delete(encode_key(image_id), db=image_db)
                txn.delete(key, db=metadata_db)
                return {"message": "✅ Document and associated image deleted"}
    return None

# Image Functions

def get_image(image_id):
    """Retrieve an image"""
    with env.begin() as txn:
        file_data = txn.get(encode_key(image_id), db=image_db)
        return file_data

def get_all_images():
    """Fetch all stored image metadata"""
    images = []
    with env.begin() as txn:
        cursor = txn.cursor(db=image_db)
        for key, _ in cursor:
            image_id = decode_key(key)
            images.append({
                "image_id": image_id,
                "image_url": f"/get-image/{image_id}"
            })
    return images

def get_image_with_details(image_id):
    """Retrieve an image with its linked metadata"""
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            document = json.loads(value_bytes.decode('utf-8'))
            if document.get("image_id") == image_id:
                document["_id"] = decode_key(key)
                return document
    return None

def get_all_images_with_details():
    """Retrieve all images with their associated documents"""
    details = []
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            document = json.loads(value_bytes.decode('utf-8'))
            if "image_id" in document:
                document["_id"] = decode_key(key)
                details.append({
                    "image_id": document["image_id"],
                    "image_url": f"/get-image/{document['image_id']}",
                    "document_details": document
                })
    return details

def upload_images_from_folder(folder_path, metadata_folder=None, exact_match=False):
    """Upload all images in a folder"""
    uploaded_files = []

    if not os.path.exists(folder_path):
        return {"error": "Folder not found"}

    metadata_dict = {}
    if metadata_folder and os.path.exists(metadata_folder):
        for meta_filename in os.listdir(metadata_folder):
            if meta_filename.lower().endswith('.json'):
                basename = os.path.splitext(meta_filename)[0]
                metadata_dict[basename] = os.path.join(metadata_folder, meta_filename)

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath) or not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        document_data = {
            "filename": filename
        }

        if metadata_folder:
            basename = os.path.splitext(filename)[0]
            matched_file = None
            if exact_match and basename in metadata_dict:
                matched_file = metadata_dict[basename]
            elif not exact_match:
                matches = [k for k in metadata_dict if basename in k or k in basename]
                if matches:
                    matched_file = metadata_dict[matches[0]]

            if matched_file:
                try:
                    with open(matched_file, 'r') as meta_file:
                        metadata = json.load(meta_file)
                        if isinstance(metadata, dict):
                            metadata["metadata_filename"] = os.path.basename(matched_file)
                            metadata["image_filename"] = filename
                            document_data = metadata
                except Exception:
                    pass
        
        upload_result = create_document_with_image(document_data, filename, file_data)
        uploaded_files.append(upload_result)

    return {
        "message": f"✅ Uploaded {len(uploaded_files)} files",
        "uploaded_files": uploaded_files
    }

def get_metadata_image_relationships(filter_by=None):
    """Get relationships between metadata and images"""
    relationships = []
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            document = json.loads(value_bytes.decode('utf-8'))
            if filter_by and not all(document.get(k) == v for k, v in filter_by.items()):
                continue
            relationship = {
                "document_id": decode_key(key),
                "image_id": document.get("image_id"),
                "image_filename": document.get("image_filename", "unknown"),
                "metadata_filename": document.get("metadata_filename", "unknown"),
                "has_valid_relationship": bool(document.get("image_id") and document.get("metadata_filename"))
            }
            for field in ["label", "category", "description"]:
                if field in document:
                    relationship[field] = document[field]
            relationships.append(relationship)
    return relationships
