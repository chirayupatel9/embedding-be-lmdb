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

def create_documents_with_images_batch(items):
    """
    Batch insert multiple documents with images using a single transaction.
    Optimized with pre-allocated results list and reduced object creation.
    """
    # Pre-allocate results list to avoid repeated resizing
    results = [None] * len(items)
    message = "✅ Document inserted with image"  # Reuse string constant
    
    with env.begin(write=True) as txn:
        for idx, item in enumerate(items):
            document_data = item.get("document_data", {})
            filename = item.get("filename", "")
            file_data = item.get("file_data")
            
            image_id = str(uuid4())
            doc_id = str(uuid4())
            
            # Store image with pre-encoded key
            txn.put(encode_key(image_id), file_data, db=image_db)
            
            # Update document_data in-place to avoid dict copy
            document_data["filename"] = filename
            document_data["image_id"] = image_id
            
            # Encode JSON once and store
            txn.put(encode_key(doc_id), json.dumps(document_data).encode('utf-8'), db=metadata_db)
            
            # Pre-allocated list assignment faster than append
            results[idx] = {
                "message": message,
                "document_id": doc_id,
                "image_id": image_id
            }
    
    return results

def get_all_documents():
    """Fetch all documents - optimized with reduced string operations"""
    documents = []
    # Pre-compute URL prefix to avoid repeated f-string creation
    url_prefix = "/get-image/"
    
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        # Single-pass cursor iteration with optimized JSON parsing
        for key, value in cursor:
            # Decode once and parse JSON directly from bytes-like object
            document = json.loads(value)
            key_str = decode_key(key)
            document["_id"] = key_str
            # Optimize URL construction - avoid f-string in hot loop
            image_id = document.get('image_id')
            if image_id:
                document["image_url"] = url_prefix + image_id
            documents.append(document)
    return documents

def get_documents_by_image_ids(image_ids):
    """Efficiently retrieve documents matching specific image_ids using cursor iteration"""
    if not image_ids:
        return []
    
    image_id_set = set(image_ids)
    documents = []
    url_prefix = "/get-image/"  # Pre-compute to avoid f-string overhead
    
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        # Single transaction cursor iteration - optimized JSON parsing and string ops
        for key, value in cursor:
            # Parse JSON directly from bytes, skip decode step
            document = json.loads(value)
            doc_image_id = document.get("image_id")
            if doc_image_id in image_id_set:
                document["_id"] = decode_key(key)
                # String concatenation faster than f-string in hot loop
                document["image_url"] = url_prefix + doc_image_id
                documents.append(document)
                # Early exit optimization: if 1:1 mapping, can break when set is exhausted
                # (commented out to handle potential multiple docs per image_id)
    
    return documents

def get_document_by_field(field, value):
    """Find document by field - optimized with early exit and reduced string ops"""
    url_prefix = "/get-image/"
    
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            # Parse JSON directly from bytes
            document = json.loads(value_bytes)
            if document.get(field) == value:
                document["_id"] = decode_key(key)
                # Optimize URL construction
                image_id = document.get("image_id")
                if image_id:
                    document["image_url"] = url_prefix + image_id
                return document
    return None

def delete_document(field, value):
    """Delete document and associated image - optimized JSON parsing"""
    with env.begin(write=True) as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            # Parse JSON directly from bytes
            document = json.loads(value_bytes)
            if document.get(field) == value:
                image_id = document.get("image_id")
                if image_id:
                    # Pre-encode key once
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

def get_images_batch(image_ids):
    """Batch retrieve multiple images using a single transaction with direct gets"""
    if not image_ids:
        return {}
    
    # Pre-encode all keys once to avoid repeated encoding overhead
    image_dict = {}
    encoded_keys_map = {}
    seen = set()
    
    # Deduplicate and pre-encode keys in one pass
    for img_id in image_ids:
        if img_id not in seen:
            seen.add(img_id)
            encoded_keys_map[img_id] = encode_key(img_id)
    
    # Single transaction with pre-encoded keys - eliminates encode overhead in hot loop
    with env.begin() as txn:
        for image_id, encoded_key in encoded_keys_map.items():
            file_data = txn.get(encoded_key, db=image_db)
            if file_data is not None:
                image_dict[image_id] = file_data
    
    return image_dict

def get_all_images():
    """Fetch all stored image metadata - optimized string operations"""
    images = []
    url_prefix = "/get-image/"
    
    with env.begin() as txn:
        cursor = txn.cursor(db=image_db)
        for key, _ in cursor:
            image_id = decode_key(key)
            # String concatenation faster than f-string in hot loop
            images.append({
                "image_id": image_id,
                "image_url": url_prefix + image_id
            })
    return images

def get_image_with_details(image_id):
    """Retrieve an image with its linked metadata - optimized JSON parsing"""
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        # Parse JSON directly from bytes, skip decode step
        for key, value_bytes in cursor:
            document = json.loads(value_bytes)
            if document.get("image_id") == image_id:
                document["_id"] = decode_key(key)
                return document
    return None

def get_all_images_with_details():
    """Retrieve all images with their associated documents - optimized string ops"""
    details = []
    url_prefix = "/get-image/"
    
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            # Parse JSON directly from bytes
            document = json.loads(value_bytes)
            image_id = document.get("image_id")
            if image_id:
                document["_id"] = decode_key(key)
                # Pre-compute URL to avoid f-string overhead
                details.append({
                    "image_id": image_id,
                    "image_url": url_prefix + image_id,
                    "document_details": document
                })
    return details

def upload_images_from_folder(folder_path, metadata_folder=None, exact_match=False):
    """Upload all images in a folder - optimized with batch writes"""
    uploaded_files = []

    if not os.path.exists(folder_path):
        return {"error": "Folder not found"}

    metadata_dict = {}
    if metadata_folder and os.path.exists(metadata_folder):
        for meta_filename in os.listdir(metadata_folder):
            if meta_filename.lower().endswith('.json'):
                basename = os.path.splitext(meta_filename)[0]
                metadata_dict[basename] = os.path.join(metadata_folder, meta_filename)

    # Collect all items for batch processing
    batch_items = []
    
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
        
        batch_items.append({
            "document_data": document_data,
            "filename": filename,
            "file_data": file_data
        })
    
    # Batch insert all items in a single transaction
    if batch_items:
        uploaded_files = create_documents_with_images_batch(batch_items)

    return {
        "message": f"✅ Uploaded {len(uploaded_files)} files",
        "uploaded_files": uploaded_files
    }

def get_metadata_image_relationships(filter_by=None):
    """Get relationships between metadata and images - optimized JSON parsing and filtering"""
    relationships = []
    # Pre-compute filter keys if filter_by exists to avoid repeated dict lookups
    filter_keys = set(filter_by.keys()) if filter_by else None
    
    with env.begin() as txn:
        cursor = txn.cursor(db=metadata_db)
        for key, value_bytes in cursor:
            # Parse JSON directly from bytes
            document = json.loads(value_bytes)
            
            # Optimize filter check - early exit if filter doesn't match
            if filter_by:
                if filter_keys and not all(document.get(k) == filter_by.get(k) for k in filter_keys):
                    continue
            
            # Build relationship dict with minimal lookups
            image_id = document.get("image_id")
            metadata_filename = document.get("metadata_filename")
            relationship = {
                "document_id": decode_key(key),
                "image_id": image_id,
                "image_filename": document.get("image_filename", "unknown"),
                "metadata_filename": metadata_filename if metadata_filename else "unknown",
                "has_valid_relationship": bool(image_id and metadata_filename)
            }
            # Optimize field copying - only iterate over fields that might exist
            for field in ["label", "category", "description"]:
                if field in document:
                    relationship[field] = document[field]
            relationships.append(relationship)

    return relationships