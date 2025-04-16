import json
import os
import time
from datafed.CommandLib import API
import json
from dotenv import load_dotenv
import numpy as np
from db_functions import * 

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
DATAFED_USER_NAME = os.environ.get("DATAFED_USER_NAME")
DATAFED_PASSWORD = os.environ.get("DATAFED_PASSWORD")
DATAFED_COLLECTION_NAME = os.environ.get("DATAFED_COLLECTION_NAME")
DATAFED_PROJECT_ID = os.environ.get("DATAFED_PROJECT_ID")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") 
GLOBUS_ENDPOINT_ID = os.environ.get("GLOBUS_ENDPOINT_ID")
GLOBUS_PATH = os.environ.get("GLOBUS_PATH")
IMAGE_PATH = os.environ.get("IMAGE_PATH")
output_dir = os.environ.get("OUTPUT_DIR")

df_api = API()
df_api.loginByPassword(DATAFED_USER_NAME, DATAFED_PASSWORD)
df_api.setContext(DATAFED_PROJECT_ID)#(os.environ.get("PROJECT_ID"))


coll_list_resp = df_api.collectionItemsList(DATAFED_COLLECTION_NAME,count=10000)#(os.environ("COLLECTION_NAME"),count = 100000)
# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for idx, each in enumerate(coll_list_resp[0].item):
    print(f"Processing item {idx},{each.id}")
    
    dv_resp = df_api.dataView(each.id)
    try:
        ig_resp = df_api.dataGet(each.id,f"{GLOBUS_ENDPOINT_ID}{GLOBUS_PATH}")
        if dv_resp and dv_resp[0].data and dv_resp[0].data[0].metadata:
            metadata_str = dv_resp[0].data[0].metadata
            try:
                res = json.loads(metadata_str)
                file_name = f"{each.id}.json"

                if not os.path.exists(file_name):
                    with open(file_name, 'w') as f:
                        json.dump(res, f, indent=4)
                    print(f"✅ JSON object saved to file: {file_name}")
                else:
                    print(f"⚠️ File '{file_name}' already exists. Not overwriting.")

                print(f"📄 Metadata loaded for item {each.id}: {res}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for item {each.id}: {e}")
                continue  # Skip to the next item if JSON is invalid
            
            image_path = f"{IMAGE_PATH}{each.id.split("/")[1]}.png"
            image_id = None
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                    # image_id = (f"{each.id}.png", img_file.read())
                    print(f"✅ Image stored with ID:")
            else:
                print(f"❌ Image not found: {image_path}")

            # Store document in MongoDB
            document = {
                "datafed_id": each.id,
                "metadata":  res,
                "image_id": str(image_id) if image_id else None  # Store image reference
            }
            print(document)
            create_document_with_image(res,f"{each.id}.png", image_data if image_data else b"")
            print(f"✅ Metadata stored for {each.id}")
            time.sleep(40)
            file_path = os.path.join(output_dir, each.id + ".json")
            with open(file_path, "w") as outfile:
                json.dump(res, outfile)
    except Exception as error:
        print("Error processing item",error)
        pass
