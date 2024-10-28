from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import json
from functions import read_json_data, create_individual_sprites
import os
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def hello_world():
    return {"message": "Hello World!"}


@app.get("/data/md08")
async def md08():
    return read_json_data('static/md08_umap_mnist_embeddings.json')


@app.get("/data/mnist")
async def mnist():
    return read_json_data('static/mnist_embeddings.json')


@app.get("/data/labels")
async def labels():
    return read_json_data('static/mnist_labels.json')


@app.get("/data/tsne")
async def tsne():
    return read_json_data('static/tsne_mnist_embeddings.json')


@app.post("/update_data")
async def update_data(request: Request):
    new_data = await request.json()
    return json.dumps(new_data)


# Define the sprite creation function (placeholder)
def create_sprites_and_save(images_path, width, height, sprite_sheet_output):
    # You need to implement the sprite creation logic here
    # This function should return some data related to the sprite (e.g., metadata)
    # For demonstration purposes, I'm returning a sample dict
    return {"status": "Sprite sheet created", "output_path": sprite_sheet_output}

class Item(BaseModel):
    image_path: str
    output_path: str | None = None
    width: int
    height: int | None = None

@app.post("/generate_sprite")
async def generate_sprite(item: Item):
    # Call the sprite creation function (you can change the parameters as needed)

    sprite_data = create_individual_sprites(item.image_path,item.output_path,item.width,item.height)#('static/train', 'static/sprite_sheet.png',64, 64, )
    # print(f'sprite data:{sprite_data}')
    # Return the sprite data as JSON
    return JSONResponse(content=sprite_data)


@app.get("/get_sprite_sheet")
async def get_sprite_sheet():
    # Serve the sprite sheet image file
    sprite_sheet_path = os.path.join('static', 'sprite_sheet.png')
    return FileResponse(sprite_sheet_path, media_type='image/png')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
