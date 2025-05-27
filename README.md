# visualizer-bke

## Extracting data from datafed
### If you have data locally you can copy the folder path and use it later for generating embedding
### Else you extract from datafed using 
- - adjust env according to your datafed configuration
- - **Must have a globus container running**
- - if you dont have globus container running you can checkout this 
- - then run the following 
```bash
python datafed_extractor.py
``` 

## Initialize docker container for mongoDb by 
```bash
docker run --name mongodb \
  -p 27017:27017 \
  --restart unless-stopped \
  -d mongodb/mongodb-community-server:latest



## Install python dependencies 
```bash
pip install -r requirements.txt
```

## Run the backend server locally 
```bash
python app.py
```
```mermaid
graph TD
    subgraph Frontend["Frontend - React/TypeScript"]
        FE["Embeddings App"]
        UI["User Interface"]
    end

    subgraph Backend["Backend - FastAPI"]
        API["API Endpoints"]
        FS["GridFS"]
        DB["MongoDB"]
        SPRITE["Sprite Sheet Generator"]
    end

    subgraph DataFlow["Data Flow"]
        Upload["Upload Images & Metadata"]
        Process["Process Data"]
        Visualize["Visualize Embeddings"]
    end

    %% Frontend to Backend connections
    FE -->|"HTTP Requests"| API
    UI -->|"User Interactions"| FE

    %% Backend internal connections
    API -->|"Store Images"| FS
    API -->|"Store Metadata"| DB
    API -->|"Generate"| SPRITE
    SPRITE -->|"Save"| FS
    SPRITE -->|"Save Metadata"| DB

    %% Data Flow connections
    Upload -->|"POST /api/upload-images-with-metadata"| API
    Process -->|"GET /api/embeddings"| API
    Visualize -->|"GET /api/get-image"| API

    %% API Endpoints
    subgraph Endpoints["API Endpoints"]
        UploadEP["/api/upload-images-with-metadata"]
        EmbeddingsEP["/api/embeddings"]
        GetImageEP["/api/get-image/{image_id}"]
        GetAllEP["/api/get-all-images"]
        GetDetailsEP["/api/get-image-details/{image_id}"]
    end

    %% Data Storage
    subgraph Storage["Storage"]
        Images["GridFS Images"]
        Metadata["MongoDB Documents"]
        SpriteSheet["Sprite Sheet"]
    end

    %% Connect endpoints to storage
    UploadEP --> Images
    UploadEP --> Metadata
    EmbeddingsEP --> SpriteSheet
    GetImageEP --> Images
    GetAllEP --> Images
    GetDetailsEP --> Metadata

    %% Style
    classDef frontend fill:#f9f,stroke:#333,stroke-width:2px
    classDef backend fill:#bbf,stroke:#333,stroke-width:2px
    classDef storage fill:#bfb,stroke:#333,stroke-width:2px
    classDef endpoint fill:#ffd,stroke:#333,stroke-width:2px

    class FE,UI frontend
    class API,FS,DB,SPRITE backend
    class Images,Metadata,SpriteSheet storage
    class UploadEP,EmbeddingsEP,GetImageEP,GetAllEP,GetDetailsEP endpoint
```

## Docker Setup

### Running Environment

To run the application in development mode:

1. Clone the repository
2. Navigate to the project directory
3. Run:

```bash
docker-compose -f docker-compose.yml up --build
```

This will:
- Mount the local code directory into the container for hot-reloading
- Run with the debug mode enabled
- Use non-authenticated MongoDB instance


## Once app is running
Running this api's in a sequence

- To load/upload data with folder path of metadata and images
```bash
/api/upload-images
```
- Followed by this run the following api with method as UMap and TSNE
```bash
/api/make_reduction/{method}
```

Later everything will be handled in UI.