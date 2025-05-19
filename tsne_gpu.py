import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cupy as cp
from cuml.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Import LMDB functions (update this to your actual imports)
from db_functions_lmdb import (
    get_all_documents as db_get_all_documents,
    get_image as db_get_image,
)

# ------------------------- Model Definition ------------------------- #
def resnet50_custom(in_channels=3, n_classes=17, dropout=0.5, weights=None):
    model = models.resnet50(weights=weights)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
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

# ------------------------- Image Loader From LMDB ------------------------- #
def load_images_from_lmdb(documents, transform):
    images = []
    filenames = []

    def load(doc):
        try:
            img_data = db_get_image(doc["image_id"])
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img_tensor = transform(img)
            return img_tensor, doc.get("filename", f"{doc['image_id']}.png")
        except Exception as e:
            print(f"Error loading image {doc.get('image_id', 'unknown')}: {str(e)}")
            return None, None

    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(load, documents))

    for img_tensor, filename in results:
        if img_tensor is not None:
            images.append(img_tensor)
            filenames.append(filename)

    return images, filenames

# ------------------------- Main Pipeline ------------------------- #
def main():
    # Settings
    pth_path = "yichen_model.pth"
    device_str = "cuda:4" if torch.cuda.is_available() else "cpu"
    batch_size = 512
    lmdb_batch_size = 4000

    # 1. Load Model
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model = resnet50_custom(in_channels=3, n_classes=17)
    checkpoint = torch.load(pth_path, map_location=device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_k = k.replace("module.", "")
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    model.fc = nn.Sequential(*list(model.fc.children())[:6])

    print("✅ Model loaded and ready.")

    # 2. Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 3. Load documents from LMDB
    documents = db_get_all_documents()
    print(f"✅ Retrieved {len(documents)} documents from LMDB.")

    embeddings = []
    file_names = []

    # 4. Process in batches
    for batch_start in tqdm(range(0, len(documents), lmdb_batch_size), desc="Processing LMDB Batches"):
        batch_docs = documents[batch_start:batch_start + lmdb_batch_size]
        images, filenames = load_images_from_lmdb(batch_docs, transform)

        if not images:
            continue

        loader = DataLoader(list(zip(images, filenames)), batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                imgs, names = batch
                imgs = imgs.to(device)
                feats = model(imgs)
                embeddings.append(feats.cpu().numpy())
                file_names.extend(names)

        torch.cuda.empty_cache()

    embeddings = np.vstack(embeddings)
    print(f"✅ Extracted embeddings shape: {embeddings.shape}")

    # 5. Run cuML t-SNE
    print("🚀 Running cuML t-SNE...")
    embeddings_gpu = cp.asarray(embeddings)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=1)
    tsne_result_gpu = tsne.fit_transform(embeddings_gpu)
    tsne_result = cp.asnumpy(tsne_result_gpu)

    print(f"✅ t-SNE completed. Result shape: {tsne_result.shape}")

    # 6. Save Results
    df = pd.DataFrame(tsne_result, columns=["x", "y"])
    df["filename"] = file_names
    df.to_csv("tsne_lmdb_output.csv", index=False)
    print("✅ t-SNE results saved to 'tsne_lmdb_output.csv'.")

    # 7. Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=10)
    plt.title("cuML t-SNE Visualization from LMDB Images")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
