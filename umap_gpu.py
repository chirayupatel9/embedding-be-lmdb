from cuml.manifold import UMAP
import cupy as cp
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

def run_umap_on_images_gpu(
    image_folder: str,
    batch_size: int = 32,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1
):
    """
    Extracts CNN features from images and performs GPU-accelerated UMAP.

    Args:
        image_folder (str): Path to images.
        batch_size (int): Batch size for processing.
        n_components (int): Output dimension of UMAP.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP.

    Returns:
        features_2d (np.ndarray): UMAP 2D coordinates.
        filenames (List[str]): Corresponding filenames.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    class ImageDataset(Dataset):
        def __init__(self, folder_path, transform=None):
            self.image_paths = [os.path.join(folder_path, f)
                                for f in os.listdir(folder_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, os.path.basename(self.image_paths[idx])

    dataset = ImageDataset(image_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(DEVICE)
    model.eval()

    features_list = []
    filenames = []

    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            out = out.view(out.size(0), -1)
            features_list.append(out.cpu().numpy())
            filenames.extend(fnames)

    features_np = np.concatenate(features_list, axis=0)
    features_cp = cp.asarray(features_np)

    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    embeddings_cp = reducer.fit_transform(features_cp)
    embeddings = cp.asnumpy(embeddings_cp)

    return embeddings, filenames
