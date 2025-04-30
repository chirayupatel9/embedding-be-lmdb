import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cupy as cp
from cuml.manifold import TSNE
import numpy as np

def run_tsne_on_images_gpu(
    image_folder: str,
    batch_size: int = 32,
    n_components: int = 2,
    perplexity: int = 30,
    learning_rate: int = 200,
    n_iter: int = 1000
):
    """
    Extracts CNN features from images in a folder and performs GPU-accelerated t-SNE using cuML.

    Args:
        image_folder (str): Path to the folder containing images (.png, .jpg).
        batch_size (int): Batch size for feature extraction.
        n_components (int): t-SNE output dimensions (2 or 3).
        perplexity (int): t-SNE perplexity.
        learning_rate (int): t-SNE learning rate.
        n_iter (int): Number of t-SNE iterations.

    Returns:
        features_2d (np.ndarray): 2D t-SNE coordinates.
        filenames (List[str]): Corresponding filenames for each point.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Custom dataset loader
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

    # Dataset and loader
    dataset = ImageDataset(image_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load ResNet model and remove classifier
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(DEVICE)
    model.eval()

    # Extract features
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

    # GPU-based t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                learning_rate=learning_rate, n_iter=n_iter)
    features_2d_cp = tsne.fit_transform(features_cp)
    features_2d = cp.asnumpy(features_2d_cp)

    return features_2d, filenames
