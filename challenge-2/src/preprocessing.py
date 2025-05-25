import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle

class SoilDataset(Dataset):
    """
    Custom dataset class for loading soil images.
    Args:
        df (pd.DataFrame): DataFrame containing image IDs.
        data_dir (str): Directory containing the images.
        transform (callable, optional): Transformations to apply to images.
    """
    def _init_(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.valid_indices = []
        for idx in range(len(self.df)):
            img_id = self.df.iloc[idx]['image_id']
            img_path = os.path.join(self.data_dir, img_id)
            try:
                with Image.open(img_path) as img:
                    img.convert('RGB')
                self.valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping invalid image {img_path}: {e}")

    def _len_(self):
        return len(self.valid_indices)

    def _getitem_(self, idx):
        actual_idx = self.valid_indices[idx]
        img_id = self.df.iloc[actual_idx]['image_id']
        img_path = os.path.join(self.data_dir, img_id)
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id

def get_transforms():
    """
    Define image transformations for preprocessing.
    Returns:
        transform (callable): Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(device):
    """
    Load pretrained EfficientNet-B0 model for feature extraction.
    Args:
        device (torch.device): Device to load the model on (CPU/GPU).
    Returns:
        model (torch.nn.Module): Pretrained model with classifier removed.
    """
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier = torch.nn.Identity()  # Remove classifier to output 1280-D features
    model = model.to(device)
    model.eval()
    return model

def extract_features(model, loader, device, desc):
    """
    Extract features from a dataset using the pretrained model.
    Args:
        model (torch.nn.Module): Pretrained model.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the model on.
        desc (str): Description for the progress bar.
    Returns:
        features (np.ndarray): Extracted features.
        img_ids (list): List of image IDs.
    """
    features = []
    img_ids = []
    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            img_ids.extend(ids)
    return np.concatenate(features, axis=0), img_ids

def preprocess_data(csv_path, data_dir, batch_size=32, n_components=100, save_pca_path='pca_model.pkl'):
    """
    Preprocess the dataset: load data, extract features, and apply PCA.
    Args:
        csv_path (str): Path to the CSV file with image IDs.
        data_dir (str): Directory containing the images.
        batch_size (int): Batch size for DataLoader.
        n_components (int): Number of PCA components.
        save_pca_path (str): Path to save the PCA model.
    Returns:
        features_pca (np.ndarray): PCA-transformed features.
        img_ids (list): List of image IDs.
        pca (PCA): Fitted PCA model.
    """
    # Load data
    df = pd.read_csv(csv_path)
    transform = get_transforms()
    dataset = SoilDataset(df, data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    # Extract features
    features, img_ids = extract_features(model, loader, device, f"Extracting Features ({data_dir.split('/')[-1]})")

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features)
    print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

    # Save PCA model for inference
    with open(save_pca_path, 'wb') as f:
        pickle.dump(pca, f)

    return features_pca, img_ids, pca

def preprocess_single_image(image_path, model, pca, device, transform):
    """
    Preprocess a single image for inference.
    Args:
        image_path (str): Path to the image.
        model (torch.nn.Module): Pretrained model.
        pca (PCA): Fitted PCA model.
        device (torch.device): Device to run the model on.
        transform (callable): Image transformations.
    Returns:
        feature_pca (np.ndarray): PCA-transformed feature.
    """
    with Image.open(image_path) as img:
        image = img.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image_tensor).cpu().numpy()
    feature_pca = pca.transform(feature)
    return feature_pca