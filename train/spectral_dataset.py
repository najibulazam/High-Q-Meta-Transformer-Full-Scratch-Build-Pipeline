# train/spectral_dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class SpectralDataset(Dataset):
    def __init__(self, geometries, spectra):
        self.geometries = torch.tensor(geometries, dtype=torch.float32)
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        
    def __len__(self):
        return len(self.geometries)
        
    def __getitem__(self, idx):
        return self.geometries[idx], self.spectra[idx]

def get_data_loaders():
    csv_path = config['project']['output_csv']
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing training file: {csv_path}")
        
    df = pd.read_csv(csv_path)
    geo_cols = ["H_um", "Px_um", "Py_um", "R1_um", "R2_um"]
    spec_cols = [col for col in df.columns if col.startswith("r_")]
    
    X = df[geo_cols].values
    Y = df[spec_cols].values
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=config['dataset']['test_size'], random_state=42
    )
    
    scaler_geo = StandardScaler()
    X_train = scaler_geo.fit_transform(X_train)
    X_val = scaler_geo.transform(X_val)
    
    train_loader = DataLoader(SpectralDataset(X_train, Y_train), batch_size=config['dataset']['batch_size'], shuffle=True)
    val_loader = DataLoader(SpectralDataset(X_val, Y_val), batch_size=config['dataset']['batch_size'], shuffle=False)
    
    return train_loader, val_loader, scaler_geo