# train/train_spec_vit.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import joblib
from train.spectral_dataset import get_data_loaders
from model.spec_vit import SpecViT

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, scaler_geo = get_data_loaders()
    
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler_geo, "saved_models/scaler_geo.pkl")
    
    model = SpecViT(
        num_parameters=config['model']['num_parameters'],
        spectrum_points=config['simulation']['wavelength_points'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        depth=config['model']['depth']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['model']['learning_rate'], weight_decay=1e-4)
    
    best_loss = float('inf')
    for epoch in range(config['model']['epochs']):
        model.train()
        t_loss = 0.0
        for geos, specs in train_loader:
            geos, specs = geos.to(device), specs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(geos), specs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * geos.size(0)
            
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for geos, specs in val_loader:
                geos, specs = geos.to(device), specs.to(device)
                v_loss += criterion(model(geos), specs).item() * geos.size(0)
                
        t_loss /= len(train_loader.dataset)
        v_loss /= len(val_loader.dataset)
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), "saved_models/best_spec_vit.pth")
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['model']['epochs']}] | Train MSE: {t_loss:.6f} | Val MSE: {v_loss:.6f}")

if __name__ == "__main__":
    train()