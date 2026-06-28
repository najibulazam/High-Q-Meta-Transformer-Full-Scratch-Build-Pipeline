# train/train_meta_vit.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from train.spectral_dataset import get_data_loaders
from model.meta_vit import MetaViT
from model.spec_vit import SpecViT

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_data_loaders()
    
    os.makedirs("saved_models", exist_ok=True)
    spec_vit_path = "saved_models/best_spec_vit.pth"
    if not os.path.exists(spec_vit_path):
        raise FileNotFoundError(f"Pretrained SpecViT model not found at '{spec_vit_path}'! Please run 'python -m train.train_spec_vit' first.")

    # Load Pre-trained SpecViT for Spectrum-Matching Loss
    forward_model = SpecViT().to(device)
    forward_model.load_state_dict(torch.load(spec_vit_path, map_location=device))
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False
        
    model = MetaViT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['model']['learning_rate'], weight_decay=1e-4)
    
    criterion_geo = nn.MSELoss()
    criterion_spec = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(config['model']['epochs']):
        model.train()
        t_loss = 0.0
        for geos, specs in train_loader:
            geos, specs = geos.to(device), specs.to(device)
            optimizer.zero_grad()
            
            pred_geos = model(specs)
            loss_geo = criterion_geo(pred_geos, geos)
            
            # Spectrum Matching Accuracy Check
            pred_specs = forward_model(pred_geos)
            loss_spec = criterion_spec(pred_specs, specs)
            
            total_loss = loss_geo + 0.5 * loss_spec
            total_loss.backward()
            optimizer.step()
            t_loss += total_loss.item() * specs.size(0)
            
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for geos, specs in val_loader:
                geos, specs = geos.to(device), specs.to(device)
                pred_geos = model(specs)
                v_loss += (criterion_geo(pred_geos, geos) + 0.5 * criterion_spec(forward_model(pred_geos), specs)).item() * specs.size(0)
                
        t_loss /= len(train_loader.dataset)
        v_loss /= len(val_loader.dataset)
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), "saved_models/best_meta_vit.pth")
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['model']['epochs']}] | Combined Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f}")

if __name__ == "__main__":
    train()