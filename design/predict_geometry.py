# design/predict_geometry.py
import os
import pandas as pd
import torch
import joblib
from model.meta_vit import MetaViT

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_csv = "dataset/target_spectrum.csv"
    model_path = "saved_models/best_meta_vit.pth"
    scaler_path = "saved_models/scaler_geo.pkl"

    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Missing target spectrum! Please run 'python -m design.target_spectrum' first. File not found: {target_csv}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing trained model! Please train MetaViT model ('python -m train.train_meta_vit') first. File not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler! Please run model training scripts first. File not found: {scaler_path}")

    target_df = pd.read_csv(target_csv)
    specs = torch.tensor(target_df.values, dtype=torch.float32).to(device)
    
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    meta_cfg = config['model']['meta_vit']
    inverse_model = MetaViT(
        spectrum_points=config['simulation']['wavelength_points'],
        num_parameters=config['model']['num_parameters'],
        embed_dim=meta_cfg['embed_dim'],
        num_heads=meta_cfg['num_heads'],
        depth=meta_cfg['depth'],
        dim_feedforward=meta_cfg['dim_feedforward']
    ).to(device)
    inverse_model.load_state_dict(torch.load(model_path, map_location=device))
    inverse_model.eval()
    
    with torch.no_grad():
        pred_scaled_geo = inverse_model(specs).cpu().numpy()
        
    scaler_geo = joblib.load(scaler_path)
    actual_geo = scaler_geo.inverse_transform(pred_scaled_geo)[0]
    
    print("\n--- Optimized Meta-Atom Specifications Found ---")
    print(f"Predicted Height (H): {actual_geo[0]:.4f} um")
    print(f"Predicted Period X (Px): {actual_geo[1]:.4f} um")
    print(f"Predicted Period Y (Py): {actual_geo[2]:.4f} um")
    print(f"Predicted Semi-minor Axis (R1): {actual_geo[3]:.4f} um")
    print(f"Predicted Semi-major Axis (R2): {actual_geo[4]:.4f} um")

if __name__ == "__main__":
    predict()