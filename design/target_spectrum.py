# design/target_spectrum.py
import os
import numpy as np
import pandas as pd
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def generate_fano_qbic_target():
    points = config['simulation']['wavelength_points']
    w_start = config['simulation']['wavelength_start_nm']
    w_end = config['simulation']['wavelength_end_nm']
    
    wavelengths = np.linspace(w_start, w_end, points)
    
    # Generate Synthetic High-Q Fano Resonance Profile
    w_0 = 6000.0  # Peak center (6 um per Section 7)
    gamma = 20.0  # Narrow linewidth for High-Q
    q_parameter = -1.2
    
    epsilon = (wavelengths - w_0) / gamma
    fano = ((q_parameter + epsilon) ** 2) / (1 + epsilon ** 2)
    target_r = (fano - np.min(fano)) / (np.max(fano) - np.min(fano))
    
    df = pd.DataFrame(target_r.reshape(1, -1), columns=[f"r_{i}" for i in range(points)])
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/target_spectrum.csv", index=False)
    print("Target high-Q Fano spectrum generated.")

if __name__ == "__main__":
    generate_fano_qbic_target()