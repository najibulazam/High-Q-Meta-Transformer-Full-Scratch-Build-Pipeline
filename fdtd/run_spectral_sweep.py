# fdtd/run_spectral_sweep.py
import os
import sys
import numpy as np
import pandas as pd
import yaml

try:
    import lumapi
except ImportError:
    lum_paths = [
        r"C:\Program Files\Lumerical\v241\api\python",
        r"C:\Program Files\Lumerical\v232\api\python",
        r"C:\Program Files\ANSYS Inc\v241\Lumerical\api\python",
        r"C:\Program Files\ANSYS Inc\v232\Lumerical\api\python"
    ]
    for path in lum_paths:
        if os.path.exists(path):
            sys.path.append(path)
            break
    import lumapi

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from fdtd.setup_qBIC_cell import build_qbic_unit_cell

def run_spectral_sweep():
    num_samples = config['dataset']['num_samples']
    w_points = config['simulation']['wavelength_points']
    output_csv = config['project']['output_csv']
    
    print(f"Initializing FDTD Session for {num_samples} iterations...")
    fd = lumapi.FDTD()
    
    dataset_rows = []
    
    for i in range(num_samples):
        H = float(np.random.uniform(0.78e-6, 0.82e-6))
        Px = float(np.random.uniform(4.02e-6, 4.07e-6))
        Py = float(np.random.uniform(2.32e-6, 2.36e-6))
        R1 = float(np.random.uniform(0.31e-6, 0.46e-6))
        R2 = float(np.random.uniform(0.1115e-6, 0.1130e-6))
        
        build_qbic_unit_cell(fd, H, Px, Py, R1, R2, theta=10.0)
        
        print(f"Running simulation index: {i+1}/{num_samples}")
        fd.run()
        
        R_result = fd.getresult("R_monitor", "R")
        reflection = np.abs(R_result['R']).flatten()
        
        row_data = {
            "H_um": H * 1e6,
            "Px_um": Px * 1e6,
            "Py_um": Py * 1e6,
            "R1_um": R1 * 1e6,
            "R2_um": R2 * 1e6
        }
        
        for idx in range(min(w_points, len(reflection))):
            row_data[f"r_{idx}"] = float(reflection[idx])
            
        dataset_rows.append(row_data)
        
        if (i + 1) % 5 == 0:
            df_temp = pd.DataFrame(dataset_rows)
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_temp.to_csv(output_csv, index=False)
            print(f"Checkpoint saved at iteration {i+1}")

    df = pd.DataFrame(dataset_rows)
    df.to_csv(output_csv, index=False)
    fd.close()
    print("FDTD spectral data generation complete.")

if __name__ == "__main__":
    run_spectral_sweep()