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
    num_samples  = config['dataset']['num_samples']
    w_points     = config['simulation']['wavelength_points']
    output_csv   = config['project']['output_csv']
    seed         = config['dataset'].get('random_seed', 42)

    rng = np.random.default_rng(seed)

    print(f"Initializing FDTD Session for {num_samples} iterations...", flush=True)
    fd = lumapi.FDTD()

    dataset_rows   = []
    wavelengths_um = None
    n_converged    = 0
    n_discarded    = 0

    for i in range(num_samples):
        H   = float(rng.uniform(0.78e-6,   0.82e-6))
        Px  = float(rng.uniform(4.02e-6,   4.07e-6))
        Py  = float(rng.uniform(2.32e-6,   2.36e-6))
        R1  = float(rng.uniform(0.31e-6,   0.46e-6))
        R2  = float(rng.uniform(1.115e-6,  1.130e-6))
        dx  = float(rng.uniform(0.95e-6,   1.05e-6))  # ellipse-center offset

        build_qbic_unit_cell(fd, H, Px, Py, R1, R2, dx, theta=10.0)

        print(f"Running simulation index: {i + 1}/{num_samples}", flush=True)
        fd.run()

        # -- Convergence check before trusting the result ---------------
        # autoshutoff status: 1 = converged (field decayed below the
        # auto-shutoff threshold), 2 = ran to max simulation time without
        # full decay. For a high-Q qBIC resonance, status 2 means the DFT
        # monitors may not have captured the full ringdown -- discard.
        status = fd.getresult("FDTD", "status")
        if status not in (1, 2):
            # Anything else (0 / error) means the run did not complete.
            print(f"  -> Run {i+1} did not complete (status={status}); skipping.", flush=True)
            n_discarded += 1
            continue
        if status == 2:
            print(f"  -> Run {i+1} ran to max simulation time (status={status}); keeping result.", flush=True)

        R_result = fd.getresult("R", "T")
        T_result = fd.getresult("T", "T")

        # Power-monitor "T" output is the *normalized* power flow through
        # the monitor plane, already real-valued. R monitor sits above the
        # source and faces +z, but it is recording power flowing in -z
        # (the reflected wave going back out the top) so Lumerical reports
        # it as negative; flip sign rather than blindly taking abs(), so a
        # genuine power-conservation bug (R+T>1) isn't silently masked.
        reflection   = (-np.asarray(R_result['T'])).flatten()
        transmission = np.asarray(T_result['T']).flatten()

        if wavelengths_um is None:
            wavelengths_um = (np.asarray(R_result['lambda']).flatten() * 1e6)

        # Sanity check: power conservation (R + T <= 1, allowing small
        # numerical slack). Flag rather than silently accept.
        total = reflection + transmission
        if np.any(total > 1.02) or np.any(total < -0.02):
            print(f"  -> Warning: R+T outside [0,1] for run {i+1} "
                  f"(min={total.min():.3f}, max={total.max():.3f}). "
                  f"Check monitor placement/orientation.", flush=True)

        row_data = {
            "H_um":  H  * 1e6,
            "Px_um": Px * 1e6,
            "Py_um": Py * 1e6,
            "R1_um": R1 * 1e6,
            "R2_um": R2 * 1e6,
            "dx_um": dx * 1e6,
        }
        for idx in range(min(w_points, len(reflection))):
            row_data[f"r_{idx}"] = float(reflection[idx])
        for idx in range(min(w_points, len(transmission))):
            row_data[f"t_{idx}"] = float(transmission[idx])

        dataset_rows.append(row_data)
        n_converged += 1

        # Save checkpoint on every converged sample so user sees real-time progress on disk
        df_temp = pd.DataFrame(dataset_rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_temp.to_csv(output_csv, index=False)
        print(f"Checkpoint saved at {n_converged} converged samples "
              f"({n_discarded} discarded so far).", flush=True)

    df = pd.DataFrame(dataset_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Save the wavelength axis once, alongside the main dataset, so r_idx /
    # t_idx columns can be mapped back to physical wavelength.
    if wavelengths_um is not None:
        wl_csv = os.path.splitext(output_csv)[0] + "_wavelengths.csv"
        pd.DataFrame({"index": np.arange(len(wavelengths_um)),
                      "wavelength_um": wavelengths_um}).to_csv(wl_csv, index=False)
        print(f"Wavelength axis saved to {wl_csv}")

    fd.close()
    print(f"FDTD spectral data generation complete. "
          f"{n_converged} converged, {n_discarded} discarded out of {num_samples}.")


if __name__ == "__main__":
    run_spectral_sweep()