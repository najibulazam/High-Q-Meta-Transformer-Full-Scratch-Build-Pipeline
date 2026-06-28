# fdtd/setup_qBIC_cell.py
import os
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def build_qbic_unit_cell(fd, H, Px, Py, R1, R2, theta=10.0):
    fd.clear()
    fd.switchtolayout()
    fd.selectall()
    fd.deleteall()
    
    # 1. Setup 3D FDTD Region
    fd.addfdtd()
    fd.setnamed("FDTD", "dimension", "3D")  # 3D Simulation for unit cell
    fd.setnamed("FDTD", "x", 0.0)
    fd.setnamed("FDTD", "y", 0.0)
    fd.setnamed("FDTD", "x span", Px)  
    fd.setnamed("FDTD", "y span", Py)  
    fd.setnamed("FDTD", "z", 1.0e-6)
    fd.setnamed("FDTD", "z span", 4.0e-6)
    
    # Set Boundary Conditions (X,Y: Periodic, Z: PML)
    fd.setnamed("FDTD", "x min bc", "Periodic")
    fd.setnamed("FDTD", "y min bc", "Periodic")
    fd.setnamed("FDTD", "z min bc", "PML")
    fd.setnamed("FDTD", "z max bc", "PML")
    fd.setnamed("FDTD", "mesh accuracy", config['simulation']['mesh_accuracy'])
    
    # 2. Add CaF2 Substrate
    fd.addrect()
    fd.setnamed("rectangle", "name", "substrate")
    fd.setnamed("substrate", "x", 0.0)
    fd.setnamed("substrate", "x span", Px * 2)
    fd.setnamed("substrate", "y", 0.0)
    fd.setnamed("substrate", "y span", Py * 2)
    fd.setnamed("substrate", "z max", 0.0)
    fd.setnamed("substrate", "z min", -2.0e-6)
    fd.setnamed("substrate", "material", "CaF2 (Calcium Fluoride) - Palik")
    
    # 3. Add Single Elliptical Germanium Resonator
    fd.addcylinder()
    fd.setnamed("cylinder", "name", "Ge_Resonator")
    fd.setnamed("Ge_Resonator", "material", "Ge (Germanium) - CRC")
    fd.setnamed("Ge_Resonator", "x", 0.0)
    fd.setnamed("Ge_Resonator", "y", 0.0)
    fd.setnamed("Ge_Resonator", "z min", 0.0)
    fd.setnamed("Ge_Resonator", "z max", H)
    fd.setnamed("Ge_Resonator", "x span", 2 * R1)
    fd.setnamed("Ge_Resonator", "y span", 2 * R2)
    fd.setnamed("Ge_Resonator", "first axis", "z")
    fd.setnamed("Ge_Resonator", "rotation", theta)
    
    # 4. Add Plane Wave Source (Oblique Angle 36 degrees)
    fd.addplane()
    fd.setnamed("source", "name", "source")
    fd.setnamed("source", "injection axis", "z")
    fd.setnamed("source", "direction", "Forward")
    fd.setnamed("source", "z", -0.5e-6)
    fd.setnamed("source", "angle theta", 36)
    fd.setnamed("source", "polarization angle", 0)
    fd.setnamed("source", "wavelength start", config['simulation']['wavelength_start_nm'] * 1e-9)
    fd.setnamed("source", "wavelength stop", config['simulation']['wavelength_end_nm'] * 1e-9)
    
    # 5. Add Reflection Power Monitor
    fd.addpower()
    fd.setnamed("monitor", "name", "R_monitor")
    fd.setnamed("R_monitor", "monitor type", "2D Z-normal")
    fd.setnamed("R_monitor", "z", -0.8e-6)
    fd.setnamed("R_monitor", "override global monitor settings", True)
    fd.setnamed("R_monitor", "frequency points", config['simulation']['wavelength_points'])