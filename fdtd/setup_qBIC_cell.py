# fdtd/setup_qBIC_cell.py
"""
Builds the qBIC dimer unit cell exactly as verified against the
hand-built reference model "Forward Design.fsp":

    FDTD region          -- 3D, periodic X/Y, PML Z
    CaF2-sub             -- rectangle, object-defined dielectric n=1.43, below z=0
    Left ellipse         -- Ge ellipsoid, rotated -theta about z, x = -dx
    Right ellipse        -- Ge ellipsoid, rotated +theta about z, x = +dx
    source               -- z-injected plane wave, direction "Backward"
                            (propagates -z, i.e. illuminates from above)
    R                     -- 2D Z-normal power monitor ABOVE the source
                            (z = +8 um in the reference), catches reflection
    T                     -- 2D Z-normal power monitor BELOW the substrate
                            (z = -1 um in the reference), catches transmission

Object names match the reference tree exactly: CaF2-sub, Left ellipse,
Right ellipse, source, R, T -- so getresult() calls and any saved .fsp
analysis scripts stay consistent.
"""
import yaml
import numpy as np

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def _check_ellipse_geometry(fd, name, expected_x, expected_r1, expected_r2, expected_h,
                             tol=1e-12):
    """
    Read back what Lumerical actually stored for an ellipsoid circle object
    and correct it if needed.

    Some Lumerical versions have a known bug (reported on the Ansys forum)
    where enabling "make ellipsoid" on a circle object doubles the stored
    x/y center coordinates relative to what was set. This silently produces
    a structure offset from the intended unit-cell position -- in the worst
    case (large dx relative to Px) pushing it outside the visible/meshed
    region entirely, which looks exactly like a "missing" resonator.

    This function checks for that and re-applies the correct x position if
    detected, and also verifies "radius 2" / "z span" actually stuck (in
    case a dependent-property write was silently dropped), raising a clear
    error instead of letting an empty/invisible structure pass silently.
    """
    actual_x = fd.getnamed(name, "x")
    if abs(actual_x - expected_x) > tol and abs(actual_x - 2 * expected_x) <= tol:
        print(f"  -> Detected known 'make ellipsoid' x-doubling bug on '{name}' "
              f"(got {actual_x*1e6:.4f} um, expected {expected_x*1e6:.4f} um); "
              f"re-applying correct x.")
        fd.setnamed(name, "x", expected_x)

    actual_r2 = fd.getnamed(name, "radius 2")
    actual_zspan = fd.getnamed(name, "z span")
    actual_enabled = fd.getnamed(name, "enabled")
    if not actual_enabled:
        raise RuntimeError(
            f"'{name}': object is disabled (enabled=False) after creation. "
            f"This object will not render or mesh even with correct geometry.")
    if actual_r2 <= 0 or abs(actual_r2 - expected_r2) > max(tol, 0.05 * expected_r2):
        raise RuntimeError(
            f"'{name}': radius 2 did not apply correctly "
            f"(got {actual_r2*1e9:.2f} nm, expected {expected_r2*1e9:.2f} nm). "
            f"Check property-setting order for this Lumerical version.")
    if actual_zspan <= 0 or abs(actual_zspan - expected_h) > max(tol, 0.05 * expected_h):
        raise RuntimeError(
            f"'{name}': z span did not apply correctly "
            f"(got {actual_zspan*1e9:.2f} nm, expected {expected_h*1e9:.2f} nm). "
            f"Check property-setting order for this Lumerical version.")


def build_qbic_unit_cell(fd, H, Px, Py, R1, R2, dx, theta=10.0):
    """
    Build the qBIC dimer unit cell in an active Lumerical FDTD session.

    Parameters
    ----------
    fd    : lumapi.FDTD  -- active FDTD session
    H     : float        -- ellipsoid height / z span (m)
    Px    : float        -- unit-cell period along X (m)
    Py    : float        -- unit-cell period along Y (m)
    R1    : float        -- ellipsoid radius 1, in-plane "minor" (m)
    R2    : float        -- ellipsoid radius 2, in-plane "major" (m)
    dx    : float        -- half-separation between the two ellipse centers
                            along X (m); centers sit at -dx and +dx
    theta : float        -- rotation magnitude (degrees); left ellipse is
                            rotated -theta, right ellipse is rotated +theta
    """
    fd.switchtolayout()
    fd.selectall()
    fd.deleteall()

    # -- 1. 3-D FDTD Region ---------------------------------------------------
    fd.addfdtd()
    # NOTE: unlike rectangle/circle/source/monitor objects, the FDTD region's
    # "name" property is inactive/read-only -- addfdtd() always creates it
    # already named "FDTD", so no rename step is needed (or possible) here.
    fd.setnamed("FDTD", "dimension",      "3D")
    fd.setnamed("FDTD", "x",              0.0)
    fd.setnamed("FDTD", "y",              0.0)
    fd.setnamed("FDTD", "x span",         Px)
    fd.setnamed("FDTD", "y span",         Py)
    fd.setnamed("FDTD", "z",              4.3e-6)
    fd.setnamed("FDTD", "z span",         11.4e-6)   # z in [-1.4, 10] um
    fd.setnamed("FDTD", "x min bc",       "Periodic")
    fd.setnamed("FDTD", "y min bc",       "Periodic")
    fd.setnamed("FDTD", "z min bc",       "PML")
    fd.setnamed("FDTD", "z max bc",       "PML")
    fd.setnamed("FDTD", "mesh accuracy",  config['simulation']['mesh_accuracy'])

    # -- 2. CaF2 Substrate -----------------------------------------------------
    # NOTE: "CaF2" is NOT in the default Lumerical material database (the
    # reference .fsp must have had a custom material added locally, or saved
    # with one baked into the file). Rather than depend on that, CaF2 is
    # defined here directly as an object-defined dielectric so the script
    # is portable across installations.
    #
    # Most published CaF2 Sellmeier fits (Malitson 1963 / Li 1980) are only
    # validated up to ~5.6 um, while this sweep runs to 9 um -- extrapolating
    # a Sellmeier fit past its validated range is its own source of error.
    # A flat real index (n ~ 1.43, CaF2's well-known near-IR/visible value)
    # is the safer default absent a vetted extended-range dispersion model.
    # If you have a specific CaF2 (n,k) dataset valid out to 9 um for your
    # paper, swap the block below for fd.addmaterial("(n,k) Material") +
    # fd.importnk2object(...) instead.
    fd.addrect()
    fd.set("name", "CaF2-sub")
    fd.setnamed("CaF2-sub",  "x",        0.0)
    fd.setnamed("CaF2-sub",  "x span",   Px)
    fd.setnamed("CaF2-sub",  "y",        0.0)
    fd.setnamed("CaF2-sub",  "y span",   Py)
    fd.setnamed("CaF2-sub",  "z",        -2.0e-6)
    fd.setnamed("CaF2-sub",  "z span",   4.0e-6)      # z in [-4, 0] um
    fd.setnamed("CaF2-sub",  "material", "<Object defined dielectric>")
    fd.setnamed("CaF2-sub",  "index",    1.43)

    # -- 3. Ge dimer: two ellipsoids, equal & opposite rotation, offset in X --
    # Built with addcircle() + "make ellipsoid" (radius / radius 2), matching
    # the reference model -- NOT an extruded polygon. This avoids any
    # polygon-vertex faceting error in the curved sidewalls.
    z_center = H / 2.0

    fd.addcircle()
    fd.set("name", "Left ellipse")
    fd.setnamed("Left ellipse", "make ellipsoid",  True)
    fd.setnamed("Left ellipse", "x",              -dx)
    fd.setnamed("Left ellipse", "y",               0.0)
    fd.setnamed("Left ellipse", "z",               z_center)
    fd.setnamed("Left ellipse", "z span",          H)
    fd.setnamed("Left ellipse", "radius",          R1)
    fd.setnamed("Left ellipse", "radius 2",        R2)
    fd.setnamed("Left ellipse", "material",        "Ge (Germanium) - CRC")
    fd.setnamed("Left ellipse", "first axis",      "z")
    fd.setnamed("Left ellipse", "rotation 1",      -theta)
    fd.setnamed("Left ellipse", "enabled",         True)
    fd.setnamed("Left ellipse", "override mesh order from material database", True)
    fd.setnamed("Left ellipse", "mesh order",      1)   # force highest priority
    _check_ellipse_geometry(fd, "Left ellipse", -dx, R1, R2, H)

    fd.addcircle()
    fd.set("name", "Right ellipse")
    fd.setnamed("Right ellipse", "make ellipsoid",  True)
    fd.setnamed("Right ellipse", "x",              dx)
    fd.setnamed("Right ellipse", "y",               0.0)
    fd.setnamed("Right ellipse", "z",               z_center)
    fd.setnamed("Right ellipse", "z span",          H)
    fd.setnamed("Right ellipse", "radius",          R1)
    fd.setnamed("Right ellipse", "radius 2",        R2)
    fd.setnamed("Right ellipse", "material",        "Ge (Germanium) - CRC")
    fd.setnamed("Right ellipse", "first axis",      "z")
    fd.setnamed("Right ellipse", "rotation 1",      theta)
    fd.setnamed("Right ellipse", "enabled",         True)
    fd.setnamed("Right ellipse", "override mesh order from material database", True)
    fd.setnamed("Right ellipse", "mesh order",      1)   # force highest priority
    _check_ellipse_geometry(fd, "Right ellipse", dx, R1, R2, H)

    # -- 4. Plane-Wave Source (normal incidence, illuminating from +z) -------
    # "direction: Backward" with injection axis z means the wave propagates
    # in -z, i.e. comes from above and travels down through the resonators
    # into the substrate -- matching the reference model exactly.
    fd.addplane()
    fd.set("name", "source")
    fd.setnamed("source", "injection axis",     "z-axis")
    fd.setnamed("source", "direction",          "Backward")
    fd.setnamed("source", "x",                  0.0)
    fd.setnamed("source", "x span",             Px * 2)
    fd.setnamed("source", "y",                  0.0)
    fd.setnamed("source", "y span",             Py * 2)
    fd.setnamed("source", "z",                  7.0e-6)
    fd.setnamed("source", "plane wave type",    "Bloch/periodic")
    fd.setnamed("source", "angle theta",        36.0)     # Incident angle 36 deg per Section 6
    fd.setnamed("source", "angle phi",          90.0)     # Wave vector in YZ plane per Section 6
    fd.setnamed("source", "polarization angle", 0.0)      # E-field along x-axis per Section 6
    fd.setnamed("source", "override global source settings", True)
    fd.setnamed("source", "wavelength start",
                config['simulation']['wavelength_start_nm'] * 1e-9)
    fd.setnamed("source", "wavelength stop",
                config['simulation']['wavelength_end_nm']   * 1e-9)

    # -- 5. R monitor: ABOVE the source, catches the backward (reflected) ----
    #       power that propagates back out the top of the FDTD region.
    # NOTE: power monitors do not have "wavelength start"/"wavelength stop"
    # as direct settable properties (that produced "property not found").
    # "override global monitor settings" must stay True because "frequency
    # points" is inactive until that flag is set -- but we do NOT set an
    # explicit wavelength range here, so the monitor still uses the source's
    # wavelength band (5.4-9 um), just sampled at our chosen point count.
    fd.addpower()
    fd.set("name", "R")
    fd.setnamed("R",     "monitor type",                      "2D Z-normal")
    fd.setnamed("R",     "x",                                 0.0)
    fd.setnamed("R",     "x span",                            Px)
    fd.setnamed("R",     "y",                                 0.0)
    fd.setnamed("R",     "y span",                            Py)
    fd.setnamed("R",     "z",                                 8.0e-6)
    fd.setnamed("R",     "override global monitor settings",  True)
    fd.setnamed("R",     "frequency points",
                config['simulation']['wavelength_points'])

    # -- 6. T monitor: BELOW the substrate, catches the forward (transmitted)
    #       power that exits the bottom of the FDTD region.
    fd.addpower()
    fd.set("name", "T")
    fd.setnamed("T",     "monitor type",                      "2D Z-normal")
    fd.setnamed("T",     "x",                                 0.0)
    fd.setnamed("T",     "x span",                            Px)
    fd.setnamed("T",     "y",                                 0.0)
    fd.setnamed("T",     "y span",                            Py)
    fd.setnamed("T",     "z",                                 -1.0e-6)
    fd.setnamed("T",     "override global monitor settings",  True)
    fd.setnamed("T",     "frequency points",
                config['simulation']['wavelength_points'])