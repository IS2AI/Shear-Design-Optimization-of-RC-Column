import numpy as np


class Column:
    def __init__(self):
        # The class Column contains all the static material, geometry and analysis parameters.

        # Geometric and analysis parameters
        self.colWidth_min = 0.2     # [m] Column minimum width
        self.cover = 0.04           # [m] Column cover to reinforcing steel NA
        self.num_fib = 16           # Number of fibers for concrete core and cover in y and z directions
        self.SecTag = 1             # Column section tag
        self.mu = 15.0              # Target ductility for analysis
        self.numIncr = 500          # Number of analysis increments

        # List of longitudinal and transverse rebar diameters
        self.d_rebars = np.asarray([0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.025, 0.028, 0.032, 0.040, 0.050]) # [m]
        self.d_rebars_tr = np.asarray([0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.025, 0.028, 0.032, 0.040, 0.050])  # [m]

        # Eurocode design parameters
        self.alpha_coef = 0.85      # Longterm effect of compressive strength
        self.nu_coef = 1.0          # Effective strength factor for fc<= 50 MPa
        self.g_c = 1.5              # Concrete safety factor
        self.g_s = 1.15             # Steel safety factor

        # Steel material parameters
        self.IDreinf = 3                # Steel material ID tag
        self.steel_weight = 0.0785      # [MN/m3] Unit weight of steel
        self.steel_weight_tr = 0.0785   # [MN/m3] Unit weight of steel
        self.fyk = 500                  # [MPa] Characteristic yield stress of longitudinal rebar
        self.fyk_tr = 500               # [MPa] Characteristic yield stress of transverse rebar
        self.Es = 210*1e3               # [MPa] Young's modulus of steel
        self.Bs = 0                     # Steel strain-hardening ratio

        # Concrete material parameters
        self.IDcon = 2                  # Concrete material ID tag
        self.concrete_weight = 0.024    # [MN/m3] Unit weight of concrete
        self.eps1U = -0.002             # Strain at maximum strength of concrete
        self.eps2U = -0.0035            # Strain at ultimate stress
