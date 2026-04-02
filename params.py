"""Shared calibration parameters for the Romer study project."""

from copy import deepcopy


SOLOW = {
    "alpha": 0.33,
    "s": 0.20,
    "n": 0.01,
    "g": 0.02,
    "delta": 0.05,
    "A0": 1.0,
    "L0": 1.0,
}

RCK = {
    "alpha": 0.33,
    "rho": 0.04,
    "theta": 2.0,
    "n": 0.01,
    "g": 0.02,
    "delta": 0.05,
    "G": 0.0,
}

DIAMOND = {
    "alpha": 0.33,
    "beta": 0.5,
    "n": 0.01,
    "delta": 1.0,
}

RBC = {
    "alpha": 0.33,
    "beta": 0.99,
    "delta": 0.025,
    "rho_z": 0.95,
    "sigma_z": 0.007,
}

NK = {
    "beta": 0.99,
    "sigma": 1.0,
    "kappa": 0.1,
    "phi_pi": 1.5,
    "phi_x": 0.5,
    "rho_v": 0.5,
}

BRASIL = {
    "alpha": 0.40,
    "s": 0.18,
    "n": 0.008,
    "g": 0.015,
    "delta": 0.05,
    "rho": 0.06,
    "theta": 2.0,
    "G": 0.0,
}

OECD_COUNTRIES = [
    "AUS", "AUT", "BEL", "CAN", "CHE", "CHL", "COL", "CRI", "CZE", "DEU",
    "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HUN", "IRL", "ISL",
    "ISR", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX", "NLD", "NOR",
    "NZL", "POL", "PRT", "SVK", "SVN", "SWE", "TUR", "USA",
]


def clone_params(base, *overrides):
    """Return a defensive copy of a parameter dictionary."""

    params = deepcopy(base)
    for override in overrides:
        params.update(override)
    return params
