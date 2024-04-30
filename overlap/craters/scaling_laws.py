# pylint: disable-msg=C0103
# pylint: disable-msg=R0913
# pylint: disable-msg=W0613

"""
Module containing methods to calculate crater diameter from impactor properties
"""

import numpy as np


def calc_transient_crater(D_imp, M_imp, v_imp, rho_imp, theta_imp):
    """
    Calculate the diameter of the transient crater.

    Parameters:
    - D_imp (float): impactor diameter
    - M_imp (float): impactor mass
    - v_imp (float): impact velocity
    - rho_imp (float): impactor bulk Density
    - theta_imp (float): impact angle

    Returns:
    - float: Diameter of the transient crater
    """

    rho_tar = 2.5e3

    g = 9.81

    D_tr = 1.44 * (M_imp ** 0.113) * ((M_imp * v_imp ** 2 / g) ** 0.22) * (D_imp ** -0.22) *\
            (np.sin(theta_imp) ** (1/3)) / (rho_tar ** (1/3))

    return D_tr


def calc_crater_diameter(D_imp, M_imp, v_imp, theta, rho_imp=0.6e3):
    """
    Calculate the final crater diameter.

    Parameters:
    - D_imp (float): impactor diameter
    - M_imp (float): impactor mass
    - v_imp (float): impact velocity
    - theta (float): Impact angle
    - rho_imp (float): impactor bulk density

    Returns:
    - float: Final crater diameter
    """

    D_tr = calc_transient_crater(D_imp, M_imp, v_imp, rho_imp, theta)

    if D_tr < 4e3:
        D_f = 1.25 * D_tr
    elif D_tr >= 4e3:
        D_f = 1.17 * (D_tr ** 1.13) * (4e3 ** -0.13)

    return D_f
