# pylint: disable-msg=C0103
# pylint: disable-msg=R0913

"""
Module docstring
"""

import numpy as np
from scipy.optimize import fsolve

def transient_crater_diameter(D_tr, D_imp, rho_imp, rho_tar, v, theta, D_sg, g):
    """
    Docstring
    """

    left_side = D_tr / (D_imp * (rho_imp / rho_tar)**0.26 * (v * np.sin(theta))**0.55)
    right_side = 1.28 / ((D_sg + D_tr) * g)**0.28

    return left_side - right_side


def calc_transient_crater(D_imp, v_imp, rho_imp, theta_imp):
    """
    Docstring
    """

    rho_tar = 3e3

    D_sg = 30
    g = 9.81

    initial_guess =  D_imp

    D_tr = fsolve(transient_crater_diameter, initial_guess,\
                  args=(D_imp, rho_imp, rho_tar, v_imp, theta_imp, D_sg, g))

    return D_tr


def calc_crater_diameter(D_imp, v_imp, theta, rho_imp=0.6e3):
    """
    Docstring
    """

    D_tr = calc_transient_crater(D_imp, v_imp, rho_imp, theta)

    if D_tr < 4e3:
        D_f = D_tr
    elif D_tr >= 4e3:
        D_f = 1. * (D_tr ** 1.18) * (4e3 ** -0.18)

    return D_f
