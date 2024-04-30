# pylint: disable-msg=E1101
# pylint: disable-msg=R0914
# pylint: disable-msg=C0103
# pylint: disable-msg=R0915

"""
Script to calculate figure 3 from Anslow+ (subm.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from overlap.utils import set_size
from overlap.atmos.frag_model import run_intergration
from overlap.craters.scaling_laws import calc_crater_diameter

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def main():
    """
    Calculates figure 3: the impact crater diameter as a function of the impactor
    diameter
    """

    D_imp = np.logspace(0, 4, 100)
    D_com = np.logspace(1, 4, 100)
    D_sto = np.logspace(1, 4, 100)
    D_iro = np.logspace(0, 4, 100)

    D_craters_com = []
    D_craters_sto = []
    D_craters_iro = []

    rho_com = 0.6e3
    rho_sto = 3.5e3
    rho_iro = 7.8e3

    Rdot0 = 0
    theta0 = 45. * np.pi / 180.
    Z0 = 100e3

    for counter, D in enumerate(D_com):

        R0 = D_com[counter] / 2
        Rdot0 = 0
        M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)

        _, vel, mass, theta, altitude, radius, _, _ = run_intergration(11.2e3, M0,\
                                                                       theta0, Z0,\
                                                                       R0, Rdot0, 1e4,\
                                                                       rho_com, 2.5e6)
        if altitude[-1] < 1:

            D_f = calc_crater_diameter(2 * radius[-1], mass[-1], vel[-1], theta[-1], rho_com)
            D_craters_com = np.append(D_craters_com, D_f)

        else:

            D_craters_com = np.append(D_craters_com, np.nan)

        R0 = D_sto[counter] / 2
        Rdot0 = 0
        M0 = rho_sto * (4 * np.pi / 3) * (R0 ** 3)

        _, vel, mass, theta, altitude, radius, _, _ = run_intergration(11.2e3, M0,\
                                                                       theta0, Z0, R0,\
                                                                       Rdot0, 1e7, rho_sto,\
                                                                       8e6)

        if altitude[-1] < 1:

            D_f = calc_crater_diameter(2 * radius[-1], mass[-1], vel[-1], theta[-1], rho_sto)
            D_craters_sto = np.append(D_craters_sto, D_f)

        else:

            D_craters_sto = np.append(D_craters_sto, np.nan)

        R0 = D_iro[counter] / 2
        Rdot0 = 0
        M0 = rho_iro * (4 * np.pi / 3) * (R0 ** 3)

        _, vel, mass, theta, altitude, radius, _, _ = run_intergration(11.2e3, M0,\
                                                                       theta0, Z0,\
                                                                       R0, Rdot0,\
                                                                       1e8, rho_iro,\
                                                                       8e6)

        if altitude[-1] < 1:

            D_f = calc_crater_diameter(2 * radius[-1], mass[-1], vel[-1], theta[-1], rho_iro)
            D_craters_iro = np.append(D_craters_iro, D_f)

        else:

            D_craters_iro = np.append(D_craters_iro, np.nan)

        print("\r" + 'Progress: ' + str(round(100 * counter / len(D_imp), 2)), end='%')

    print("\r")

    D_f_com = []
    D_f_ast = []
    D_f_iro = []

    for D in D_imp:

        D_f = calc_crater_diameter(D, np.pi * rho_com * D ** 3 / 6, 11.2e3, 45 * np.pi / 180, 0.6e3)
        D_f_com = np.append(D_f_com, D_f)
        D_f = calc_crater_diameter(D, np.pi * rho_sto * D ** 3 / 6, 11.2e3, 45 * np.pi / 180, 3.5e3)
        D_f_ast = np.append(D_f_ast, D_f)
        D_f = calc_crater_diameter(D, np.pi * rho_iro * D ** 3 / 6, 11.2e3, 45 * np.pi / 180, 7.8e3)
        D_f_iro = np.append(D_f_iro, D_f)

    fig_width, fig_height = set_size('thesis', 1, (1, 1))
    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(D_imp / 1e3, D_f_com / 1e3, color='tab:gray', ls='--', zorder=0)
    plt.plot(D_imp / 1e3, D_f_ast / 1e3, color='tab:gray', ls='--', zorder=0)
    plt.plot(D_imp / 1e3, D_f_iro / 1e3, color='tab:gray', ls='--', zorder=0)

    plt.plot(D_com / 1e3, D_craters_com / 1e3, color=cm.bamako((3 - 2.5) / 3),\
             zorder=5, label=r'Comet')
    plt.plot(D_sto / 1e3, D_craters_sto / 1e3, color=cm.bamako((3 - 1.5) / 3),\
             zorder=1, label=r'Stone')
    plt.plot(D_iro / 1e3, D_craters_iro / 1e3, color=cm.bamako((3 - 0.5) / 3),\
             zorder=5, label=r'Iron')

    plt.minorticks_on()

    plt.xlabel(r'$D_{\rm imp}$ [km]', fontsize=13)
    plt.ylabel(r'$D_{\rm crater}$ [km]', fontsize=13)

    plt.xlim(5e-3, 5)
    plt.ylim(1e-1, 5e1)

    plt.legend(fontsize=11)

    plt.xscale('log')
    plt.yscale('log')

#    plt.savefig('comet_crater_production_constraints.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
