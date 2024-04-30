# pylint: disable-msg=E1101
# pylint: disable-msg=R0914
# pylint: disable-msg=C0103
# pylint: disable-msg=R0915
# pylint: disable-msg=W0703
# pylint: disable-msg=W0632

"""
Module Docstring
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from cmcrameri import cm
from overlap.montecarlo.check_intersection import calc_frac_overlaps
from overlap.montecarlo.generate_craters import EarthMoon
from overlap.utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def calc_f_overlap(N):
    """
    Docstring
    """

    try:
        _, earth_1e3, _ = read_files()

    except Exception:
        npoints = 10_000
        sfd_index = 2.0
        rmin = 1e3
        rmax = 300e3

        calc_frac_overlaps(sfd_index, rmin, rmax, npoints, EarthMoon.EARTH)

        _, earth_1e3, _ = read_files()

    return 1 - earth_1e3 ** N


def read_files():
    """
    Docstring
    """

    earth_1e2 = np.loadtxt('./overlap/montecarlo/Earth_1e2.txt', unpack=True)
    earth_1e3 = np.loadtxt('./overlap/montecarlo/Earth_1e3.txt', unpack=True)
    earth_1e4 = np.loadtxt('./overlap/montecarlo/Earth_1e4.txt', unpack=True)

    return earth_1e2, earth_1e3, earth_1e4


def crater_rate_vals():
    """
    Docstring
    """

    t = np.linspace(0, 4.5, 10_000) * 1e9

    A_m = 4 * np.pi * (1737 ** 2)

    a = 1.23e-15
    b = 7.85e-9
    c = 1.30e-12
    dndt_marchi = (a * b * np.exp(b * t) + c) * A_m * 20 * (1e3 / 1e2) ** 2

    a = 7.26e-31
    b = 16.7e-9
    c = 1.19e-12
    dndt_robbins = (a * b * np.exp(b * t) + c) * A_m * 20 * (1e3 / 1e2) ** 2

    return dndt_marchi, dndt_robbins


def main():
    """
    Docstring
    """

    R_ins = np.logspace(-4, 5, 500)
    taus = np.logspace(1, 6, 500)

    r_grid, t_grid = np.meshgrid(R_ins, taus)

    steady_state3 = np.zeros_like(r_grid)

    for i, R_in in enumerate(R_ins):

        for j, tau in enumerate(taus):

            steady_state3[j, i] =  R_in * tau * calc_f_overlap(R_in * tau)

    dndt_marchi, dndt_robbins = crater_rate_vals()

    fig_width, fig_height = set_size('thesis', 1, (1, 1))

    _ = plt.figure(figsize=(fig_width, fig_height))

    contour = plt.contourf(r_grid, t_grid, steady_state3 / 100, norm=LogNorm(),\
                           extend='both', levels=np.logspace(-6, 3, 100), cmap=cm.bamako)

    cbar = plt.colorbar(contour, ticks=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3], )
    cbar.set_label("Steady-state number of overlapping\n craters suitable for prebiotic chemistry",\
                    fontsize=12, rotation=270, labelpad=25)

    cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=[0.2, 0.4, 0.6, 0.8],\
                                    numticks=10))
    formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(np.log10(x))}}}$")
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.contour(r_grid, t_grid, steady_state3 / 100, [.9], linestyles=[':'],\
                colors=['k'], linewidths=[1.])

    for c in contour.collections:
        c.set_edgecolor("face")

    plt.axvline(dndt_marchi[0], color=cm.bamako(0.8), ls='--', linewidth=0.9)
    plt.axvline(dndt_marchi[8900],  color='k', ls='--', linewidth=0.9)
    plt.axvline(dndt_robbins[8900], color='k', ls='-.', linewidth=0.9)

    plt.text(3.5e-2, 2e1, r"$1.0\,{\rm Gyr}$ crater rate", rotation=90,\
             color=cm.bamako(0.9), fontsize=11)
    plt.text(2.0e+2, 9e3, r"$4.0\,{\rm Gyr}$ crater rate", rotation=90,\
             color='k', fontsize=11)

    plt.text(9e2, 5e1, "1 overlapping crater", color="k", rotation=-54, fontsize=10)

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel(r"Crater rate [yr$^{-1}$]", fontsize=12)
    plt.ylabel(r"Ferrocyanide salt lifetime [yr]", fontsize=12)

    ax = plt.gca()

    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=9))
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    ax.tick_params(axis='x', which='both', labelsize=11)
    ax.tick_params(axis='y', which='both', labelsize=11)

    legend_lines = [
        plt.Line2D([], [], color='tab:gray', linestyle='--', linewidth=1.0,\
                   label=r'Marchi et al. 2009'),
        plt.Line2D([], [], color='tab:gray', linestyle='-.', linewidth=1.0,\
                   label=r'Robbins 2014')
    ]

    plt.legend(handles=legend_lines, loc='upper left', fontsize=10)

#    plt.savefig('./paper_figures/ferro_lifetime_crater_rate_contour.pdf',\
#                format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
