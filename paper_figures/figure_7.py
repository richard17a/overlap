# pylint: disable-msg=E1101
# pylint: disable-msg=R0914
# pylint: disable-msg=C0103
# pylint: disable-msg=W0703
# pylint: disable-msg=W0632

"""
Module Docstring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from cmcrameri import cm
from overlap.montecarlo.check_intersection import calc_frac_overlaps
from overlap.montecarlo.generate_craters import EarthMoon
from overlap.utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def tanh_function(x, a, b):
    """
    Docstring
    """

    return a * np.tanh(b * x)


def read_files():
    """
    Docstring
    """

    earth_1e2 = np.loadtxt('./overlap/montecarlo/Earth_1e2.txt', unpack=True)
    earth_1e3 = np.loadtxt('./overlap/montecarlo/Earth_1e3.txt', unpack=True)
    earth_1e4 = np.loadtxt('./overlap/montecarlo/Earth_1e4.txt', unpack=True)

    return earth_1e2, earth_1e3, earth_1e4


def fit_overlap_curves():
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

    nums = np.logspace(5, 12, 1000)

    params_3, _ = curve_fit(tanh_function, nums, 1 - earth_1e3 ** nums, p0=[1, 1e-8])

    return params_3


def crater_rate_vals():
    """
    Docstring
    """

    t = np.linspace(0, 4.5, 10_000)

    a = 1.23e-15
    b = 7.85
    c = 1.30e-3
    N1_marchi = a * (np.exp(b * t) - 1) + c * t

    a = 7.26e-31
    b = 16.7
    c = 1.19e-3
    N1_robbins = a * (np.exp(b * t) - 1) + c * t

    A_m = 4 * np.pi * (1737 ** 2)
    N1_marchi = A_m * np.array(N1_marchi) * 15
    N1_robbins = A_m * np.array(N1_robbins) * 15

    dndt_marchi = np.abs(np.gradient(N1_marchi, t)) / 1e9 * (1e3 / 1e2) ** 2
    dndt_robbins = np.abs(np.gradient(N1_robbins, t)) / 1e9 * (1e3 / 1e2) ** 2

    return t, dndt_marchi, dndt_robbins


def main():
    """
    Docstring
    """

    t, dndt_marchi, dndt_robbins = crater_rate_vals()
    params_3 = fit_overlap_curves()

    steady_state_1e1_1e3 = np.zeros_like(t)
    steady_state_1e2_1e3 = np.zeros_like(t)
    steady_state_1e3_1e3 = np.zeros_like(t)

    steady_state_1e1_1e3_lf = np.zeros_like(t)
    steady_state_1e2_1e3_lf = np.zeros_like(t)
    steady_state_1e3_1e3_lf = np.zeros_like(t)

    for counter, _ in enumerate(t):

        steady_state_1e1_1e3[counter] = dndt_marchi[counter] * 1e1 *\
            tanh_function(dndt_marchi[counter] * 1e1, params_3[0], params_3[1])
        steady_state_1e2_1e3[counter] = dndt_marchi[counter] * 1e2 *\
            tanh_function(dndt_marchi[counter] * 1e2, params_3[0], params_3[1])
        steady_state_1e3_1e3[counter] = dndt_marchi[counter] * 1e3 *\
            tanh_function(dndt_marchi[counter] * 1e3, params_3[0], params_3[1])

        steady_state_1e1_1e3_lf[counter] = dndt_robbins[counter] * 1e1 *\
            tanh_function(dndt_robbins[counter] * 1e1, params_3[0], params_3[1])
        steady_state_1e2_1e3_lf[counter] = dndt_robbins[counter] * 1e2 *\
            tanh_function(dndt_robbins[counter] * 1e2, params_3[0], params_3[1])
        steady_state_1e3_1e3_lf[counter] = dndt_robbins[counter] * 1e3 *\
            tanh_function(dndt_robbins[counter] * 1e3, params_3[0], params_3[1])

    f_comet = 0.01

    fig_width, fig_height = set_size('thesis', 1, (1, 1))
    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(t, steady_state_1e1_1e3 * f_comet, ls='-',\
             label=r'$\tau_{\rm Fe(CN)_6} = 10\,$yr', c=cm.batlow(0))
    plt.plot(t, steady_state_1e2_1e3 * f_comet, ls='-',\
             label=r'$\tau_{\rm Fe(CN)_6} = 100\,$yr', c=cm.batlow(0.33))
    plt.plot(t, steady_state_1e3_1e3 * f_comet, ls='-',\
             label=r'$\tau_{\rm Fe(CN)_6} = 1000\,$yr', c=cm.batlow(0.66))

    plt.plot(t, steady_state_1e1_1e3_lf * f_comet, ls='--', c=cm.batlow(0))
    plt.plot(t, steady_state_1e2_1e3_lf * f_comet, ls='--', c=cm.batlow(0.33))
    plt.plot(t, steady_state_1e3_1e3_lf * f_comet, ls='--', c=cm.batlow(0.66))

    plt.axhspan(1e-13, 1e0, color='tab:red', alpha=0.1, zorder=0)
    plt.axvspan(4.5, 4., ymin=12/16, color='tab:gray', alpha=0.25, zorder=0)

    plt.yscale('log')

    plt.xlim(3, 4.5)
    plt.ylim(1e-12, 1e4)

    ax = plt.gca()

    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=11))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100))

    plt.xlabel('Age [Gyr]', fontsize=12)
    plt.ylabel(r'$N_{\rm overlap}$', fontsize=13)

    initial_legend = plt.legend(fontsize=11)

    extra_legend_handles = [
        plt.Line2D([], [], color='tab:gray', linestyle='-', label='Marchi et al. 2009'),
        plt.Line2D([], [], color='tab:gray', linestyle='--', label='Robbins 2014')
    ]
    extra_legend = plt.legend(handles=extra_legend_handles, fontsize=11, loc='lower right')

    plt.gca().add_artist(extra_legend)
    plt.gca().add_artist(initial_legend)

    plt.minorticks_on()

    # plt.savefig('./paper_figures/steady_state_craters.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()