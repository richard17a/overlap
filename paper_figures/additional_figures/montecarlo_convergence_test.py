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
import matplotlib
import matplotlib.pyplot as plt
from cmcrameri import cm
from overlap.montecarlo.generate_craters import generate_craters, EarthMoon
from overlap.montecarlo.check_intersection import count_overlaps
from overlap.utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def calc_probs():
    """
    Docstring
    """

    npoints_array = np.logspace(2, 5, 10)
    sfd_index = 2.0
    rmin = 1e2
    rmax = 300e3

    probs = []

    for counter, npoints in enumerate(npoints_array):

        print("Progress:", str(round(100 * counter / len(npoints_array), 2)), end="%\r")

        craters_i = generate_craters(sfd_index, rmin, rmax, int(npoints), EarthMoon.EARTH)
        craters_f = generate_craters(sfd_index, rmin, rmax, int(npoints), EarthMoon.EARTH)

        num_overlaps2, _, _ = count_overlaps(craters_i, craters_f, int(npoints))

        prob = num_overlaps2 / int(npoints)

        probs = np.append(probs, prob)

    np.savetxt('./paper_figures/additional_figures/npoints.txt', npoints_array)
    np.savetxt('./paper_figures/additional_figures/probs.txt', probs)

    return npoints_array, probs


def main():
    """
    Docstring
    """

    fig_width, fig_height = set_size('thesis', 1, (1, 1))

    try:
        npoints_array = np.loadtxt('./paper_figures/additional_figures/npoints.txt', unpack=True)
        probs = np.loadtxt('./paper_figures/additional_figures/probs.txt', unpack=True)

    except Exception:
        _, probs = calc_probs()
        probs = probs ** 4e8

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(npoints_array, probs / probs[-1], marker='o', color=cm.bamako(0.2))
    plt.axhline(1, ls='--', color='tab:gray', zorder=0)

    plt.xlabel(r'$N_{\rm craters}$', fontsize=13)
    plt.ylabel(r'$f_{\rm overlap}/f_{\rm overlap}(N_{\rm max})$', fontsize=13)

    plt.minorticks_on()

    plt.savefig('./paper_figures/additional_figures/convergence_out.pdf',\
                format='pdf', bbox_inches='tight')

    plt.show()

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(npoints_array, probs / probs[-1], marker='o', color=cm.bamako(0.2))
    plt.axhline(1, ls='--', color='tab:gray', zorder=0)

    plt.xscale('log')

    plt.xlabel(r'$N_{\rm craters}$', fontsize=13)
    plt.ylabel(r'$f_{\rm overlap}/f_{\rm overlap}(N_{\rm max})$', fontsize=13)

    plt.minorticks_on()

    plt.savefig('./paper_figures/additional_figures/convergence_out_log.pdf',\
                format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
