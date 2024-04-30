# pylint: disable-msg=E1101
# pylint: disable-msg=R0914
# pylint: disable-msg=C0103
# pylint: disable-msg=R0915
# pylint: disable-msg=W0703

"""
Script to calculate figure 4 from Anslow+ (subm.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from overlap.montecarlo.check_intersection import calc_frac_overlaps
from overlap.montecarlo.generate_craters import EarthMoon
from overlap.utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def read_files():
    """
    Read the overlap probabilities from the Monte Carlo simulation output files

    Returns:
    - tuple: A tuple containing three floats (for varying D_min)
    """

    earth_1e2 = np.loadtxt('./overlap/montecarlo/Earth_1e2.txt', unpack=True)
    earth_1e3 = np.loadtxt('./overlap/montecarlo/Earth_1e3.txt', unpack=True)
    earth_1e4 = np.loadtxt('./overlap/montecarlo/Earth_1e4.txt', unpack=True)

    moon_1e2 = np.loadtxt('./overlap/montecarlo/Moon_1e2.txt', unpack=True)
    moon_1e3 = np.loadtxt('./overlap/montecarlo/Moon_1e3.txt', unpack=True)
    moon_1e4 = np.loadtxt('./overlap/montecarlo/Moon_1e4.txt', unpack=True)

    return earth_1e2, earth_1e3, earth_1e4, moon_1e2, moon_1e3, moon_1e4


def main():
    """
    Calculates figure 4: the fraction of overlapping craters (as a function of the
    number of craters on the surface)
    """

    fig_width, fig_height = set_size('thesis', 1, (1, 1))

    try:
        earth_1e2, earth_1e3, earth_1e4, moon_1e2, moon_1e3, moon_1e4 = read_files()

    except Exception:
        npoints = 10_000
        sfd_index = 2.0
        rmin = 1e3
        rmax = 300e3

        calc_frac_overlaps(sfd_index, rmin, rmax, npoints, EarthMoon.EARTH)
        calc_frac_overlaps(sfd_index, rmin, rmax, npoints, EarthMoon.MOON)

        earth_1e2, earth_1e3, earth_1e4, moon_1e2, moon_1e3, moon_1e4 = read_files()

    nums = np.logspace(5, 12, 1000)
    # nums = np.logspace(3, 9, 1000)

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(nums, 1 - earth_1e2 ** nums, label=r'$D_{\rm min} = 10^2\,{\rm m}$',\
             color=cm.bamako(0.8))
    plt.plot(nums, 1 - earth_1e3 ** nums, label=r'$D_{\rm min} = 10^3\,{\rm m}$',\
             color=cm.bamako(0.5))
    plt.plot(nums, 1 - earth_1e4 ** nums, label=r'$D_{\rm min} = 10^4\,{\rm m}$',\
             color=cm.bamako(0.2))

    plt.plot(nums, 1 - moon_1e2 ** nums, color=cm.bamako(0.8), ls='--')
    plt.plot(nums, 1 - moon_1e3 ** nums, color=cm.bamako(0.5), ls='--')
    plt.plot(nums, 1 - moon_1e4 ** nums, color=cm.bamako(0.2), ls='--')

    plt.xscale('log')

    plt.xlim(1e5, 1e12)
    # plt.xlim(1e3, 1e9)

    plt.xlabel(r'$N_{\rm craters}$', fontsize=13)
    plt.ylabel(r'$f_{\rm overlap}$', fontsize=13)

    initial_legend = plt.legend(fontsize=11)

    extra_legend_handles = [
        plt.Line2D([], [], color='tab:gray', linestyle='-', label='Earth'),
        plt.Line2D([], [], color='tab:gray', linestyle='--', label='Moon')
    ]
    extra_legend = plt.legend(handles=extra_legend_handles, fontsize=11, loc='lower right')

    plt.gca().add_artist(extra_legend)
    plt.gca().add_artist(initial_legend)

    plt.minorticks_on()

    # plt.savefig('f_overlap_D_min_basic.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
