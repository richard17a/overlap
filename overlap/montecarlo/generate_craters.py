# pylint: disable-msg=C0103
# pylint: disable-msg=R0903
# pylint: disable-msg=R0914
# pylint: disable-msg=E1101

"""
Module Docstring
"""

from enum import Enum
import numpy as np
import numba
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm


class EarthMoon(Enum):
    """
    Docstring
    """
    EARTH = 0
    MOON = 1


class Crater:
    """
    Docstring
    """
    def __init__(self, x, y, z, r):
        """
        Docstring
        """
        self.x = x
        self.y = y
        self.z = z
        self.r = r


@numba.jit(nopython=True)
def generate_radius(sfd_index, rmin, rmax):
    """
    Docstring
    """

    while True:
        u = np.random.random()

        r = u ** (-1/sfd_index)
        r = rmin * r

        if r <= rmax:
            break

    return r


@numba.jit(nopython=True)
def generate_radii(sfd_index, rmin, rmax, npoints):
    """
    Docstring
    """

    radii = np.empty(npoints)

    for i in range(npoints):
        radii[i] = generate_radius(sfd_index, rmin, rmax)

    return radii


def generate_latitudes(npoints):
    """
    Docstring
    """

    latitudes = np.random.rand(npoints) * np.pi - np.pi / 2

    return latitudes


def generate_longitudes(npoints):
    """
    Docstring
    """

    longitudes = np.random.rand(npoints) * 2 * np.pi

    return longitudes


def generate_craters(sfd_index, rmin, rmax, npoints, body=EarthMoon.EARTH):
    """
    Docstring
    """

    latitudes = generate_latitudes(npoints)
    longitudes = generate_longitudes(npoints)

    cos_lat = np.cos(latitudes)
    sin_lat = np.sin(latitudes)
    cos_lon = np.cos(longitudes)
    sin_lon = np.sin(longitudes)

    craters = np.empty((npoints, 3))

    rads = generate_radii(sfd_index, rmin, rmax, npoints)

    if body == EarthMoon.EARTH:
        x = 6371e3 * cos_lat * cos_lon
        y = 6371e3 * cos_lat * sin_lon
        z = 6371e3 * sin_lat

        craters = np.array([Crater(xi, yi, zi, ri) for xi, yi, zi, ri in zip(x, y, z, rads)],\
                           dtype=object)
    elif body == EarthMoon.MOON:
        x = 1737e3 * cos_lat * cos_lon
        y = 1737e3 * cos_lat * sin_lon
        z = 1737e3 * sin_lat

        craters = np.array([Crater(xi, yi, zi, ri) for xi, yi, zi, ri in zip(x, y, z, rads)],\
                           dtype=object)

    crater_dtype = np.dtype([('x', np.float64), ('y', np.float64),\
                             ('z', np.float64), ('r', np.float64)])

    craters = np.array([(crater.x, crater.y, crater.z, crater.r) for crater in craters],\
                       dtype=crater_dtype)

    return craters


def main():
    """
    Docstring
    """

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    fig_width, fig_height = 5.9055, 3.6498

    npoints = 1_000_000
    sfd_index = 2.0

    rmin = 1e2
    rmax = 300e3

    craters = generate_craters(sfd_index, rmin, rmax, npoints)
    craters_small = generate_craters(sfd_index, 1e1, rmax, npoints)

    lons = generate_longitudes(npoints)
    lats = generate_latitudes(npoints)

    radii = np.array([crater['r'] for crater in craters])
    radii_small = np.array([crater['r'] for crater in craters_small])

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.hist(lons / (2 * np.pi), bins=100, histtype='step', density=True)

    plt.xlabel(r'Longitude / $2 \pi$')
    plt.ylabel(r'Frequency')

    plt.minorticks_on()

    plt.show()

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.hist(lats / (np.pi / 2), bins=100, histtype='step', density=True)

    plt.xlabel(r'Latitude / $\pi/2$')
    plt.ylabel('Frequency', fontsize=11)

    plt.minorticks_on()

    plt.show()

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.hist(radii, bins=np.logspace(np.min(np.log10(radii)), np.max(np.log10(radii)), 200),\
             density=True, histtype='step', color=cm.bamako(0.2))
    plt.hist(radii_small, bins=np.logspace(np.min(np.log10(radii_small)),\
                                           np.max(np.log10(radii_small)), 200), density=True,\
                                            histtype='step', color=cm.bamako(0.5))

    plt.hist(radii_small, bins=np.logspace(1, 2, 200), density=True, color='tab:gray',\
             zorder=0, alpha=0.25, lw=0, histtype='stepfilled')

    plt.text(1.5e1, 1e-8, "(very)\ninefficient\ncratering")

    plt.axvline(300e3, ls=':', c='k')

    plt.text(1.5e5, 5e-3, r"$D_{\rm max}$", rotation=90, fontsize=12)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$D_{\rm crater}$ [m]', fontsize=13)
    plt.ylabel(r'$N(>D)$ (normalised)', fontsize=13)

    ax = plt.gca()
    ax.yaxis.get_minor_locator().set_params(numticks=20)

    # plt.savefig('impactors_sfd_comparison.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
