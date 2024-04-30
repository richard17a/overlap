# pylint: disable-msg=C0103
# pylint: disable-msg=C0200
# pylint: disable-msg=R0914

"""
Module containing main functions for the monte carlo overlap
"""

import numpy as np
from numba import jit
from overlap.montecarlo.generate_craters import generate_craters, EarthMoon


@jit(nopython=True)
def calculate_distances(crater_1, craters, body):
    """
    Calculate the distances between a reference crater and a population of craters.

    Parameters:
    - crater_1: crater
    - craters (numpy.ndarray): array of craters
    - body (EarthMoon): Earth/Moon classifier

    Returns:
    - numpy.ndarray: distances between crater_1 and craters

    """

    x_coords = craters['x']
    y_coords = craters['y']
    z_coords = craters['z']

    dot_products = crater_1['x'] * x_coords + \
                   crater_1['y'] * y_coords + \
                   crater_1['z'] * z_coords

    if body == EarthMoon.EARTH:
        distances = 6371e3 * np.arccos(dot_products / (6371e3 ** 2))
    elif body == EarthMoon.MOON:
        distances = 1737e3 * np.arccos(dot_products / (1737e3 ** 2))

    return distances


@jit(nopython=True)
def check_within_radii(crater_1, craters, D_min=1e3, body=EarthMoon.EARTH):
    """
    Check if a crater intersects with the population of craters on the surface.

    Parameters:
    - crater_1: crater
    - craters (numpy.ndarray): crater population
    - D_min (float): Minimum crater diameter
    - body (EarthMoon): Earth/Moon classifier

    Returns:
    - numpy.ndarray: array of intersecting craters

    """

    if crater_1.r > D_min:

        radii = craters['r']

        distances = calculate_distances(crater_1, craters, body)

        radii_sum = crater_1.r + radii

        return distances[(distances < radii_sum) & (crater_1.r > radii)]

    return np.zeros(0)


@jit(nopython=True)
def count_overlaps(craters_i, craters_f, npoints, body=EarthMoon.EARTH):
    """
    Count the number of overlaps in a population of craters with random locations and sizes.

    Parameters:
    - craters_i (numpy.ndarray): population of craters
    - craters_f (numpy.ndarray): population of craters
    - body (EarthMoon): Earth/Moon classifier

    Returns:
    - tuple: the cumulative number of overlapping craters in the population
    """

    num_overlaps2 = 0
    num_overlaps3 = 0
    num_overlaps4 = 0
    for counter in range(len(craters_i)):
        crater_i = craters_i[counter]

        intersect_distances2 = check_within_radii(crater_i, craters_f, 1e2, body)
        num_overlaps2 += (npoints - len(intersect_distances2)) / npoints

        intersect_distances3 = check_within_radii(crater_i, craters_f, 1e3, body)
        num_overlaps3 += (npoints - len(intersect_distances3)) / npoints

        intersect_distances4 = check_within_radii(crater_i, craters_f, 1e4, body)
        num_overlaps4 += (npoints - len(intersect_distances4)) / npoints

    return num_overlaps2, num_overlaps3, num_overlaps4


def calc_frac_overlaps(sfd_index, rmin, rmax, npoints, body=EarthMoon.EARTH):
    """
    Calculates the fraction of overlapping craters in the population.

    Parameters:
    - sfd_index (float): index of the power-law SFD
    - rmin (float): Minimum radius
    - rmax (float): maximum radius
    - npoints (int): Number of craters
    - body (EarthMoon): Earth/Moon classifier

    Returns:
    - tuple: p_single - the probability of two craters randomly overlapping on the
            surface of the Earth/Moon (equation 2.2)

    """

    craters_i = generate_craters(sfd_index, rmin, rmax, npoints, body)
    craters_f = generate_craters(sfd_index, rmin, rmax, npoints, body)

    print('Generated craters')

    num_overlaps2, num_overlaps3, num_overlaps4 = count_overlaps(craters_i, craters_f,\
                                                                npoints, body)

    prob2 = num_overlaps2 / npoints
    prob3 = num_overlaps3 / npoints
    prob4 = num_overlaps4 / npoints

    print(prob2, prob3, prob4)

    f_path = './overlap/montecarlo/'
    if body == EarthMoon.EARTH:
        f_name_1e2 = 'Earth_1e2.txt'
        f_name_1e3 = 'Earth_1e3.txt'
        f_name_1e4 = 'Earth_1e4.txt'

    elif body == EarthMoon.MOON:
        f_name_1e2 = 'Moon_1e2.txt'
        f_name_1e3 = 'Moon_1e3.txt'
        f_name_1e4 = 'Moon_1e4.txt'

    np.savetxt(f_path + f_name_1e2, [prob2])
    np.savetxt(f_path + f_name_1e3, [prob3])
    np.savetxt(f_path + f_name_1e4, [prob4])
