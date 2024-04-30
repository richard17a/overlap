# pylint: disable-msg=E1101
# pylint: disable-msg=R0914
# pylint: disable-msg=C0103
# pylint: disable-msg=W0703
# pylint: disable-msg=W0632
# pylint: disable-msg=R0913

"""
Script to calculate figure 6 from Anslow+ (subm.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
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

    return earth_1e2, earth_1e3, earth_1e4


def calc_f_overlap(N):
    """
    Calculate the fraction of overlapping craters, according to equation 2.3

    Parameters:
    - N: Number of craters

    Returns:
    - f_overlap: Fraction of overlapping craters
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


def crater_rate_vals():
    """
    Return the crater rate as a function of time for the Robbins (2014) and Marchi+ (2009)
    chronologies

    Returns:
    - tuple: Two numpy arrays containing crater rates for Marchi+ and Robbins chronologies
            respectively
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


def model(y, t, a, b, c, tau):
    """
    ODE model describing the change in crater counts over time.

    Parameters:
    - y: Array containing current values of N, N_overlap
    - t: Array of time
    - a, b, c: Parameters describing crater rate
    - tau: Ferrocyanide salt lifetime

    Returns:
    - dydt: Array containing dN/dt, dN_overlap/dt
    """

    N, No = y

    R_in = crater_rate(t, a, b, c)

    dNdt = R_in - N / tau

    dNodt = calc_f_overlap(N) * R_in - No / tau

    return [dNdt, dNodt]


def crater_rate(t, a, b, c):
    """
    Calculates the (scaled) crater rate on the Earth for a given impact chronology

    Parameters:
    - t: Time
    - a, b, c: Parameters describing crater rate

    Returns:
    - R_in: crater rate
    """

    R_in = a * b * np.exp(b * t) + c

    A_m = 4 * np.pi * (1737 ** 2)

    R_in = R_in * A_m * 13.5 * (1e3 / 1e2) ** 2

    return R_in


def calc_N_overlap(t, solution, a, b, c):
    """
    Calculates the number of overlapping craters as a function of time

    Parameters:
    - t: time
    - solution: Solution of the ODE model (N, N_overlap)
    - a, b, c: Parameters describing impact chronology

    Returns:
    - t_disc: time
    - N_overlap: Number of overlapping craters
    """

    t_disc = np.linspace(3.4, 4.5, 4500) * 1e9
    N_overlap = np.zeros(len(t_disc))

    for counter, _ in enumerate(t_disc):

        if counter < len(t_disc) - 1:

            delta_N = (a * (np.exp(b * t_disc[counter + 1]) - 1) + c * t_disc[counter + 1]) -\
                        (a * (np.exp(b * t_disc[counter]) - 1) + c * t_disc[counter])
            A_m = 4 * np.pi * (1737 ** 2)
            delta_N = delta_N * A_m * 13.5 * (1e3 / 1e2) ** 2

            N = solution[:, 0][(t > t_disc[counter]) & (t <= t_disc[counter + 1])]

            prob_overlap = calc_f_overlap(np.mean(N))

            N_overlap[counter] = delta_N * prob_overlap

        else:

            delta_N = (a * (np.exp(b * t_disc[-1]) - 1) + c * t_disc[-1]) -\
                        (a * (np.exp(b * t_disc[-2]) - 1) + c * t_disc[-2])
            A_m = 4 * np.pi * (1737 ** 2)
            delta_N = delta_N * A_m * 13.5 * (1e3 / 1e2) ** 2

            N = solution[:, 0][(t > t_disc[counter - 1])]

            prob_overlap = calc_f_overlap(np.mean(N))

            N_overlap[counter] = delta_N * prob_overlap

    return t_disc, N_overlap


def main():
    """
    Calculates figure 6: the cumulative number of overlapping craters on Earth
    suitable for prebiotic chemistry.
    """

    fig_width, fig_height = set_size('thesis', 1, (1, 1))

    f_comet = 0.01
    f_success = 0.01
    f_crust = 0.2

    t = np.linspace(3.4, 4.5, 1_000_000) * 1e9
    y0 = [0, 0]

    a = 7.26e-31
    b = 16.7e-9
    c = 1.19e-12

    solution_1e1 = odeint(model, y0, t, args=(a, b, c, 1e1))
    t_disc_1e1, N_overlap_1e1 = calc_N_overlap(t, solution_1e1, a, b, c)

    solution_1e2 = odeint(model, y0, t, args=(a, b, c, 1e2))
    t_disc_1e2, N_overlap_1e2 = calc_N_overlap(t, solution_1e2, a, b, c)

    solution_1e3 = odeint(model, y0, t, args=(a, b, c, 1e3))
    t_disc_1e3, N_overlap_1e3 = calc_N_overlap(t, solution_1e3, a, b, c)

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(t_disc_1e1 / 1e9, f_success * f_crust * f_comet * np.cumsum(N_overlap_1e1),\
            color=cm.batlow(0), label=r'$\tau_{\rm Fe(CN)_6} = 10\,$yr')

    plt.plot(t_disc_1e2 / 1e9, f_success * f_crust * f_comet * np.cumsum(N_overlap_1e2),\
            color=cm.batlow(0.33), label=r'$\tau_{\rm Fe(CN)_6} = 100\,$yr')

    plt.plot(t_disc_1e3 / 1e9, f_success * f_crust * f_comet * np.cumsum(N_overlap_1e3),\
            color=cm.batlow(0.66), label=r'$\tau_{\rm Fe(CN)_6} = 1000\,$yr')

    a = 1.23e-15
    b = 7.85e-9
    c = 1.30e-12

    solution = odeint(model, y0, t, args=(a, b, c, 1e1))
    t_disc_1e1, N_overlap_1e1 = calc_N_overlap(t, solution, a, b, c)

    solution = odeint(model, y0, t, args=(a, b, c, 1e2))
    t_disc_1e2, N_overlap_1e2 = calc_N_overlap(t, solution, a, b, c)

    solution = odeint(model, y0, t, args=(a, b, c, 1e3))
    t_disc_1e3, N_overlap_1e3 = calc_N_overlap(t, solution, a, b, c)

    plt.plot(t_disc_1e1 / 1e9, f_success * f_crust * f_comet * np.cumsum(N_overlap_1e1), ls='--',\
            color=cm.batlow(0))

    plt.plot(t_disc_1e2 / 1e9, f_success * f_crust * f_comet * np.cumsum(N_overlap_1e2), ls='--',\
            color=cm.batlow(0.33))

    plt.plot(t_disc_1e3 / 1e9, f_success * f_crust * f_comet * np.cumsum(N_overlap_1e3), ls='--',\
            color=cm.batlow(0.66))

    plt.axhspan(1e-13, 1e0, color='tab:red', alpha=0.1, zorder=0)
    plt.axvspan(4.5, 4., color='tab:gray', alpha=0.25, zorder=0)

    plt.yscale('log')

    plt.xlim(3.5, 4.5)
    plt.ylim(1e-8, 1e4)

    plt.xlabel('Age [Gyr]', fontsize=12)
    plt.ylabel('Cumulative number of overlapping\ncraters suitable for prebiotic chemistry',\
                fontsize=12)

    initial_legend = plt.legend(fontsize=11, loc='upper left')

    extra_legend_handles = [
        plt.Line2D([], [], color='tab:gray', linestyle='-', label='Robbins 2014'),
        plt.Line2D([], [], color='tab:gray', linestyle='--', label='Marchi et al. 2009')
    ]
    extra_legend = plt.legend(handles=extra_legend_handles, fontsize=11, loc='lower right')

    plt.gca().add_artist(extra_legend)
    plt.gca().add_artist(initial_legend)

    plt.minorticks_on()

    plt.gca().yaxis.get_major_locator().set_params(numticks=8)
    plt.gca().yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])

#    plt.savefig('./steady_state_craters.pdf', format='pdf', bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    main()
