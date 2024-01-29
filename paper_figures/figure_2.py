# pylint: disable-msg=C0103
# pylint: disable-msg=E1101

"""
Docstring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from overlap.utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fig_width, fig_height = set_size('thesis', 1, (1, 1))


def get_data():
    """
    Docstring NEED TO CHECK ALL OF THESE VALUES (le Feuvre 2011)
    """

    age = np.array([0.025, 0.053, 0.109, 0.80, 3.15, 3.22, 3.30,\
                    3.41, 3.58, 3.75, 3.80, 3.85, 3.85, 3.85]) * 1e3
    N_crater = np.array([0.169, 0.390, 0.824, 5.77, 29.7, 30.2, 27.7,\
                         32.4, 60.1, 93.6, 84.5, 301, 298, 306]) * 1e-4

    t_contentious = np.array([4.35]) * 1e3
    N_contentious = np.array([7.851e-1])

    return age, t_contentious, N_crater, N_contentious


def main():
    """
    Docstring
    """

    t = np.linspace(0, 4570, 10_000)

    a = 5.44e-14
    b = 6.93e-3
    c = 8.38e-7
    N1_neukum = a * (np.exp(b * t) - 1) + c * t

    a = 1.23e-15
    b = 7.85e-3
    c = 1.30e-6
    N1_marchi = a * (np.exp(b * t) - 1) + c * t

    a = 1.893e-26
    b = 14.44e-3
    c = 7.960e-7
    N1_leFeuvreWieczorek = a * (np.exp(b * t) - 1) + c * t

    a = 7.26e-31
    b = 16.7e-3
    c = 1.19e-6
    N1_robbins = a * (np.exp(b * t) - 1) + c * t

    N1_neukum = np.array(N1_neukum)
    N1_marchi = np.array(N1_marchi)
    N1_leFeuvreWieczorek = np.array(N1_leFeuvreWieczorek)
    N1_robbins = np.array(N1_robbins)

    age, t_contentious, N_crater, N_contentious = get_data()

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(t, N1_neukum, label=r'Neukum & Ivanov 1994',  color=cm.bamako((4 - 0.5) / 4))
    plt.plot(t, N1_marchi, label=r'Marchi et al. 2009',  color=cm.bamako((3.5 - 0.5) / 4))
    plt.plot(t, N1_leFeuvreWieczorek, label=r'Le Feuvre & Wieczorek 2011',\
             color=cm.bamako((2.5 - 0.5) / 4))
    plt.plot(t, N1_robbins, label=r'Robbins 2014', color=cm.bamako((0 - 0.5) / 4))

    plt.plot(age, N_crater, 'x', color='tab:brown')
    plt.plot(t_contentious, N_contentious, 'x', color='tab:red')

    plt.axvline(3920, color='tab:gray')
    plt.text(3650, 2e1, "unconstrained for\nages >3.92 Gyr", fontsize=11,\
             rotation=90, ha='center', va='center')

    plt.legend(fontsize=11, loc='upper left')

    plt.yscale('log')

    plt.xlim(0, 4500)
    plt.ylim(1e-5, )

    ax = plt.gca()

    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100))

    plt.xticks([0, 1000, 2000, 3000, 4000], [0, 1, 2, 3, 4])

    plt.xlabel('Age [Gyr]', fontsize=12)
    plt.ylabel(r'$N(1\,{\rm km}, t)\,$ [km$^{-2}$]', fontsize=13)

    plt.savefig('crater_chronology_methods_1km.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
