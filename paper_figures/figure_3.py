# pylint: disable-msg=R0914
# pylint: disable-msg=C0103
# pylint: disable-msg=C0200
# pylint: disable-msg=E1101

"""
Module Docstring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmcrameri import cm
from overlap.utils import set_size
from overlap.atmos.frag_model import run_intergration

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def main():
    """
    Docstring
    """

    fig_width, fig_height = set_size('thesis', 1, (1, 1))
    _, ax = plt.subplots(figsize=(1.25 * fig_width, fig_height))

    rho_m = 0.6e3
    V0 = 11.2e3
    R0 = np.array([25, 50, 100, 150])
    Rdot0 = 0
    M0 = rho_m * (4 * np.pi / 3) * (R0 ** 3)
    theta0 = 45. * np.pi / 180.
    Z0 = 100e3

    zorders = [0, 1, 2, 3]

    for i in range(len(R0)):
        _, vel, mass, _, altitude, _, _, dEkin_dh =\
            run_intergration(V0, M0[i], theta0, Z0, R0[i], Rdot0, 1e4, rho_m, 2.5e6)

        ax.plot(vel / 1e3, altitude / 1e3, zorder=6 + zorders[-(i + 1)],\
                label=r'$R_{\rm imp}$ = ' + str(R0[i]) + r'$\,{\rm m}$',\
                    color=cm.bamako((len(R0) - i - 0.5) / len(R0)))

        _, vel, mass, _, altitude, _, _, dEkin_dh =\
            run_intergration(1.5 * V0, M0[i], theta0, Z0, R0[i], Rdot0, 1e4, rho_m, 2.5e6)

        ax.plot(vel / 1e3, altitude / 1e3, zorder=3 + zorders[-(i + 1)],\
                color=cm.bamako((len(R0) - i - 0.5) / len(R0)), ls=':')

        _, vel, mass, _, altitude, _, _, dEkin_dh =\
            run_intergration(2 * V0, M0[i], theta0, Z0, R0[i], Rdot0, 1e4, rho_m, 2.5e6)

        ax.plot(vel / 1e3, altitude / 1e3, zorder=zorders[-(i + 1)],\
                color=cm.bamako((len(R0) - i - 0.5) / len(R0)), ls='--')

    ax.set_ylim(0, 50)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="30%", pad=0.2)

    for i in range(len(R0)):
        _, vel, mass, _, altitude, _, _, dEkin_dh =\
            run_intergration(V0, M0[i], theta0, Z0, R0[i], Rdot0, 1e4, rho_m, 2.5e6)

        cax.plot(dEkin_dh / mass[0] / 1e3, altitude / 1e3, zorder=6 + zorders[-(i + 1)],\
                    label=r'$R_{\rm imp}$ = ' + str(R0[i]) + r'$\,{\rm m}$',\
                    color=cm.bamako((len(R0) - i - 0.5) / len(R0)))

    cax.set_ylim(0, 50)

    ax.set_xlim(0, 25)
    ax.set_xticks([0, 5, 10, 15, 20, 25])

    cax.set_yticklabels([])

    ax.minorticks_on()
    cax.minorticks_on()

    ax.legend()

    ax.set_ylabel(r'Altitude [km]', fontsize=12)
    ax.set_xlabel(r'Velocity [km/s]', fontsize=12)

    cax.set_xlabel('Specific energy\ndeposition [J/km/kg]', fontsize=12)

    ax.text(23.4, 46, r'(a)', fontsize=12)
    cax.text(6.1, 46, r'(b)', fontsize=12)

    # plt.savefig('comet_survival_trajectories.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
