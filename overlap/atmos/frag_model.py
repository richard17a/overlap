# pylint: disable-msg=C0103
# pylint: disable-msg=R0913
# pylint: disable-msg=R0914
# pylint: disable-msg=W0612
# pylint: disable-msg=W0613

"""
This module containts the numerical model (based on Chyba+1993) used to calculate
the atmospheric entry of comets
"""

import numpy as np
from scipy.integrate import solve_ivp


def differential_equations(t, y, sigma_imp, rho_imp, eta, rho_atm0):
    """
    Differential equations describing the trajectory and deformation of impactor

    Parameters:
    - t: Time
    - y: State vector containing
        [velocity, mass, angle, altitude, radius, radius rate of change]
    - sigma_imp: impactor tensile strength
    - rho_imp: impactor density
    - eta: impactor specific heat of ablation
    - rho_atm0: atmospheric surface density

    Returns:
    - list: Derivative of state vector
    """

    ### ------ model constants ------ ###
    C_d = 0.7   	    # drag coefficient
    C_h = 0.02          # heat transfer efficiency
    C_l = 0.001         # lift coefficient
    M_E = 5.97e24       # Earth mass
    R_E = 6371e3        # Earth radius
    G = 6.67e-11        # Gravitational constant
    sigma = 5.6704e-8   # stefan-boltzmann constant
    T = 25000           # max shock temperature
    H = 7.2e3           # atmospheric scale height
    ### ----------------------- ###

    V, M, theta, Z, R, W = y

    rho_a = rho_atm0 * np.exp(- Z / H)
    A = np.pi * R ** 2
    g = G * M_E / (R_E + Z) ** 2

    dVdt = - C_d * rho_a * A * V**2 / M + g * np.sin(theta)
    dthetadt = (M * g * np.cos(theta) - 0.5 * C_l * rho_a * A * V**2) /\
               (M * V) - V * np.cos(theta) / (R_E + Z)
    dZdt = - V * np.sin(theta)
    dMdt = - np.minimum(sigma * T**4, 0.5 * C_h * rho_a * V**3) * A / eta

    if 0.25 * C_d * rho_a * V**2 > sigma_imp:
        W_dot = C_d * rho_a * V**2 / (2 * rho_imp * R)
    else:
        W_dot = 0.0

    R_dot = dMdt / (2 * np.pi * rho_imp * R ** 2) + W

    return [dVdt, dMdt, dthetadt, dZdt, R_dot, W_dot]


def event_Z_crossing(t, y):
    """
    Event that will trigger when impactor altitude crosses 0 - i.e. reaches
    the surface
    """

    return y[3]


def event_mass_zero(t, y):
    """
    Event that will trigger when impactor has lost all mass due to ablation
    """

    return y[1]


def event_pancake(t, y, R0):
    """
    Event that will trigger when size of pancake exceeds 6 * initial radius -
    following Collins+2005
    """

    return 6 * R0 - y[4]


def event_dVdt_zero(t, y, rho_atm0):
    """
    Event will trigger when object reaches terminal velocity
    """

    C_d = 0.7       # drag coefficient
    H = 7.2e3       # atmospheric scale height
    G = 6.67e-11    # gravitational constant
    M_E = 5.97e24   # Earth mass
    R_E = 6371e3    # Earth radius

    V, M, theta, Z, R, Rdot = y
    rho_a = rho_atm0 * np.exp(- Z / H)
    A = np.pi * R ** 2
    g = G * M_E / (R_E + Z) ** 2

    term_vel = np.sqrt((M * g * np.sin(theta)) / (C_d * rho_a * A))

    out_num = 1

    if (np.isclose(term_vel, V, rtol=1e-01)) & (V < 1e3):
        print(term_vel, V)
        out_num = -1

    return out_num


def run_intergration(V0, M0, theta0, Z0, R0, Rdot0, sigma_imp, rho_imp, eta, rho_atm0=1.225):
    """
    Solve the set of differential equations for atmospheric entry of comets.

    Parameters:
    - V0: initial velocity
    - M0: initial mass
    - theta0: initial impact angle
    - Z0: initial altitude
    - R0: initial radius
    - Rdot0: initial rate of change of radius (always 0)
    - sigma_imp: impactor tensile strength
    - rho_imp: impactor bluk density
    - eta: impactor specific heat of avlation
    - rho_atm0: atmospheric surface density

    Returns:
    - tuple: time, velocity, mass, angle, altitude, radius,
             ram pressure, (atmospheric) energy deposition
    """

    t_span = (0, 500)

    def event_pancake_with_R0(t, y):
        """
        Event triggered when size of pancake exceeds 6 * initial radius -
        here we input the initial radius of the impactor
        """

        return event_pancake(t, y, R0)

    def event_dVdt_zero_rhoatm0(t, y):
        """
        Event triggered when object reaches terminal velocity -
        here we input the atmosphric surface density
        """

        return event_dVdt_zero(t, y, rho_atm0)

    event_Z_crossing.terminal = True
    event_Z_crossing.direction = -1

    event_mass_zero.terminal = True
    event_mass_zero.direction = -1

    event_dVdt_zero_rhoatm0.terminal = True
    event_dVdt_zero_rhoatm0.direction = -1

    event_pancake_with_R0.terminal = True
    event_pancake_with_R0.direction = -1

    events = [event_Z_crossing, event_mass_zero, event_dVdt_zero_rhoatm0, event_pancake_with_R0]


    sol_iso = solve_ivp(
        fun=lambda t, y: differential_equations(t, y, sigma_imp, rho_imp, eta, rho_atm0),
        t_span=t_span,
        y0=[V0, M0, theta0, Z0, R0, Rdot0],
        method='RK45',
        dense_output=True,
        events=events,
        max_step=1e-2
    )

    t = sol_iso.t

    vel = sol_iso.sol(t)[0][:len(t)]
    mass = sol_iso.sol(t)[1][:len(t)]
    theta = sol_iso.sol(t)[2][:len(t)]
    altitude = sol_iso.sol(t)[3][:len(t)]
    radius = sol_iso.sol(t)[4][:len(t)]

    C_d = 0.7           # drag coefficient
    C_h = 0.02          # heat transfer efficiency
    M_E = 5.97e24       # Earth mass
    R_E = 6371e3        # Earth radius
    G = 6.67e-11        # Gravitational constant
    sigma = 5.6704e-8   # stefan-boltzmann constant
    T = 25000           # max shock temperature
    H = 7.2e3           # atmospheric scale height

    g = G * M_E / (R_E + altitude) ** 2
    rho_a = rho_atm0 * np.exp(- altitude / H)
    A = np.pi * radius ** 2

    dVdt = - C_d * rho_a * A * vel**2 / mass + g * np.sin(theta)
    dMdt = - np.minimum(sigma * T**4, 0.5 * C_h * rho_a * vel**3) * A / eta
    dZdt = - vel * np.sin(theta)

    ram_pressure = 0.7 * rho_atm0 * np.exp(-altitude / H) * vel ** 2 / 2
    Ekindot = mass * vel * dVdt + 0.5 * vel**2 * dMdt
    dEkindh = Ekindot / dZdt

    return t, vel, mass, theta, altitude, radius, ram_pressure, dEkindh
