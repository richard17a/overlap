# pylint: disable-msg=C0103
# pylint: disable-msg=R0913
# pylint: disable-msg=R0914
# pylint: disable-msg=W0612
# pylint: disable-msg=W0613

"""
Module docstring
"""

import numpy as np
from scipy.integrate import solve_ivp


def differential_equations(t, y, sigma_imp, rho_imp, eta, rho_atm0):
    """
    Docstring
    """

    ### ------ Defining constants ------ ### (COME BACK AND ADD DESCRIPTIONS HERE)
    C_d = 0.7
    C_h = 0.02
    C_l = 0.001
    M_E = 5.97e24
    R_E = 6371e3
    G = 6.67e-11
    sigma = 5.6704e-8
    T = 25000
    H = 7.2e3
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

#    if 0.25 * C_d * rho_a * V**2 > sigma_imp:
#        R_dot = Rdot
#        R_ddot = C_d * rho_a * V**2 / (2 * rho_imp * R)
#    else:
#        R_dot = 0.0
#        R_ddot = 0.0
#
#    return [dVdt, dMdt, dthetadt, dZdt, R_dot, R_ddot]
    if 0.25 * C_d * rho_a * V**2 > sigma_imp:
        W_dot = C_d * rho_a * V**2 / (2 * rho_imp * R)
    else:
        W_dot = 0.0

    R_dot = dMdt / (2 * np.pi * rho_imp * R ** 2) + W

    return [dVdt, dMdt, dthetadt, dZdt, R_dot, W_dot]


def event_Z_crossing(t, y):
    """
    Event triggered when altitude crosses 0
    """

    return y[3]


def event_mass_zero(t, y):
    """
    Event triggered when all mass ablated
    """

    return y[1]


def event_pancake(t, y, R0):
    """
    Event triggered when size of pancake exceeds 6 * initial radius
    """

    return 6 * R0 - y[4]


def event_dVdt_zero(t, y, rho_atm0):
    """
    Event triggered when object reaches terminal velocity
    """

    C_d = 0.7
    H = 7.2e3
    G = 6.67e-11
    M_E = 5.97e24
    R_E = 6371e3

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
    Docstring
    """

    # Time span for integration
    t_span = (0, 500)

    def event_pancake_with_R0(t, y):
        """
        Event triggered when size of pancake exceeds 6 * initial radius
        """

        return event_pancake(t, y, R0)

    def event_dVdt_zero_rhoatm0(t, y):
        """
        Event triggered when object reaches terminal velocity
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

    # Solve the differential equations
    sol_iso = solve_ivp(
        fun=lambda t, y: differential_equations(t, y, sigma_imp, rho_imp, eta, rho_atm0),
        t_span=t_span,
        y0=[V0, M0, theta0, Z0, R0, Rdot0],
        method='RK45',  # You can choose other integration methods as well
        dense_output=True,
        events=events,
        max_step=1e-2  # Adjust the maximum step size
    )

    t = sol_iso.t

    vel = sol_iso.sol(t)[0][:len(t)]
    mass = sol_iso.sol(t)[1][:len(t)]
    theta = sol_iso.sol(t)[2][:len(t)]
    altitude = sol_iso.sol(t)[3][:len(t)]
    radius = sol_iso.sol(t)[4][:len(t)]

    C_d = 0.7
    C_h = 0.02
    M_E = 5.97e24
    R_E = 6371e3
    G = 6.67e-11
    sigma = 5.6704e-8
    T = 25000
    H = 7.2e3
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
