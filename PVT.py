import math


# PVT block
def ethanol_viscosity_from_temperature(T):
    """
    Calculation of the viscosity of liquid ethanol as a function of temperature by exponential correlation
    :param T: temperature of ethanol, K
    :return: ethanol viscosity
    """
    A = 0.00201 * 1e-6
    B = 1614
    C = 0.00618
    D = 1.132 * (-1e-5)
    return A * math.exp(B / T + C * T + D * T * T)


# PVT block
def n2_viscosity_from_temperature(T):
    """
    Calculation of the viscosity of nitrogen gas as a function of temperature according to the Sutherland formula
    :param T: temperature of nitrogen, K
    :return: viscosity of nitrogen gas
    """
    VISCOSITY_INIT = 1.7e-5
    T_INIT = 273
    S = 104.7
    return (VISCOSITY_INIT * (T / T_INIT) ** 1.5) * (T_INIT + S) / (T + S)
