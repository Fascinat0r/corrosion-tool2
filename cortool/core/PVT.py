import math

from substance import Substance


def viscosity_from_temperature(substance: Substance, temperature: float) -> float:
    if substance == Substance.ETHANOL:
        # Параметры для этанола
        A = 0.00201 * 1e-6
        B = 1614
        C = 0.00618
        D = -1.132 * 1e-5
        return A * math.exp(B / temperature + C * temperature + D * temperature ** 2)
    elif substance == Substance.NITROGEN:
        # Параметры для азота по формуле Сазерленда
        VISCOSITY_INIT = 1.7e-5
        T_INIT = 273
        S = 104.7
        return (VISCOSITY_INIT * (temperature / T_INIT) ** 1.5) * (T_INIT + S) / (temperature + S)
    # Добавление новых условий для других веществ
    return None
