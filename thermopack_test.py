import copy
import math
import sys
from dataclasses import dataclass, field
from typing import List

import thermopack.cubic

sys.path.insert(0, '../pycThermopack/')
from thermopack.cubic import cubic


@dataclass
class tube_point:
    name: str = 'mixture'  # name: string, name of the material
    phase_name: List[str] = field(default_factory=lambda: ['gas'])  # phase_name: list of strings, name of phases
    number_of_fluids: int = 1  # number_of_fluids: integer, number of phases
    temperature: float = 300.0  # T: double, local temperature, Kelvin
    pressure: float = 101325.0  # p: double, local pressure, Pascal
    velocity: float = 0.5  # v: double, mixture velocity, m/s
    diameter: float = 0.1  # D: double, tube local diameter, m
    length: float = 10  # length: double, tube length, m. Zero if it's endpoint
    section_type = 1  # section type: integer, type of the section for losses calculation, not used yet - enum!
    molar_composition: List[float] = field(
        default_factory=lambda: [1.0])  # molar_composition: list of doubles, molar composition [probably constant]
    molar_masses: List[float] = field(
        default_factory=lambda: [1.0])  # molar_masses: list of doubles, molar masses [constant overall]
    vapor_components: List[float] = field(default_factory=lambda: [
        0.5])  # vapor_components: list of doubles, vapor distribution over components (sum is 1)
    liquid_components: List[float] = field(default_factory=lambda: [
        0.5])  # liquid_components: list of doubles, liquid distribution over components (sum is 1)
    components_density: List[float] = field(
        default_factory=lambda: [1.0])  # components_density: list of doubles, density distribution over components
    overall_density: float = 1.0  # overall_density: double, overall density, kg/m^3
    overall_vapor_fraction: float = 0.5  # overall_vapor_fraction: double, vapor distribution over mixture
    overall_liquid_fraction: float = 0.5  # overall_liquid_fraction: double, liquid distribution over mixture
    liquid_viscosities: List[float] = field(default_factory=lambda: [
        1.0e-3])  # liquid_viscosities: list of doubles, viscosity of liquid parts over components
    vapor_viscosities: List[float] = field(default_factory=lambda: [
        1.0e-3])  # vapor_viscosities:  list of doubles, viscosity of vapor parts over components
    liquid_overall_viscosity: float = 1.0e-3  # liquid_overall_viscosity: double, viscosity of liquid part
    vapor_overall_viscosity: float = 1.0e-3  # vapor_overall_viscosity: double, viscosity of vapor part
    overall_viscosity: float = 1.0e-3  # overall_viscosity: double, viscosity of mixture
    flow_mode: str = "bubble"  # flow_mode: string, name of selected flow flow_mode
    flow_mode_key: float = 1.0  # flow_mode_key: double, currently XTT, later - other number to characterize flow_mode
    flow_mode_friction_factor: float = 1.0  # flow_mode_friction_factor: double, currently from XTT
    reynolds_number: float = 10000.0  # reynolds_number: double, Reynolds number for ...


def calculate_xtt(liquid_density, gas_density, liquid_viscosity, gas_viscosity, velocity, diameter):
    return ((1.096 / liquid_density) ** 0.5) * ((liquid_density / gas_density) ** 0.25) * (
            (gas_viscosity / liquid_viscosity) ** 0.1) * ((velocity / diameter) ** 0.5)


def calculate_xtt(point: tube_point):
    liquid_density = point.components_density[0]
    gas_density = point.components_density[1]
    liquid_viscosity = point.liquid_viscosities[0]
    gas_viscosity = point.liquid_viscosities[1]
    velocity = point.velocity
    diameter = point.diameter
    return ((1.096 / liquid_density) ** 0.5) * ((liquid_density / gas_density) ** 0.25) * (
            (gas_viscosity / liquid_viscosity) ** 0.1) * ((velocity / diameter) ** 0.5)


def calculate_viscosity(liquid_viscosity, gas_viscosity, friction_factor):
    return ff * liquid_viscosity + (1 - friction_factor) * gas_viscosity


def calculate_Re(velocity, diameter, overall_density, overall_viscosity):
    return velocity * diameter * overall_density / overall_viscosity


def calculate_Re(point: tube_point):
    return point.velocity * point.diameter * point.overall_density / point.overall_viscosity


def return_lambda(Re):
    if Re < 2300:
        return 64 / Re
    else:
        return 0.316 / (Re ** 0.25)


def return_pressure_loss(velocity, diameter, length, lam, density):
    xi = lam * length / diameter
    return (xi * velocity ** 2) * 0.5 * density


def get_overall_density(point: tube_point):  # необходимо дописать учёт агрегатного состояния
    return sum(point.molar_composition[i] * point.components_density[i] for i in range(point.number_of_fluids))


def return_mode(xtt):
    if xtt < 10: return 'bubble'
    if 10 <= xtt < 100:
        return 'plug'
    if 100 <= xtt < 1000:
        return 'slug'
    if 1000 <= xtt < 10000:
        return 'annular'
    if 10000 <= xtt:
        return 'mist'
    return 'undefined'


# liquid to solid viscosity calculation:

def return_friction_factor(xtt):
    if xtt < 10:
        return 1
    if 10 <= xtt < 100:
        return 0.9
    if 100 <= xtt < 1000:
        return 0.8
    if 1000 <= xtt < 10000:
        return 0.7
    if 10000 <= xtt:
        return 0.6
    return 0


# PVT block
def ethanol_viscosity_from_temperature(T):
    """
    Calculation of the viscosity of liquid ethanol as a function of temperature by exponential correlation
    :param T: temperature of ethanol
    :return: ethanol viscosity
    """
    A = 0.00201 * 1e-6
    B = 1614
    C = 0.00618
    D = 1.132 * (-1e-5)
    return A * math.exp(B / T + C * T + D * T * T)


# PVT block
def n2_viscosity_from_temp(T):
    """
    Calculation of the viscosity of nitrogen gas as a function of temperature according to the Sutherland formula
    :param T: temperature of nitrogen
    :return: viscosity of nitrogen gas
    """
    VISCOSITY_INIT = 1.7e-5
    T_INIT = 273
    S = 104.7
    return (VISCOSITY_INIT * (T / T_INIT) ** 1.5) * (T_INIT + S) / (T + S)


def update_point_state(point: tube_point, rk_fluid: thermopack.cubic.cubic):
    """
    Updates tube point parameters after changing local temperature and pressure
    :param point:
    :param rk_fluid:
    :return: tube point with the updated state
    """
    x, y, vap_frac, liq_frac, phase_key = rk_fluid.two_phase_tpflash(point.temperature, point.pressure,
                                                                     point.molar_composition)
    point.vapor_components = x
    point.liquid_components = y
    point.overall_vapor_fraction = vap_frac
    point.overall_liquid_fraction = liq_frac
    temp, = rk_fluid.specific_volume(point.temperature, point.pressure, point.molar_composition, 1)
    density_1 = point.molar_masses[0] / temp
    temp, = rk_fluid.specific_volume(point.temperature, point.pressure, point.molar_composition, 2)
    density_2 = point.molar_masses[1] / temp
    point.components_density = [density_1, density_2]
    point.overall_density = get_overall_density(point)
    ethanol_viscosity = ethanol_viscosity_from_temperature(point.temperature)
    n2_viscosity = n2_viscosity_from_temp(point.temperature)
    point.liquid_viscosities = [ethanol_viscosity, n2_viscosity]
    point.vapor_viscosities = [ethanol_viscosity, n2_viscosity]
    return point


def start_point(point: tube_point):  # initialization of start point, done by hand
    rk_fluid = cubic('N2,ETOH', 'SRK')  # obsolete

    point.temperature = 320.0  # Kelvin
    point.pressure = 3.5 * 101325  # Pascal
    point.molar_composition = [0.5, 0.5]  # Molar composition
    point.molar_masses = [0.03, 0.028]
    point.velocity = 5.0  # [m/s]
    point.diameter = 0.08  # [m]
    point.length = 100

    point = update_point_state(point, rk_fluid)
    return point


def define_tube_params(point: tube_point, diameter, length, density_old):
    q = point.velocity * point.diameter * point.diameter * density_old
    new_velocity = q / (diameter * diameter * point.overall_density)  # mass balance, pi/4 is skipped
    # due to presence in both parts of equation
    point.diameter = diameter
    point.length = length
    point.velocity = new_velocity
    return point


def pvt_block(point: tube_point, new_pressure, new_temperature):
    rk_fluid = cubic('N2,ETOH', 'SRK')  # obsolete

    point.temperature = new_temperature
    point.pressure = new_pressure

    point = update_point_state(point, rk_fluid)
    return point


point = tube_point()
start_point(point)
tube_diameters = [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.1]
tube_lengths = [1000, 200, 450, 1000, 200, 300, 200, 300, 1000, 0]
tube_points = [point]
for i in range(1, len(tube_diameters)):
    xtt = calculate_xtt(tube_points[i - 1])
    # print ('i', i, ' xtt ', xtt)
    flow_mode = return_mode(xtt)
    print('Found ', flow_mode, ' flow_mode at', i, '! xtt= ', xtt)
    ff = return_friction_factor(xtt)
    tube_points[i - 1].overall_viscosity = calculate_viscosity(tube_points[i - 1].liquid_viscosities[0],
                                                               tube_points[i - 1].liquid_viscosities[1], ff)
    tube_points[i - 1].reynolds_number = calculate_Re(tube_points[i - 1])
    print('Reynolds number for ', i, ' is ', tube_points[i - 1].reynolds_number, 'lambda is ', return_lambda(tube_points[i - 1].reynolds_number))
    diff = return_pressure_loss(tube_points[i - 1].velocity, tube_points[i - 1].diameter, tube_points[i - 1].length,
                                return_lambda(tube_points[i - 1].reynolds_number), tube_points[i - 1].overall_density)
    P1 = tube_points[i - 1].pressure - diff
    T1 = tube_points[i - 1].temperature - i * 0.3
    print(P1, T1)
    point_2 = copy.deepcopy(tube_points[i - 1])

    density = point_2.overall_density
    point_2 = pvt_block(point_2, P1, T1)
    point_2 = define_tube_params(point_2, tube_diameters[i], tube_lengths[i], density)
    tube_points.append(point_2)

# end of PVT block
# start of flow_mode part
# xtt = calculate_xtt(p)
# flow_mode = return_mode(xtt)
# print ('Found ', flow_mode, ' flow_mode!; xtt= ', xtt)
# end of flow_mode part
# start of friction part
# ff = return_friction_factor(xtt)

# p.overall_viscosity = calculate_visc(p.liquid_viscosities[0], p.liquid_viscosities[1], ff)

# p.Re = calculate_Re(p)

# print ('Reynolds number is ', p.Re)
# diff = return_pressure_loss(p.v,p.D,length, return_lambda(p.Re), p.overall_density)
# print (p.p, diff)
# P1 = p.p-diff
# T1 = p.T-20.0
# print ('New pressure is', P1)

# p2 = copy.deepcopy(p)
# pvt_block(p2, P1, T1)
# xtt2 = calculate_xtt(p2)
# mode2 = return_mode(xtt2)
# print ('New flow_mode is ', mode2, ' !; xtt= ', xtt2)
# pressure loss calculated, now time for calculating flow_mode again
# print ('dp', p.p, P1, 'D', p.D, 'length', length)
# wss = 0.25*(diff)*p.D/length
# print ('Average WSS is about', wss, 'Pa')

# tube = [p,p2]

# rho_1_1, =  rk_fluid.specific_volume(T1, P1, z, 1)
# rho_1_1= m_ethanol/rho_1_1
# rho_2_1, =  rk_fluid.specific_volume(T1, P1, z, 2)
# rho_2_1 = m_n2/rho_2_1
# rho_tot_1 = z[0]*rho_1+z[1]*rho_2


# xtt = calculate_xtt(rho_1_1, rho_2_1, mu_1, mu_2, v1, D)
# flow_mode = return_mode(xtt)
# print ('New flow_mode is ', flow_mode, ' !; xtt= ', xtt)
# wss = 0.25*(p-P1)*D/length
# print ('Average WSS is about', wss, 'Pa')
