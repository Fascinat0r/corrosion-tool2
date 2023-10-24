from typing import List

import pandas as pd
from thermopack.cubic import cubic

from PVT import *


class Tube_point:
    name: str = 'mixture'  # name: string, name of the material
    phase_name: List[str] = ['gas']  # phase_name: list of strings, name of phases
    number_of_fluids: int = 1  # number_of_fluids: integer, number of phases
    temperature: float = 300.0  # T: double, local temperature, Kelvin
    pressure: float = 101325.0  # p: double, local pressure, Pascal
    velocity: float = 0.5  # v: double, mixture velocity, m/s
    diameter: float = 0.1  # D: double, tube local diameter, m
    length: float = 10  # length: double, tube length, m. Zero if it's endpoint
    section_type = 1  # section type: integer, type of the section for losses calculation, not used yet - enum!
    molar_composition: List[float] = [1.0]  # molar_composition: list of doubles, molar composition [probably constant]
    molar_masses: List[float] = [1.0]  # molar_masses: list of doubles, molar masses [constant overall]
    vapor_components: List[float] = [
        0.5]  # vapor_components: list of doubles, vapor distribution over components (sum is 1)
    liquid_components: List[float] = [
        0.5]  # liquid_components: list of doubles, liquid distribution over components (sum is 1)
    components_density: List[float] = [1.0]  # components_density: list of doubles, density distribution over components
    overall_density: float = 1.0  # overall_density: double, overall density, kg/m^3
    overall_vapor_fraction: float = 0.5  # overall_vapor_fraction: double, vapor distribution over mixture
    overall_liquid_fraction: float = 0.5  # overall_liquid_fraction: double, liquid distribution over mixture
    liquid_viscosities: List[float] = [
        1.0e-3]  # liquid_viscosities: list of doubles, viscosity of liquid parts over components
    vapor_viscosities: List[float] = [
        1.0e-3]  # vapor_viscosities:  list of doubles, viscosity of vapor parts over components
    liquid_overall_viscosity: float = 1.0e-3  # liquid_overall_viscosity: double, viscosity of liquid part
    vapor_overall_viscosity: float = 1.0e-3  # vapor_overall_viscosity: double, viscosity of vapor part
    overall_viscosity: float = 1.0e-3  # overall_viscosity: double, viscosity of mixture
    flow_mode: str = "bubble"  # flow_mode: string, name of selected flow flow_mode
    flow_mode_key: float = 1.0  # flow_mode_key: double, currently XTT, later - other number to characterize flow_mode
    flow_mode_friction_factor: float = 1.0  # flow_mode_friction_factor: double, currently from XTT
    reynolds_number: float = 10000.0  # reynolds_number: double, Reynolds number for ...
    roughness = 0.01  # roughness: шероховатость внутренней поверхности трубы
    mass = 0.1
    angle = 0  # angle: угол наклона трубы

    def start_point_from_excel(self, excel_path):  # initialization of start pointы
        df = pd.read_excel(excel_path)
        _ = df['Unnamed: 1']

        self.temperature = _[0]
        self.pressure = _[2]
        self.molar_composition = [_[7], _[13]]
        self.molar_masses = [_[6], _[12]]
        self.velocity = _[3]
        self.diameter = _[22]
        self.length = _[20]
        self.vapor_viscosities = _[9]
        self.liquid_viscosities = _[15]
        self.components_density = [_[8], _[14]]
        self.roughness = _[21]
        self.mass = _[1]
        self.update_point_state()

    def define_tube_params(self, diameter, length, density_old):
        q = self.velocity * self.diameter * self.diameter * density_old
        new_velocity = q / (diameter * diameter * self.overall_density)  # mass balance, pi/4 is skipped
        # due to presence in both parts of equation
        self.diameter = diameter
        self.length = length
        self.velocity = new_velocity

    def update_point_state(self):
        """
        Updates tube point parameters after changing local temperature and pressure
        """
        rk_fluid = cubic('N2,ETOH', 'SRK')  # obsolete
        x, y, vap_frac, liq_frac, phase_key = rk_fluid.two_phase_tpflash(self.temperature, self.pressure,
                                                                         self.molar_composition)
        self.vapor_components = x
        self.liquid_components = y
        self.overall_vapor_fraction = vap_frac
        self.overall_liquid_fraction = liq_frac

        temp, = rk_fluid.specific_volume(self.temperature, self.pressure, self.molar_composition, 1)
        density_1 = self.molar_masses[0] / temp

        temp, = rk_fluid.specific_volume(self.temperature, self.pressure, self.molar_composition, 2)
        density_2 = self.molar_masses[1] / temp

        self.components_density = [density_1, density_2]
        self.overall_density = self.calculate_overall_density()
        ethanol_viscosity = ethanol_viscosity_from_temperature(self.temperature)
        n2_viscosity = n2_viscosity_from_temperature(self.temperature)
        self.liquid_viscosities = [ethanol_viscosity, n2_viscosity]
        self.vapor_viscosities = [ethanol_viscosity, n2_viscosity]

    def calculate_Re(self):
        """
        Calculates the Reynolds number based on the total density and total viscosity of the medium.
        :return: Reynolds number
        """
        return self.velocity * self.diameter * self.overall_density / self.overall_viscosity

    def calculate_xtt(self):
        """
        Calculates the parameter by which the flow mode can be obtained.
        NOTE: The simplest correlation has been applied, which will require adjustments in the future
        :return: xtt - Lockhart-Martinelli parameter
        """
        liquid_density = self.components_density[0]
        gas_density = self.components_density[1]
        liquid_viscosity = self.liquid_viscosities[0]  # ? liquid_overall_viscosity?
        gas_viscosity = self.liquid_viscosities[1]  # ?
        velocity = self.velocity
        diameter = self.diameter
        return ((1.096 / liquid_density) ** 0.5) * ((liquid_density / gas_density) ** 0.25) * (
                (gas_viscosity / liquid_viscosity) ** 0.1) * ((velocity / diameter) ** 0.5)

    def calculate_overall_density(self):  # необходимо дописать учёт агрегатного состояния
        return sum(self.molar_composition[i] * self.components_density[i] for i in range(self.number_of_fluids))

    def calculate_lambda(self):
        if self.reynolds_number < 2300:
            return 64 / self.reynolds_number
        else:
            return 0.316 / (self.reynolds_number ** 0.25)

    def calculate_pressure_loss(self):
        xi = self.calculate_lambda() * self.length / self.diameter
        return (xi * self.velocity ** 2) * 0.5 * self.overall_density

    def calculate_viscosity(self, friction_factor):
        liquid_viscosity = self.liquid_viscosities[0]  # ? liquid_overall_viscosity?
        gas_viscosity = self.liquid_viscosities[1]  # ?
        return friction_factor * liquid_viscosity + (1 - friction_factor) * gas_viscosity


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
    """
    Outputs the friction factor to calculate the viscosity.
    :param xtt:
    :return:
    """
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

