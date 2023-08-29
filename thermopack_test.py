import copy
import sys
sys.path.insert(0, '../pycThermopack/')
from Tube_point import tube_point

def calculate_xtt(liquid_density, gas_density, liquid_viscosity, gas_viscosity, velocity, diameter):
    return ((1.096 / liquid_density) ** 0.5) * ((liquid_density / gas_density) ** 0.25) * (
            (gas_viscosity / liquid_viscosity) ** 0.1) * ((velocity / diameter) ** 0.5)


def calculate_viscosity(liquid_viscosity, gas_viscosity, friction_factor):
    return friction_factor * liquid_viscosity + (1 - friction_factor) * gas_viscosity


def calculate_Re(velocity, diameter, overall_density, overall_viscosity):
    return velocity * diameter * overall_density / overall_viscosity


def calculate_lambda(Re):
    if Re < 2300:
        return 64 / Re
    else:
        return 0.316 / (Re ** 0.25)


def return_pressure_loss(velocity, diameter, length, lam, density):
    xi = lam * length / diameter
    return (xi * velocity ** 2) * 0.5 * density


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


def start_point(point: tube_point):  # initialization of start point, done by hand

    point.temperature = 320.0  # Kelvin
    point.pressure = 3.5 * 101325  # Pascal
    point.molar_composition = [0.5, 0.5]  # Molar composition
    point.molar_masses = [0.03, 0.028]
    point.velocity = 5.0  # [m/s]
    point.diameter = 0.08  # [m]
    point.length = 100

    point.update_point_state()
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
    point.temperature = new_temperature
    point.pressure = new_pressure
    point.update_point_state()
    return point


point = tube_point()
start_point(point)
tube_diameters = [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.1]
tube_lengths = [1000, 200, 450, 1000, 200, 300, 200, 300, 1000, 0]
tube_points = [point]
for i in range(1, len(tube_diameters)):
    xtt = tube_points[i - 1].calculate_xtt()
    # print ('i', i, ' xtt ', xtt)
    flow_mode = return_mode(xtt)
    print('Found ', flow_mode, ' flow_mode at', i, '! xtt= ', xtt)
    friction_factor = return_friction_factor(xtt)
    tube_points[i - 1].overall_viscosity = calculate_viscosity(tube_points[i - 1].liquid_viscosities[0],
                                                               tube_points[i - 1].liquid_viscosities[1],
                                                               friction_factor)
    tube_points[i - 1].reynolds_number = tube_points[i - 1].calculate_Re()
    print('Reynolds number for ', i, ' is ', tube_points[i - 1].reynolds_number, 'lambda is ',
          calculate_lambda(tube_points[i - 1].reynolds_number))
    diff = point.calculate_pressure_loss()
    P1 = tube_points[i - 1].pressure - diff
    T1 = tube_points[i - 1].temperature - i * 0.3
    print('P1: ', P1, ', T1: ', T1)
    point_2 = copy.deepcopy(tube_points[i - 1])

    density = point_2.overall_density
    point_2 = pvt_block(point_2, P1, T1)
    point_2 = define_tube_params(point_2, tube_diameters[i], tube_lengths[i], density)

    tube_points.append(point_2)
    print()

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
