import copy
import sys
import time

sys.path.insert(0, '../pycThermopack/')
from Tube_point import *


@deprecated
def pvt_block(point: Tube_point, new_pressure, new_temperature):
    point.temperature = new_temperature
    point.pressure = new_pressure
    update_point_state(point)
    return point


@deprecated
def main(path: str):
    point = Tube_point()
    start_point_from_excel(point, path)
    tube_diameters = [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.1]
    tube_lengths = [1000, 200, 450, 1000, 200, 300, 200, 300, 1000, 0]
    tube_points = [point]

    re_nums = list()
    list_p1 = list()
    list_t1 = list()

    for i in range(1, len(tube_diameters)):
        xtt = calculate_xtt(tube_points[i - 1])
        # print ('i', i, ' xtt ', xtt)
        flow_mode = return_mode(xtt)
        print('Found ', flow_mode, ' flow_mode at', i, '! xtt= ', xtt)
        friction_factor = return_friction_factor(xtt)
        tube_points[i - 1].overall_viscosity = calculate_viscosity(tube_points[i - 1], friction_factor)
        tube_points[i - 1].reynolds_number = calculate_Re(tube_points[i - 1])
        re_nums.append(tube_points[i - 1].reynolds_number)
        print('Reynolds number for ', i, ' is ', tube_points[i - 1].reynolds_number, 'lambda is ',
              calculate_lambda(tube_points[i - 1]))
        diff = calculate_pressure_loss(point)
        P1 = tube_points[i - 1].pressure - diff
        T1 = tube_points[i - 1].temperature - i * 0.3
        list_p1.append(P1)
        list_t1.append(T1)
        print('P1: ', P1, ', T1: ', T1)
        next_point = copy.deepcopy(tube_points[i - 1])

        density = next_point.overall_density
        next_point = pvt_block(next_point, P1, T1)
        define_tube_params(next_point, tube_diameters[i], tube_lengths[i], density)

        tube_points.append(next_point)
        print()


if __name__ == "__main__":
    start = time.time()
    main("xlsm/Imput_data.xlsm")
    end = time.time()
    print("The time of execution of above program is :", (end - start) * 10 ** 3, "ms")
