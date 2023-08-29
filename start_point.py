import json

from Tube_point import tube_point


def start_point(point: tube_point):  # initialization of start point, done by hand

    # point.temperature = 320.0  # Kelvin
    # point.pressure = 3.5 * 101325  # Pascal
    # point.molar_composition = [0.5, 0.5]  # Molar composition
    # point.molar_masses = [0.03, 0.028]
    # point.velocity = 5.0  # [m/s]
    # point.diameter = 0.08  # [m]
    # point.length = 100
    fill_variables_from_json("jsons/default.json", point)
    point.update_point_state()
    return point


def fill_variables_from_json(json_file, point):
    with open(json_file, 'r') as file:
        data = json.load(file)

    point.temperature = data.get('temperature', point.temperature)
    point.pressure = data.get('pressure', point.pressure)
    point.molar_composition = data.get('molar_composition', point.molar_composition)
    point.molar_masses = data.get('molar_masses', point.molar_masses)
    point.velocity = data.get('velocity', point.velocity)
    point.diameter = data.get('diameter', point.diameter)
    point.length = data.get('length', point.length)
