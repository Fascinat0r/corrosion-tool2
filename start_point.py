import json

from Tube_point import Tube_point


def start_point(point: Tube_point, path: str):  # initialization of start point, done by hand

    fill_variables_from_json(path, point)
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
