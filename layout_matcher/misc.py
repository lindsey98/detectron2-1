import numpy as np
import re
import yaml

def preprocess(shot_size, compos):
    '''
    :param shot_size:
    :param compos:
    :return: rescaled compos within range of [0, 100]
    '''
    image_size = np.array(list(shot_size[:2][::-1] * 2))
    box_after = np.asarray(list(map(lambda x: np.divide(x, image_size) * 100, compos)))
    return box_after

def read_coord(coord_path):
    '''
    Read coordinates txt
    :param coord_path:
    :return: coordinates array Nx4, confidence array Nx1
    '''
    coords = [x.strip().split('\t')[0] for x in open(coord_path).readlines()]
    confidence = [x.strip().split('\t')[1] for x in open(coord_path).readlines()]

    coords_arr = []
    for coord in coords:
        testbox = list(map(float, re.search(r'\((.*?)\)', coord).group(1).split(",")))
        coords_arr.append(testbox)

    coords_arr = np.asarray(coords_arr)
    return coords_arr, confidence


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config