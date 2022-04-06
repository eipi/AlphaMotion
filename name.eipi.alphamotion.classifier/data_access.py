import pandas as pd

import constants
import os
import xml.etree.ElementTree as et
import csv


# example of data that AndroSensor return
# os.listdir()

def read_xml_file(filename, path=''):
    file = os.path.realpath(
        os.path.join(os.path.join(os.getcwd(), os.pardir),
                     os.path.join(os.path.join(constants.DATA_BASE_PATH, path), filename)))
    # pd.read_xml(path_or_buffer=file, parser='etree').head()
    tree = et.parse(file)
    root = tree.getroot()
    print(root)
    for e in root.iter():
        print(e.tag + ' :: ' + e.text)


def read_csv_file_raw(filename, path=''):
    file = os.path.realpath(
        os.path.join(os.path.join(os.getcwd(), os.pardir),
                     os.path.join(os.path.join(constants.DATA_BASE_PATH, path), filename)))
    with open(file, mode='r') as f:
        return oleksander_parser(f)


def read_csv_file_to_xyz(filename, path=''):
    custom_path = os.path.join(os.path.join(constants.DATA_BASE_PATH, path), filename)
    data = pd.read_csv_file_raw(os.path.realpath(custom_path))
    x = data.accelerometer_X.values
    y = data.accelerometer_Y.values
    z = data.accelerometer_Z.values
    return x, y, z


def read_csv_file_to_numpy(filename, path=''):
    custom_path = os.path.join(os.path.join(constants.DATA_BASE_PATH, path), filename)
    data = pd.read_csv(os.path.realpath(custom_path))
    return data.to_numpy()


def oleksander_parser(file):
    samples = []
    #        for line in csv.reader(f):
    #            print(line)
    dict_reader = csv.DictReader(file)
    for line in dict_reader:
        line_x = float(line['accelerometer_X'])
        line_y = float(line['accelerometer_Y'])
        line_z = float(line['accelerometer_Z'])
        vector = [line_x, line_y, line_z]
        samples.append(vector)
        # print('X=' + str(line_x) + ', Y=' + str(line_y) + ', Z=' + str(line_z))
    return samples


def hg93_parser(file):
    samples = []
    #        for line in csv.reader(f):
    #            print(line)
    dict_reader = csv.DictReader(file)
    for line in dict_reader:
        line_x = float(line['accelerometer_X'])
        line_y = float(line['accelerometer_Y'])
        line_z = float(line['accelerometer_Z'])
        vector = [line_x, line_y, line_z]
        samples.append(vector)
        # print('X=' + str(line_x) + ', Y=' + str(line_y) + ', Z=' + str(line_z))
    return samples


parser_mapping = {'oleksander': oleksander_parser(), 'hg93': hg93_parser()}
