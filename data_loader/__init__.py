import json


from data_loader.Vaihingen_loader import VaihingenLoader
from data_loader.Potsdam_loader import PotsdamLoader
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'Vaihingen': VaihingenLoader,
        'Potsdam':PotsdamLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
