import json


def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        output = json.load(file)
    return output


def save_config(path, target):
    with open(path, 'w') as fp:
        json.dump(target, fp)
