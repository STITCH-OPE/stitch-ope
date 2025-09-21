import json
import os
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))


def load_json(paths):
    files = []
    for json_path in paths:
        with open(json_path) as f:
            d = json.load(f)
            files.append(d)
    return files


def plot_estimates(names, jsons):
    values = {}
    for name, json_file in zip(names, jsons):
        values[name] = json_file['mean']
        values[name]['correct'] = json_file['correct']
    df = pd.DataFrame.from_dict(values)
    df = df.transpose()
    df.plot(kind='bar')
    plt.savefig('mean.pdf')
    

def plot_log_rmse(names, jsons):
    values = {}
    for name, json_file in zip(names, jsons):
        values[name] = json_file['log_rmse']
    df = pd.DataFrame.from_dict(values)
    df = df.transpose()
    df.plot(kind='bar')
    plt.savefig('rmse.pdf')


names = ['acrobot', 'cartpole', 'mountaincar']
paths = [
    os.path.join(path, name, 'outputs', 'log_rmses_200_50_1.0.json') 
    for name in names
]

jsons = load_json(paths)
plot_estimates(names, jsons)
plot_log_rmse(names, jsons)
