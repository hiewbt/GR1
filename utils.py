import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot(y, U, v):
    fig, ax = plt.subplots()
    ax.scatter(y[:, 0], y[:, 1], c=np.argmax(U, axis=0))
    ax.scatter(v[:, 0], v[:, 1], marker='X', c='purple')
    return fig


def show_centroids(root, vs):
    for i, v in enumerate(vs, 1):
        root.write(f'v{i}:\t' + str(v))

    
def prepare_iris(path, C):
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    
    df = pd.read_csv(path).drop('Id', axis=1)
    N = len(df)
    x = df.drop(columns[-1], axis=1).to_numpy(float)
    
    indices = pd.factorize(df[columns[-1]])[0]
    U_bar = np.zeros([N, C])
    for row, idx in zip(U_bar, indices):
        row[idx] = 1
    
    return x, U_bar
    
def unsupervisedPrepare_iris(path, C):
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    
    df = pd.read_csv(path).drop('Id', axis=1)
    x = df.drop(columns[-1], axis=1).to_numpy(float)
    return x