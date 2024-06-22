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
    percentSupervision = 90
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    
    df = pd.read_csv(path).drop('Id', axis=1)
    N = len(df)
    amount = int(N * percentSupervision / 100)
    x = df.drop(columns[-1], axis=1).to_numpy(float)
    
    indices = pd.factorize(df[columns[-1]])[0]
    supervise = np.random.choice(N, amount, replace=False)
    U_bar = np.zeros([N, C])
    for i in supervise:
        U_bar[i][indices[i]] = 0.51

    return x, U_bar
    
def unsupervisedPrepare_iris(path, C):
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    
    df = pd.read_csv(path).drop('Id', axis=1)
    x = df.drop(columns[-1], axis=1).to_numpy(float)
    return x

def true_labels():
    df = pd.read_csv('D:\Code\GR\GR1\iris.csv')
    indices = pd.factorize(df['Species'])[0]
    return indices
    
    

