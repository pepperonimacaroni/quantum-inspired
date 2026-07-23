import numpy as np

def load_sequence():
    x = np.linspace(0, 10 * np.pi, 500)
    data = np.sin(x)
    return data

#Regression Demo:
def data_creation():
    X = np.linspace(-2, 2, 200).reshape(-1, 1)
    Y = X ** 2
    return X, Y

def create_windows(data, window_size):
    X = []
    Y = []
    i = 0

    while i < len(data) - window_size:
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])
        i += 1

    return np.array(X), np.array(Y).reshape(-1,1)