import numpy as np
import matplotlib.pyplot as plt


def softmax(x, dim):
    exp = np.exp(x)
    return exp / exp.sum(axis=dim, keepdims=True)


class UnsupervisedFuzzyClusterer:
    def __init__(self, N, C, D, m, A=...):
        self.U = np.random.randn(C, N)
        self.U = softmax(self.U, dim=0)
        self.v = np.random.randn(C, D)
        self.A = A if A != ... else np.identity(D)
        self.m = m

    def compute_J(self, y):
        s = y[:, None, ...] - self.v         # NxCxD
        d2 = np.einsum('ncad, dd, ncda -> nc', s[..., None, :], self.A, s[..., None])

        um = self.U ** self.m
        j = um.T * d2
        return j.mean()

    def update_centeroids(self, y):
        um = self.U ** self.m
        scale_factor = um.sum(axis=-1, keepdims=True)  # Cx1
        self.v = np.dot(um, y) / scale_factor

    def update_membership_matrix(self, y):
        s = y[:, None, ...] - self.v         # NxCxD
        d = np.einsum('ncad, dd, ncda -> nc', s[..., None, :], self.A, s[..., None]).T ** 0.5

        t = 2 / (self.m - 1)
        self.U = 1 / ((d[:, None, ...] / d) ** t).sum(axis=1)                          # CxCxN

    def fit(self, y, eps):
        losses = []
        
        while True:
            self.update_centeroids(y)
            U_old = self.U
            self.update_membership_matrix(y)
            loss = self.compute_J(y).item()
            losses.append(loss)

            print(f'loss: {loss:.4f}', end='\r')

            d = np.linalg.norm(self.U - U_old, 'fro')
            if d <= eps:
                break
        print()
        return losses
    

class SemiSupervisedFuzzyClusterer:
    def __init__(self, N, C, D, m):
        U = np.random.randn(N, C)
        self.U = softmax(U, dim=1)
        self.v = np.random.randn(C, D)
        self.m = m

    def compute_J(self, x, U_bar):
        t1 = np.abs(self.U - U_bar) ** self.m
        t2 = np.sum((x[:, None, ...] - self.v) ** 2, axis=-1)
        return np.sum(t1 * t2)

    def update_centeroids(self, x, U_bar):
        t = np.abs(self.U - U_bar) ** self.m
        num = np.sum(t[..., None] * x[:, None, ...], axis=0)
        den = np.sum(t, axis=0)[..., None]
        self.v = num / den

    def update_U(self, x, U_bar):
        e = 1 / (1 - self.m)
        d = np.sum((x[:, None, ...] - self.v) ** 2, axis=-1)

        multiplier = np.sum(1 - U_bar, axis=-1, keepdims=True)
        num = d ** e
        den = np.sum(num, axis=-1, keepdims=True)

        self.U = U_bar + multiplier * num / den
    
    def fit(self, x, U_bar, eps):
        losses = []
        
        while True:
            self.update_centeroids(x, U_bar)
            U_old = self.U
            self.update_U(x, U_bar)
            loss = self.compute_J(x, U_bar).item()
            losses.append(loss)

            print(f'loss: {loss:.4f}', end='\r')

            d = np.linalg.norm(self.U - U_old, 'fro')
            if d <= eps:
                break
        print()
        return losses