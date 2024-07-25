import numpy as np


# euclidean distance
def compute_dist(*data, dim, single=True):
    if single:
        x = y = data[0]
    else:
        x = data[0]
        y = data[1]
    costm = np.zeros((x.shape[0], y.shape[0]))
    for d in range(dim):
        costm += np.power(np.subtract.outer(x[:, d], y[:, d]), 2)
    return costm

# compute entropy term
def get_entropy(a, reg):
    return -np.sum(a * np.log(a + 1e-20)) * reg

# compute KL divergence
def get_kl(a, b):
    return np.sum(a * np.log(a / (b + 1e-20)))

# compute entropy regularized balanced optimal transport map
def ot_balanced(a, b, costm, reg, n_iter=1000):
    tmap = np.exp(-costm / reg)
    for i in range(n_iter):
        tmap = np.diag(a) @ np.diag(1 / tmap.sum(axis=1)) @ tmap
        tmap = tmap @ np.diag(1 / tmap.sum(axis=0)) @ np.diag(b)
    return tmap

# compute entropy regularized balanced optimal transport map with log stabilization
def ot_balanced_log_stabilized(a, b, costm, reg, reg_list, n_iter=1000, tau=50, n_per_reg=20):
    def diag_iter(costm, a, b, u, v, reg, n_iter, get_scaling=False):
        K0 = np.exp((np.add.outer(u, v) - costm) / reg)
        b0 = np.ones(b.shape[0])
        u0 = u
        v0 = v
        for i in range(n_iter):
            a0 = a / (K0 @ b0)
            b0 = b / (K0.T @ a0)
            if np.max(np.abs(np.log(np.hstack((a0, b0))))) > tau:
                # print('detected')
                u0 = u0 + reg * np.log(a0)
                v0 = v0 + reg * np.log(b0)
                K0 = np.exp((np.add.outer(u0, v0) - costm) / reg)
                b0 = np.ones(b.shape[0])
        if get_scaling:
            return u0, v0, a0, b0, K0
        else:
            return u0, v0
    def update_uv(a, b, costm, reg, u, v):
        return diag_iter(costm, a, b, u, v, reg, n_iter=n_per_reg)
    u = np.zeros(a.shape[0])
    v = np.zeros(b.shape[0])
    for r in reg_list:
        # print('r=' + str(r))
        u, v = update_uv(a, b, costm, r, u, v)
    u, v, a0, b0, K = diag_iter(costm, a, b, u, v, reg, n_iter=n_iter, get_scaling=True)
    return np.diag(a0) @ K @ np.diag(b0)

# compute entropy regularized unbalanced optimal transport map
def ot_unbalanced(a, b, costm, reg, reg1, reg2, n_iter=1000):
    K = np.exp(-costm / reg)
    v = np.repeat(1, len(b))
    for i in range(n_iter):
        u = (a / (K @ v)) ** (reg1 / (reg + reg1))
        v = (b / (np.transpose(K) @ u)) ** (reg2 / (reg + reg2))
    tmap = np.diag(u) @ K @ np.diag(v)
    return tmap / np.sum(tmap)

# compute entropy regularized unbalanced optimal transport map with log stabilization
def ot_unbalanced_log_stabilized(a, b, costm, reg, reg1, reg2, reg_list, n_iter=1000, tau=50, n_per_reg=20):
    def prox_0(a, b, K, reg, reg_m):
        return a / (K @ b)
    def prox_pos(a, b, K, reg, reg_m):
        return (a / (K @ b)) ** (reg_m / (reg + reg_m))
    def diag_iter(costm, a, b, u, v, reg, reg1, reg2, n_iter, get_scaling=False):
        prox_a = prox_0 if reg1 == 0 else prox_pos
        prox_b = prox_0 if reg2 == 0 else prox_pos
        K0 = np.exp((np.add.outer(u, v) - costm) / reg)
        b0 = np.ones(b.shape[0])
        u0 = u
        v0 = v
        for i in range(n_iter):
            a0 = prox_a(a, b0, K0, reg, reg1)
            b0 = prox_b(b, a0, K0.T, reg, reg2)
            if np.max(np.abs(np.log(np.hstack((a0, b0))))) > tau:
                # print('detected')
                u0 = u0 + reg * np.log(a0)
                v0 = v0 + reg * np.log(b0)
                K0 = np.exp((np.add.outer(u0, v0) - costm) / reg)
                b0 = np.ones(b.shape[0])
        if get_scaling:
            return u0, v0, a0, b0, K0
        else:
            return u0, v0
    def update_uv(a, b, costm, reg, reg1, reg2, u, v):
        return diag_iter(costm, a, b, u, v, reg, reg1, reg2, n_iter=n_per_reg)
    u = np.zeros(a.shape[0])
    v = np.zeros(b.shape[0])
    for r in reg_list:
        # print('r=' + str(r))
        u, v = update_uv(a, b, costm, r, reg1, reg2, u, v)
    u, v, a0, b0, K = diag_iter(costm, a, b, u, v, reg, reg1, reg2, n_iter=n_iter, get_scaling=True)
    tmap = np.diag(a0) @ K @ np.diag(b0)
    return tmap / np.sum(tmap)

