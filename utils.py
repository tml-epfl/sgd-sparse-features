import numpy as np
import torch
from fc_nets import FCNet2Layers, FCNet, compute_grad_matrix


def get_iters_eval(n_iter_power, x_log_scale, n_iters_first=101, n_iters_next=151):
    num_iter = int(10**n_iter_power) + 1

    iters_loss_first = np.array(range(100))
    if x_log_scale:
        iters_loss_next = np.unique(np.round(np.logspace(0, n_iter_power, n_iters_first)))
    else:
        iters_loss_next = np.unique(np.round(np.linspace(0, num_iter, n_iters_next)))[:-1]
    iters_loss = np.unique(np.concatenate((iters_loss_first, iters_loss_next)))
    
    return num_iter, iters_loss


def get_data_two_layer_relu_net(n, d, m_teacher, init_scales_teacher, seed):
    np.random.seed(seed + 1) 
    torch.manual_seed(seed + 1) 

    n_test = 1000
    H = np.eye(d)
    X = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n)).float()
    X = X / torch.sum(X**2, 1, keepdim=True)**0.5
    X_test = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n_test)).float()
    X_test = X_test / torch.sum(X_test**2, 1, keepdim=True)**0.5

    # generate ground truth labels
    with torch.no_grad():
        net_teacher = FCNet2Layers(n_feature=d, n_hidden=m_teacher)
        net_teacher.init_gaussian(init_scales_teacher)
        net_teacher.layer1.weight.data = net_teacher.layer1.weight.data / torch.sum((net_teacher.layer1.weight.data)**2, 1, keepdim=True)**0.5
        net_teacher.layer2.weight.data = torch.sign(net_teacher.layer2.weight.data)

        y, y_test = net_teacher(X), net_teacher(X_test)

        print('y', y[:20, 0])
    
    return X, y, X_test, y_test, net_teacher


def get_data_multi_layer_relu_net(n, d, m_teacher, init_scales_teacher, seed):
    np.random.seed(seed + 1) 
    torch.manual_seed(seed + 1) 

    n_test = 1000
    H = np.eye(d)
    X = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n)).float()
    X = X / torch.sum(X**2, 1, keepdim=True)**0.5
    X_test = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n_test)).float()
    X_test = X_test / torch.sum(X_test**2, 1, keepdim=True)**0.5

    # generate ground truth labels
    with torch.no_grad():
        net_teacher = FCNet(n_feature=d, n_hidden=m_teacher)
        net_teacher.init_gaussian(init_scales_teacher)
        y, y_test = net_teacher(X), net_teacher(X_test)
        print('y:', y[:, 0])
    
    return X, y, X_test, y_test, net_teacher


def effective_rank(v):
    v = v[v != 0]
    v /= v.sum()
    return -(v * np.log(v)).sum()


def rm_too_correlated(net, X, V, corr_threshold=0.99):
    V = V.T
    idx_keep = np.where((V > 0.0).sum(0) > 0)[0]
    V_filtered = V[:, idx_keep]  # filter out zeros
    corr_matrix = np.corrcoef(V_filtered.T) 
    corr_matrix -= np.eye(corr_matrix.shape[0])

    idx_to_delete, i, j = [], 0, 0
    while i != corr_matrix.shape[0]:
        if (np.abs(corr_matrix[i]) > corr_threshold).sum() > 0:
            corr_matrix = np.delete(corr_matrix, (i), axis=0)
            corr_matrix = np.delete(corr_matrix, (i), axis=1)
            # print('delete', j)
            idx_to_delete.append(j)
        else:
            i += 1
        j += 1
    assert corr_matrix.shape[0] == corr_matrix.shape[1]
    idx_keep = np.delete(idx_keep, [idx_to_delete])
    
    return V[:, idx_keep].T

def compute_grad_matrix_dim(net, X, corr_threshold=0.99):
    grad_matrix = compute_grad_matrix(net, X)
    grad_matrix_sq_norms = np.sum(grad_matrix**2, 0)
    m = 100
    v_j = []
    for j in range(m):
        v_j.append(grad_matrix_sq_norms[[j, m+j, 2*m+j]])  # matrix: w1, w2, w3, w4
    V = np.vstack(v_j)

    V_reduced = rm_too_correlated(net, X, V, corr_threshold=corr_threshold)
    grad_matrix_dim = V_reduced.shape[0]
    return grad_matrix_dim


def project(u, v, u0, v0, u1, v1, u2, v2):
    u, v = u - u0, v - v0
    alpha = (u @ u1 + v @ v1) / (np.sum(u1**2) + np.sum(v1**2))
    beta = (u @ u2 + v @ v2) / (np.sum(u2**2) + np.sum(v2**2))
    return alpha, beta

