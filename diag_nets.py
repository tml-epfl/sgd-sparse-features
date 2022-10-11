import numpy as np


def loss(X, y_hat, y):
    return np.linalg.norm(y_hat - y)**2 / (2*X.shape[0])



def valley_projection(X, y, gamma, u0, v0, num_iter):
    u, v = u0, v0
    for i in range(num_iter):
        y_hat = X @ (u * v)
        Xerr = X.T@(y_hat - y)
        grad_u, grad_v = (Xerr * v) / X.shape[0], (Xerr * u) / X.shape[0]
        
        u = u - gamma * grad_u
        v = v - gamma * grad_v

    return u, v


def GD(X, y, X_test, y_test, gamma, u0, v0, iters_loss, num_iter, thresholds=[-1], decays=[2], wd=0, balancedness=0, normalized_gd=False, weight_avg=0.0, return_all_params=False, valley_project=False):
    train_losses, test_losses = [], []
    us, vs = [], []
    u, v, u_avg, v_avg = u0, v0, u0, v0
    for i in range(num_iter):
        if i in iters_loss:
            if valley_project:
                u_avg, v_avg = valley_projection(X, y, 0.5, u, v, 2000)
            train_losses += [loss(X, X @ (u_avg * v_avg), y)]
            test_losses += [loss(X_test, X_test @ (u_avg * v_avg), y_test)]
            us, vs = us + [u_avg], vs + [v_avg]
        
        y_hat = X @ (u * v)
        Xerr = X.T@(y_hat - y)
        grad_u, grad_v = (Xerr * v) / X.shape[0], (Xerr * u) / X.shape[0]
        if normalized_gd and (np.linalg.norm(grad_u) > 0 and np.linalg.norm(grad_v) > 0):
            grad_u, grad_v = grad_u / np.linalg.norm(grad_u), grad_v / np.linalg.norm(grad_v)
        
        u = u - gamma * grad_u - wd*u - balancedness*u*(2*(np.abs(u)>np.abs(v))-1) 
        v = v - gamma * grad_v - wd*v - balancedness*v*(2*(np.abs(v)>np.abs(u))-1) 
        u_avg, v_avg = weight_avg*u_avg + (1-weight_avg)*u, weight_avg*v_avg + (1-weight_avg)*v

        if i in thresholds:
            for threshold, decay in zip(thresholds, decays):
                if i == threshold:
                    gamma = gamma / decay

    if return_all_params:
        return train_losses, test_losses, u, v, us, vs
    else:
        return train_losses, test_losses, u, v


def SGD(X, y, X_test, y_test, gamma, u0, v0, iters_loss, num_iter, thresholds=[-1], decays=[2], weight_avg=0.0, return_all_params=False, valley_project=False):
    train_losses, test_losses = [], []
    us, vs = [], []
    u, v, u_avg, v_avg = u0, v0, u0, v0
    for i in range(num_iter):
        if i in iters_loss:
            if valley_project:
                u_avg, v_avg = valley_projection(X, y, 0.5, u, v, 2000)
            train_losses += [loss(X, X @ (u_avg * v_avg), y)]
            test_losses += [loss(X_test, X_test @ (u_avg * v_avg), y_test)]
            us, vs = us + [u_avg], vs + [v_avg]

        i_t = np.random.randint(X.shape[0])
        error = X[i_t] @ (u * v) - y[i_t] 
        Xerr = error * X[i_t]
        grad_u, grad_v = Xerr * v,  Xerr * u
        
        u = u - gamma * grad_u   # gradient step
        v = v - gamma * grad_v   # gradient step
        u_avg, v_avg = weight_avg*u_avg + (1-weight_avg)*u, weight_avg*v_avg + (1-weight_avg)*v

        if i in thresholds:
            for threshold, decay in zip(thresholds, decays):
                if i == threshold:
                    gamma = gamma / decay
    
    if return_all_params:
        return train_losses, test_losses, u, v, us, vs
    else:
        return train_losses, test_losses, u, v


def n_SAM_GD(X, y, X_test, y_test, gamma, u0, v0, iters_loss, num_iter, rho):
    train_losses, test_losses = [], []
    u, v = u0, v0
    for i in range(num_iter):
        y_hat = X @ (u * v)
        Xerr = X.T @ (y_hat - y)
        u_sam, v_sam = u + rho * (Xerr * v) / X.shape[0], v + rho * (Xerr * u) / X.shape[0]
        
        Xerr_sam = X.T @ (X @ (u_sam * v_sam) - y)
        grad_u_sam, grad_v_sam = (Xerr_sam * v_sam) / X.shape[0], (Xerr_sam * u_sam) / X.shape[0]
        
        u = u - gamma * grad_u_sam   # gradient step
        v = v - gamma * grad_v_sam   # gradient step

        if i in iters_loss:
            train_losses += [loss(X, X @ (u * v), y)]
            test_losses += [loss(X_test, X_test @ (u * v), y_test)]

    return train_losses, test_losses, u, v


def one_SAM_GD(X, y, X_test, y_test, gamma, u0, v0, iters_loss, num_iter, rho, loss_derivative_only=False):
    train_losses, test_losses = [], []
    u, v = u0, v0
    for i in range(num_iter):
        y_hat = X @ (u * v)
        r = y_hat - y
        grad_u_sam, grad_v_sam = np.zeros_like(u), np.zeros_like(v)
        for k in range(X.shape[0]):
            u_sam_k = u + 2 * rho * r[k] * X[k] * v
            v_sam_k = v + 2 * rho * r[k] * X[k] * u
            if not loss_derivative_only:
                grad_u_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * v_sam_k * X[k] / X.shape[0]
                grad_v_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * u_sam_k * X[k] / X.shape[0]
            else:
                grad_u_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * v * X[k] / X.shape[0]
                grad_v_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * u * X[k] / X.shape[0]

        u = u - gamma * grad_u_sam   # gradient step
        v = v - gamma * grad_v_sam   # gradient step

        if i in iters_loss:
            train_losses += [loss(X, X @ (u * v), y)]
            test_losses += [loss(X_test, X_test @ (u * v), y_test)]

    return train_losses, test_losses, u, v


def dln_hessian(u, v, X, y, normalized=False):
    beta = u * v
    if normalized:
        u = np.abs(beta)**0.5 * np.sign(u)
        v = np.abs(beta)**0.5 * np.sign(v)
    # print(beta[:10], u[:10], v[:10])
    n, d = X.shape
    H = np.zeros([2*d, 2*d])
    H[:d, :d] = np.diag(v) @ (X.T@X) @ np.diag(v)
    H[d:, :d] = np.diag(v) @ (X.T@X) @ np.diag(u) + np.diag(X.T@(X@(u*v) - y))
    H[:d, d:] = np.diag(u) @ (X.T@X) @ np.diag(v) + np.diag(X.T@(X@(u*v) - y))
    H[d:, d:] = np.diag(u) @ (X.T@X) @ np.diag(u)
    return H / n


def dln_hessian_eigs(u, v, X, y, normalized=False):
    hess = dln_hessian(u, v, X, y, normalized)
    eigs, _ = np.linalg.eig(hess)
    return eigs


def dln_grad_loss(u, v, X, y):
    Xerr = X.T @ (X @ (u * v) - y)
    grad_u, grad_v = (Xerr * v) / X.shape[0], (Xerr * u) / X.shape[0]
    return np.concatenate([grad_u, grad_v])


def dln_avg_individual_grad_loss(u, v, X, y, residuals=True):
    n = X.shape[0]
    r = X @ (u * v) - y

    if not residuals:
        r = np.ones_like(r)

    sum = 0
    for i in range(n):
        sum += np.sum((r[i] * X[i] * v)**2) / n
        sum += np.sum((r[i] * X[i] * u)**2) / n
    
    return sum


def compute_grad_matrix_ranks(us, vs, X, l0_threshold_grad_matrix=0.0001):
    grad_matrix_ranks = []
    for u, v in zip(us, vs):
        grad_matrix = np.hstack([X * u, X * v])
        svals = np.linalg.svd(grad_matrix)[1] 
        rank = (svals / svals[0] > l0_threshold_grad_matrix).sum()
        grad_matrix_ranks.append(rank)
    return grad_matrix_ranks

