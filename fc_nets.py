from unittest import TestResult
import numpy as np
import torch
import torch.nn.functional as F
import copy


class FCNet2Layers(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output=1, biases=[False, False]):
        super(FCNet2Layers, self).__init__()
        self.layer1 = torch.nn.Linear(n_feature, n_hidden, bias=biases[0])     
        self.layer2 = torch.nn.Linear(n_hidden, n_output, bias=biases[1])
    
    def init_gaussian(self, init_scales):
        self.layer1.weight.data = init_scales[0] * torch.randn_like(self.layer1.weight)
        self.layer2.weight.data = init_scales[1] * torch.randn_like(self.layer2.weight)

    def init_gaussian_clsf(self, init_scales):
        self.layer1.weight.data = init_scales[0] * torch.randn_like(self.layer1.weight)
        self.layer2.weight.data = ((torch.randn_like(self.layer2.weight) > 0).float() - 0.5) * 2 * (self.layer1.weight.data**2).sum(1)**0.5

    def init_blanc_et_al(self, init_scales):
        self.layer1.weight.data = 2.5 * (-1 + 2 * torch.round(torch.rand_like(self.layer1.weight))) * init_scales[0]
        self.layer1.bias.data = 2.5 * init_scales[0] * torch.randn_like(self.layer1.bias)
        self.layer2.weight.data = 4.0 * init_scales[1] * torch.randn_like(self.layer2.weight)

    def features(self, x, normalize=True, scaled=False):
        x = F.relu(self.layer1(x)) 
        if scaled:
            x = x * self.layer2.weight
        if normalize:
            x /= (x**2).sum(1, keepdim=True)**0.5
            x[torch.isnan(x)] = 0.0
        return x.data.numpy()

    def feature_sparsity(self, X, corr_threshold=0.99):
        phi = self.features(X)
        idx_keep = np.where((phi > 0.0).sum(0) > 0)[0]
        phi_filtered = phi[:, idx_keep]  # filter out zeros
        corr_matrix = np.corrcoef(phi_filtered.T) 
        corr_matrix -= np.eye(corr_matrix.shape[0])

        idx_to_delete, i, j = [], 0, 0
        while i != corr_matrix.shape[0]:
            # print(i, corr_matrix.shape, (np.abs(corr_matrix[i]) > corr_threshold).sum())
            if (np.abs(corr_matrix[i]) > corr_threshold).sum() > 0:
                corr_matrix = np.delete(corr_matrix, (i), axis=0)
                corr_matrix = np.delete(corr_matrix, (i), axis=1)
                # print('delete', j)
                idx_to_delete.append(j)
            else:
                i += 1
            j += 1
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        # print(idx_to_delete, idx_keep)
        idx_keep = np.delete(idx_keep, [idx_to_delete])
        sparsity = (phi[:, idx_keep] != 0).sum() / (phi.shape[0] * phi.shape[1])
        
        return sparsity

    def forward(self, x):
        z = F.relu(self.layer1(x)) 
        z = self.layer2(z)         
        return z


class FCNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, biases=True):
        super(FCNet, self).__init__()
        self.n_hidden = [n_feature] + n_hidden + [1]  # add the number of input and output units
        self.biases = biases
        self.layers = torch.nn.ModuleList()
        for i in range(len(self.n_hidden) - 1):
            self.layers.append(torch.nn.Linear(self.n_hidden[i], self.n_hidden[i+1], bias=self.biases))
    
    def init_gaussian(self, init_scales):
        for i in range(len(self.n_hidden) - 1):
            self.layers[i].weight.data = init_scales[i] * torch.randn_like(self.layers[i].weight)

    def forward(self, x):
        for i in range(len(self.n_hidden) - 2):
            x = F.relu(self.layers[i](x)) 
        x = self.layers[-1](x)
        return x

    def features(self, x, normalize=True, scaled=False, n_hidden_to_take=-1):
        for i in range(n_hidden_to_take if n_hidden_to_take > 0 else len(self.n_hidden) - 2):
            x = F.relu(self.layers[i](x)) 
        if scaled and n_hidden_to_take in [-1, len(self.n_hidden)]:
            x = x * self.layers[-1].weight
        if normalize:
            x /= (x**2).sum(1, keepdim=True)**0.5
            x[torch.isnan(x)] = 0.0
        return x.data.numpy()

    def feature_sparsity(self, X, n_hidden_to_take=-1, corr_threshold=0.99):
        phi = self.features(X, n_hidden_to_take=n_hidden_to_take)
        idx_keep = np.where((phi > 0.0).sum(0) > 0)[0]
        phi_filtered = phi[:, idx_keep]  # filter out zeros
        corr_matrix = np.corrcoef(phi_filtered.T) 
        corr_matrix -= np.eye(corr_matrix.shape[0])

        idx_to_delete, i, j = [], 0, 0
        while i != corr_matrix.shape[0]:
            # print(i, corr_matrix.shape, (np.abs(corr_matrix[i]) > corr_threshold).sum())
            if (np.abs(corr_matrix[i]) > corr_threshold).sum() > 0:
                corr_matrix = np.delete(corr_matrix, (i), axis=0)
                corr_matrix = np.delete(corr_matrix, (i), axis=1)
                # print('delete', j)
                idx_to_delete.append(j)
            else:
                i += 1
            j += 1
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        # print(idx_to_delete, idx_keep)
        idx_keep = np.delete(idx_keep, [idx_to_delete])
        sparsity = (phi[:, idx_keep] != 0).sum() / (phi.shape[0] * phi.shape[1])
        
        return sparsity

    def n_highly_corr(self, X, n_hidden_to_take=-1, corr_threshold=0.99):
        phi = self.features(X, n_hidden_to_take=n_hidden_to_take)
        idx_keep = np.where((phi > 0.0).sum(0) > 0)[0]
        phi_filtered = phi[:, idx_keep]  # filter out zeros
        corr_matrix = np.corrcoef(phi_filtered.T) 
        corr_matrix -= np.eye(corr_matrix.shape[0])

        idx_to_delete, i, j = [], 0, 0
        while i != corr_matrix.shape[0]:
            # print(i, corr_matrix.shape, (np.abs(corr_matrix[i]) > corr_threshold).sum())
            if (np.abs(corr_matrix[i]) > corr_threshold).sum() > 0:
                corr_matrix = np.delete(corr_matrix, (i), axis=0)
                corr_matrix = np.delete(corr_matrix, (i), axis=1)
                # print('delete', j)
                idx_to_delete.append(j)
            else:
                i += 1
            j += 1
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        # print(idx_to_delete, idx_keep)
        idx_keep = np.delete(idx_keep, [idx_to_delete])
        sparsity = (phi[:, idx_keep] != 0).sum() / (phi.shape[0] * phi.shape[1])
        
        return phi.shape[1] - len(idx_keep)


def moving_average(net, net_avg, weight_avg):
    for param, param_avg in zip(net.parameters(), net_avg.parameters()):
        param_avg.data = weight_avg*param_avg.data + (1-weight_avg)*param.data


def train_fc_net(X, y, X_test, y_test, gamma, batch_size, net, iters_loss, num_iter, thresholds=[-1], decays=[-1], iters_percentage_linear_warmup=0.0, gamma_warmup_factor_max=1.0, warmup_exponent=1.0, weight_avg=0.0, clsf=False, gauss_ln_scale=0.0):
    assert iters_percentage_linear_warmup <= decays[0], 'we should decay the step size only after warmup'
    train_losses, test_losses, nets_avg = [], [], []
    net, net_avg = copy.deepcopy(net), copy.deepcopy(net)

    loss_f = (lambda y_pred, y: torch.mean(torch.log(1 + torch.exp(-y_pred * y)))) if clsf else (lambda y_pred, y: torch.mean((y_pred - y)**2))
    # loss_f = lambda y_pred, y: torch.mean(torch.log(1 + torch.exp(-y_pred * y)))

    optimizer = torch.optim.SGD(net.parameters(), lr=gamma)  #, momentum=0.9)
    for i in range(num_iter):
        if i in iters_loss:
            train_losses += [loss_f(net_avg(X), y)]
            test_losses += [loss_f(net_avg(X_test), y_test)]
            nets_avg.append(copy.deepcopy(net_avg))
            if torch.isnan(train_losses[-1]):
                return train_losses, test_losses, nets_avg
        
        if i <= int(iters_percentage_linear_warmup * num_iter) and int(iters_percentage_linear_warmup * num_iter) > 0:
            optimizer.param_groups[0]['lr'] = gamma + (gamma_warmup_factor_max - 1) * gamma * (i / int(num_iter))**warmup_exponent  
            # optimizer.param_groups[0]['lr'] = gamma + gamma * gamma_warmup_factor_max * (i / int(iters_percentage_linear_warmup * num_iter))**warmup_exponent
        
        if i in thresholds:
            for threshold, decay in zip(thresholds, decays):
                if i == threshold:
                    optimizer.param_groups[0]['lr'] /= decay

        indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        batch_x, batch_y = X[indices], y[indices]
        if gauss_ln_scale > 0.0:  # label noise with schedule (note: supports only one threshold at the moment)
            batch_y += torch.randn_like(batch_y) * gauss_ln_scale / (decay if i > thresholds[0] else 1.0)
        loss = loss_f(net(batch_x), batch_y) 

        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()        

        moving_average(net, net_avg, weight_avg) 
    
    return train_losses, test_losses, nets_avg


def compute_grad_matrix(net, X):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
    grad_matrix_list = []
    for i in range(X.shape[0]):
        h = net(X[[i]])
        optimizer.zero_grad()   
        h.backward()       

        grad_total_list = []
        for param in net.parameters():
            grad_total_list.append(param.grad.flatten().data.numpy())
        grad_total = np.concatenate(grad_total_list)  
        grad_matrix_list.append(grad_total)

    grad_matrix = np.vstack(grad_matrix_list)
    return grad_matrix


def compute_grad_matrix_ranks(nets, X, l0_threshold_grad_matrix=0.0001):
    n_params = sum([np.prod(param.shape) for param in nets[-1].parameters()])
    X_eval = X[:n_params]
    grad_matrix_ranks = []
    for net in nets:
        svals = np.linalg.svd(compute_grad_matrix(net, X_eval))[1] 
        rank = (svals / svals[0] > l0_threshold_grad_matrix).sum()
        grad_matrix_ranks.append(rank)
    return grad_matrix_ranks

