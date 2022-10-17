from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F


def compute_loss(batches, model, cuda=True, noisy_examples='default', loss_f=F.cross_entropy, n_batches=-1):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    for i, (X, X_augm2, y, _, ln) in enumerate(batches):
        if n_batches != -1 and i > n_batches:  # limit to only n_batches
            break
        if cuda:
            X, y = X.cuda(), y.cuda()

        if noisy_examples == 'none':
            X, y = X[~ln], y[~ln]
        elif noisy_examples == 'all':
            X, y = X[ln], y[ln]
        else:
            assert noisy_examples == 'default'

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=model.half_prec):
            output = model(X)
            loss = loss_f(output, y)

        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex

    return 1 - robust_acc, avg_loss


def compute_feature_sparsity(batches, model, return_block, corr_threshold=0.95, n_batches=-1, n_relu_max=1000):
    with torch.no_grad():
        features_list = []
        for i, (X, X_augm2, y, _, ln) in enumerate(batches):
            if n_batches != -1 and i > n_batches:
                break
            X, y = X.cuda(), y.cuda()
            features = model(X, return_features=True, return_block=return_block).cpu().numpy()
            features_list.append(features)

        phi = np.vstack(features_list)
        phi = phi.reshape(phi.shape[0], np.prod(phi.shape[1:]))

        sparsity = (phi > 0).sum() / (phi.shape[0] * phi.shape[1])
        
        if phi.shape[1] > n_relu_max:  # if there are too many neurons, we speed it up by random subsampling
            random_idx = np.random.choice(phi.shape[1], n_relu_max, replace=False)
            phi = phi[:, random_idx]

        idx_keep = np.where((phi > 0.0).sum(0) > 0)[0]
        phi_filtered = phi[:, idx_keep]  # filter out always-zeros
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
        sparsity_rmdup = (phi[:, idx_keep] > 0).sum() / (phi.shape[0] * phi.shape[1])
        
        n_highly_corr = phi.shape[1] - len(idx_keep)
        return sparsity, sparsity_rmdup, n_highly_corr
