from logging import lastResort
import torch
import numpy as np
import math
from utils import clamp, get_random_delta, project_lp


def get_lr_schedule(lr_schedule_type, n_epochs, lr, warmup_factor=1.0, warmup_exp=1.0):
    if lr_schedule_type == 'cyclic':
        lr_schedule = lambda epoch: np.interp([epoch], [0, n_epochs * 2 // 5, n_epochs], [0, lr, 0])[0]
    elif lr_schedule_type in ['piecewise', 'piecewise_10_100']:
        def lr_schedule(t):
            """
            Following the original ResNet paper (+ warmup for resnet34).
            t is the fractional number of epochs that is passed which starts from 0.
            """
            # if 100 epochs in total, then warmup lasts for exactly 2 first epochs
            # if t / n_epochs < 0.02 and model in ['resnet34']:
            #     return lr_max / 10.
            if t / n_epochs < 0.5:
                return lr
            elif t / n_epochs < 0.75:
                return lr / 10.
            else:
                return lr / 100.
    elif lr_schedule_type in 'piecewise_01epochs':
        def lr_schedule(t):
            if warmup_exp > 1.0:
                if t / n_epochs < 0.1:
                    return warmup_exp**(t/n_epochs*100) * lr
                elif t / n_epochs < 0.9:
                    return warmup_exp**(0.1*100) * lr / 10.
                else:
                    return warmup_exp**(0.1*100) * lr / 100.
            elif warmup_exp < 1.0:
                if t / n_epochs < 0.1:
                    return (1 + (t/n_epochs*100)**warmup_exp) * lr
                elif t / n_epochs < 0.9:
                    return (1 + (0.1*100)**warmup_exp) * lr / 10.
                else:
                    return (1 + (0.1*100)**warmup_exp) * lr / 100.
            else:
                if t / n_epochs < 0.1:
                    return np.interp([t], [0, 0.5 * n_epochs], [lr, warmup_factor*lr])[0]  # note: we interpolate up to 0.5*t to be compatible with toy ReLU net experiments
                elif t / n_epochs < 0.9:
                    return 0.1 / 0.5 * warmup_factor*lr / 10.
                else:
                    return 0.1 / 0.5 * warmup_factor*lr / 100.
    elif lr_schedule_type in 'piecewise_03epochs':
        def lr_schedule(t):
            if warmup_exp > 1.0:
                if t / n_epochs < 0.3:
                    return warmup_exp**(t/n_epochs*100) * lr
                elif t / n_epochs < 0.9:
                    return warmup_exp**(0.3*100) * lr / 10.
                else:
                    return warmup_exp**(0.3*100) * lr / 100.
            elif warmup_exp < 1.0:
                if t / n_epochs < 0.3:
                    return (1 + (t/n_epochs*100)**warmup_exp) * lr
                elif t / n_epochs < 0.9:
                    return (1 + (0.3*100)**warmup_exp) * lr / 10.
                else:
                    return (1 + (0.3*100)**warmup_exp) * lr / 100.
            else:
                if t / n_epochs < 0.3:
                    return np.interp([t], [0, 0.5 * n_epochs], [lr, warmup_factor*lr])[0]  # note: we interpolate up to 0.5*t to be compatible with toy ReLU net experiments
                elif t / n_epochs < 0.9:
                    return 0.3 / 0.5 * warmup_factor*lr / 10.
                else:
                    return 0.3 / 0.5 * warmup_factor*lr / 100.
    elif lr_schedule_type in 'piecewise_05epochs':
        def lr_schedule(t):
            if warmup_exp > 1.0:
                if t / n_epochs < 0.5:
                    return warmup_exp**(t/n_epochs*100) * lr
                elif t / n_epochs < 0.9:
                    return warmup_exp**(0.5*100) * lr / 10.
                else:
                    return warmup_exp**(0.5*100) * lr / 100.
            elif warmup_exp < 1.0:
                if t / n_epochs < 0.5:
                    return (1 + (t/n_epochs*100)**warmup_exp) * lr
                elif t / n_epochs < 0.9:
                    return (1 + (0.5*100)**warmup_exp) * lr / 10.
                else:
                    return (1 + (0.5*100)**warmup_exp) * lr / 100.
            else:
                if t / n_epochs < 0.5:
                    return np.interp([t], [0, 0.5 * n_epochs], [lr, warmup_factor*lr])[0]  # note: we interpolate up to 0.5*t to be compatible with toy ReLU net experiments
                elif t / n_epochs < 0.9:
                    return 0.5 / 0.5 * warmup_factor*lr / 10.
                else:
                    return 0.5 / 0.5 * warmup_factor*lr / 100.
    elif lr_schedule_type == 'cosine':
        # cosine LR schedule without restarts like in the SAM paper
        # (as in the JAX implementation used in SAM https://flax.readthedocs.io/en/latest/_modules/flax/training/lr_schedule.html#create_cosine_learning_rate_schedule)
        return lambda epoch: lr * (0.5 + 0.5*math.cos(math.pi * epoch / n_epochs))
    elif lr_schedule_type == 'inverted_cosine':
        return lambda epoch: lr - lr * (0.5 + 0.5*math.cos(math.pi * epoch / n_epochs))
    elif lr_schedule_type == 'constant':
        return lambda epoch: lr
    else:
        raise ValueError('wrong lr_schedule_type')
    return lr_schedule


def get_delta_pgd(model, scaler, loss_f, X, y, eps, n_steps_pgd_curr, step_size_pgd_curr, args, use_pred_label=False):
    if args.attack_init == 'zero':
        delta = torch.zeros_like(X, requires_grad=True)
    elif args.attack_init == 'random':
        delta = get_random_delta(X.shape, eps, args.at_norm, requires_grad=True)
        if args.dataset != 'gaussians_binary' and args.model != 'linear':
            delta = clamp(X + delta.data, 0, 1) - X
    else:
        raise ValueError('wrong args.attack_init')

    if args.universal_at:  # note: it's not the same as just averaging deltas bc of the normalization / sign step
        delta = delta[0:1, :, :, :]

    for _ in range(n_steps_pgd_curr):
        with torch.cuda.amp.autocast(enabled=model.half_prec):
            logits = model(X + delta)
            y_adv = logits.max(1)[1].data if use_pred_label else y
            loss = loss_f(logits, y_adv)

        grad = torch.autograd.grad(scaler.scale(loss), delta)[0]
        grad = grad.detach() / scaler.get_scale()

        if args.at_norm == 'l2':
            grad_norms = (grad ** 2).sum([1, 2, 3], keepdim=True) ** 0.5
            grad_norms[grad_norms == 0] = np.inf  # to prevent division by zero
            delta_next = delta.data + step_size_pgd_curr * grad / grad_norms  # step of normalized gradient ascent
        elif args.at_norm == 'linf':
            delta_next = delta.data + step_size_pgd_curr * torch.sign(grad)
        else:
            raise ValueError('wrong args.at_norm')
        delta.data = project_lp(delta_next, args.at_norm, eps)
        if args.dataset != 'gaussians_binary' and args.model != 'linear':
            delta.data = clamp(X + delta.data, 0, 1) - X

    return delta.detach()


def warmup_schedule(n_epochs_warmup, iteration, batch_size, n_train_effective, binary=True):
    n_iters_max = n_epochs_warmup * n_train_effective // batch_size
    coeff = min(iteration / n_iters_max if n_iters_max != 0 else 1.0, 1.0)
    if binary:
        coeff = math.floor(coeff)
    return coeff


def change_bn_mode(model, bn_train):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if bn_train:
                module.train()
            else:
                module.eval()


def moving_average(net1, net2, alpha=0.999):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    with torch.no_grad():
        model.train()
        momenta = {}
        model.apply(reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for x, _, _, _, _ in loader:
            x = x.cuda(non_blocking=True)
            b = x.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(x)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.eval()


