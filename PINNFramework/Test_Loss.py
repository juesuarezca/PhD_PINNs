from .LossTerm import LossTerm
from torch import Tensor
from torch.nn import Module
import torch
import numpy as np
import ot
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import mygrad as mg
class My_Loss(LossTerm):
    def __init__(self, dataset,quad_weights=[] ,norm='Wass', weight=1.):
        super(My_Loss, self).__init__(dataset, norm, weight)
        self.quad_weights = quad_weights
        self.norm = 'Wass'
    def forward(self, x, model, gt_y):
        """
                This function returns the loss for the initial condition
                L_0 = norm(model(x), gt_y)

                Args:
                x (Tensor) : position of initial condition
                model (Module): model that represents the solution
                gt_y (Tensor): ground true values for the initial state
                """
        prediction = model(x)
        ini_residual = prediction - gt_y
        if self.norm == 'Mse':
            zeros = torch.zeros(ini_residual.shape, device=ini_residual.device)
            loss = torch.nn.MSELoss()(ini_residual, zeros)
        elif self.norm == 'Quad':
            quad_loss = (np.sum([torch.sum(torch.square(ini_residual[i])) * self.quad_weights[i] for i in
                                 range(len(ini_residual))]) ** (1 / 2))
            loss = quad_loss
            print(loss)
        elif self.norm == 'Wass':
            M = [[(i - j) ** 2 for i in range(len(prediction))] for j in range(len(prediction))]
            min_u = np.abs(min((mg.Tensor(prediction.detach().numpy()[:, 0])))) + 0.01
            C_x = np.max(mg.Tensor(prediction.detach().numpy()[:, 0]) + min_u)
            u_r = (mg.Tensor(prediction.detach().numpy()[:, 0] )+ min_u) / C_x
            min_gt = np.abs(min((gt_y[:, 0]))) + 0.01
            D_x = torch.sum(gt_y[:, 0] + min_gt)
            v_r = ((gt_y[:, 0] + min_gt) / D_x).detach().numpy()
            # quad_loss = (np.sum([torch.sum(torch.square(ini_residual[i] * self.quad_weights[i])) for i in
            #                     range(len(ini_residual))]) ** (1 / 2))
            loss = ot.sinkhorn2(np.array(u_r), np.array(v_r), np.array(M), 0.8)[0]
            # ot.sinkhorn_unbalanced2(u_r, v_r, np.array(M), 0.9, 0.9)[0]
            print('loss', loss)
        elif self.norm == 's_e':
            loss = np.sum(np.square(ini_residual))
            return loss
        else:
            raise ValueError('Loss not defined')
        return loss * self.weight

class My_Loss2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, nu, dist, lam=1e-3, N=100):
        assert mu.dim() == 2 and nu.dim() == 2 and dist.dim() == 2
        bs = mu.size(0)
        d1, d2 = dist.size()
        assert nu.size(0) == bs and mu.size(1) == d1 and nu.size(1) == d2
        log_mu = mu.log()
        log_nu = nu.log()
        log_u = torch.full_like(mu, -math.log(d1))
        log_v = torch.full_like(nu, -math.log(d2))
        for i in range(N):
            log_v = sinkstep(dist, log_nu, log_u, lam)
            log_u = sinkstep(dist.t(), log_mu, log_v, lam)

        # this is slight abuse of the function. it computes (diag(exp(log_u))*Mt*exp(-Mt/lam)*diag(exp(log_v))).sum()
        # in an efficient (i.e. no bxnxm tensors) way in log space
        distances = (-sinkstep(-dist.log()+dist/lam, -log_v, log_u, 1.0)).logsumexp(1).exp()
        ctx.log_v = log_v
        ctx.log_u = log_u
        ctx.dist = dist
        ctx.lam = lam
        return distances

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out[:, None] * ctx.log_u * ctx.lam, grad_out[:, None] * ctx.log_v * ctx.lam, None, None, None