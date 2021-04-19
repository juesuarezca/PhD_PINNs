import torch
from torch import Tensor as Tensor
from torch.nn import Module as Module
from torch.nn import MSELoss, L1Loss
from .LossTerm import LossTerm
import numpy as np


class PDELoss(LossTerm):
    def __init__(self, dataset, pde, quad_weights=[], norm='L2', weight=1.):
        """
        Constructor of the PDE Loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            pde (function): function that represents residual of the PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(PDELoss, self).__init__(dataset, norm, weight)
        self.dataset = dataset
        self.pde = pde
        self.quad_weights = quad_weights
        self.norm = norm
    def __call__(self, x: Tensor, model: Module, **kwargs):
        """
        Call function of the PDE loss. Calculates the norm of the PDE residual

        x: residual points
        model: model that predicts the solution of the PDE
        """
        x.requires_grad = True  # setting requires grad to true in order to calculate
        u = model.forward(x)
        pde_residual = self.pde(x, u, **kwargs)
        if self.norm == 'Mse':
            zeros = torch.zeros(pde_residual.shape, device=pde_residual.device)
            loss = torch.nn.MSELoss()(pde_residual, zeros)
        elif self.norm == 'Quad':
            quad_loss = (np.sum([torch.sum(torch.square(pde_residual[i])) * self.quad_weights[i] for i in
                                 range(len(pde_residual))]) ** (1 / 2))
            loss = quad_loss
        elif self.norm == 'Wass':
            prediction = u
            M = [[(i - j) ** 2 for i in range(len(prediction))] for j in range(len(prediction))]
            min_u = abs(min((prediction.detach().numpy()[:, 0]))) + 0.01
            C_x = np.sum(prediction.detach().numpy()[:, 0] + min_u)
            u_r = (prediction.detach().numpy()[:, 0] + min_u) / C_x
            min_gt = abs(min((gt_y[:, 0]))) + 0.01
            D_x = torch.sum(gt_y[:, 0] + min_gt)
            v_r = ((gt_y[:, 0] + min_gt) / D_x).detach().numpy()
            # quad_loss = (np.sum([torch.sum(torch.square(ini_residual[i] * self.quad_weights[i])) for i in
            #                     range(len(ini_residual))]) ** (1 / 2))
            loss = ot.sinkhorn2(np.array(u_r), np.array(v_r), np.array(M), 0.8)[0]
            # ot.sinkhorn_unbalanced2(u_r, v_r, np.array(M), 0.9, 0.9)[0]
            print('loss', loss)
        else:
            raise ValueError('Loss not defined')
        return loss*0