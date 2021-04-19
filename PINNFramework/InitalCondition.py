from .LossTerm import LossTerm
from torch import Tensor
from torch.nn import Module
import torch
import numpy as np
import ot
import matplotlib.pyplot as plt
class InitialCondition(LossTerm):
    def __init__(self, dataset,quad_weights=[] ,norm='L2', weight=1.):
        """
        Constructor for the Intial condition

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(InitialCondition, self).__init__(dataset, norm, weight)
        self.quad_weights = quad_weights
        self.norm = norm
    def __call__(self, x: Tensor, model: Module, gt_y: Tensor):
        """
        This function returns the loss for the initial condition
        L_0 = norm(model(x), gt_y)

        Args:
        x (Tensor) : position of initial condition
        model (Module): model that represents the solution
        gt_y (Tensor): ground true values for the initial state
        """
        prediction = model(x)
        ini_residual = prediction-gt_y
        if self.norm == 'Mse':
            zeros = torch.zeros(ini_residual.shape, device=ini_residual.device)
            loss = torch.nn.MSELoss()(ini_residual,zeros)
        elif self.norm== 'Quad':
            quad_loss = (np.sum([torch.sum(torch.square(ini_residual[i])) * self.quad_weights[i] for i in
                                 range(len(ini_residual))]) ** (1 / 2))
            loss = quad_loss
        elif self.norm == 'Wass':
            M = [[(i-j)**2 for i in range(len(prediction))] for j in range(len(prediction))]
            min_u = abs(min((prediction.detach().numpy()[:,0])))+0.01
            C_x = np.sum(prediction.detach().numpy()[:,0]+min_u)
            u_r = (prediction.detach().numpy()[:,0]+ min_u)/C_x
            min_gt = abs(min((gt_y[:,0])))+0.01
            D_x = torch.sum(gt_y[:,0] + min_gt)
            v_r = ((gt_y[:,0]+ min_gt)/D_x).detach().numpy()
            #quad_loss = (np.sum([torch.sum(torch.square(ini_residual[i] * self.quad_weights[i])) for i in
            #                     range(len(ini_residual))]) ** (1 / 2))
            loss = ot.sinkhorn2(np.array(u_r), np.array(v_r), np.array(M),0.8)[0]
            #ot.sinkhorn_unbalanced2(u_r, v_r, np.array(M), 0.9, 0.9)[0]
            print('loss',loss)
        else:
            raise ValueError('Loss not defined')

        return loss*self.weight