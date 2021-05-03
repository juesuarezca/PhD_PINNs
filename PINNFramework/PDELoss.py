import torch
from torch import Tensor as Tensor
from torch.nn import Module as Module
from torch.nn import MSELoss, L1Loss
from .LossTerm import LossTerm
import numpy as np
import ot
from torch.autograd import Variable
import geomloss
class PDELoss(LossTerm):
    def __init__(self, dataset, pde, func_left, func_right,quad_weights=[], norm='L2', weight=1.,
                reg_param_w=0.2):
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
        self.func_left = func_left
        self.func_right = func_right
        self.reg_par = reg_param_w
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
            quad_loss = (np.sum([torch.square(pde_residual[i]) * self.quad_weights[i] for i in
                                 range(len(pde_residual))]) ** (1 / 2))
            loss = quad_loss
        elif self.norm == 'Wass':
            mu = self.func_left(x,u,**kwargs)[:,0]
            nu = self.func_right(x,u,**kwargs)
            prediction = mu
            gt_y = nu
            min_u = torch.abs(min((prediction))) + 0.01
            C_x = torch.sum(prediction + min_u)
            u_r = (prediction + min_u) / C_x
            min_gt = torch.abs(min((gt_y))) + 0.01
            D_x = torch.sum(gt_y + min_gt)
            v_r = (gt_y + min_gt) / D_x
            M = torch.Tensor(
                [[(x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2 for i in         range(len(prediction))] for j in
                 range(len(prediction))])
            def sinkhorn_normalized(x, y, epsilon, n, niter):

                Wxy = sinkhorn_loss(x, y, epsilon, n, niter)
                Wxx = sinkhorn_loss(x, x, epsilon, n, niter)
                Wyy = sinkhorn_loss(y, y, epsilon, n, niter)
                return 2 * Wxy - Wxx - Wyy

            def cost_matrix(x, y, p=2):
                print(x.shape)
                "Returns the matrix of $|x_i-y_j|^p$."
                x_col = x.unsqueeze(1)
                y_lin = y.unsqueeze(0)
                c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
                return c

            def sinkhorn_loss(mu, nu, M, epsilon, niter):
                """
                Given two emprical measures with n points each with locations x and y
                outputs an approximation of the OT cost with regularization parameter epsilon
                niter is the max. number of steps in sinkhorn loop
                """
                # The Sinkhorn algorithm takes as input three variables :
                C = Variable(M)  # Wasserstein cost function
                rho = 1  # (.5) **2          # unbalanced transport
                tau = -.8  # nesterov-like acceleration
                lam = rho / (rho + epsilon)  # Update exponent
                thresh = 10 ** (-1)  # stopping criterion

                # Elementary operations .....................................................................
                def ave(u, u1):
                    "Barycenter subroutine, used by kinetic acceleration through extrapolation."
                    return tau * u + (1 - tau) * u1

                def M(u, v):
                    "Modified cost for logarithmic updates"
                    "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
                    return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

                def lse(A):
                    "log-sum-exp"
                    return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

                # Actual Sinkhorn loop ......................................................................
                u, v, err = 0. * mu, 0. * nu, 0.
                actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

                for i in range(niter):
                    u1 = u  # useful to check the update
                    u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
                    v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
                    # accelerated unbalanced iterations
                    # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
                    # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
                    err = (u - u1).abs().sum()

                    actual_nits += 1
                    if (err < thresh).data.numpy():
                        break
                U, V = u, v
                pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
                cost = torch.sum(pi * C)  # Sinkhorn cost

                return cost
            #u_rr = prediction
            #v_rr = gt_y
            #loss = geomloss.SamplesLoss().forward(torch.reshape(u_rr,(len(u_rr),1)),torch.reshape(v_rr,(len(v_rr),1)))
            loss = sinkhorn_loss(u_r, v_r, M, self.reg_par, 200)
        else:
            raise ValueError('Loss not defined')
        return loss
