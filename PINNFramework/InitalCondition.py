from .LossTerm import LossTerm
from torch import Tensor
from torch.nn import Module
import torch
import numpy as np
import ot
import matplotlib.pyplot as plt
from torch.autograd import Variable
print('modified')
class InitialCondition(LossTerm):
    def __init__(self, dataset,quad_weights=[] ,norm='Mse', weight=1.):
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
        elif self.norm == 'Wass2':
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
        elif self.norm == 'Wass':
            M = torch.Tensor([[(x[i,0] - x[j,0]) ** 2+(x[i,1] - x[j,1]) ** 2 for i in range(len(prediction))] for j in range(len(prediction))])
            min_u = torch.abs(min((prediction[:, 0]))) + 0.01
            C_x = torch.sum(prediction[:, 0] + min_u)
            u_r = (prediction[:, 0] + min_u) / C_x
            min_gt = torch.abs(min((gt_y[:, 0]))) + 0.01
            D_x = torch.sum(gt_y[:, 0] + min_gt)
            v_r = (gt_y[:, 0] + min_gt) / D_x

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
                # both marginals are fixed with equal weights
                # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
                # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
                #mu = Variable(output, requires_grad = False)
                #nu = Variable(target, requires_grad=False)
                #mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
                #nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)

                # Parameters of the Sinkhorn algorithm.
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

            loss = sinkhorn_loss(u_r, v_r, M, 0.05, 200)
        else:
            raise ValueError('Loss not defined')

        return loss*self.weight
