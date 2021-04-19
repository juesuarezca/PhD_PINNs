from .LossTerm import LossTerm
from torch.autograd import grad
from torch import ones
import torch
import numpy as np
import ot

class BoundaryCondition(LossTerm):
    def __init__(self, name, dataset, weight=1.):
        self.name = name
        super(BoundaryCondition, self).__init__(dataset, weight)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("The call function of the Boundary Condition has to be implemented")


class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary conditions: y(x) = func(x).
    """

    def __init__(self, func, dataset ,name, quad_weights=0,  norm = 'L2',weight=1.):
        super(DirichletBC, self).__init__(name, dataset, weight)
        self.func = func
        self.norm = norm
        self.quad_weights = quad_weights
    def __call__(self, x, model):
        prediction = model(x)  # is equal to y
        ini_residual = (prediction - self.func(x))
        if self.norm == 'Mse':
            zeros = torch.zeros(ini_residual.shape, device=ini_residual.device)
            loss = torch.nn.MSELoss()(ini_residual,zeros)
        elif self.norm== 'Quad':
            quad_loss = (np.sum([torch.sum(torch.square(ini_residual[i])) * self.quad_weights[i] for i in
                                 range(len(ini_residual))]) ** (1 / 2))
            loss = quad_loss
        elif self.norm == 'Wass':
            gt_y= self.func(x)
            M = [[(i-j)**2 for i in range(len(prediction))] for j in range(len(prediction))]
            min_u = abs(min((prediction.detach().numpy()[:,0])))+0.01
            C_x = np.sum(prediction.detach().numpy()[:,0]+min_u)
            u_r = (prediction.detach().numpy()[:,0]+ min_u)/C_x
            min_gt = abs(min((gt_y[:,0])))+0.01
            D_x = torch.sum(gt_y[:,0] + min_gt)
            v_r = ((gt_y[:,0]+ min_gt)/D_x).detach().numpy()
            loss = ot.sinkhorn2(np.array(u_r), np.array(v_r), np.array(M),0.8)[0]
            #ot.sinkhorn_unbalanced2(u_r, v_r, np.array(M), 0.9, 0.9)[0]
            print('loss',loss)
        else:
            raise ValueError('Loss not defined')

        #loss = self.weight * self.norm(prediction, self.func(x))*0 #Just for testing the initial condition Loss
        return self.weight*loss*0

class NeumannBC(BoundaryCondition):
    """
    Neumann boundary conditions: dy/dn(x) = func(x).
    """

    def __init__(self, func, dataset, input_dimension, output_dimension, name, norm='L2',weight=1.):
        super(NeumannBC, self).__init__(name, dataset, norm, weight)
        self.func = func
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def __call__(self, x, model):
        grads = ones(x.shape, device=model.device)
        y = model(x)[:, self.output_dimension]
        grad_y = grad(y, x, create_graph=True, grad_outputs=grads)[0]
        y_dn = grad_y[:, self.input_dimension]
        return self.weight * self.norm(y_dn, self.func(x))


class RobinBC(BoundaryCondition):
    """
    Robin boundary conditions: dy/dn(x) = func(x, y).
    """

    def __init__(self, func, dataset, input_dimension, output_dimension, name, norm='L2', weight=1.):
        super(RobinBC, self).__init__(name, dataset, norm, weight)
        self.func = func
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def __call__(self, x, y, model):
        y = model(x)[:, self.output_dimension]
        grads = ones(y.shape, device=y.device)
        grad_y = grad(y, x, create_graph=True, grad_outputs=grads)[0]
        y_dn = grad_y[:, self.input_dimension]
        return self.weight * self.norm(y_dn, self.func(x, y))


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary condition
    """

    def __init__(self, dataset, output_dimension, name, degree=None, input_dimension=None,  norm='L2', weight=1.):
        super(PeriodicBC, self).__init__(name, dataset, norm, weight)
        if degree is not None and input_dimension is None:
            raise ValueError("If the degree of the boundary condition is defined the input dimension for the "
                             "derivative has to be defined too ")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.degree = degree

    def __call__(self, x_lb, x_ub, model):
        x_lb.requires_grad = True
        x_ub.requires_grad = True
        y_lb = model(x_lb)[:, self.output_dimension]
        y_ub = model(x_ub)[:, self.output_dimension]
        grads = ones(y_lb.shape, device=y_ub.device)
        if self.degree is None:
            return self.weight * self.norm(y_lb, y_ub)
        elif self.degree == 1:
            y_lb_grad = grad(y_lb, x_lb, create_graph=True, grad_outputs=grads)[0]
            y_ub_grad = grad(y_ub, x_ub, create_graph=True, grad_outputs=grads)[0]
            y_lb_dn = y_lb_grad[:, self.input_dimension]
            y_ub_dn = y_ub_grad[:, self.input_dimension]
            return self.weight * self.norm(y_lb_dn, y_ub_dn)

        else:
            raise NotImplementedError("Periodic Boundary Condition for a higher degree than one is not supported")
