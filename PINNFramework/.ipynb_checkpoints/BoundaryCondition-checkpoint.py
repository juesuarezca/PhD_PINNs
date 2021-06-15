from .LossTerm import LossTerm
from torch.autograd import grad
from torch import ones
import torch
import numpy as np
import ot
from torch.autograd import Variable
import geomloss
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

    def __init__(self, func, dataset ,name, quad_weights=0, sob_weights = [],  norm = 'L2',weight=1., reg_param_w=0.2):
        super(DirichletBC, self).__init__(name, dataset, weight)
        self.func = func
        self.norm = norm
        self.quad_weights = quad_weights
        self.reg_par = reg_param_w
        self.sob_weights = sob_weights
    def __call__(self, x, model):
        if self.norm == 'Sobolev_1'or self.norm == 'Quad_new':
            bdy_set = x
            bdy_weights, Hkw_bdy= self.sob_weights
            bdy_weights = torch.Tensor(bdy_weights)
            lb_line = np.linspace(min(bdy_set),min(bdy_set),len(bdy_set))
            ub_line = np.linspace(max(bdy_set),max(bdy_set),len(bdy_set))
            zero_time = np.linspace(0,0,len(bdy_set))
            x_1= torch.Tensor(list(zip(bdy_set,lb_line,zero_time)))
            x_2 = torch.Tensor(list(zip(bdy_set,ub_line,zero_time)))
            x_3 = torch.Tensor(list(zip(lb_line,bdy_set,zero_time)))
            x_4 = torch.Tensor(list(zip(ub_line,bdy_set,zero_time)))
            f_1 = -model(x_1)[:,0]+self.func(x_1)
            f_2 = -model(x_2)[:,0]+self.func(x_2)
            f_3 = -model(x_3)[:,0]+self.func(x_3)
            f_4 = -model(x_4)[:,0]+self.func(x_4)
            if self.norm == 'Quad_new':
                L2_ip = (((f_1)**2).dot(bdy_weights)+
                         ((f_2)**2).dot(bdy_weights)+
                        ((f_3)**2).dot(bdy_weights)+
                        ((f_4)**2).dot(bdy_weights))
                loss = L2_ip
            else:
                L2_ip = (((f_1)**2).dot(bdy_weights)+
                         ((f_2)**2).dot(bdy_weights)+
                        ((f_3)**2).dot(bdy_weights)+
                        ((f_4)**2).dot(bdy_weights))
                cxc_1 = torch.outer(f_1,f_1)
                cxc_2 = torch.outer(f_2,f_2)
                cxc_3 = torch.outer(f_3,f_3)
                cxc_4 = torch.outer(f_4,f_4)
                #print(min(cxc_1.flatten()),min(cxc_2.flatten()),min(cxc_3.flatten()),min(cxc_4.flatten()))
                if len(Hkw_bdy)>0:
                    #print(np.sum([torch.sum(cxc_1*Hkw_bdy[i]) for i in range(len(Hkw_bdy))]),
                    #np.sum([torch.sum(cxc_2*Hkw_bdy[i]) for i in range(len(Hkw_bdy))]),
                    #np.sum([torch.sum(cxc_3*Hkw_bdy[i]) for i in range(len(Hkw_bdy))]),
                    #np.sum([torch.sum(cxc_4*Hkw_bdy[i]) for i in range(len(Hkw_bdy))]))
                    H_k = (np.sum([torch.sum(cxc_1*Hkw_bdy[i]) for i in range(len(Hkw_bdy))])+
                    np.sum([torch.sum(cxc_2*Hkw_bdy[i]) for i in range(len(Hkw_bdy))])+
                    np.sum([torch.sum(cxc_3*Hkw_bdy[i]) for i in range(len(Hkw_bdy))])+
                    np.sum([torch.sum(cxc_4*Hkw_bdy[i]) for i in range(len(Hkw_bdy))]))
                    loss = L2_ip+ H_k
                else:
                    loss = L2_ip
                print(loss)
        else:
            prediction = model(x)[:,0] # is equal to y
            ini_residual = (prediction - self.func(x))
            gt_y = self.func(x)
            if self.norm == 'Mse':
                zeros = torch.zeros(ini_residual.shape, device=ini_residual.device)
                loss = torch.mean(torch.square(ini_residual))#torch.nn.MSELoss()(ini_residual,zeros)
            elif self.norm== 'Quad':
                quad_loss =(np.sum([torch.square(ini_residual[i]) * self.quad_weights[i] for i in
                                     range(len(ini_residual))]))
                loss = quad_loss
            elif self.norm == 'Wass2':
                gt_y= self.func(x)
                M = [[(i-j)**2 for i in range(len(prediction))] for j in range(len(prediction))]
                min_u = abs(min((prediction.detach().numpy())))+0.01
                min_gt = abs(min((gt_y)))+0.01
                min_mu = min(min_u, min_gt)
                C_x = np.sum(prediction.detach().numpy()+min_mu)
                u_r = (prediction.detach().numpy()+ min_mu)/C_x
                D_x = torch.sum(gt_y + min_mu)
                v_r = ((gt_y+ min_mu)/D_x).detach().numpy()
                loss = ot.sinkhorn2(np.array(u_r), np.array(v_r), np.array(M),0.8)[0]
                #ot.sinkhorn_unbalanced2(u_r, v_r, np.array(M), 0.9, 0.9)[0]
                print('loss',loss)
            elif self.norm == 'Wass':
                M = torch.Tensor(
                    [[(x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2 for i in range(len(prediction))] for j in
                     range(len(prediction))])
                min_u = min(prediction)
                min_gt = min(gt_y)
                min_mu = abs(min(min_u,min_gt))+0.01
                C_x = torch.sum(prediction + min_mu)
                u_r = (prediction + min_mu) / C_x
                D_x = torch.sum(gt_y + min_mu)
                v_r = (gt_y + min_mu) / D_x
                def sinkhorn_normalized(x, y, epsilon, n, niter):

                    Wxy = sinkhorn_loss(x, y, epsilon, n, niter)
                    Wxx = sinkhorn_loss(x, x, epsilon, n, niter)
                    Wyy = sinkhorn_loss(y, y, epsilon, n, niter)
                    return 2 * Wxy - Wxx - Wyy

                def cost_matrix(x, y, p=2):
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

                loss = sinkhorn_loss(u_r, v_r, M, self.reg_par, 200)
                #u_rr = prediction
                #v_rr = gt_y
                #loss = geomloss.SamplesLoss().forward(torch.reshape(u_rr,(len(u_rr),1)),torch.reshape(v_rr,(len(v_rr),1)))
            else:
                raise ValueError('Loss not defined')
        return loss*0

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
