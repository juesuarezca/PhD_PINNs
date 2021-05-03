import sys
import numpy as np
import scipy
import scipy.io
from pyDOE import lhs
from torch import Tensor, ones, stack, load
from torch.autograd import grad
from torch.utils.data import Dataset
#import mintegpy as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py as h5
import os
import torch
import PINNFramework as pf
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D 
import sys
#Read the input variables
e_l, DEG, n_epoch = [int(sys.argv[1]), int(sys.argv[2]),  int(sys.argv[3])]
#Fixed Data
DIM = 2
LP=2
POINTKIND = 'gauss_leg'#'leja'
USEDATA = False
#Doamin Bounds
lb = np.array([-1.0, -1.0, 0.0])
ub = np.array([1.0, 1.0, 0])
#Create 2-D Dataset from the analytical solution
def Herm_pol(n):
    p =  sp.Symbol('p')
    Hn = sp.lambdify(p,sp.hermite(n, p))
    return Hn

def eigenvalue (ev):
    a =  sp.Symbol('a')
    b =  sp.Symbol('b')
    c =  sp.Symbol('c')
    Hna = sp.hermite(ev,a)
    Hnb = sp.hermite(ev,b)
    Psi = (1/((2**e_l*sp.factorial(ev))**(1/2))*(np.pi**(-1/4))*
     sp.exp(-(a**2+b**2)/2)*Hna*Hnb)
    return int(sp.simplify((-1/2*(sp.diff(Psi,a,a)+sp.diff(Psi,b,b))+
                            1/2*(a**2+b**2)*Psi)/Psi))
lam = eigenvalue(e_l)
def Psi (x,y,t,f):
    x = torch.Tensor(x)
    y= torch.Tensor(y)
    Hn= Herm_pol(e_l)
    psi_t = torch.exp(torch.complex(torch.Tensor([0]),torch.Tensor([0])))
    return 1/((2**e_l*scipy.math.factorial(e_l))**(1/2))*(np.pi**(-1/4))*torch.exp(-(x**2+y**2)/2)*Hn(x)*Hn(y)
def set_gen_1d(POLYDEG,n_bdy):
    unscaled_pts = np.polynomial.legendre.leggauss(POLYDEG)
    scaled_pts = []
    weights = []
    for i in range(3):
        if (n_bdy[1][i]-n_bdy[0][i])/2 == 0:
            scaled_pts.append([n_bdy[1][i]])
            weights.append([1])
        else:
            m = (n_bdy[1][i]-n_bdy[0][i])/2
            b = (n_bdy[0][i]+n_bdy[1][i])/2
            scaled_pts.append(unscaled_pts[0]*m+b)
            weights.append(unscaled_pts[1]*m)
    Grid = np.array([[[[scaled_pts[0][i],scaled_pts[1][j],scaled_pts[2][k]]
                   for i in range(len(scaled_pts[0]))]for j in range(len(scaled_pts[1]))]
                 for k in range(len(scaled_pts[2]))]).reshape(len(scaled_pts[0])
                                                              *len(scaled_pts[1])*
                                                              len(scaled_pts[2]),3)
    Weights = np.array([[[weights[0][i]*weights[1][j]*weights[2][k]
                   for i in range(len(weights[0]))]for j in range(len(weights[1]))]
                 for k in range(len(weights[2]))]).reshape(len(weights[0])
                                                              *len(weights[1])*
                                                              len(weights[2]))
    return Grid, Weights


def MSE_set(n_bdy,  nb, x_dim=1, y_dim=1, nsteps=1, dt=0.1):
    x_0 = np.linspace(n_bdy[0][0], n_bdy[1][0], x_dim)
    y_0 = np.linspace(n_bdy[0][1], n_bdy[1][1], y_dim)
    X, Y = np.meshgrid(x_0, y_0)
    X_0 = X.reshape(-1)
    Y_0 = Y.reshape(-1)
    idx_x = np.random.choice(len(X_0), nb, replace=False)
    x = np.array([X_0[idx_x]]).T
    y = np.array([Y_0[idx_x]]).T
    T = np.zeros(x.shape)
    return np.array(np.concatenate([x,y,T], axis=1))


class BoundaryConditionDataset(Dataset):

    def __init__(self, boundary_set) -> object:
        """_bdy, x_dim, y_dim, nsteps, d
        Constructor of the Boundary condition dataset, with x_bdy an array with the
        lower and uper bound in the x direction and respectively y_bdy. Only for square domain.
        """
        self.Bdy_training = boundary_set
    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        return Tensor(self.Bdy_training).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1
class InitialConditionDataset(Dataset):

    def __init__(self,initial_set, norm = 'L2', n_bdy=[], x_dim=1, y_dim=1, nx =1 ):
        """
        Constructor of the boundary condition dataset
        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        if norm =='Mse' or norm== 'Wass':
            x_0 = np.linspace(n_bdy[0][0], n_bdy[1][0], x_dim)
            y_0 = np.linspace(n_bdy[0][1], n_bdy[1][1], y_dim)
            X, Y = np.meshgrid(x_0, y_0)
            X_0 = X.reshape(-1)
            Y_0 = Y.reshape(-1)
            idx_x = np.random.choice(len(X_0), nx, replace=False)
            self.x = np.array([X_0[idx_x]]).T
            self.y = np.array([Y_0[idx_x]]).T
            sol = Psi(self.x, self.y, 0, 1)  # Psi(x,y,t=0,f)
            self.u = sol
            self.t = np.array([np.zeros(len(self.x))]).T
        elif norm == 'Quad':
            X = np.array(initial_set[0])
            Y = np.array(initial_set[1])
            #X, Y = np.meshgrid(x_0, y_0)
            self.x = np.array([X.reshape(-1)]).T
            self.y = np.array([Y.reshape(-1)]).T
            sol = Psi(self.x,self.y,0,1)
            self.u = sol
            self.t = np.array([np.zeros(len(self.x))]).T
        else:
            raise ValueError('Norm not defined')
    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self,idx):
        x = np.concatenate([self.x,self.y,self.t],axis=1)
        y = self.u
        return Tensor(x).float(), Tensor(y).float()

class PDEDataset(Dataset):
    def __init__(self, residual_set):
        self.xf = np.array(residual_set).T
    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        return Tensor(self.xf).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1
if __name__ == "__main__":
    # Domain bounds
    #Create Datasets for the different Losses
    def schroedinger1d(x, u):
        omega = 1
        pred = u
        u = pred[:, 0]
        #v = pred[:, 1]
        grads = ones(u.shape, device=pred.device)  # move to the same device as prediction
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
        #grad_v = grad(v, x, create_graph=True, grad_outputs=grads)[0]

        # calculate first order derivatives
        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        u_t = grad_u[:, 2]
        # v_x = grad_v[:, 0]
        #v_y = grad_v[:, 1]
        #v_t = grad_v[:, 2]
        # calculate second order derivatives
        grad_u_x = grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        #grad_v_x = grad(v_x, x, create_graph=True, grad_outputs=grads)[0]
        u_xx = grad_u_x[:, 0]
        u_yy = grad_u_x[:, 1]
        #v_xx = grad_v_x[:, 0]
        #v_yy = grad_v_x[:, 1]
        f_r = -1/2*(u_xx - u_yy) + 1/2*(x[:, 0] ** 2 + x[:, 1] ** 2)*u - lam*u
        #print(torch.mean((-1/2*(u_xx - u_yy) + 1/2*(x[:, 0] ** 2 + x[:, 1] ** 2)*u)/u),torch.std((-1/2*(u_xx - u_yy) + 1/2*(x[:, 0] ** 2 + x[:, 1] ** 2)*u)/u))
        #f_u = -1 * u_t - 0.5 * v_xx - 0.5 * v_yy + omega * 0.5 * (x[:, 0] ** 2) * v + omega * 0.5 * (x[:, 1] ** 2) * v
        # fv is the imaginary part of the schrodinger equation
        #f_v = -1 * v_t + 0.5 * u_xx + 0.5 * u_yy - omega * 0.5 * (x[:, 0] ** 2) * u - omega * 0.5 * (x[:, 1] ** 2) * u
        return f_r#stack([f_u, f_v], 1)  # concatenate real part and imaginary part
    def res_left(x, u):
        omega = 1
        pred = u
        u = pred#pred[:, 0]
        f_left =  lam*u
        return f_left
    def res_right(x, u):
        omega = 1
        pred = u
        u = pred[:, 0]
        #v = pred[:, 1]
        grads = ones(u.shape, device=pred.device)  # move to the same device as prediction
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
        #grad_v = grad(v, x, create_graph=True, grad_outputs=grads)[0]

        # calculate first order derivatives
        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        u_t = grad_u[:, 2]
        #v_x = grad_v[:, 0]
        #v_y = grad_v[:, 1]
        #v_t = grad_v[:, 2]
        # calculate second order derivatives
        grad_u_x = grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        #grad_v_x = grad(v_x, x, create_graph=True, grad_outputs=grads)[0]
        u_xx = grad_u_x[:, 0]
        u_yy = grad_u_x[:, 1]
        #v_xx = grad_v_x[:, 0]
        #v_yy = grad_v_x[:, 1]
        f_right = -1/2*(u_xx - u_yy) + 1/2*(x[:, 0] ** 2 + x[:, 1] ** 2)*u       
        return f_right
    def hom_dir(x):
        P = Psi(x[:,0], x[:,1], x[:,2], 1)
        #P_r = P.real
        #P_im = P.imag
        return P# torch.stack((P_r,P_im),1)
    def Dataset_loss (Norm, bounds , n_points, deg = 1):
        [lb, ub] = bounds
        residual_bdy = [[lb[0], lb[1], lb[2]], [ub[0], ub[1], ub[2]]]
        initial_bdy = [[lb[0], lb[1], lb[2]], [ub[0], ub[1], lb[2]]]
        boundary_bdy = [[[lb[0], lb[1], lb[2]], [lb[0], ub[1], ub[2]]], [[ub[0], lb[1], lb[2]], [ub[0], ub[1], ub[2]]],
                        [[lb[0], lb[1], lb[2]], [ub[0], lb[1], ub[2]]],
                        [[lb[0], ub[1], lb[2]], [ub[0], ub[1], ub[2]]]]
        if Norm == 'Mse' or Norm == 'Wass':
            boundary_set = np.concatenate([MSE_set(boundary_bdy[i], nb= int(n_points/4),
                                                       x_dim=1000, y_dim=1000, nsteps=1, dt=0.1) for i in range(4)], axis=0)
            residual_set = MSE_set(residual_bdy, nb= n_points, x_dim=1000, y_dim=1000, nsteps=1, dt=0.1).T
            initial_set = []
            Datasets = [[boundary_set, residual_set, initial_set],[[0],[0],[0]]]
        elif Norm == 'Quad':
            I_C_1d = set_gen_1d(deg, initial_bdy)
            initial_set = I_C_1d[0].T
            initial_weights = I_C_1d[1]
            boundary_set = np.concatenate([set_gen_1d(deg, boundary_bdy[i])[0] for i in range(4)], axis=0)
            boundary_weights = np.concatenate([set_gen_1d(deg, boundary_bdy[i])[1] for i in range(4)], axis=0)
            RS_1d = set_gen_1d(deg, residual_bdy)
            residual_set = RS_1d[0].T
            residual_weights = RS_1d[1]
            Datasets = [[boundary_set, residual_set, initial_set],[boundary_weights, residual_weights, initial_weights]]
        else:
            raise(ValueError('Loss not defined'))
        
        ## Crete Loss functions
        #Boundary term
        bc_dataset = BoundaryConditionDataset(Datasets[0][0])
        dirichlet_bc = pf.DirichletBC(func=hom_dir, dataset=bc_dataset,
                                      quad_weights=Datasets[1][0]  , name='Dirichlet BC', 
                                      norm=Norm,reg_param_w = 0.2)
        # Residual Terms
        pde_dataset = PDEDataset(Datasets[0][1])
        pde_loss = pf.PDELoss(pde_dataset, schroedinger1d,
                              func_left = res_left, func_right = res_right,
                              quad_weights=Datasets[1][1], norm=Norm,reg_param_w = 0.2)
        #Initial Condition Term
        ic_dataset = InitialConditionDataset(Datasets[0][2], norm=Norm, n_bdy=initial_bdy, x_dim=1000, y_dim=1000, nx=n_points)
        initial_condition = pf.InitialCondition(ic_dataset,  quad_weights=Datasets[1][2], 
                                                norm=Norm)
        #test_loss = tl.My_Loss(ic_dataset,  quad_weights=Datasets[1][2], norm=Norm)
        return [dirichlet_bc, pde_loss, initial_condition], [bc_dataset, pde_dataset, ic_dataset], Datasets[1]
    
# Call the datasets functions, losses and weights for the training and for the performance measure
folder = 'Results_Simulation/03.05/'+str(e_l)+'_'
# Call the datasets functions, losses and weights for the training and for the performance measure
[dirichlet_bc_2, pde_loss_2, initial_condition_2], [bc_dataset_2, pde_dataset_2, ic_dataset_2], [boundary_weights_2,
                                                                                     residual_weights_2,
                                                                                     initial_weights_2] = Dataset_loss('Quad', [lb, ub], 1, deg=DEG)
[dirichlet_bc, pde_loss, initial_condition], [bc_dataset, pde_dataset, ic_dataset], [boundary_weights,
                                                                                     residual_weights,
                                                                                     initial_weights] = Dataset_loss('Mse', [lb, ub], len(initial_weights_2), deg=1)
[dirichlet_bc_3, pde_loss_3, initial_condition_3], [bc_dataset_3, pde_dataset_3, ic_dataset_3], [boundary_weights_3,
                                                                                     residual_weights_3,
                                                                                     initial_weights_3] = Dataset_loss('Wass', [lb, ub], len(initial_weights_2), deg=1)
model_1 = pf.models.MLP(input_size=3, output_size=1, hidden_size=50, num_hidden=4, lb=lb, ub=ub)
model_2 = pf.models.MLP(input_size=3, output_size=1, hidden_size=50, num_hidden=4, lb=lb, ub=ub)
model_3 = pf.models.MLP(input_size=3, output_size=1, hidden_size=50, num_hidden=4, lb=lb, ub=ub)


performance_var = [initial_condition_2, [dirichlet_bc], pde_loss]
pinn_1 = pf.PINN(model_1, 3, 1, pde_loss,initial_condition, performance_var, [dirichlet_bc], use_gpu=False)
loss_1 = pinn_1.fit(n_epoch, 'Adam', 1e-3,pinn_path=folder+'best_model_Mse.pt')
pinn_2 = pf.PINN(model_2, 3, 1, pde_loss, initial_condition_2, performance_var, [dirichlet_bc] ,use_gpu=False)
loss_2= pinn_2.fit(n_epoch, 'Adam', 1e-3,pinn_path=folder+'best_model_Quad.pt')
pinn_3 = pf.PINN(model_3, 3, 1, pde_loss, initial_condition_3, performance_var, [dirichlet_bc] ,use_gpu=False)
loss_3 = pinn_3.fit(n_epoch, 'Adam', 1e-3,pinn_path = folder+'best_model_Wass.pt')


#Produce plots
#!mkdir /Users/juanesteban/PhD_PINNs/Results_Simulation/27.04/
fig = plt.figure()
# ax2 = fig.add_subplot(2, 1, 1)
plt.semilogy(loss_1.numpy(), label='MSE Loss')
plt.semilogy(loss_2.numpy(), label='Quadrature Loss')
plt.semilogy(loss_3.numpy(), label='Wasserstein Loss')
plt.xlabel('Epoch')
plt.ylabel('Wasserstein Performance')
plt.title('$\lambda$: '+str(lam)+', Degree: '+str(DEG))
plt.legend()
plt.savefig(folder + 'Loss_'+str(lam)+'_'+str(DEG)+'_n_epoch_'+str(n_epoch)+'.png')

x_t = np.linspace(lb[0], ub[0])
y_t = np.linspace(lb[1], ub[1])
t = 0
X_c = torch.tensor([[[i, j, t] for i in x_t] for j in y_t])
X_c1 = X_c.reshape(50*50,3)
X_c1.requires_grad = True
X_m,Y_m = np.meshgrid(x_t,y_t)
#print(schroedinger1d(X_c, pinn_1(X_c)))
pinn_1.load_model(folder+'best_model_Mse.pt')
PRED_1 = pinn_1(X_c.float())
lam_pred = torch.mean(torch.div(res_right(X_c1,pinn_1(X_c1.float())),
                                pinn_1(X_c1.float())[:,0])).detach().numpy()
lam_pred = "{:.2f}".format(lam_pred)
fig = plt.figure()
ax = fig.gca(projection='3d')#fig.add_subplot(2, 1, 2, projection='3d')
c1 = ax.plot_surface(X_m, Y_m, PRED_1[:,:,0].detach().numpy(),label='Trained Psi',
                    color='blue')
c3 = ax.plot_wireframe(X_m, Y_m, Psi(X_m,Y_m,0,1),label ='Real Psi',color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('$\lambda$: '+str(lam)+', Deg: '+str(DEG)+', $\lambda$_p: '+str(lam_pred))
plt.savefig(folder + 'Surface_Mse_'+str(e_l)+'_'+str(DEG)+'_n_epoch_'+str(n_epoch)+'.png')

pinn_2.load_model(folder+'best_model_Quad.pt')
PRED_2 = pinn_2(X_c.float())
lam_pred = torch.mean(torch.div(res_right(X_c1,pinn_2(X_c1.float())),
                                pinn_1(X_c1.float())[:,0])).detach().numpy()
lam_pred = "{:.2f}".format(lam_pred)
fig = plt.figure()
ax = fig.gca(projection='3d')#
c2 = ax.plot_surface(X_m, Y_m, PRED_2[:, :, 0].detach().numpy(), label='Trained Psi_wass',
                    color='green')
c3 = ax.plot_wireframe(X_m, Y_m, Psi(X_m,Y_m,0,1),label ='Real Psi',color = 'red')
plt.title('$\lambda$: '+str(lam)+', Deg: '+str(DEG)+', $\lambda$_p: '+str(lam_pred))
plt.savefig(folder + 'Surface_Quad_'+str(e_l)+'_'+str(DEG)+'_n_epoch_'+str(n_epoch)+'.png')

pinn_3.load_model(folder + 'best_model_Wass.pt')
PRED_3 = model_3(X_c.float())
lam_pred = torch.mean(torch.div(res_right(X_c1,pinn_3(X_c1.float())),
                                pinn_1(X_c1.float())[:,0])).detach().numpy()
lam_pred = "{:.2f}".format(lam_pred)
fig = plt.figure()
ax = fig.gca(projection='3d')#
c2 = ax.plot_surface(X_m, Y_m, PRED_3[:, :, 0].detach().numpy(), label='Trained Psi_wass',
                    color='orange')
c3 = ax.plot_wireframe(X_m, Y_m, Psi(X_m,Y_m,1,e_l),label ='Real Psi',color = 'red')
plt.title('$\lambda$: '+str(lam)+', Deg: '+str(DEG)+', $\lambda$_p: '+str(lam_pred))
plt.savefig(folder + 'Surface_Wass_'+str(e_l)+'_'+str(DEG)+'_n_epoch_'+str(n_epoch)+'.png')
