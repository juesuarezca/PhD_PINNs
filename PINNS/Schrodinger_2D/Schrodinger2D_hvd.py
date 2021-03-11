import numpy as np
import time
import torch
import torch.nn as nn
import torch.autograd
import h5py
import torch.optim as optim
import scipy.io
from torch.autograd import Variable
import torch.optim as optim
from enum import Enum
from SchrodingerBalancedECDataset import SchrodingerEquationDataset
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib


class SchrodingerNet(nn.Module):
    def __init__(self, numLayers, numFeatures, lb, ub, samplingX, samplingY, activation=torch.tanh):
        """
        This function creates the components of the Neural Network and saves the datasets
        :param x0: Position x at time zero
        :param u0: Real Part of the solution at time 0 at position x
        :param v0: Imaginary Part of the solution at time 0 at position x
        :param tb: Time Boundary
        :param X_f: Training Data for partial differential equation
        :param layers: Describes the structure of Neural Network
        :param lb: Value of the lower bound in space
        :param ub: Value of the upper bound in space
        """
        torch.manual_seed(1234)
        super(SchrodingerNet, self).__init__()
        self.numLayers = numLayers
        self.numFeatures = numFeatures
        self.lin_layers = nn.ModuleList()
        self.activation = activation
        self.lb = torch.Tensor(lb).float().cuda()
        self.ub = torch.Tensor(ub).float().cuda()
        
        #calculate matrix for trepze rule 
        #matrix construction follows from : http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html
        W = np.zeros((samplingX, samplingY))
        W[0, 0] = 1
        W[0, samplingY - 1] = 1
        W[samplingX - 1, samplingY - 1] = 1
        W[samplingX - 1, 0] = 1

        for idx in range(1, samplingX - 1):
            W[idx, 0] = 2
            W[idx, samplingY - 1] = 2

        for idx in range(1, samplingY - 1):
            W[0, idx] = 2
            W[samplingX - 1, idx] = 2

        for i in range(1, samplingX - 1):
            for j in range(1, samplingY - 1):
                W[i, j] = 4

        W = W.reshape(-1)
        self.W = torch.Tensor(W).float().cuda()

        # building the neural network
        self.init_layers()

        # Creating Weight Matrix for energy conservation mechanism

    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
        self.lin_layers.append(nn.Linear(3, self.numFeatures))
        for _ in range(self.numLayers):
            inFeatures = self.numFeatures
            self.lin_layers.append(nn.Linear(inFeatures, self.numFeatures))
        self.lin_layers.append(nn.Linear(inFeatures, 2))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def net_uv(self, x, y, t):
        """
        Function that calculates the nn output at postion (x,y) at time t
        :param x: position
        :param t: time
        :return: Approximated solutions and their gradients
        """

        #torch.cuda.empty_cache()
        dim = x.shape[0] #defines the shape of the gradient

        # save input in variabeles is necessary for gradient calculation
        x = Variable(x, requires_grad=True).cuda()
        y = Variable(y, requires_grad=True).cuda()
        t = Variable(t, requires_grad=True).cuda()

        X = torch.stack([x, y, t], 1)

        UV = self.forward(X)
        u = UV[:, 0]
        v = UV[:, 1]
        grads = torch.ones([dim]).cuda()

        # huge change to the tensorflow implementation this function returns all neccessary gradients
        J_U = torch.autograd.grad(u,[x,y,t], create_graph=True, grad_outputs=grads)
        J_V = torch.autograd.grad(v,[x,y,t], create_graph=True, grad_outputs=grads)
    
        u_x = J_U[0].reshape([dim])
        u_y = J_U[1].reshape([dim])
        u_t = J_U[2].reshape([dim])

        v_x = J_V[0].reshape([dim])
        v_y = J_V[1].reshape([dim])
        v_t = J_V[2].reshape([dim])

        u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        v_xx = torch.autograd.grad(v_x, x, create_graph=True, grad_outputs=grads)[0]

        u_yy = torch.autograd.grad(u_y, y, create_graph=True, grad_outputs=grads)[0]
        v_yy = torch.autograd.grad(v_y, y, create_graph=True, grad_outputs=grads)[0]

        u_xx = u_xx.reshape([dim])
        v_xx = v_xx.reshape([dim])
        u_yy = u_yy.reshape([dim])
        v_yy = v_yy.reshape([dim])

        return u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t
    def forward(self, x):
        """
        This function is the forward of a simple multilayer perceptron
        """
        #normalize input in range between -1 and 1 for better training convergence
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(0, len(self.lin_layers) - 2):
            x = self.lin_layers[i](x)
            x = self.activation(x)
            # x = torch.sin(x)
            # x = F.tanh(x)
        x = self.lin_layers[-1](x)

        return x

    def net_pde(self, x, y, t, omega=1.):
        """
        Calculates the quality of the pde estimation
        :param x postion x
        :param t time t
        :param omega frequency of the harmonic oscillator 
        """
        #get predicted solution and the gradients 
        u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = self.net_uv(x, y, t)
        x = x.view(-1)
        y = y.view(-1)

        #calculate loss for real and imaginary part seperatly 
        # fu is the real part of the schrodinger equation
        f_u = -1 * u_t - 0.5 * v_xx - 0.5 * v_yy + omega* 0.5 * (x ** 2) * v + omega * 0.5 *  (y ** 2) * v
        # fv is the imaginary part of the schrodinger equation 
        f_v = -1 * v_t + 0.5 * u_xx + 0.5 * u_yy - omega* 0.5 * (x ** 2) * u - omega * 0.5 * (y ** 2) * u
        return u, v, f_u, f_v

    def solution_loss(self, x, y, t, u0, v0):
        """
        Supervised loss for training the initial condition 
        """
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        inputX = torch.stack([x, y, t], 1)
        UV = self.forward(inputX)
        u = UV[:, 0]
        v = UV[:, 1]

        u0 = u0.view(-1)
        v0 = v0.view(-1)

        loss = torch.mean((u0 - u) ** 2) + torch.mean(v ** 2)
        return loss

  

    def ec_pde_loss(self, x0, y0, t0, u0, v0, xf, yf, tf, xe, ye, te, c, samplingX, samplingY,activateEnergyLoss=True, alpha=1.):
    
        #reshape all inputs into correct shape 
        x0 = x0.view(-1)
        y0 = y0.view(-1)
        t0 = t0.view(-1)
        xf = xf.view(-1)
        yf = yf.view(-1)
        tf = tf.view(-1)
        xe = xe.view(-1)
        ye = ye.view(-1)
        te = te.view(-1)

        n0 = x0.shape[0]
        nf = xf.shape[0]

        inputX = torch.cat([x0, xf, xe])
        inputY = torch.cat([y0, yf, ye])
        inputT = torch.cat([t0, tf, te])
        
        

        u, v, f_u, f_v = self.net_pde(inputX, inputY, inputT)

        solU = u[:n0]
        solV = v[:n0]

        eU = u[n0 + nf:]
        eV = v[n0 + nf:]
        eH = eU ** 2 + eV ** 2

        lowerX = self.lb[0]
        higherX = self.ub[0]

        lowerY = self.lb[1]
        higherY = self.ub[1]

        disX = (higherX - lowerX) / samplingX
        disY = (higherY - lowerY) / samplingY

        u0 = u0.view(-1)
        v0 = v0.view(-1)
        integral = 0.25 * disX * disY * torch.sum(eH * self.W)
        # calculte integral over field for energy conservation
        eLoss = (integral - c) ** 2

        pdeLoss = alpha * torch.mean((solU - u0) ** 2) + \
                  alpha * torch.mean((solV - v0) ** 2) + \
                  torch.mean(f_u ** 2) + \
                  torch.mean(f_v ** 2)
        if activateEnergyLoss:
            pdeLoss = pdeLoss + eLoss

        if epoch % 30 == 0:
            #write into tensorboard
            if log_writer:
                log_writer.add_scalar('Solution U', torch.mean((solU - u0) ** 2), epoch)
                log_writer.add_scalar('Solution V', torch.mean((solV - v0) ** 2), epoch)
                log_writer.add_scalar('Real PDE', torch.mean(f_u ** 2), epoch)
                log_writer.add_scalar('Imaginary PDE', torch.mean(f_v ** 2), epoch)
                log_writer.add_scalar('Energy Loss', eLoss, epoch)
                log_writer.add_scalar('PDE Loss', pdeLoss, epoch)
                log_writer.add_scalar('Integral', integral, epoch)

        return pdeLoss


def writeIntermediateState(timeStep, model, epoch, nx, ny, fileWriter,csystem):
    """
    Functions that write intermediate solutions to tensorboard
    """
    if fileWriter:
        x, y, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
        x = torch.Tensor(x).float().cuda()
        y = torch.Tensor(y).float().cuda()
        t = torch.Tensor(t).float().cuda()

        inputX = torch.stack([x, y, t], 1)
        UV = model.forward(inputX).detach().cpu().numpy()

        u = UV[:, 0].reshape((nx, ny))
        v = UV[:, 1].reshape((nx, ny))

        h = u ** 2 + v ** 2

        fig = plt.figure()
        plt.imshow(u, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('Real_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(v, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('Imaginary_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(h, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('Norm_' + str(timeStep), fig, epoch)
        plt.close(fig)


def valLoss(model, timeStep, csystem):
    """
    The validation loss is the MSE between predicted and correct solution.
    The loss is calculated seperatly for real and imaginary part 
    """
    x, y, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()
    uPred = UV[:, 0]
    vPred = UV[:, 1]

    # load label data
    uVal, vVal = SchrodingerEquationDataset.loadFrame(pData, timeStep)
    uVal = np.array(uVal)
    vVal = np.array(vVal)

    valLoss = np.mean((uVal - uPred) ** 2) + np.mean((vVal - vPred) ** 2)
    return valLoss


def writeValidationLoss(model, writer, timeStep, epoch, csystem):
    if writer:
        loss = valLoss(model, timeStep,csystem)
        writer.add_scalar("ValidationLoss_" + str(timeStep), loss, epoch)


def save_checkpoint(model, optimizer, path, epoch):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path + 'model_' + str(epoch))


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    # static parameter
    nx = 200
    ny = 200
    nt = 1000
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    dt = 0.001
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx": nx , "ny": ny, "nt": nt, "dt": dt}

    pData = '/projects/p_electron/stiller/schrodinger/data/schrodinger_reduced_3/'

    batchSizeInit = 2500  #for the balanced dataset is not needed

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str)
    parser.add_argument("--batchsize", dest="batchsize", type=int)
    parser.add_argument("--numbatches", dest="numBatches", type=int)
    parser.add_argument("--initsize", dest="initSize", type=int)
    parser.add_argument("--numlayers", dest="numLayers", type=int)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int)
    parser.add_argument("--epochssolution", dest="epochsSolution", type=int)
    parser.add_argument("--epochsPDE", dest="epochsPDE", type=int)
    parser.add_argument("--energyloss", dest="energyLoss",type=int)
    parser.add_argument("--pretraining", dest="pretraining", type=int)
    parser.add_argument("--alpha",dest="alpha",type=float)
    parser.add_argument("--lhs",dest="lhs",type=int)

    args = parser.parse_args()

    if hvd.rank() == 0: 
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 +  args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
    
    print("Rank",hvd.rank(),"Local Rank", hvd.local_rank())
    
    #adapter of commandline parameters

    modelPath = '/projects/p_electron/stiller/schrodinger/thesis/models/' + args.identifier + '/'
    logdir = '/projects/p_electron/stiller/schrodinger/thesis/tensorboard/' + args.identifier + '/'
    batchSizePDE = args.batchsize
    useGPU = True
    numBatches = args.numBatches
    initSize = args.initSize
    numLayers = args.numLayers
    numFeatures = args.numFeatures
    numEpochsSolution = args.epochsSolution
    numEpochsPDE = args.epochsPDE
    activateEnergyLoss = args.energyLoss
    pretraining = args.pretraining
    #postprocessing = args.postprocessing

    #create modelpath
    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True) 
    # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None

    # create dataset
    ds = SchrodingerEquationDataset(pData, coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, initSize, numBatches, batchSizePDE, shuffle=True, useGPU=True,do_lhs=args.lhs)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)

    activation = torch.tanh

    print("Memory Allocated before",torch.cuda.memory_allocated('cuda:0') / 1e9)
    model = SchrodingerNet(numLayers, numFeatures, ds.lb, ds.ub, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, torch.tanh).cuda()
    print("Memory Allocated after",torch.cuda.memory_allocated('cuda:0') / 1e9)

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)

    if pretraining:
        for epoch in range(numEpochsSolution):
            for x0, y0, t0, Ex_u, Ex_v, xf, yf, tf, xe, ye, te in train_loader:
                optimizer.zero_grad()
                # calculate loss
                loss = model.solution_loss(x0, y0, t0, Ex_u, Ex_v)
                loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                print("Loss at Epoch " + str(epoch) + ": " + str(loss.item()))
                sys.stdout.flush()
                if log_writer:
                    log_writer.add_scalar("Initital Loss",loss.item(),epoch)


    # save model after initial training
    if log_writer:
        state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
        torch.save(state, modelPath + 'model_init_' + str(0))

    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = 7e-6

    for epoch in range(numEpochsPDE):

        for x0, y0, t0, Ex_u, Ex_v, xf, yf, tf, xe, ye, te in train_loader:
            optimizer.zero_grad()

            # calculate loss
            
            loss = model.ec_pde_loss(x0,
                                     y0,
                                     t0,
                                     Ex_u,
                                     Ex_v,
                                     xf,
                                     yf,
                                     tf,
                                     xe,
                                     ye,
                                     te,
                                     1.,
                                     numOfEnergySamplingPointsX,
                                     numOfEnergySamplingPointsY,
                                     activateEnergyLoss,
                                     args.alpha)
            loss.backward()
            optimizer.step()

        if epoch % 30 == 0:
            writeIntermediateState(0, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(250, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(500, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(750, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(1000, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeValidationLoss(model, log_writer, 250, epoch,coordinateSystem)
            writeValidationLoss(model, log_writer, 500, epoch,coordinateSystem)
            writeValidationLoss(model, log_writer, 750, epoch,coordinateSystem)
            writeValidationLoss(model, log_writer, 1000, epoch,coordinateSystem)
            sys.stdout.flush()

            print("PDE Loss at Epoch: ", epoch + 1, loss.item())
            if log_writer:
                log_writer.add_histogram('First Layer Grads', model.lin_layers[0].weight.grad.view(-1, 1), epoch)
                save_checkpoint(model, optimizer, modelPath, epoch)

    if log_writer:
        hParams = {'numLayers': numLayers,
                   'numFeatures': numFeatures,
                   'ResidualPoints': numBatches * batchSizePDE,
                   'alpha':args.alpha,
                   'ELoss': activateEnergyLoss}

        valLoss0 = valLoss(model, 0, coordinateSystem)
        valLoss250 = valLoss(model, 250, coordinateSystem)
        valLoss500 = valLoss(model, 500, coordinateSystem)

        metric = {'hparam/SimLoss': loss.item(),
                  'hparam/valLoss0': valLoss0,
                  'hparam/valLoss250': valLoss250,
                  'hparam/valLoss500': valLoss500}

        log_writer.add_hparams(hParams, metric)
