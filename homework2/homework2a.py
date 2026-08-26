import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.transform import rescale
from scipy import io
import torch.optim
from torch import optim
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from collections import OrderedDict

#from moduless import utils

import time

def get_mgridVideo(N, dim=2):
    L = 3
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    a = np.array([])
    b = np.array([])
    c = np.array([])
    mgrid1 = np.zeros((N*N,dim+1),dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    xv1 = np.float32(xv.reshape(-1, 1))
    yv1 = np.float32(yv.reshape(-1, 1))
    mgrid1[:,0] = (xv1[:,0])
    mgrid1[:,1] = (yv1[:,0]) 
    idx = np.linspace(1,-1,L)

    for l in range(L):
        a = np.float32(np.concatenate([a, mgrid1[:,0]],axis=None))
        b = np.float32(np.concatenate([b, mgrid1[:,1]],axis=None))
        c = np.float32(np.concatenate([c, mgrid1[:,2]+idx[l]],axis=None))

    mgrid2 = np.vstack((a,b,c))
    mgrid2 = mgrid2.transpose()
    mgrid = torch.from_numpy(mgrid2)
    return mgrid 

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return (torch.sin(1*self.omega_0 * self.linear(input)))*torch.exp(0*(self.omega_0 * self.linear(input))**2)
        #w0 = 4.0
        #s0 = 4.0
        #return (torch.sin(w0*self.omega_0 * self.linear(input)))*torch.exp(-s0*(self.omega_0 * self.linear(input))**2)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return (torch.sin(1*intermediate)*torch.exp(-0*(intermediate**2))), intermediate
        #w0 = 4.0
        #s0 = 4.0
        #return (torch.sin(w0*intermediate)*torch.exp(-s0*(intermediate**2))), intermediate
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def load(N,L):

    # load data
    P = Patterns(N,L)
    G = P.BayerFilter()
    #Data = io.loadmat(DataPath1)
    #Data = Data['data']
    Data1 = data.astronaut()
    Data = resize(Data1, (256, 256), anti_aliasing=True)
    Xinput = Data.astype(np.float32)
    [a,b,c]=np.shape(Xinput)
    tp = np.min([a,b])
    Xinput = Xinput[0:tp,0:tp,:]
    temp = np.zeros((N,N,L))
    BM = np.size(Xinput,0)
    for i in range(L):
        temp[:,:,i] = rescale(Xinput[:,:,i], N/BM, anti_aliasing=True)*255

    X = (np.rint(temp)).astype(np.float32)
    return X

def get_mosaic_tensor(N):
    L = 3
    X = load(N,L)
    P = Patterns(N,L)
    MSFA = P.BayerFilter()
    S = D3CASSI(X,MSFA)
    Y = S.D3CASSIsampling() # compute measurements
    #S = CCASSI(X,MSFA)
    #Y = S.CCASSIsampling() # compute measurements

    savepath2 = 'C:/Users/nelson/OneDrive/Documenten/implicitRepresentation-main/Reconstruction/measu_video_stick.mat'
    io.savemat(savepath2, {'mesu': Y})
    Xbar = S.unfoldingD3CASSI()
    #Xbar = S.unfoldingCCASSI()
    [M,N,L] = np.shape(Xbar)
    I = np.array([],dtype=np.float32)
    x = np.array([],dtype=np.float32)
    y = np.array([],dtype=np.float32)
    z = np.array([],dtype=np.float32)
    idx =np.linspace(-1,1,L)

    for l in range(L):
        J = Xbar[:,:,l]*(Xbar[:,:,l]>0)
        b = J[J>0]-np.finfo(float).eps
        I = np.concatenate([I, b],axis=None)
        y1,x1 = np.nonzero((J>0))
        z1 = y1*0 + idx[l]
        x = np.concatenate([x, x1],axis=0)
        y = np.concatenate([y, y1],axis=0)
        z = np.concatenate([z, z1],axis=0)
        
    I1 = 2.*(I - np.min(I))/np.ptp(I)-1
    I1 = np.float32(I1.reshape(-1, 1))
    I1 = torch.from_numpy(I1)

    # Coordinates
    x = 2.*(x - np.min(x))/np.ptp(x)-1
    x = x.reshape(-1, 1)
    y = 2.*(y - np.min(y))/np.ptp(y)-1
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    mgrid1 = np.zeros((np.size(z),3),dtype=np.float32)
    mgrid1[:,0] = x[:,0]
    mgrid1[:,1] = y[:,0]
    mgrid1[:,2] = z[:,0]
    mgrid = torch.from_numpy(mgrid1)
        
    return I1, mgrid
  
class D3CASSI:
    def __init__(self,X,C):
        self.X = X
        self.C = C
        self.Y = []
    def D3CASSIsampling(self):
        [m, n, l] = np.shape(self.X)
        Y = np.zeros((m, n))
        for i in range(l):
            # Load ith multispectral image
            X1 = self.X[:, :, i]
            # Load ith coded aperture
            C1 = self.C==(i+1)
            # To compute the compressive measurement without prism dispersion
            Y = Y + X1*C1
        Y = (np.rint(Y)).astype(np.float32)
        self.Y = Y+np.finfo(float).eps
        return self.Y
    def unfoldingD3CASSI(self):
        [m, n] = np.shape(self.C)
        [m, n1] = np.shape(self.Y)
        l = np.uint8(np.max(self.C))
        J = np.zeros((m,n,l),dtype=np.float32)
        for j in range(l):
            T = self.C==(j+1)
            J[:,:,j] = J[:,:,j] + (self.Y[:,:]*T)
        return J  
    
class CCASSI:
    def __init__(self,X,C):
        self.X = X
        self.C = C
        self.Y = []
    
    def CCASSIsampling(self):
        [m, n, l] = np.shape(self.X)
        Y = np.zeros((m, n+l-1))
        print(Y.shape)
        for i in range(l):
            # Load ith multispectral image
            X1 = self.X[:, :, i]
            # Load ith coded aperture
            C1 = self.C==1  #np.random.rand(m,n)<0.5
            # To compute the compressive measurement with prism dispersion
            Y[:,i:(n+i)] = Y[:,i:(n+i)] + (X1*C1)
        Y = (np.rint(Y)).astype(np.float32)
        savepath3 = 'C:/Users/nelson/OneDrive/Documenten/implicitRepresentation-main/Reconstruction/mask32.mat'
        io.savemat(savepath3, {'mask': C1})
        self.Y = Y   
        return self.Y
    
    def unfoldingCCASSI(self):
        [m, n] = np.shape(self.C)
        [m, n1] = np.shape(self.Y)
        l = n1-n+1
        J = np.zeros((m,n,l),dtype=np.float32)
        for j in range(l):
            T = self.C==(1)
            # if j == 1 or j == range(l):
            J[:,:,j] = J[:,:,j] + (self.Y[:,j:n+j]*T)
            # else:
            #     J[:,:,j] = J[:,:,j] + (self.Y[:,j:n+j]*T) + (self.Y[:,j-1:n+j-1]*T) + (self.Y[:,j+1:n+j+1]*T)
        return J

class Patterns:
    def __init__(self,N,L):
        self.N = N
        self.L = L    
    def BayerFilter(self):
        N = self.N
        M = int(self.N//2)
        K = np.array([[1, 2], [2, 3]])
        G = np.kron(np.ones((M,M)), K)
        G = G[0:N,0:N]
        return G
    def RollingShutter(self):
        DataPath1 = 'C:/Users/nelson/OneDrive/Documenten/implicitRepresentation-main/Kernel/kernel'+str(L)+'.mat'
        Data = io.loadmat(DataPath1) # X
        K = Data['G4']
        G = K
        return G

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        I1, xyz = get_mosaic_tensor(sidelength)
        # Intensities
        self.pixels = I1
        #xyz = get_mgridMosaic(sidelength, 2)
        # coordinates of the measurement N x 3
        self.coords = xyz

    def __len__(self):
        return 1#len(self.coords)

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

n = 256
L = 3
ref = ImageFitting(n)
dataloader = DataLoader(ref, batch_size=5000, pin_memory=True, num_workers=0)

lossfn = nn.L1Loss() 
lossfn = lossfn.cuda()
img_siren = Siren(in_features=3, out_features=1, hidden_features=256, 
                  hidden_layers=4, outermost_linear=True)
img_siren.cuda()

total_steps =  1500# 7860# Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 10
lr = 4.2e-5
optim = torch.optim.Adam(lr=lr, params=img_siren.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

batch_size = 5000 #args.siren_batch_size
scheduler = lr_scheduler.ExponentialLR
#scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for step in range(total_steps):

    if step%300 == 0:
        lr = lr*0.95
        opimizer = torch.optim.Adam(lr=lr, params=img_siren.parameters())

    #batch_idx = torch.randperm(1,batch_size)[0:batch_size]
    batch_idx = torch.randint(0, model_input.size(dim=1), (1, batch_size))
    model_output, coords = img_siren(model_input[:,batch_idx,:]) 
    loss = ((model_output - ground_truth[:,batch_idx,:])**2).mean()
    #if loss < 0.000000000010:
    #    step = total_steps
        #loss = lossfn(model_output, ground_truth)
    
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.12f, learning rate %0.8f" % (step, loss,lr))
        img_grad = gradient(model_output, coords)
        img_laplacian = laplace(model_output, coords)

    opimizer.zero_grad()
    loss.backward()
    opimizer.step()
    #scheduler.step()
    #print(f"iter: {iter} - loss: {loss.item()}")

xyz = get_mgridVideo(n, 2)
img_siren.eval()
RGBImage2 = np.zeros((n, n, L))
for i in range(L):
    model_output, coordstest = img_siren(xyz[i*n*n:(i+1)*n*n,:].cuda())
    a =model_output[0*n*n:(1)*n*n,0].reshape(n,n,1) #model_output
    RGBImage2[:,:,L-i-1] = a.cpu().view(n,n).detach().numpy()

RGB=1
if RGB==1:
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    m = (n+L-1)
    RGBImage = np.zeros((n, n))
    X = load(n,L)
    P = Patterns(n,L)
    MSFA = P.BayerFilter()
    S = D3CASSI(X,MSFA)
    mosaic = S.D3CASSIsampling() # compute measurements
    axes[0].imshow(mosaic)
    axes[0].set_title("Measurement")

    RGBImage2 = (RGBImage2+np.max(RGBImage2))/(np.max(RGBImage2)-np.min(RGBImage2))
    RGBImage2 = abs(RGBImage2/np.max(RGBImage2))
    RGBImageImp = (np.rint(RGBImage2*255)).astype(np.uint8)

    # Filename
    filename = 'savedImageImplicit2.mat'
  
    # Saving the image
    savepath = 'C:/Users/nelson/OneDrive/Documenten/implicitRepresentation-main/Reconstruction/Rec_video_stick.mat'
    savepath1 = 'C:/Users/nelson/OneDrive/Documenten/implicitRepresentation-main/Reconstruction/gt_video_stick.mat'
    io.savemat(savepath, {'Xrec': RGBImage2})

    axes[1].imshow(RGBImageImp[:,:,:]/255, vmin=0, vmax=1)
    axes[1].set_title("Reconstructed RGB")

    Data1 = data.astronaut()
    Data = resize(Data1, (256, 256), anti_aliasing=True)
    [a1,b1,c1]=np.shape(Data)
    Xinput = Data.astype(np.float32)

    [a,b,c]=np.shape(Xinput)
    tp = np.min([a,b])
    Xinput = Xinput[0:tp,0:tp,:]
    gt = np.zeros((n,n,L))
    BM = np.size(Xinput,0)
    gt = rescale(Xinput, n/BM, anti_aliasing=True)/np.max(Xinput)

    psnr = peak_signal_noise_ratio(gt, RGBImage2, data_range=1)
    RGBImage22 = np.float32(RGBImage2)
    #ssim_score = ssim(gt, RGBImage22, multichannel=True, data_range=RGBImage2.max() - RGBImage2.min())
    io.savemat(savepath1, {'ground': gt})
    print("Implicit PSNR:", psnr)
    #print("Implicit SSIM:", ssim_score)
    axes[2].imshow(gt[:,:,:], vmin=0, vmax=1)
    axes[2].set_title("Groundtruth RGB")

    #axes[3].imshow(bilinear, vmin=0, vmax=1)
    #axes[3].set_title("Billinear")
    #bilinear = np.float64(bilinear)/255

      # Filename
    filename = 'gt.png'
  
    # Saving the image
    #gt1 = gt*255
    #cv2.imwrite(filename, gt1)

    #psnr = peak_signal_noise_ratio(gt, bilinear, data_range=1)
    #print("Bilinear PSNR:", psnr)
    plt.show()

elif RGB==0:
    fig, axes = plt.subplots(1,4, figsize=(18,6))
    RGBImage = np.zeros((n, n))
    a =ground_truth[0,0:n*n,0].reshape(n,n,1) #model_output
    RGBImage[:,:] = a.cpu().view(n,n).detach().numpy()
    axes[0].imshow((RGBImage+1)/2, vmin=0, vmax=1)
    axes[0].set_title("Measurement")

    a =model_output[0,0:n*n,0].reshape(n,n,1) #model_output
    RGBImage[:,:] = a.cpu().view(n,n).detach().numpy()
    axes[1].imshow((RGBImage+1)/2, vmin=0, vmax=1)
    axes[1].set_title("Reconstruction Mosaic")

    a =img_grad[0,0:n*n,0].reshape(n,n,1) #model_output
    RGBImage[:,:] = a.cpu().view(n,n).detach().numpy()
    axes[2].imshow((RGBImage+1)/2, vmin=0, vmax=1)
    axes[2].set_title("Reconstruction Mosaic")

    a =img_laplacian[0,0:n*n,0].reshape(n,n,1) #model_output
    RGBImage[:,:] = a.cpu().view(n,n).detach().numpy()
    axes[3].imshow((RGBImage+1)/2, vmin=0, vmax=1)
    axes[3].set_title("Reconstruction Mosaic")

    plt.show()