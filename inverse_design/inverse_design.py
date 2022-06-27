####################################################
# design structure with one hologram for each polarization
# with Fresnel near-to-farfield transformation
####################################################
# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from scipy.io import loadmat, savemat
from model_p3 import Model
from glob import glob
import matplotlib.pyplot as plt
import time
from torch.fft import fft2, ifft2, fftshift, ifftshift

####################################################
# Setup the device

if torch.cuda.is_available():
    device = torch.device('cuda:2')
else:
    device = torch.device('cpu')
device = torch.device('cpu')
####################################################
# Load data

file_lst = glob('../data/data3d_*.mat')

file_name = file_lst[0]
file_init = loadmat(file_name)

# number of structs
N_str = len(file_lst)
str_shape = file_init['R'].shape
N = file_init['R'].shape[0]
Ng = file_init['E'].shape[1]//N

####################################################
# Define some parameters

R = 3

zpd_size = R
bnd_size = R

pd_size = zpd_size + bnd_size

window_size = 70

graph_shape = [window_size + 2 * pd_size, window_size + 2 * pd_size]

# don't forget to count the number of neighbours at som point. It'll become a hassle!
node_lst = []
for i in range(graph_shape[0]):
    for j in range(graph_shape[1]):
        if i == graph_shape[0]//2 and j == graph_shape[1]//2:
            N_nbr = 0
        for k in range(-R, R+1):
            for l in range(-R, R+1):
                if k**2 + l**2 <= R**2:
                    if (0 <= i + k < graph_shape[0]) and (0 <= j + l < graph_shape[1]):
                        if i == graph_shape[0] // 2 and j == graph_shape[1] // 2:
                            N_nbr += 1
                        node_lst.append([(i + k) * graph_shape[1] + (j + l), i * graph_shape[1] + j])
nodes_lst = (np.array(node_lst).transpose()).tolist()

# define features for whether an element is on the boundary or not
bnd_ = 2*torch.ones(graph_shape)
bnd_[:zpd_size, :] = 0
bnd_[-zpd_size:, :] = 0
bnd_[:, :zpd_size] = 0
bnd_[:, -zpd_size:] = 0
bnd_[zpd_size:pd_size, zpd_size:-zpd_size] = 1
bnd_[-pd_size:-zpd_size, zpd_size:-zpd_size] = 1
bnd_[zpd_size:-zpd_size, zpd_size:pd_size] = 1
bnd_[zpd_size:-zpd_size, -pd_size:-zpd_size] = 1

# create the position tensors
pos_y, pos_x = np.meshgrid(np.arange(graph_shape[1]), np.arange(graph_shape[0]))
pos_ = torch.FloatTensor((np.stack((pos_x, pos_y), axis=0)).transpose(1, 2, 0).reshape(-1, 2))

# define the mask for the main window

mask_ = np.ones((graph_shape[0], graph_shape[1]), dtype=bool)
mask_[:pd_size, :] = False
mask_[-pd_size:, :] = False
mask_[:, :pd_size] = False
mask_[:, -pd_size:] = False
mask_ = mask_.reshape(-1)

###########################################
# Training

# initialize parameters
lr = 3e-5  # learning rate
N_epochs = 10000  # number of epochs
batch_size = 1
in_d = 5  # input dimension
h_d = 1600  # hidden layer dimension
out_d = 6 * Ng**2  # output dimension
nhl = 5  # number of hidden layers
l2_reg = 1e-5  # l2-reg multiplier


# fix the mask
mask_ = mask_.repeat(batch_size)


# create the graph network
graph_model = Model(in_d, h_d, out_d, N_nbr, num_layers=nhl).to(device)


# create the CNN network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 24, 5, padding='same')
        self.conv2 = nn.Conv2d(24, 48, 5, padding='same')
        self.conv3 = nn.Conv2d(48, 24, 5,padding='same')
        self.conv4 = nn.Conv2d(24, 6, 5, padding='same')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


conv_net = Net().to(device)


# the file paths for the model
PATH_best = "./p3best.pt"
checkpoint = torch.load(PATH_best, map_location=device)
graph_model.load_state_dict(checkpoint['model_state_dict'])
conv_net.load_state_dict(checkpoint['model_state_dict2'])


###########################################
# Load Targets

file_im2 = loadmat('./image_m.mat')
mask_im = (torch.tensor(file_im2['img'],dtype=torch.bool)).to(device)


###########################################
# Initialize Design

num_nodes = len(nodes_lst[0])
rad_ = np.random.choice([6.0,8.0,10.0], (graph_shape))
hts_ = np.random.choice([14.0,16.0,18.0], (graph_shape))
dsp_ = np.random.choice([-2.0,0.0,2.0], (2,graph_shape[0],graph_shape[1]))
mask_bnd = np.zeros((graph_shape[0], graph_shape[1]), dtype=bool)
mask_bnd[:pd_size, :] = True
mask_bnd[-pd_size:, :] = True
mask_bnd[:, :pd_size] = True
mask_bnd[:, -pd_size:] = True
mask_bnd = mask_bnd.reshape(-1)
x_ = torch.FloatTensor((np.concatenate((np.expand_dims(rad_, axis=0), np.expand_dims(hts_, axis=0), dsp_, np.expand_dims(bnd_, axis=0)), axis=0)).transpose(1, 2, 0).reshape(-1, 5))
x_.requires_grad = True
lr_x = 1e4

###########################################
# Simulation Parameters

# wavlength
wvn = 600e-9
# wave number
k_0 = 2*np.pi/wvn
# simulation resolution
dl = wvn/30
# distance from the near-field
h_val = 30*wvn

cost_file = open("str_cost.txt", "w")

# farfield size
Nx, Ny = mask_im.shape

# padding required
Nxp = (Nx-window_size*Ng)//2
Nyp = (Ny-window_size*Ng)//2

# Fresnel kernel
ps_x = np.arange(-Nx//2, Nx//2)*dl
ps_y = np.arange(-Ny//2, Ny//2)*dl
px, py = np.meshgrid(ps_x, ps_y)
pr_ = np.sqrt(px**2+py**2+h_val**2)
#filt_ = h_val/(pr_**2)*(1/(2*np.pi*pr_)-1j/wvn)*np.exp(1j*2*np.pi*pr_/wvn)
filt_ = h_val/(pr_**2)*(-1j/wvn)*np.exp(1j*2*np.pi*pr_/wvn)
prop_shift = torch.tensor(np.fft.fft2(np.fft.ifftshift(filt_)), device=device)
#prop_shift = np.fft.fft2(np.fft.ifftshift(filt_))
g = dgl.graph((nodes_lst[0], nodes_lst[1]), device=device)
g.ndata['pos'] = pos_.to(device)

def fwd_eval(x, fwd=True):
    g.ndata['h'] = x.to(device)
    out_1 = graph_model.forward(g)[mask_].reshape(1, window_size**2, Ng**2, 6)
    out_2 = torch.zeros(1, 6, window_size*Ng, window_size*Ng, device=device)
    for x_id in range(window_size):
        for y_id in range(window_size):
            out_2[0, :, x_id*Ng:(x_id+1)*Ng, y_id*Ng:(y_id+1)*Ng] = out_1[0, x_id*window_size+y_id].T.reshape(6,Ng,Ng)
    out_3 = conv_net(out_2)[0]
    
    En = F.pad(torch.stack([out_3[0]+1j*out_3[1],out_3[2]+1j*out_3[3],out_3[4]+1j*out_3[5]]),(Nxp, Nxp, Nyp, Nyp))
      
    Ef = ifft2(fft2(En) * prop_shift)*dl*dl
    
    out_4 = (Ef.abs()**2).sum(0)
    
    loss = (out_4[mask_im].sum()/out_4.sum()-1)**2 + 0.01* ((out_4[mask_im]-out_4[mask_im].mean())**2).sum()/mask_im.sum()
    
    print(loss.item())
    cost_file.write("%f\n" % loss.item())
    
    if fwd:
        loss.backward()
    else:
        return Ef.detach().cpu().numpy()
        
for iter_ in range(500):
    fwd_eval(x_)
    with torch.no_grad():
        x_[:,:-1] -= lr_x * x_.grad[:,:-1]
        x_[x_[:,0]<6.0,0] = 6.0
        x_[x_[:,0]>10.0,0] = 10.0
        x_[x_[:,1]<14.0,1] = 14.0
        x_[x_[:,1]>18.0,1] = 18.0
        x_[x_[:,2]<-2.0,2] = -2.0
        x_[x_[:,2]>2.0,2] = 2.0
        x_[x_[:,3]<-2.0,3] = -2.0
        x_[x_[:,3]>2.0,3] = 2.0
        x_[mask_bnd,:-1] = 0

        x_.grad.zero_()

    if not iter_%10:
        lr_x *= 0.9
        savemat('./str_dsn.mat',{'x':x_.detach().cpu().numpy()})

savemat('./str_dsn.mat',{'x':x_.detach().cpu().numpy()})
print('hello')