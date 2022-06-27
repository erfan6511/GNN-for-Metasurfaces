####################################################
# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from scipy.io import loadmat
from model_p3 import Model
from glob import glob

####################################################
# Setup the device

if torch.cuda.is_available():
    device = torch.device('cuda:2')
else:
    device = torch.device('cpu')

####################################################
# Load data

file_lst = glob('./data/data3d_*.mat')

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

window_size = 7

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
epoch_lim = 1000


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

# Define the loss
mse_loss = nn.MSELoss()


# set up the optimizer
optimizer = torch.optim.Adam(list(conv_net.parameters())+list(graph_model.parameters()), lr=lr, weight_decay=l2_reg)


# the file paths for the model
PATH_ = "./p3train.pt"
PATH_best = "./p3best.pt"
cost_file = open("p3cost.txt", "w")
tst_cost_file = open("p3tcost.txt", "w")

best_cost = 1e6


def train_test(tr_chk=True):
    t_cost = 0
    pan_num = 10
    # load panels
    if tr_chk:
        pan_arr = np.random.choice(np.arange(int(0.9*N_str)),pan_num)
    else:
        pan_arr = int(0.9*N_str) + np.random.choice(np.arange(int(0.1*N_str)),pan_num)
    E_lst = []
    H_lst = []
    D_lst = []
    R_lst = []
    for file_id in pan_arr:
        file_ = loadmat(file_lst[file_id])
        # save the fields
        Ex = file_['E'][0]
        Ey = file_['E'][1]
        Ez = file_['E'][2]
        E = 5 * np.stack((np.real(Ex), np.imag(Ex), np.real(Ey), np.imag(Ey), np.real(Ez), np.imag(Ez)), axis=0)
        # process the fields
        E_lst.append(E)
        R_lst.append(file_['R'])
        H_lst.append(file_['H'])
        D_lst.append(file_['D'])

    for in_iter in range(100):
        # choose one of the panels
        str_id = np.random.choice(np.arange(pan_num))

        g_lst = []
        y_ = []
        # choose the ids for selecting the panel
        p_ids = np.random.randint(0, str_shape[0]-graph_shape[0], (2, batch_size))
        for b_id in range(batch_size):
            rad_ = R_lst[str_id][p_ids[0, b_id]:p_ids[0, b_id]+graph_shape[0],
                                 p_ids[1, b_id]:p_ids[1, b_id]+graph_shape[1]]
            hts_ = H_lst[str_id][p_ids[0, b_id]:p_ids[0, b_id]+graph_shape[0],
                                 p_ids[1, b_id]:p_ids[1, b_id]+graph_shape[1]]
            dsp_ = D_lst[str_id][:, p_ids[0, b_id]:p_ids[0, b_id]+graph_shape[0],
                                 p_ids[1, b_id]:p_ids[1, b_id]+graph_shape[1]]
            x_ = torch.FloatTensor((np.concatenate((np.expand_dims(rad_, axis=0), np.expand_dims(hts_, axis=0), dsp_, np.expand_dims(bnd_, axis=0)),
                                                   axis=0)).transpose(1, 2, 0).reshape(-1, 5))
            y_.append(E_lst[str_id][:, Ng * (p_ids[0, b_id] + pd_size):
                                    Ng * (p_ids[0, b_id] + pd_size + window_size),
                                    Ng * (p_ids[1, b_id] + pd_size):
                                    Ng * (p_ids[1, b_id] + pd_size + window_size)])
            g = dgl.graph((nodes_lst[0], nodes_lst[1]), device=device)
            g.ndata['h'] = x_.to(device)
            g.ndata['pos'] = pos_.to(device)
            g_lst.append(g)

        bg = dgl.batch(g_lst)
        optimizer.zero_grad()

        out_1 = graph_model.forward(bg)[mask_].reshape(batch_size, window_size**2, Ng**2, 6)
        out_2 = torch.zeros(batch_size, 6, window_size*Ng, window_size*Ng, device=device)
        for b_id in range(batch_size):
            for x_id in range(window_size):
                for y_id in range(window_size):
                    out_2[b_id, :, x_id*Ng:(x_id+1)*Ng, y_id*Ng:(y_id+1)*Ng] = out_1[b_id, x_id*window_size+y_id].T.reshape(6,Ng,Ng)
        loss = mse_loss(conv_net(out_2) , torch.FloatTensor(np.array(y_)).to(device))

        t_cost += loss.item()

        if tr_chk:
            loss.backward()
            optimizer.step()

    if not tr_chk:
        graph_model.train()

    return t_cost/100


cost_lst = []
cost_tst_lst = []

for epoch in range(epoch_lim):
    # train
    ep_cost = train_test()
    cost_lst.append(ep_cost)

    cost_file.write("%f\n" % ep_cost)
    print('iter_num = {}, epoch_cost = {}'.format(epoch, ep_cost))
    if not (epoch % 10):
        # test
        tst_cost = train_test(tr_chk=False)
        cost_tst_lst.append(tst_cost)

        tst_cost_file.write("%f\n" % tst_cost)
        print('batch_num = {}, test_cost = {}'.format(epoch, tst_cost))
        print('***********************************')

    if not (epoch % 50):
        for param_group in optimizer.param_groups:
            lr *= 0.95
            param_group['lr'] = lr

        if tst_cost <= best_cost:
            best_cost = tst_cost
            torch.save({'epoch': (epoch + 1),
                        'model_state_dict': graph_model.state_dict(),
                        'model_state_dict2': conv_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': tst_cost,
                        'cost_arr': cost_lst,
                        'test_cost': cost_tst_lst}, PATH_best)
        torch.save({'epoch': (epoch + 1),
                    'model_state_dict': graph_model.state_dict(),
                    'model_state_dict2': conv_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': tst_cost,
                    'cost_arr': cost_lst,
                    'test_cost': cost_tst_lst}, PATH_)

print('Saving Model...')
# test
tst_cost = train_test(tr_chk=False)

tst_cost_file.write("%f\n" % tst_cost)
print('batch_num = {}, test_cost = {}'.format(epoch, tst_cost))
print('***********************************')

if tst_cost <= best_cost:
    best_cost = tst_cost
    torch.save({'epoch': (epoch + 1),
                'model_state_dict': graph_model.state_dict(),
                'model_state_dict2': conv_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tst_cost,
                'cost_arr': cost_lst,
                'test_cost': cost_tst_lst}, PATH_best)
torch.save({'epoch': (epoch + 1),
            'model_state_dict': graph_model.state_dict(),
            'model_state_dict2': conv_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tst_cost,
            'cost_arr': cost_lst,
            'test_cost': cost_tst_lst}, PATH_)
