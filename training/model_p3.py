# the final layer has to generate 6 panels (a real and an imaginary one for each polarization)
# Imports

import torch
import torch.nn as nn


class GnnLayer(nn.Module):
    def __init__(self, in_feats, out_feats, neighbr, bias=False, activation=False):
        super(GnnLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neighbor = neighbr
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(size=(neighbr*in_feats, out_feats)))
        self.bias_bool = bias
        # define the network inside the nodes
        self.relu = nn.LeakyReLU()

        if activation:
            self.activation = activation
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.out_feats)))
                nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
        else:
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.out_feats)))
                nn.init.xavier_uniform_(self.bias)
            nn.init.xavier_uniform_(self.weight)

    def forward(self, g):
        def msg_func(edges):
            return {'m': edges.src['h'], 'src_pos': edges.src['pos']}

        def red_func(nodes):
            msgs = nodes.mailbox['m']
            if msgs.shape[1] == self.neighbor:
                tmp_ = nodes.data['pos'].unsqueeze(1).repeat_interleave(self.neighbor, dim=1) - nodes.mailbox['src_pos']
                # distance tensor
                dist_tensor = torch.norm(tmp_, dim=2, keepdim=True)
                dist_tensor[dist_tensor == 0] = 0.5
                wow = (msgs / dist_tensor.view(-1, self.neighbor, 1)).view(-1, self.neighbor * self.in_feats)
                return {'h1': torch.matmul(wow, self.weight)}
            else:
                return {'h1': torch.zeros(msgs.shape[0], self.out_feats).to(torch.device('cuda:2'))}

        def apply_func(nodes):
            h1 = nodes.data['h1']
            if self.bias_bool:
                h1 = h1 + self.bias
            if self.activation:
                h1 = self.relu(h1)
            return {'h': h1}

        g.update_all(msg_func, red_func, apply_func)


class Model(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, neighbor, num_layers=1):
        super(Model, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.neighbor = neighbor
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.build_model()

    def build_model(self):
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        for l_id in range(self.num_layers):
            h2h = self.build_hidden_layer(l_id)
            self.layers.append(h2h)
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        return GnnLayer(self.in_feats, self.h_feats, self.neighbor, bias=True)

    def build_hidden_layer(self, lay_):
        return GnnLayer(self.h_feats, self.h_feats, self.neighbor, bias=True,  activation=True)

    def build_output_layer(self):
        return GnnLayer(self.h_feats, self.out_feats, self.neighbor, bias=True)

    def forward(self, g):
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')
