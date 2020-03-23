import torch
import torch.nn as nn
import torch.nn.functional as F

class MPNN(nn.Module):
    """
        A class for a module of a message-passing graph neural network.
        Based on Gilmer et al. (2017) and Battaglia et al. (2018)

        Parameters
        ----------

        V_attributes : int
            The number of attributes associated with each vertice

        E_attributes : int
            The number of attributes associated with each edge

        edge_update_nn : torch.nn.Module
            A neural network for calculating edge updates. (a.k.a. messages)
            Should take a 1d-tensor with a length of (2 * V_attributes + E_attributes)
            Should output a 1d-tensor with a length of E_hidden
    
        vertice_update_nn : torch.nn.Module
            A neural network for calculating vertice updates. 
            Should take a 1d-tensor with a length of (V_attributes + E_hidden)
            Should output a 1d-tensor with a length of V_hidden

        output_update_nn : torch.nn.Module
            A neural network for calculating output (a.k.a. global attribute or readout)
            Should take a 1d-tensor with a length of V_hidden
            Should output a 1d-tensor with a length of V_hidden

        V_hidden : int (Optional)
            The number of features of the updated vertice attributes. Defaults to V_attributes
        E_hidden : int (Optional)
            The number of features of the updated edge attributes. Defaults to E_attributes
    """
    def __init__(self,  V_attributes, E_attributes, edge_update_nn, vertice_update_nn, output_update_nn, V_hidden=None, E_hidden=None):
        super(MPNN, self).__init__()
        self.E_attributes = E_attributes
        self.V_attributes = V_attributes

        self.edge_update_nn = edge_update_nn
        self.vertice_update_nn = vertice_update_nn
        self.output_update_nn = output_update_nn

        if V_hidden is None:
            self.V_hidden = V_attributes
        else:
            self.V_hidden = V_hidden

        if E_hidden is None:
            self.E_hidden = E_attributes
        else:
            self.E_hidden = E_hidden
    

    def forward(self, data):
        E = data[0]
        E_V = data[1]
        V = data[2]

        E_n = E.shape[0]
        V_n = V.shape[0]

        E_new = torch.zeros([E_n, self.E_hidden])
        V_new = torch.zeros([V_n, self.V_hidden])

        for i in range(E_n):
            V0 = V[E_V[i][0]]
            V1 = V[E_V[i][1]]
            E_new[i] = self.edge_update_nn.forward(torch.cat([V0, V1, E[i]]))

        V_agregated = torch.zeros(self.V_hidden)
        
        for i in range(V_n):
            #Calculating agregated edges value for a given vertice
            E_agregated = torch.zeros(self.E_hidden)
            for j in range(E_n):
                if E[j][1] == i:
                    E_agregated += E_new[j]
            V_new[i] = self.vertice_update_nn.forward(torch.cat([E_agregated, V[i]])) 
            V_agregated += V_new[i]
        
        u = self.output_update_nn.forward(V_agregated)

        return E_new, V_new, u

    def get_weights(self):
        d = {}
        d["edge_update"] = self.edge_update_nn.get_weights() 
        d["vertice_update"] = self.vertice_update_nn.get_weights()
        d["output_update"] = self.output_update_nn.get_weights()
        return d

class DoubleLayerMLP(nn.Module):
    """
    A simple double-layer perceptron class to serve as our update function in the MNPP
    """
    def __init__(self, inp, hid, out):
        super(DoubleLayerMLP, self).__init__()
        self.l1 = nn.Linear(inp, hid)
        self.l2 = nn.Linear(hid, out)

    def forward(self, x):
        return self.l2 (F.relu(self.l1(x)))

    def get_weights(self):
        d = {}
        d['w1'] = self.l1.weight.data
        d['b1'] = self.l1.bias.data
        d['w2']= self.l2.weight.data
        d['b2'] = self.l2.bias.data
        return d

