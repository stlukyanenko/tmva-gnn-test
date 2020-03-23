from models import MPNN, DoubleLayerMLP
from mao import MAOdataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path


class MultiMNNP_MOA(nn.Module):
    """
        A class for a complex a message-passing graph neural network for MOA classification.
        Based on the generalization of  Gilmer et al. (2017) proposed by Battaglia et al. (2018)
        
        Extensibely uses and besed on the MNNP module from ./model.py
        
    """
    def __init__(self):
        super(MultiMNNP_MOA, self).__init__()
        V_attributes = 6
        E_attributes = 6

        edge_update_nn = DoubleLayerMLP(V_attributes + V_attributes + E_attributes, 64, E_attributes)
        vertice_update_nn = DoubleLayerMLP(V_attributes + E_attributes, 32, V_attributes)
        output_update_nn = DoubleLayerMLP(V_attributes, 32, 1)

        self.l1 = MPNN(V_attributes, E_attributes, edge_update_nn, vertice_update_nn, output_update_nn)
        self.l2 = MPNN(V_attributes, E_attributes, edge_update_nn, vertice_update_nn, output_update_nn)
        self.l3 = MPNN(V_attributes, E_attributes, edge_update_nn, vertice_update_nn, output_update_nn)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, data):
        _, V_new, _ = self.l1.forward(data)
        _, V_new, _ = self.l2.forward((V_new, data[1], data[2]))
        E, V_new, u = self.l3.forward((V_new, data[1], data[2]))
        
        return self.sigmoid(u)

if __name__ == '__main__':
    # Our model is simple enough to train it on CPU 
    # And also I overused my colab and now cannot connect to a gpu there

    device = torch.device('cpu')
    dataset = MAOdataset(os.path.dirname(os.path.abspath(__file__)) + '/MAO/')
    
    train_set, test_set = torch.utils.data.random_split(dataset, [dataset.__len__() - dataset.__len__()//4 , dataset.__len__()//4 ])

    #dataloaders = {}
    #dataloaders['train'] = torch.utils.data.DataLoader(train_set, shuffle=True)
    #dataloaders['test'] = torch.utils.data.DataLoader(test_set, shuffle=True)

    model = MultiMNNP_MOA()

    model.to(device)
        
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01, weight_decay=1e-3)

    model.train()
    for epoch in range(100):
        print ("Epoch: " + str(epoch) )
        for x, y in train_set:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = F.binary_cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            
    model.eval()

    correct = 0
    for x, y in train_set:
        y_pred = model(x) > 0.5
        if y_pred == y:
            correct += 1
    print ("Acc: " + str(correct/len(train_set)))



