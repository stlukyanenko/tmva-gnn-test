import torch
import torch.nn as nn
import os.path
import json 

from models import MPNN, DoubleLayerMLP 


def load_weights_mlp(model, weights_dict, name):
    model.l1.weight.data = torch.Tensor(weights_dict[name]["w1"])
    model.l1.bias.data = torch.Tensor(weights_dict[name]["b1"])
    model.l2.weight.data = torch.Tensor(weights_dict[name]["w2"])
    model.l2.bias.data = torch.Tensor(weights_dict[name]["b2"])

E_n = 10
V_n = 20
V_attributes = 3
E_attributes = 3

graph_n = 1

edge_update_nn = DoubleLayerMLP(V_attributes + V_attributes + E_attributes, E_attributes, E_attributes)
vertice_update_nn = DoubleLayerMLP(V_attributes + E_attributes, V_attributes, V_attributes)
output_update_nn = DoubleLayerMLP(V_attributes, V_attributes, V_attributes)

model = MPNN(V_attributes, E_attributes, edge_update_nn, vertice_update_nn, output_update_nn)

weights_dict = json.load(open('weights.json'))

load_weights_mlp(model.edge_update_nn, weights_dict, "edge_update")
load_weights_mlp(model.vertice_update_nn, weights_dict, "vertice_update")
load_weights_mlp(model.output_update_nn, weights_dict, "output_update")

graph_dict = json.load(open('test_graph.json'))

E = torch.Tensor(graph_dict["E"])
V = torch.Tensor(graph_dict["V"])
V_E = torch.tensor(graph_dict["V_E"], dtype=torch.int32)

data = (E,V_E,V)
print(model.forward(data))
