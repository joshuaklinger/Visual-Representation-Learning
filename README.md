# Visual-Representation-Learning
A Deep Learning Model trained on the ModelNet40 dataset to diffuse point clouds to random Gaussian noise and to reconstruct them from random Gaussian noise


Download the dataset via:
from torch_geometric.datasets import ModelNet
dataset = ModelNet(root="data/ModelNet40", name='40', train=True)
test_dataset = ModelNet(root="data/ModelNet40", name='40', train=False)
