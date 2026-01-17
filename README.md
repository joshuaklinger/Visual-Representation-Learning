# Visual-Representation-Learning: Learning protein colocalization with Diffusion Models
A Deep Learning Model trained on the ModelNet40 dataset to diffuse point clouds to random Gaussian noise and to reconstruct them from random Gaussian noise


This project implements a diffusion-based generative model for 3D point clouds that combines:

* PointNet Encoder: Extracts compact latent representations from unordered point sets

* Diffusion Process: State-of-the-art generative modeling with gradual noise addition/removal

* Autoencoder Framework: Learns meaningful representations for efficient point cloud generation

* Trained on the ModelNet40 dataset, the model can generate diverse 3D shapes including airplanes, chairs, sofas, vases, and more.



Download the dataset via:

```
pip install -r requirements.txt
```

### Download ModelNet40
It can take 20 minutes using 

```python
from torch_geometric.datasets import ModelNet
dataset = ModelNet(root="data/ModelNet40", name='40', train=True)
test_dataset = ModelNet(root="data/ModelNet40", name='40', train=False)
```

### URL to start TensorBoard
```
http://localhost:6006
```
