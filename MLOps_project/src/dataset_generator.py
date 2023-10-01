import torch
import numpy as np
import torch.nn.functional as F

def generate(n_objects = 10, length = 100, random_seed = 0):
    np.random.seed(random_seed)
    # P = np.random.rand(n_objects, n_objects) * 1.5
    # P = torch.tensor(P)
    # P =F.softmax(P, dim = 1).numpy()  # row is probability distribution
    P = np.random.normal(loc = 1, scale= 2, size = (n_objects, n_objects))
    P = torch.tensor(P)
    P =F.softmax(P, dim = 1).numpy()  # row is probability distribution

    page = np.random.choice(range(n_objects))
    for i in range(length):
        page = np.random.choice(range(n_objects), p = P[page])
        yield page