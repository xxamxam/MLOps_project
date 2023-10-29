import os
import sys

import numpy as np
import torch
from conf.config import Data


def load_mnist(cfg: Data, flatten=False, train=True):
    # provide Data part of config
    """taken from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py"""
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source=cfg.sourse, path=cfg.path):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, path + filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(cfg.path + filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(cfg.path + filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(cfg.path + filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(cfg.path + filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return np.copy(data)

    # We can now download and read the training and test set images and labels.
    if train:
        X = load_mnist_images(cfg.train_data_file)
        y = load_mnist_labels(cfg.train_labels_file)
    else:
        X = load_mnist_images(cfg.test_data_file)
        y = load_mnist_labels(cfg.test_labels_file)

    if flatten:
        X = X.reshape([X.shape[0], -1])

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X, y


def get_loader(X, y, batch_size=64):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).long()
    )
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    return train_loader


def save_all(model, model_parameters, save_path, save_name):
    model_dict = model.state_dict()
    tmp_save = [model_dict, model_parameters]
    torch.save(
        tmp_save, save_path + save_name
    )  # не state_dict потому что у модели есть параметры, которые влияют на архитектуру
