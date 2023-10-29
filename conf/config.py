from dataclasses import dataclass


@dataclass
class Data:
    name: str
    path: str
    sourse: str
    train_data_file: str
    train_labels_file: str
    test_data_file: str
    test_labels_file: str


@dataclass
class ModelParameters:
    k: int


@dataclass
class Model:
    name: str
    save_path: str
    save_name: str
    parameters: ModelParameters


@dataclass
class Training:
    train_part: float
    batch_size: int
    epochs: int
    optimizer: str
    device: str


@dataclass
class Infer:
    model_name: str
    model_parh: str
    infer_save_path: str
    infer_name: str
    batch_size: int


@dataclass
class Config:
    data: Data
    model: Model
    training: Training
    infer: Infer
