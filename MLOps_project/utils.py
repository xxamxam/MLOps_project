import torch


def save_all(model, model_parameters, save_path, save_name):
    model_dict = model.state_dict()
    tmp_save = [model_dict, model_parameters]
    torch.save(
        tmp_save, save_path + save_name
    )  # не state_dict потому что у модели есть параметры, которые влияют на архитектуру
