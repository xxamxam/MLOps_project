from functools import lru_cache

import hydra
import numpy as np
from conf.config import Config
from data import load_mnist
from hydra.core.config_store import ConfigStore
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_server(img: np.array):
    triton_client = get_client()
    input_tensor = InferInput(
        name="IMAGES", shape=img.shape, datatype=np_to_triton_dtype(img.dtype)
    )
    input_tensor.set_data_from_numpy(img)

    infer_output = InferRequestedOutput("CLASS_PROBS")
    query_response = triton_client.infer(
        "onnx-model", [input_tensor], outputs=[infer_output]
    )
    rez = query_response.as_numpy("CLASS_PROBS")
    return rez


cs = ConfigStore.instance()
cs.store(name="infer_config", node=Config)


def draw_num(num):
    # num is 1x28X28 array
    num = num[0]
    for i in range(28):
        for j in range(28):
            print("*" if (num[i][j] != 0) else " ", end="")
        print()


@hydra.main(config_path="./../conf", config_name="config", version_base="1.3")
def main(cfg: Config):
    X_test, y_test = load_mnist(cfg.data, train=False)
    X_test, y_test = X_test[:32], y_test[:32]
    rez = call_triton_server(X_test)
    accuracy = np.mean(y_test == np.argmax(rez, axis=1))
    print(f"tested_numbers: {np.argmax(rez, axis = 1)}")
    print(f"model accuracy is {accuracy}")

    print("second number in dataset is 2:")
    draw_num(X_test[1])

    return


if __name__ == "__main__":
    main()
