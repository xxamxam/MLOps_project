import fire
import numpy as np
import requests
from conf.config import Config
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore


cs = ConfigStore.instance()
cs.store(name="config", group="first", node=Config)


# @hydra.main(config_path="./../conf", config_name="config", version_base="1.3")
def main(servind_addr: str):
    with initialize(version_base="1.3", config_path="./../conf"):
        cfg = compose(
            config_name="config", overrides=[f"infer.inference_addr={servind_addr}"]
        )
    url = cfg.infer.inference_addr + "/invocations"
    X = np.random.rand(2, *cfg.model.onnx_parameters.input_shape)
    data = {"inputs": X.tolist()}
    headers = {"content-type": "application/json"}
    r = requests.post(url, json=data, headers=headers)
    print(r)
    print(r.text)


if __name__ == "__main__":
    fire.Fire(main)
