import os
import sys

import torch.distributed as dist
import uvicorn
from fastapi_server import build_app
from src.engine import Engine
from utils.configs import ServerConfigs


def main():

    cfg = ServerConfigs().from_cli()
    print("Command Line argument parsing completed")

    device = cfg.device
    if device.startswith("cuda:"):
        gpu_index = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        device = "cuda:0"
        cfg.device = device

    engine = Engine(
        model=cfg.model,
        dtype=cfg.dtype,
        device=device,
        config=cfg,
    )
    print("Engine instantiated..")

    app = build_app(engine)

    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        workers=1
    )
    print("Uvicorn sever is ready for serving")


if __name__ == "__main__":
    main()