import os
import torch.distributed as dist
import uvicorn
from fastapi_server import build_app
from engine import Engine
from configs import ServerConfigs


def main():

    cfg = ServerConfigs().from_cli()
    print("Command Line argument parsing completed")

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    engine = Engine(
        model=cfg.model,
        dtype=cfg.dtype,
        device=cfg.device
    )
    print("Engine instantiated..")

    engine.load()

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