import uvicorn
from fastapi_server import build_app
from engine import Engine
from configs import ServerConfigs


def main():

    cfg = ServerConfigs().from_cli()
    print("Command Line argument parsing completed")

    engine = Engine(
        model=cfg.model,
        dtype=cfg.dtype,
        device=cfg.device
    )
    print("Engine instantiated")

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