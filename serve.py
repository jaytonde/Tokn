import os
import sys


def _ensure_clean_cuda_libs():
    """Remove a system CUDA toolkit from LD_LIBRARY_PATH before torch is imported.

    pip torch wheels ship their own cuBLAS/NCCL/cuDNN and locate them via RPATH.
    If a system CUDA (e.g. CUDA_HOME=/data1/cuda) is on LD_LIBRARY_PATH it shadows
    those bundled libs, which can be ABI-incompatible with this torch build and
    makes every cuBLAS GEMM fail with CUBLAS_STATUS_INVALID_VALUE.

    We strip the system CUDA lib dirs and re-exec the interpreter once so the
    cleaned environment takes effect before any CUDA library is loaded. This is
    independent of the shell/user (e.g. a root .bashrc that re-exports the path).
    """
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld or os.environ.get("_TOKN_LD_CLEANED"):
        return

    bad = set()
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root:
            bad.add(os.path.join(root, "lib64"))
            bad.add(os.path.join(root, "lib"))

    def is_bad(p: str) -> bool:
        # Drop explicit CUDA_HOME libs, or any non-wheel path that looks like a
        # system CUDA toolkit. Paths inside a pip "nvidia/.../lib" are kept.
        if p in bad:
            return True
        if "cuda" in p.lower() and os.sep + "nvidia" + os.sep not in p:
            return True
        return False

    cleaned = ":".join(p for p in ld.split(":") if p and not is_bad(p))
    if cleaned != ld:
        os.environ["LD_LIBRARY_PATH"] = cleaned
        os.environ["_TOKN_LD_CLEANED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_clean_cuda_libs()

import torch.distributed as dist
import uvicorn
from fastapi_server import build_app
from engine import Engine
from configs import ServerConfigs


def main():

    cfg = ServerConfigs().from_cli()
    print("Command Line argument parsing completed")

    device = cfg.device
    if device.startswith("cuda:"):
        gpu_index = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        device = "cuda:0"

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29502")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    engine = Engine(
        model=cfg.model,
        dtype=cfg.dtype,
        device=device
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