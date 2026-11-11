"""Prepare a child process without mutating the user's installed environment."""
import os
from pathlib import Path
import sysconfig
import sys


def runtime_environment(cache_dir, gpu=None, prefer_env_cuda=False):
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    locations = {"HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
                 "HF_MODULES_CACHE": "huggingface/modules", "TRANSFORMERS_CACHE": "huggingface/hub",
                 "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch",
                 "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton", "NUMBA_CACHE_DIR": "numba",
                 "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp"}
    for name, relative in locations.items():
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OPENVLA_SKIP_CHECKPOINT_SYNC="1", TF_FORCE_GPU_ALLOW_GROWTH="true")
    if gpu is not None:
        if not str(gpu).isdigit():
            raise ValueError("gpu must be a single physical GPU index")
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if prefer_env_cuda:
        if os.name != "posix":
            raise ValueError("--prefer-env-cuda currently supports Linux only")
        nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
        libraries = [nvidia / "cudnn/lib", nvidia / "cublas/lib", Path(sys.prefix) / "lib"]
        if not all(p.is_dir() for p in libraries):
            raise ValueError("Current Python environment lacks expected NVIDIA libraries")
        env.pop("LD_PRELOAD", None)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries))
    return env
