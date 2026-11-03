"""Run the paired check in a fresh process with this environment's cuDNN libraries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a new name")
    library_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    libraries = [library_root / "cudnn/lib", library_root / "cublas/lib", Path(sys.prefix) / "lib"]
    if not all(path.is_dir() for path in libraries):
        parser.error("Expected NVIDIA library directories are missing from this Python environment")
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-u", "-B", str(Path(__file__).with_name("server_pruner_pair.py")),
               "--output", str(output), *remaining]
    output.parent.mkdir(parents=True, exist_ok=True)
    logfile = output.with_name(output.name + ".launcher.log")
    statusfile = output.with_name(output.name + ".launcher.json")
    if statusfile.exists():
        parser.error("launcher report already exists")
    with logfile.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
        finally:
            child.stdout.close()
    with statusfile.open("x", encoding="utf-8") as stream:
        json.dump({"command": command, "exit_code": code, "log": str(logfile),
                   "LD_LIBRARY_PATH": environment["LD_LIBRARY_PATH"],
                   "note": "Nonzero exit, including native-library abort, is not a successful check"}, stream, indent=2)
    print("Launcher exit code:", code, "Log:", logfile)
    return 0 if code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
