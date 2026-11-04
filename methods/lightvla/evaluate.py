"""Unified evaluation for the audited LightVLA Spatial checkpoint/backend."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--closed-loop", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if args.closed_loop:
        from .closed_loop import main as execute
    else:
        from .evaluation import main as execute
    return execute(rest)
