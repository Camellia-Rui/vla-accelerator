"""Lazy dispatch to supported inference implementations."""
import argparse


def main(argv=None, evaluate=False):
    parser = argparse.ArgumentParser(description="Run an explicitly supported method", add_help=False)
    parser.add_argument("--method", required=True, choices=("lightvla", "vla_pruner"))
    args, rest = parser.parse_known_args(argv)
    if args.method == "lightvla":
        if evaluate:
            parser.error("LightVLA unified evaluation is not implemented; use its documented evaluator")
        from methods.lightvla.cli import main as execute
    else:
        from methods.vla_pruner.cli import main as execute
        if evaluate:
            rest = ["--evaluate", *rest]
    return execute(rest)
