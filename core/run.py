"""Lazy dispatch to supported inference implementations."""
import argparse


def main(argv=None, evaluate=False):
    parser = argparse.ArgumentParser(description="Run an explicitly supported method", add_help=False)
    parser.add_argument("--method", required=True, choices=("lightvla", "vla_pruner", "specprune"))
    args, rest = parser.parse_known_args(argv)
    if args.method == "specprune":
        if not evaluate:
            parser.error("SpecPrune currently exposes experimental evaluate only; deployment run is not validated")
        from methods.specprune.evaluation import main as execute
    elif args.method == "lightvla":
        if evaluate:
            from methods.lightvla.evaluate import main as execute
        else:
            from methods.lightvla.cli import main as execute
    else:
        from methods.vla_pruner.cli import main as execute
        if evaluate:
            rest = ["--evaluate", *rest]
    return execute(rest)
