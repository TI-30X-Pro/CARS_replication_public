"""Entry point for `python -m scripts.decision_tree`."""
import argparse

from . import ad4che, nusc
from . import round as round_mod


SUBCOMMANDS = [
    ("nusc",   nusc),
    ("round",  round_mod),
    ("ad4che", ad4che),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.decision_tree",
        description="Adversary-selection decision-tree training",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, mod in SUBCOMMANDS:
        help_line = (mod.__doc__ or "").splitlines()[0] if mod.__doc__ else ""
        sp = sub.add_parser(name, help=help_line, description=help_line)
        mod.add_args(sp)
        sp.set_defaults(_run=mod.run)
    args = parser.parse_args()
    args._run(args)


if __name__ == "__main__":
    main()
