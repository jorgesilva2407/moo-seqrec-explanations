"""Single entry point: ``seqrec <command> ...``.

To add a command, create a module in ``seqrec/commands/`` exposing ``add_arguments(parser)``
and ``run(args)``, then add one line to ``COMMANDS`` below.
"""

from __future__ import annotations

import argparse

from seqrec.commands import evaluate, explain_bench, explain_report, explain_run, prepare_data, train, upload

COMMANDS = (
    ("prepare-data", prepare_data),
    ("train", train),
    ("evaluate", evaluate),
    ("upload", upload),
    ("explain-bench", explain_bench),
    ("explain-run", explain_run),
    ("explain-report", explain_report),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seqrec", description="SASRec on MovieLens-20M")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name, module in COMMANDS:
        summary = (module.__doc__ or "").strip().splitlines()[0] if module.__doc__ else ""
        command_parser = sub.add_parser(name, help=summary, description=summary)
        module.add_arguments(command_parser)
        command_parser.set_defaults(handler=module.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args) or 0)
