"""Upload a finished run (best.pt, metrics, model card) to the Hugging Face Hub."""

from __future__ import annotations

import argparse

from seqrec import paths
from seqrec.training import publish


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser, processed=False)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--repo-id", required=True, help="Hub repo, e.g. user/sasrec-ml20m")
    parser.add_argument("--folder", default="", help="repo folder for the run, e.g. fp32 (default: repo root)")
    parser.add_argument("--public", action="store_true", help="create the repo public (default: private)")


def run(args: argparse.Namespace) -> int:
    publish.check_repo_id(args.repo_id)
    publish.require_token()
    run_dir = paths.run_dir(paths.resolve_out_dir(args.out_dir), args.run_name)
    url = publish.upload_run(run_dir, args.repo_id, private=not args.public, folder=args.folder)
    print(f"uploaded {run_dir} to {url}")
    return 0
