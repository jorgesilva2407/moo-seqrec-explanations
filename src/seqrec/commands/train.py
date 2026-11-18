"""Train SASRec on the processed data (requires CUDA)."""

from __future__ import annotations

import argparse

import torch

from seqrec import paths
from seqrec.data import load_histories, load_meta
from seqrec.data.tensors import build_padded, plan_storage
from seqrec.models.sasrec import SASRec, SASRecConfig
from seqrec.training import trainer
from seqrec.training import publish
from seqrec.training.checkpoint import load_training_state

# Arguments stored in checkpoints / config.json (everything that defines the run).
TRAIN_ARG_NAMES = (
    "seed", "epochs", "batch_size", "lr", "patience", "compile", "val_subset", "grad_clip", "precision",
    "hidden_size", "num_layers", "num_heads", "dropout", "processed_name",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser)
    parser.add_argument("--run-name", default=None, help="default: <timestamp>-seed<seed>")
    parser.add_argument("--resume", action="store_true", help="resume <run-name> from its last.pt")
    parser.add_argument("--extend-epochs", type=int, default=0,
                        help="with --resume: raise the epoch limit (continue a run that stopped at its limit)")
    parser.add_argument("--upload-last", action="store_true",
                        help="also upload last.pt, so another machine can resume the run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("auto", "fp32"), default="auto",
                        help="auto = bf16 (else fp16) autocast on CUDA; fp32 = no autocast")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="clip the global gradient norm to this value (0 = off, the default)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-subset", type=int, default=0, help="validate on N random users per epoch (0 = all)")
    parser.add_argument("--compile", action="store_true", help="use torch.compile (off by default)")
    parser.add_argument("--upload-to", default=None, metavar="REPO_ID",
                        help="after training, upload the run to this Hugging Face repo (needs HF_TOKEN)")
    parser.add_argument("--upload-folder", default="", help="repo folder for the run, e.g. fp32 (default: repo root)")
    parser.add_argument("--upload-public", action="store_true", help="create the Hub repo public (default: private)")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA is not available, refusing to train on CPU. Check the GPU with "
            "'just gpu-check' and that the host driver supports the image's CUDA version."
        )
    return torch.device("cuda", torch.cuda.current_device())


def run(args: argparse.Namespace) -> int:
    device = require_cuda()
    if args.upload_to:  # fail before hours of training, not after
        publish.check_repo_id(args.upload_to)
        publish.require_token()
    data_dir = paths.resolve_data_dir(args.data_dir)
    out_dir = paths.resolve_out_dir(args.out_dir)

    if args.resume and not args.run_name:
        raise SystemExit("ERROR: --resume needs --run-name.")
    run_name = args.run_name or paths.default_run_name(args.seed)
    run_dir = paths.run_dir(out_dir, run_name)
    last_path = run_dir / "last.pt"

    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"ERROR: nothing to resume, {last_path} does not exist.")
        saved = load_training_state(last_path)
        train_args = saved["train_args"]
        if args.extend_epochs:
            train_args["epochs"] = args.extend_epochs
        config = SASRecConfig.from_dict(saved["model_config"])
        processed = paths.processed_dir(data_dir, train_args["processed_name"])
        meta = load_meta(processed)
        if saved["data_hash"] != meta["data_hash"]:
            raise SystemExit("ERROR: processed data changed since this run started (data hash differs).")
    else:
        if last_path.exists():
            raise SystemExit(f"ERROR: {run_dir} already has a last.pt; use --resume or another --run-name.")
        processed = paths.processed_dir(data_dir, args.processed_name)
        meta = load_meta(processed)
        train_args = {name: getattr(args, name) for name in TRAIN_ARG_NAMES}
        config = SASRecConfig(
            num_items=meta["num_items"],
            max_length=meta["max_length"],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            ffn_size=args.hidden_size,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = trainer.setup_logging(run_dir)
    logger.info("run %s in %s", run_name, run_dir)
    logger.info("device: %s", torch.cuda.get_device_name(device))

    histories = load_histories(processed)
    data = build_padded(histories, config.max_length, config.num_items)
    storage, reason = plan_storage(data.nbytes(), device)
    logger.info("data placement: %s", reason)
    data = data.to_storage(storage, device)

    torch.manual_seed(int(train_args["seed"]))  # model init is reproducible from the seed
    model = SASRec(config)
    logger.info("parameters: %d", sum(p.numel() for p in model.parameters()))
    trainer.fit(
        model=model,
        model_config=config,
        data=data,
        train_args=train_args,
        run_dir=run_dir,
        device=device,
        data_hash=meta["data_hash"],
        resume=args.resume,
        logger=logger,
    )
    if args.upload_to:
        try:
            url = publish.upload_run(run_dir, args.upload_to, private=not args.upload_public, folder=args.upload_folder,
                                     include_last=args.upload_last)
            logger.info("uploaded run to %s", url)
        except (Exception, SystemExit):  # training result is safe on disk; report loudly
            logger.exception("UPLOAD FAILED (the run is intact in %s; retry with `just upload`)", run_dir)
            return 1
    return 0
