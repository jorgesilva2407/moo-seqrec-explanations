"""Training loop: full cross-entropy over all items, Adam, early stopping on validation NDCG@10."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from seqrec.data.tensors import PaddedData
from seqrec.models.loader import KEY_STATE
from seqrec.models.sasrec import SASRec, SASRecConfig
from seqrec.training.checkpoint import (
    build_checkpoint,
    capture_rng_state,
    library_versions,
    load_training_state,
    restore_rng_state,
    save_checkpoint,
)
from seqrec.training.metrics import DEFAULT_KS, evaluate_split

CSV_FIELDS = ["epoch", "train_loss", "val_ndcg@10", "val_hr@10", "lr", "epoch_seconds", "is_best"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(run_dir: Path) -> logging.Logger:
    """Logger writing to the console and ``<run_dir>/train.log`` (handlers reset on each call)."""
    logger = logging.getLogger("seqrec.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(run_dir / "train.log", mode="a"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def choose_precision(device: torch.device, mode: str = "auto") -> tuple[str, torch.dtype | None, bool]:
    """(name, autocast dtype, use GradScaler).

    mode "auto": bf16 if supported, else fp16 + scaler, CPU: fp32. mode "fp32": no autocast.
    """
    if mode not in ("auto", "fp32"):
        raise ValueError(f"precision must be 'auto' or 'fp32', got {mode!r}")
    if device.type == "cuda" and mode == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16, False
        return "fp16", torch.float16, True
    return "fp32", None, False


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_logits_fn(model: SASRec, device: torch.device, amp_dtype: torch.dtype | None):
    """Inputs long [B, L] -> float32 logits [B, N]. The encoder may run under autocast, the final
    projection is always fp32 so that low precision does not create ties in the ranking."""

    def logits_fn(inputs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            with _autocast(device, amp_dtype):
                hidden = model.encode(inputs)[:, -1]
            return model.item_logits(hidden.float())

    return logits_fn


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def fit(
    *,
    model: SASRec,
    model_config: SASRecConfig,
    data: PaddedData,
    train_args: dict,
    run_dir: Path,
    device: torch.device,
    data_hash: str,
    resume: bool,
    logger: logging.Logger,
) -> dict:
    """Train (or resume) and return the final metrics (also written to final_metrics.json)."""
    seed = int(train_args["seed"])
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_args["lr"]))
    precision, amp_dtype, use_scaler = choose_precision(device, train_args.get("precision", "auto"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logger.info("precision: %s (seed %d)", precision, seed)

    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    start_epoch, best_metric, best_epoch, bad_epochs, finished = 1, -1.0, 0, 0, False
    if resume:
        state = load_training_state(last_path)
        model.load_state_dict(state[KEY_STATE])
        optimizer.load_state_dict(state["optimizer_state"])
        if use_scaler and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        bad_epochs = int(state["bad_epochs"])
        finished = bool(state["finished"])
        restore_rng_state(state["rng_state"])
        logger.info("resumed from %s at epoch %d (best %.5f @ %d)", last_path, start_epoch, best_metric, best_epoch)

    _write_config_once(run_dir, model_config, train_args, seed, data_hash, precision)

    encoder = torch.compile(model) if train_args.get("compile") else model
    epochs, patience = int(train_args["epochs"]), int(train_args["patience"])
    batch_size = int(train_args["batch_size"])
    grad_clip = float(train_args.get("grad_clip", 0.0))  # 0 = off (also for runs saved before the option)
    val_users = _val_subset(data.num_users, int(train_args.get("val_subset", 0)), seed)
    logits_fn = make_logits_fn(model, device, amp_dtype)
    n_train = data.train.shape[0]

    csv_path = run_dir / "metrics.csv"
    new_csv = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if new_csv:
            writer.writeheader()

        def checkpoint_payload(epoch: int, done: bool) -> dict:
            return build_checkpoint(
                model, model_config, train_args, seed, data_hash,
                extra={
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if use_scaler else {},
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "finished": done,
                    "rng_state": capture_rng_state(),
                },
            )

        epoch = start_epoch - 1
        for epoch in range(start_epoch, epochs + 1):
            if finished:
                break
            t0 = _sync_time(device)
            model.train()
            gen = torch.Generator()
            gen.manual_seed(seed * 100003 + epoch)
            perm = torch.randperm(n_train, generator=gen)
            loss_sum = torch.zeros((), device=device)
            grad_norm_max = torch.zeros((), device=device)
            token_count = 0
            for start in tqdm(range(0, n_train, batch_size), desc=f"epoch {epoch}", leave=False, disable=not sys.stderr.isatty()):
                inputs, labels = data.train_batch(perm[start : start + batch_size])
                valid = labels != 0
                count = int(valid.sum().item())
                if count == 0:
                    continue
                with _autocast(device, amp_dtype):
                    hidden = encoder(inputs)  # [B, L, H]
                    logits = model.item_logits(hidden[valid])  # [M, N]
                loss = F.cross_entropy(logits.float(), labels[valid] - 1)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)  # no-op without a scaler (bf16)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    grad_norm_max = torch.maximum(grad_norm_max, norm.detach().float())
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.detach() * count
                token_count += count
            train_loss = float(loss_sum.item()) / max(token_count, 1)

            model.eval()
            val = evaluate_split(logits_fn, data, "val", DEFAULT_KS, user_index=val_users)
            seconds = _sync_time(device) - t0
            val_ndcg, val_hr = val["ndcg@10"], val["hr@10"]

            improved = val_ndcg > best_metric
            if improved:
                best_metric, best_epoch, bad_epochs = val_ndcg, epoch, 0
                save_checkpoint(best_path, build_checkpoint(
                    model, model_config, train_args, seed, data_hash,
                    extra={"epoch": epoch, "val_ndcg@10": val_ndcg, "val_hr@10": val_hr},
                ))
            else:
                bad_epochs += 1
            stop = bad_epochs >= patience
            save_checkpoint(last_path, checkpoint_payload(epoch, stop or epoch == epochs))

            writer.writerow({
                "epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_ndcg@10": f"{val_ndcg:.6f}",
                "val_hr@10": f"{val_hr:.6f}", "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": f"{seconds:.1f}", "is_best": int(improved),
            })
            csv_file.flush()
            logger.info(
                "epoch %d | loss %.4f | val NDCG@10 %.4f HR@10 %.4f | %.1fs%s%s",
                epoch, train_loss, val_ndcg, val_hr, seconds, " *" if improved else "",
                f" | max grad norm {float(grad_norm_max.item()):.2f}" if grad_clip > 0 else "",
            )
            if stop:
                logger.info("early stopping: no improvement for %d epochs (best epoch %d)", patience, best_epoch)
                break

    # Final evaluation with the best weights, fp32 end to end, all users.
    model.load_state_dict(load_training_state(best_path)[KEY_STATE])
    model.eval()
    fp32_fn = make_logits_fn(model, device, None)
    final = {
        "best_epoch": best_epoch,
        "val": evaluate_split(fp32_fn, data, "val", DEFAULT_KS),
        "test": evaluate_split(fp32_fn, data, "test", DEFAULT_KS),
    }
    _write_json(run_dir / "final_metrics.json", final)
    logger.info("final metrics: %s", json.dumps(final))
    return final


def _sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _val_subset(num_users: int, size: int, seed: int) -> torch.Tensor | None:
    if size <= 0 or size >= num_users:
        return None
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.sort(rng.choice(num_users, size=size, replace=False)))


def _write_config_once(run_dir, model_config, train_args, seed, data_hash, precision) -> None:
    path = run_dir / "config.json"
    if path.exists():
        return
    _write_json(path, {
        "model_config": model_config.to_dict(),
        "train_args": train_args,
        "seed": seed,
        "data_hash": data_hash,
        "precision": precision,
        "versions": library_versions(),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
    })
