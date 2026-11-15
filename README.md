# seqrec: SASRec (full cross-entropy) on MovieLens-20M

PyTorch implementation of the SASRec+ recipe (Klenitskiy & Vasilev, RecSys 2023) trained with
full cross-entropy over all items, packaged in a Docker image for training on vast.ai.
Every repeatable task is a `just` recipe (`just --list`). Nothing in this repository has been run
yet: treat the first runs as the real test.

Reproduction target: the paper reports ML-20M SASRec+ HR@10 about 0.298 and NDCG@10 about 0.183
(with max length 200; this project deliberately uses 128). Being close is the aim, not identical.

Layout: `src/seqrec/{paths,cli}.py`, `commands/` (one module per command), `data/`, `models/`
(`sasrec.py`, `loader.py`), `training/`, `tests/`, `justfile`, `Dockerfile`, `.github/workflows/docker.yml`.

## 1. Install just (written for just 1.40; the `[group]` attribute needs 1.27 or newer)

- macOS: `brew install just`
- any OS with Rust/cargo: `cargo install just`
- no install: `uvx rust-just --list` (use `uvx rust-just <recipe>` in place of `just <recipe>`)

`just --list` prints every recipe, grouped into `local` (your laptop) and `remote` (the GPU
machine, inside the container). Override variables on the command line, e.g.
`just tag=v0.1.0 docker-build` or `just --set run_name my-run train`.

## 2. One-time repository setup

Edit `image` and `repo_url` at the top of the `justfile` (replace `OWNER/REPO`, lowercase), then:

```
just lock                 # creates uv.lock (commit it: the Docker build uses --frozen)
git init
git add -A && git commit -m "Initial commit"
# create an empty GitHub repository (web UI, or `gh repo create OWNER/REPO --private --source .`)
git remote add origin git@github.com:OWNER/REPO.git
git branch -M main
git push -u origin main
```

`.gitignore` excludes `.venv`, caches, `data/`, `outputs/`, `workspace/`, `pulled/`, checkpoints
(`*.pt`), `*.npz` and `*.zip` (anchored at the repository root, so `src/seqrec/data/` is tracked).

## 3. Watch CI and publish the package

The `docker` workflow runs **only manually** (no push, tag or pull-request triggers): start it with
`just ci-build main` or from the Actions tab ("Run workflow"), then watch it with `gh run watch`. It builds the image and pushes
`ghcr.io/OWNER/REPO` (tags `latest` and `sha-<short>`). The package appears under the repository's
"Packages" sidebar. Either make it public (package page, Package settings, Change visibility) or
keep it private and give vast.ai registry credentials (user, plus a personal access token with
`read:packages`). `just push-image-info` prints the URLs and these steps.

## 4. Release

```
just release v0.1.0       # checks a clean tree, asks to confirm, tags and pushes
```

The tag push does not build anything. Run `just ci-build v0.1.0` to build it; that publishes
`ghcr.io/OWNER/REPO:0.1.0` and `:sha-<short>`. (`latest` is only set when the build runs on `main`.)

## 5. Create the vast.ai instance

Use the image `ghcr.io/OWNER/REPO:<tag>` as a custom template/image with:

- an NVIDIA GPU with at least 16 GB of memory (24 GB preferred);
- a host whose driver supports CUDA 12.6 or newer (the image is built on CUDA 12.6.3 with the
  `cu126` PyTorch wheels; PyTorch bundles the CUDA libraries, only the host driver matters);
- a persistent disk of at least 30 GB, with the volume mounted at `/workspace`;
- SSH launch mode.

Exact UI labels may differ; check the vast.ai documentation. To use another CUDA version change the
`pytorch-cuXXX` index name and URL in `pyproject.toml` and the `CUDA_IMAGE` tag in the `Dockerfile`
together, then `just lock`.

## 6. On the instance (over SSH)

New shells start in `/opt/seqrec` (where the `justfile` lives); otherwise `cd /opt/seqrec`.

```
just gpu-check            # nvidia-smi + torch CUDA check
just test
just prepare-data-smoke   # 2000-user subset into <data>/processed_smoke (downloads ML-20M if absent)
just train-smoke          # 2 epochs; read the seconds per epoch in the log
just prepare-data         # full data into <data>/processed
just train-bg             # detached tmux session "train"; output also in the run's console.log
just logs                 # follow the newest train.log
just attach               # attach to tmux (detach: Ctrl-b d)
just resume <run-name>    # after an interruption: continues from last.pt
```

**Upload to Hugging Face at the end of training.** Create a Hugging Face token with write access
(a fine-grained token limited to the target repo is best) and set it as the environment variable
`HF_TOKEN` of the vast.ai instance (or `export HF_TOKEN=...` in your SSH session before
`just train-bg`). Never put it in the image or the repository. Then:

```
just upload_to=YOUR_USER/sasrec-ml20m train-bg      # uploads best.pt, metrics and a model card when done
just upload_to=YOUR_USER/sasrec-ml20m resume <run>  # same, after an interruption
just upload <run-name> YOUR_USER/sasrec-ml20m       # upload an existing run by hand
just upload_folder=fp32 upload <run-name> YOUR_USER/sasrec-ml20m   # put it under fp32/ (root README untouched)
```

The published model (`AbyssWanderer/sasrec-ml20m`, private) keeps one folder per run: `bf16/` (first run, bf16
autocast, collapsed after its best epoch) and `fp32/` (stable run); each holds `best.pt`, `config.json`,
`final_metrics.json`, `metrics.csv` and `train.log`. Load one with `hf_hub_download(repo, "fp32/best.pt")`.

The repo is created private; add `upload_visibility=--upload-public` to make it public. The token is
checked before training starts, so a missing token fails immediately. If the upload itself fails, the
run stays intact on disk and `just upload` retries it. Revoke the token when you are done. The image
must include the `upload` dependency group (the justfile and CI build it by default).

To use a zip you downloaded yourself: put it at `/workspace/data/raw/ml-20m.zip`
(or run `seqrec prepare-data --zip-path <file>`). Runs live in `/workspace/outputs/runs/<run-name>/`
with `best.pt`, `last.pt`, `config.json`, `metrics.csv`, `train.log`, `final_metrics.json`.
`just clean-smoke` removes only the smoke outputs (after confirmation).

## 7. Evaluate and use the loader

```
just evaluate /workspace/outputs/runs/<run-name>/best.pt
```

This prints HR/NDCG@{1,5,10,20} on validation and test (full ranking, history items excluded) and saves
them to `<out-dir>/evaluations/`. The trainer already writes `final_metrics.json` for the best checkpoint.

To test `models/loader.py`:

```python
from seqrec.models.loader import load
scorer = load("runs/<run-name>/best.pt", device="cpu")
logits = scorer.logits(seqs)        # LongTensor[B, L<=128] -> float32 [B, N]; column j is item j+1
print(scorer.num_items, scorer.max_length, scorer.padding_id, scorer.checkpoint_id)
```

`just test` runs the save/load round-trip test (also on CPU).

## 8. Before destroying the instance

The instance disk disappears with the instance. From your laptop:

```
just pull-run <host> <ssh-port> <run-name>     # rsync to ./pulled/<run-name>/ (+ data_meta.json)
```

## 9. Rebuilding after a code change

Commit, `git push`, run `just ci-build main` and wait for the Actions run to finish, then on the instance
pull the new tag (stop and recreate the instance from the new tag; keep the `/workspace` volume).

## 10. Extending the project

To add a command (for example an explanation runner): create `src/seqrec/commands/<name>.py` with
`add_arguments(parser)` and `run(args)`, and add one line to `COMMANDS` in `cli.py`. Add a matching
recipe to the `justfile` (group `remote`, calling `{{seqrec}} <name> {{paths_flags}} ...`). The new
code may import only `seqrec.models.loader`, `seqrec.data` and `seqrec.paths` (not `training/` or
`commands/`). Put its outputs next to `runs/`, e.g. `<out-dir>/explanations/<run-name>/`, via a helper
added to `paths.py`. Add its dependencies as a separate uv group in `pyproject.toml`
(`[dependency-groups] explain = [...]`), run `just lock`, and build the image with
`--build-arg DEPENDENCY_GROUPS="dev explain"` (`just groups="dev explain" docker-build`; for CI add the
arg under `build-args` in the workflow).

## 11. Explanations (in progress)

`src/seqrec/explain/` holds the counterfactual-explanation part: `instances.py` (sampling of
(history, target) pairs), `backgrounds.py` (the fills, fixed per instance) and `verifier.py` (margins
under necessity and sufficiency interventions, counted in SASRec passes). It needs the `explain`
dependency group. Before any long run, on the GPU instance:

```
just explain-bench                      # downloads fp32/best.pt from the Hub (private repo: set HF_TOKEN)
just explain-bench --checkpoint /workspace/outputs/runs/<run-name>/best.pt
```

It prints the SASRec passes per second for several chunk sizes and how well the estimates with
M = 5, 10, 20, 40 backgrounds agree with a 200-background reference, and saves the numbers to
`<out-dir>/explanations/bench/`. The search methods and the experiment runner are not written yet.

## 12. References and credits

The model, loss and hyperparameters follow A. Klenitskiy and A. Vasilev, "Turning Dross Into Gold Loss:
is BERT4Rec really better than SASRec?" (RecSys 2023, arXiv 2309.07602), and the original SASRec by
W.-C. Kang and J. McAuley (ICDM 2018). The authors' public repository
(github.com/antklen/sasrec-bert4rec-recsys23) was consulted as a reference for preprocessing and
evaluation details; no code was copied from it.

```bibtex
@inproceedings{klenitskiy2023turning,
  title     = {Turning Dross Into Gold Loss: is {BERT4Rec} really better than {SASRec}?},
  author    = {Klenitskiy, Anton and Vasilev, Alexey},
  booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems (RecSys '23)},
  year      = {2023},
  note      = {arXiv:2309.07602}
}
@inproceedings{kang2018self,
  title     = {Self-Attentive Sequential Recommendation},
  author    = {Kang, Wang-Cheng and McAuley, Julian},
  booktitle = {2018 IEEE International Conference on Data Mining (ICDM)},
  year      = {2018}
}
@article{harper2015movielens,
  title   = {The {MovieLens} Datasets: History and Context},
  author  = {Harper, F. Maxwell and Konstan, Joseph A.},
  journal = {ACM Transactions on Interactive Intelligent Systems},
  volume  = {5},
  number  = {4},
  year    = {2015}
}
```

## 13. Results (to fill in)

| run | seed | epochs (best) | val HR@10 | val NDCG@10 | test HR@10 | test NDCG@10 | s/epoch | checkpoint id |
|-----|------|---------------|-----------|-------------|------------|--------------|---------|---------------|
|     |      |               |           |             |            |              |         |               |
