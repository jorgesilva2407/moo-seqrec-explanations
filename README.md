# seqrec: SASRec (full cross-entropy) on MovieLens-20M

PyTorch implementation of the SASRec+ recipe (Klenitskiy & Vasilev, RecSys 2023) trained with
full cross-entropy over all items, packaged in a Docker image for training on vast.ai.
Every repeatable task is a `just` recipe (`just --list`). The trained model is explained in §11
(counterfactual explanations by NSGA-II over history masks), which also says how to reproduce the
results and build the course article.

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

## 11. Explanations

`src/seqrec/explain/` holds the counterfactual-explanation part and needs the `explain` dependency
group: `instances.py` (sampling of (history, target) pairs), `backgrounds.py` (the fills, fixed per
instance), `verifier.py` (margins under necessity and sufficiency interventions, counted in SASRec
passes), `methods.py` (NSGA-II and the baselines behind one ask/tell protocol) and `scheduler.py`
(runs many jobs in shared GPU batches, enforces the budget, writes one result file per job).

```
just explain-bench                         # passes per second, GPU memory per chunk size, choice of M
just explain-run --stage main --results-repo AbyssWanderer/moo-seqrec-explanations-ml20m
just explain-run --stage main --shard 1/4 --results-repo ...       # one of 4 machines
just explain-run --stage tune-pop50 --instance-set tune --per-stratum 10 --methods nsga2 --pop-size 50
```

Both download `fp32/best.pt` from the Hub (private repo: `HF_TOKEN`). `explain-run` writes
`<out-dir>/explanations/<stage>/<checkpoint-id>/<method>/u<user>_t<target>_s<seed>.npz` (layout in
`scheduler.py`), skips jobs whose file exists, and with `--results-repo` first downloads the stage
from the dataset and uploads every 10 minutes, so an interrupted or sharded run resumes anywhere.

Background ablations: `--fill {popularity,users,padding}` chooses the fill of the search (`users` =
snippets of other users' histories, `padding` = the positions are deleted), `--eval-fill` the fill
that re-scores the fronts (default: the same), and `--instance-m` the number of popularity
backgrounds of the check that selects instances, so every stage uses the same instances.

GECE baseline: `just explain-gece --stage main` runs GECE (Scarcelli et al., arXiv 2508.03606)
through the implementation in TRACE, copied unmodified into `src/seqrec/explain/third_party/trace/`
(github.com/devUuung/TRACE, commit ab14d96, MIT). Every sequence GECE queries is mapped to the
positions it changed and those position sets are scored by the verifier, in the same file layout
(method `gece`). TRACE's GECE stops at its first valid counterfactual; `--gece-restarts` starts it
again until the pass budget is spent (method `gece_restarts`).

Unattended on vast.ai: `just vast-template sha-xxxxxxx` creates a template whose onstart command is
`vast/onstart.sh`. It runs `prepare-data` and then the recipe in the `SASREC_JOB` environment variable
(default: the benchmark), e.g. `--env '-e SASREC_JOB=explain-run+--stage+main+--shard+0/4'` on
`vastai create instance` (`+` stands for a space and `,` separates jobs: vast.ai drops values with spaces). The log is uploaded to `logs/` in the dataset when the job ends.

### Article

`report/` holds the course article (Portuguese, ACM format: `acmart` with `sigconf,nonacm`, the class
that ships with MacTeX/TeX Live, so no template files are vendored).

```
report/
  Makefile               latexmk through MacTeX (/Library/TeX/texbin); output in report/build/
  src/artigo.tex         preamble, title, abstract; inputs the sections
  src/secoes/*.tex       one file per section
  src/referencias.bib    bibliography (ACM-Reference-Format, BibTeX)
  src/figuras/*.pdf      generated figures      } written by scripts/figuras.py from results/*.json
  src/tabelas/*.tex      generated table bodies } and dados/exemplo.json; do not edit by hand
  dados/exemplo.json     the worked example (written by scripts/exemplo.py from the result files)
  scripts/               figuras.py, exemplo.py and the analyses listed under "Reproducing the results"
```

```
just report            # build report/build/artigo.pdf
just report-figures    # regenerate figures and tables from results/*.json, then build
just report-watch      # rebuild on every save
just report-check      # fail on undefined references/citations; overfull boxes and page count
just report-clean
```

Numbers quoted in the running text of `src/secoes/resultados.tex` were copied from `results/*.json`;
after re-running an experiment, regenerate the tables and re-check those sentences.

### Reproducing the results

Fixed inputs: checkpoint `fp32/best.pt` of `AbyssWanderer/sasrec-ml20m` (checkpoint id `e37b97aa9645`),
processed data with hash `806838bcc2ea…` (`just prepare-data`; `explain-run` warns if the hash differs),
instance seed 0, `K = 10`, 10^5 SASRec passes per job, fronts re-scored on 100 independent
backgrounds. Every result file is in the dataset `AbyssWanderer/moo-seqrec-explanations-ml20m`
(private), one folder per stage, so the reports can be rebuilt without re-running the searches.
Images used: `sha-973992d` (benchmark, tuning, main, scalarised GA and operator stages),
`sha-6951e5c` (background ablations), `sha-3ff8376` (GECE), `sha-79e6c1a` (seeded NSGA-II and the
scalarised GA re-run on the main instances). Add `--results-repo <dataset>` to every
`explain-run` / `explain-gece` below to resume from and upload to the dataset.

Searches (GPU; `T` stands for `--instance-set tune --per-stratum 10 --seeds 0 1 2`):

| Stage(s) | Command |
|---|---|
| benchmark | `just explain-bench --per-stratum 10` |
| `tune-p{P}-m{M}`, P in 25 50 100 200, M in 5 10 20 | `just explain-run --stage tune-pP-mM T --methods nsga2 --pop-size P --m M --instance-m M` |
| `main` | `just explain-run --stage main --pop-size 25` (optionally `--shard i/4` on four machines) |
| `sga-p{P}-d{D}`, P in 10 25 50, D in 2 3 4 | `just explain-run --stage sga-pP-dD T --methods scalar_ga --scalar-pop-size P --scalar-divisions D --m 10 --pop-size 25` |
| `op-init-uniform`, `op-no-crossover`, `op-mut0.5`, `op-mut2`, `op-mut4` | `just explain-run --stage <name> T --methods nsga2 --m 10 --pop-size 25` plus `--init uniform`, `--no-crossover` or `--mutation-scale 0.5 / 2 / 4` |
| `abl-users` | `just explain-run --stage abl-users --fill users --eval-fill popularity --m 10 --methods nsga2 scalar_ga greedy ablation --pop-size 25` |
| `abl-padding` | the same with `--stage abl-padding --fill padding --m 1 --budget 10000` (deletion has no randomness: one background, the same 5,000 masks per job) |
| `tune-seeded` | `just explain-run --stage tune-seeded T --methods nsga2_seeded --m 10 --pop-size 25` (NSGA-II seeded with the ablation ranking) |
| `main` (method `nsga2_seeded`) | `just explain-run --stage main --methods nsga2_seeded --pop-size 25` |
| `main-sga-p25-d4`, `main-sga-p10-d2` | `just explain-run --stage main-sga-pP-dD --methods scalar_ga --scalar-pop-size P --scalar-divisions D --pop-size 25` (tuned scalarised GA on the main instances) |
| `main` (methods `gece`, `gece_restarts`) | `just explain-gece --stage main` and `just explain-gece --stage main --gece-restarts` |

The tuning stages ran before `--instance-m` existed, when the instance check used `--m` backgrounds;
`--instance-m M` reproduces that. All later stages use the default of 10.

Analyses (laptop; they write the files in `results/`, which the article's tables and figures read).
`R` stands for `--results-repo AbyssWanderer/moo-seqrec-explanations-ml20m`, which downloads the stages
into `<out-dir>/explanations` first; reports are saved under `<out-dir>/explanations/reports/`.

| File in `results/` | Command |
|---|---|
| `tuning-stage.json` | `just explain-report --by stage --name tuning R --stages tune-p25-m5 … tune-p200-m20` (the 12 stages) |
| `main-method.json` and its two convergence plots (earlier snapshots, kept for reference: `main-method.json` and `main-with-gece-method.json` predate the seeded NSGA-II) | `just explain-report --stages main --by method --full --name main R --methods nsga2 scalar_ga random recency similarity ablation greedy enumeration` |
| `main-final-method.json` (the article's main table and convergence figure) | `just explain-report --stages main main-sga-p25-d4 main-sga-p10-d2 --by method --tag-stages --full --name main-final R --pairs nsga2_seeded,greedy nsga2_seeded,nsga2 nsga2,scalar_ga@main-sga-p25-d4 nsga2,scalar_ga@main-sga-p10-d2 scalar_ga@main-sga-p25-d4,scalar_ga scalar_ga@main-sga-p10-d2,scalar_ga nsga2_seeded,scalar_ga@main-sga-p10-d2 nsga2_seeded,scalar_ga@main-sga-p25-d4 greedy,nsga2 ablation,nsga2 greedy,ablation` |
| `nsga2-seeded-tuning-stage.json` | `just explain-report --by stage --full --name nsga2-seeded-tuning R --stages tune-p25-m10 tune-seeded` |
| `main-with-gece-method.json` | `just explain-report --stages main --by method --name main-with-gece R` |
| `scalar-ga-tuning-stage.json` | `just explain-report --by stage --name scalar-ga-tuning R --stages sga-p10-d2 … sga-p50-d4` (the 9 stages) |
| `nsga2-operators-stage.json` and its plots | `just explain-report --by stage --full --name nsga2-operators R --stages tune-p25-m10 op-init-uniform op-no-crossover op-mut0.5 op-mut2 op-mut4` |
| `background-ablation.json` | `uv run --group explain python report/scripts/ablacao.py <out-dir>/explanations` (needs `main`, `abl-users`, `abl-padding` downloaded) |
| `gece-returned.json` | `uv run --group explain python report/scripts/gece_retornado.py <out-dir>/explanations` |
| `exact-front.json` | `uv run --group explain python report/scripts/frente_exata.py --processed <data-dir>/processed [--device mps]` (runs the model: about 35 minutes on an M2 laptop) |
| `report/dados/exemplo.json` | `uv run --group explain python report/scripts/exemplo.py --explanations <out-dir>/explanations --processed <data-dir>/processed --zip <data-dir>/raw/ml-20m.zip` |

Hypervolumes are scaled inside each instance over the fronts being compared, so a method's value
changes with the set of methods or stages in the report (for example `main-method.json` against
`main-with-gece-method.json`); compare values only inside one file. The searches are deterministic
given the seeds on one device type; results from a different GPU or from MPS can differ in the
last digits. `exact-front.json` and the worked example were produced on a laptop (MPS).

### Round 2 (response to the review): what was run and how to reproduce it

Launch scripts for every vast.ai machine of this round are in `vast/round2/` (offer ids and template
hashes in them are specific to that day and account: search fresh offers and create a template with
`just vast-template <image tag>`). Images: `sha-b185f3d` (first machines) and `sha-d7c46c6` (adds
`six`, needed by the Kaggle CLI; everything later). Run the launchers from bash, not zsh.

**ML-20M** (results in `AbyssWanderer/moo-seqrec-explanations-ml20m`):

| Stage(s) | What |
|---|---|
| `tune-moead-{tch,pbi}-p{5,6}`, `tune-nsga3-p{5,6}`, `tune-smsemoa-p{25,50}` | held-out tuning of the new MOEAs (selection by total HV): MOEA/D Tchebicheff 5 partitions, NSGA-III 6, SMS-EMOA population 50 |
| `main` (methods `nsga2_nc`, `shapley`, `moead`, `nsga3`, `smsemoa`) | the new methods on the 99 main instances, 5 seeds |
| `redraw` | `--instance-set redraw`: 99 new instances disjoint from main and tuning users; nsga2, nsga2_seeded, greedy, ablation, scalar_ga, 3 seeds |
| `redraw-moea` | SMS-EMOA and MOEA/D on the redrawn set |
| `exact-front/exact-front-n16.json` | `just exact-front --length 16 --methods nsga2 random greedy moead nsga3 smsemoa` (GPU) |

**Zvuk** (data: Kaggle `alexxl/zvuk-dataset`, CC BY 4.0, Shevchenko et al. KDD 2024; needs
`KAGGLE_API_TOKEN`; results in `AbyssWanderer/moo-seqrec-explanations-zvuk`, models in
`AbyssWanderer/sasrec-zvuk`):

1. Preprocessing: `just prepare-data --dataset zvuk --processed-name processed_zvuk --max-items 50000
   --max-users 100000` (listens >= 30 s, first listen per user and track, 50k most-listened tracks,
   5-core, 100k users) -> 100,000 users, 50,000 items, 13,237,170 interactions, data hash
   `1943379b600c...`; shared as `processed/processed_zvuk` in the Zvuk dataset (`just fetch-processed`).
2. SASRec search (full CE, fp32, grad clip 1.0, lr 1e-3, batch 128, length 128, 1 head): round 1 =
   hidden {128, 256} x layers {2, 3} x dropout {0.1, 0.2, 0.3}, 8 epochs each (`search/<run>` in the
   model repo, with `last.pt`); round 2 = the three best by validation NDCG@10 resumed with
   `train --resume --extend-epochs 40` (patience 5) to `final/<run>`. Selected: `final/zvuk-h256-l3-d0.1`
   (best validation NDCG@10 0.2686; test NDCG@10 0.2471, HR@10 0.3344).
3. `explain-bench` on that model (`bench/` in the Zvuk dataset), then stages `main` (nsga2,
   nsga2_seeded, greedy, ablation, scalar_ga) and `main-sms` (SMS-EMOA) on 45 instances (15 per
   stratum), 3 seeds, three shards; check of robustness to the model choice: stages `alt-model` and
   `alt-model-sms` with `final/zvuk-h256-l3-d0.2` on 15 instances.

**Local analyses** (laptop; write `results/*.json`; `<ml>` / `<zv>` = local copies of the two
datasets' `explanations`, `<data>` = processed data directories):

| File in `results/` | Command |
|---|---|
| `final-ml20m-method.json` (+ convergence plots) | `just explain-report --stages main main-sga-p25-d4 main-sga-p10-d2 --by method --tag-stages --full --name final-ml20m R --pairs nsga2,smsemoa nsga2,nsga3 nsga2,moead nsga2,scalar_ga moead,scalar_ga smsemoa,scalar_ga nsga2,nsga2_nc nsga2,shapley shapley,ablation nsga2_seeded,greedy nsga2_seeded,nsga2 greedy,nsga2 smsemoa,greedy nsga2_seeded,smsemoa nsga2,scalar_ga@main-sga-p25-d4 nsga2,scalar_ga@main-sga-p10-d2 nsga2_seeded,scalar_ga@main-sga-p10-d2 smsemoa,scalar_ga@main-sga-p25-d4` |
| `final-redraw-method.json` | `just explain-report --stages redraw redraw-moea --by method --name final-redraw R --pairs nsga2_seeded,greedy nsga2_seeded,nsga2 greedy,nsga2 nsga2,smsemoa nsga2,scalar_ga moead,scalar_ga` |
| `final-zvuk-method.json` | `just explain-report --stages main main-sms --by method --full --name final-zvuk --results-repo AbyssWanderer/moo-seqrec-explanations-zvuk --pairs nsga2_seeded,greedy nsga2_seeded,nsga2 greedy,nsga2 nsga2,smsemoa nsga2,scalar_ga` |
| `final-zvuk-alt-method.json` | `just explain-report --stages alt-model alt-model-sms --by method --name final-zvuk-alt --results-repo AbyssWanderer/moo-seqrec-explanations-zvuk` |
| `new-moea-tuning-stage.json` | `just explain-report --by stage --name new-moea-tuning R --stages tune-p25-m10 tune-moead-tch-p5 tune-moead-tch-p6 tune-moead-pbi-p5 tune-moead-pbi-p6 tune-nsga3-p5 tune-nsga3-p6 tune-smsemoa-p25 tune-smsemoa-p50` |
| `nsga2-seeded-tuning-stage.json` | (round 1) `just explain-report --by stage --full --name nsga2-seeded-tuning R --stages tune-p25-m10 tune-seeded` |
| `exact-front-n16.json` | downloaded from `exact-front/` in the ML-20M dataset (written by the GPU run above) |
| `offline-checks.json` | `uv run --group explain python report/scripts/verificacoes.py <ml>` (needs `main` and `bench/`) |
| `conflict-m100.json` | `uv run --group explain python report/scripts/conflito_m100.py --processed <data>/processed --device mps` (~1 h on an M2) |
| `model-checks.json` | `uv run --group explain python report/scripts/sanidade_modelo.py --name zvuk --processed <data>/processed_zvuk --hub-repo AbyssWanderer/sasrec-zvuk --hub-file final/zvuk-h256-l3-d0.1/best.pt --device mps`, and the same with `--name ml-20m --processed <data>/processed --hub-repo AbyssWanderer/sasrec-ml20m --hub-file fp32/best.pt` (30,000 random test users) |

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

## 13. Results

SASRec runs (full ranking over the catalog; the explanations use the fp32 checkpoint):

| run | seed | epochs (best) | val HR@10 | val NDCG@10 | test HR@10 | test NDCG@10 | checkpoint id |
|-----|------|---------------|-----------|-------------|------------|--------------|---------------|
| fp32 | 42 | 49 | 0.3272 | 0.1988 | 0.2944 | 0.1806 | `e37b97aa9645` |

Explanation results are in `results/` and in the article (`report/`).
