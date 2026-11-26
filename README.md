# CEBS

Core implementation of **Counterfactual Event-Binding Sanitization** for
multivariate time-series forecasting. [中文说明](README_zh.md)

CEBS uses localized input responses, recurrent supervision evidence, and trusted
calibration to produce fixed whole-window and local-label masks. These masks are
frozen before victim training and can be reused across forecasting architectures.

The code release contains the defense implementation only. It does not contain
ablation sweeps, baseline-comparison launchers, multiseed/rate/adaptive experiment
queues, cost profiling, paper drafts, or cloud-management scripts.

## Supplementary material

The [Supplementary Material (PDF)](supplementary/CEBS_Supplementary_Material.pdf)
accompanies *Counterfactual Event-Binding Sanitization for Backdoor Defense in
IoT Forecasting* and provides the algorithm details, experimental settings, and
complete supplementary results.

## Installation

Use Python 3.10 and a platform-appropriate PyTorch 2.3.0 wheel (CUDA 12.1 for GPU
execution), then install the requirements:

```bash
python -m pip install -r requirements.txt
python defense_cebs.py --help
```

FEDformer and TimesNet are included as victim models. Detection uses a
compatible infected FEDformer proxy and requires CUDA. Native PyTorch checkpoints
must come from a trusted source.

### Optional SimpleTM predictor

For a SimpleTM victim, obtain [SimpleTM from upstream](https://github.com/vsingh-group/SimpleTM)
and add its checkout root, which contains `model/` and `layers/`, to `PYTHONPATH`.
CEBS imports `model.SimpleTM.Model` when `victim` is set to `SimpleTM`. The
model settings are in `configs/models.yaml`. The tested upstream revision is
[`3c77d82`](https://github.com/vsingh-group/SimpleTM/tree/3c77d820837b726afb03c943235ea95bc924243d).

## Prepare data

See [data/README.md](data/README.md) for dataset sources and exact NPZ/checkpoint
formats. Supply observed training records, acquired trusted blocks, a clean
validation segment, and the infected proxy. These are not replaced by cached
historical masks or Oracle attack annotations.

Edit `configs/pems03_backtime.yaml` to set input paths, victim, and output location.
The model configurations are in `configs/models.yaml`. The released implementation
uses a 12-step history and 12-step horizon. Scalar normalization is shared with the
proxy and trusted inputs.

The example configuration selects support–horizon pairs for the base selector
at runtime. In the paper's PEMS03–BackTime evaluation, the base selector
uses the six pairs frozen before evaluation and listed in Section S1 of the
supplementary material.

## Run CEBS

```bash
# Read-only input checks; does not start detection or training.
python defense_cebs.py --config configs/pems03_backtime.yaml --stage check

# Compute fresh masks, then train the defended victim.
python defense_cebs.py --config configs/pems03_backtime.yaml --stage all

# Or execute the two stages separately.
python defense_cebs.py --config configs/pems03_backtime.yaml --stage detect
python defense_cebs.py --config configs/pems03_backtime.yaml --stage train
```

Use a fresh `outputs/` subdirectory for a new detection run. The training stage
checks that the data, trusted pool, proxy, dataset/period, and final mask file match
the completed detection record. The trainer uses masked Smooth L1 loss, saves
recoverable checkpoints, and selects the lowest clean-validation MAE with earliest
tie-breaking. An optional evaluation NPZ is read only after training.

In `outputs/<run>/detection/final_masks.npz`, `hard_mask` is Boolean `(N,)` and
`local_mask` is Boolean `(N,12,C)`; **True means exclude**. Local entries are cleared
on hard-deleted windows, and retained windows must have valid supervision.

To reuse an intervention on another victim, retain the same detection inputs and
output directory, change only `victim`, and run `--stage train`. Each architecture
has a separate training subdirectory. Do not regenerate detection masks separately
and call that frozen-mask reuse.

## Layout

```text
defense_cebs.py   main defense entry point
cebs/            evidence, calibration, mask composition, training, and I/O
models/          FEDformer, TimesNet, and their required layers
configs/         method/model and input-path configuration
data/            acquisition and format instructions (no raw data/weights)
supplementary/   supplementary material for the paper (PDF)
tests/           lightweight core software checks
```

The implementation is separated from the original experiment registry. Core
numerical definitions and the native training loop were retained; obsolete script
entry points and experiment-specific deployment paths are not required. Historical
paper-table replay assets and full experimental workflows are maintained separately.
CPU checks do not constitute a new GPU end-to-end reproduction.

## Checks and sources

```bash
python -m unittest discover -s tests -v
```

We acknowledge [BackTime](https://github.com/xiaolin-cs/BackTime),
[TimeGuard](https://github.com/qducnguyen/TimeGuard),
[Time-Series-Library](https://github.com/thuml/Time-Series-Library), and
[SimpleTM](https://github.com/vsingh-group/SimpleTM). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and license status.
