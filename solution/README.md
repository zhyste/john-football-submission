# Possession labels from a network trained on other matches

This folder holds the team's solution. `TECHNICAL_DOCUMENT.md` describes the approach, the
choices and the results; this file is a short guide to the code.

The system labels every graded frame of the match as `H`, `A` or `D`. A neural network reads the
tracking and gives P(dead) and P(home | alive) for each frame, and a switch-cost decoder turns
these into possessions. The network is trained on 29 public matches from other competitions. No
label of the competition match trains a model.

## Requirements

- Python 3.11.18 and [`uv`](https://docs.astral.sh/uv/); every package is pinned in `uv.lock`.
- The competition data bundle (only the tactical tracking is read), in `data/` or wherever
  `MTAH_DATA_DIR` points (see `DATA.md`). It is not in the repository.
- No GPU is needed to regenerate the submission. Training uses an NVIDIA GPU if there is one; on
  Windows and Linux `uv` installs the CUDA build of PyTorch, which also runs on a CPU.

## Run

All commands run from the repository root.

```bash
uv sync --locked
uv run --locked python solution/run_v19.py --no-score
uv run --locked python evaluation/validate_submission.py evaluation/results/rawnet_v19.csv
```

This writes the submitted file `evaluation/results/rawnet_v19.csv` from the saved networks in
`weights/rawnet/`, in a few minutes on a CPU. Expected SHA-256:

```
53234d8cdadc312cfb86607ca60520712116bf796b419867927babd5f9c8d754
```

Prediction always runs on the CPU, so the file is the same with or without a GPU. The first run
also parses the tracking into `data/cache/`. Without `--no-score` the script prints the scores
on the published 5 minutes.

## Train again

```bash
uv run --locked python solution/external.py --download      # 29 outside matches, 4.5 GB
uv run --locked python solution/raw_model.py fold --fold A --tag hybrid40 --set hybrid=true --epochs 40
uv run --locked python solution/raw_model.py fold --fold B --tag hybrid40 --set hybrid=true --epochs 40
uv run --locked python solution/raw_report.py --chosen hybrid40 --cost 3
uv run --locked python solution/raw_model.py final --tag hybrid40 --seeds 0,1,2 --set hybrid=true --epochs 40
```

The two `fold` runs (2 seeds each) give the scores on held-out outside matches, and `raw_report.py`
collects them into `weights/rawnet/report.json`. The `final` run trains the submitted networks on
all 29 matches: about 27 min on an NVIDIA RTX 3060, about 4 h on a CPU. Training is
deterministic, so the same machine and package versions give the same weights; other hardware
gives equivalent but not identical weights, which is why the trained networks are committed.
`raw_model.py` picks CUDA, then the Apple GPU, then the CPU; `RAWNET_DEVICE=cpu` forces one.
`raw_model.py speed` times ten training steps.

## Files

| File | Role |
|---|---|
| `external.py` | Download and parse the outside matches (SkillCorner, Metrica, DFL) |
| `ext_features.py` | 89 features from player and ball positions, the same for every match |
| `raw_model.py` | The network (set encoder over players and ball, the 89 features, dilated TCN), outside labels, ball removal, training, decoder call |
| `raw_report.py` | Collects the fold results of the network into `weights/rawnet/report.json` |
| `decode.py` | Viterbi decoder with a switch cost |
| `load.py`, `atd_fix.py` | Read the competition tracking; correct the two goalkeeper slots |
| `run_v19.py` | Writes the submitted file (CPU prediction) |
| `weights/rawnet/hybrid40_final_*` | The submitted networks (3 seeds), feature standardisation and settings |
| `weights/rawnet/hybrid_final_*` | Final networks of the 24-epoch candidate, not submitted |
| `weights/rawnet/fold_*.json`, `report.json`, `log_*.txt` | Fold results and training logs |

## Outside data

29 public matches, downloaded by `external.py` into `external_data/` (gitignored, never
redistributed): SkillCorner open data (MIT License), Metrica Sports sample data (no formal
licence; the source is acknowledged as its README asks) and DFL / Sportec open data (CC BY 4.0,
Bassek et al., *Sci Data* 12, 195, 2025). Details in section 2.1 of the technical document.

## Results

| Test | spellF1 |
|---|---|
| Held-out outside matches (selection score, 12 cases) | 0.693 |
| Published 5 minutes | 0.621 |
