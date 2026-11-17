# Technical document: possession labels from a network trained on other matches

Team Name: John Football

Members: Zachary Chua, Tan Chun Yee, Coen Tan

Code: folder `solution/` of this repository (guide: `solution/README.md`).
Submitted file: `evaluation/results/rawnet_v19.csv`
(SHA-256 `53234d8cdadc312cfb86607ca60520712116bf796b419867927babd5f9c8d754`).

The submitted file `john_football.csv` is `evaluation/results/rawnet_v19.csv` renamed to the team
name; the SHA-256 above applies to both.

To regenerate the submitted file (details in section 6):

```bash
uv sync --locked                                          # Python 3.11.18, packages from uv.lock
uv run --locked python solution/run_v19.py --no-score     # data bundle in data/ (see DATA.md)
uv run --locked python evaluation/validate_submission.py evaluation/results/rawnet_v19.csv
```

## 1. Summary

The system labels every graded frame of the match as `H`, `A` or `D` in two steps:

1. **A neural network** ("rawnet") reads the provided tracking (player and ball positions) and
   gives, for each frame, P(ball is dead) and P(home has it | ball is alive). It has a set
   encoder over the players and the ball of each frame, and a temporal network over the half.
2. **A switch-cost decoder** (Viterbi) turns the per-frame probabilities into possessions.

Frames outside the two halves are `X`. The file never writes `X` on a graded frame.

The network is trained only on 29 public matches from other competitions that come with real
possession labels. The model, its settings and the switch cost are chosen on outside matches
that the network did not train on. **No label of the competition match trains a model.**

| Test | spellF1 | macroF1 |
|---|---|---|
| Published 5 minutes (`evaluation/gt/`) | 0.621 | 0.725 |
| Held-out outside matches (mean of 12 cases, the selection score) | 0.693 | |

## 2. The approach, and why

### 2.1 Outside data

| Source | Matches | Tracking | Possession labels |
|---|---|---|---|
| SkillCorner open data (A-League 2024/25, MIT licence), `github.com/SkillCorner/opendata` | 20 | From the TV picture, 10 fps | `possession.group` per frame; `D` outside the phases of play |
| Metrica Sports sample data, `github.com/metrica-sports/sample-data` | 2 | Optical, 25 fps | Team of the latest on-ball event; `D` from ball-out, foul and goal events to the restart |
| DFL / Sportec open data (Bundesliga and 2. Bundesliga 2022/23, CC BY 4.0), Bassek et al., *Sci Data* 12, 195 (2025), `doi.org/10.1038/s41597-025-04505-y` | 7 | Optical, 25 fps | `BallPossession` and `BallStatus` per frame |

**Licences.** We use the outside data only to train and validate the network. None of it is in
the repository (`external_data/` is gitignored); `external.py --download` fetches it from the
original sources.

| Source | Licence | Terms that apply to us, and how we meet them |
|---|---|---|
| SkillCorner open data | MIT License (declared in the repository, `github.com/SkillCorner/opendata`) | Use, modification and redistribution allowed; the copyright and licence notice must accompany copies. We do not redistribute the data, and we credit SkillCorner here. |
| Metrica Sports sample data | No formal licence. The repository README (`github.com/metrica-sports/sample-data`, "Legal stuff") asks users to "be responsible with the use of this data" and, for any public use, to "acknowledge the source" | We use it only for training and validation, do not redistribute it, and acknowledge Metrica Sports here. |
| DFL / Sportec open data | Creative Commons Attribution 4.0 (CC BY 4.0), figshare record `doi.org/10.6084/m9.figshare.28196177` | Use, adaptation and redistribution allowed with attribution. We credit the dataset: Bassek, M., Rein, R., Weber, H. and Memmert, D., "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12, 195 (2025), and do not redistribute it. |

`solution/external.py --download` fetches all of them (4.5 GB) and converts them to one common
format. Four adjustments make them resemble the competition match (`LabelRules`, `HideRules`
and `Config` in `solution/raw_model.py`):

- **SkillCorner counts for little.** Its tracking is estimated from the broadcast picture: the
  player it names as the ball owner is a median 1.1 m from the ball. The competition match has a
  fixed camera that sees all players, like the optical sources. SkillCorner pieces are drawn a
  quarter of the time and weighted so that SkillCorner has the weight of 0.05 per frame
  against 1 for the optical sources. The weight was chosen on held-out matches with CatBoost on
  the 89 features (all sources at weight 1: 0.612; weight 0.05: 0.652) and kept for the network.
- **Frames with no tracked player are not graded in training.** 27% of SkillCorner's graded
  frames are replays with no tracking, and 98% of those are dead ball. The competition match has
  no such frame, so the network must not learn "no tracking means dead". At prediction time a
  frame with no player gets P(dead) = 0.98.
- **The DFL ball is removed in a realistic pattern.** DFL has the ball on 100% of frames; the
  competition tracking has it on 59% of graded frames and loses it mostly in stoppages. We hide
  the DFL ball through 90% of the dead-ball periods and in short periods covering 10% of live
  play, which leaves it visible on 56 to 59% of frames.
- **Long ball removal and team swap** (ideas taken from a teammate's BiGRU model on the
  `coen-baseline` branch, which is not part of this pipeline). Training pieces have the ball
  hidden for periods of 2 to 90 s, and home and away exchanged, so the network learns to work
  from player movement alone and cannot prefer one side (section 2.3).

### 2.2 Hand-made features

The network gets the raw positions (section 2.3) and also a row of 89 features computed from
player and ball positions only (`solution/ext_features.py`), by the same code for every match.
All pitches are scaled to 105 m x 68 m, the home team attacks +x in both halves, and each half
is read on a 5 frames-per-second grid so that matches recorded at 10 and 25 fps give comparable
numbers. The competition match has five such grids, which together cover every frame. The
feature groups are:

- **team shape**: centroid, spread, front and back lines, speed, and home-minus-away
  differences, also averaged over 3 s and 8 s;
- **ball**: position, distance to the nearest line, speed, whether it is seen, and the time
  since and until it is seen;
- **contact**: which team has a player within 1.5 m of the ball, votes of that over 1 to 15 s
  into the past and the future, and the team of the last and the next sustained contact;
- **ball against the team lines**: whether the ball is ahead of or behind each team.

Features that differ between sources for reasons unrelated to possession are left out (the
number of tracked players, the minute of the half).

### 2.3 The network

`solution/raw_model.py` (`RawNet`), 680,900 parameters. It works on one half at a time, on the
same 5 fps grid as the features, with the pitch oriented so that home attacks +x. It has two
stages: a **set encoder** turns each frame, an unordered set of players plus the ball, into one
vector; a **temporal network** then reads the sequence of frame vectors over the whole half.

```
 per frame (5 fps)                                                 per half
 ---------------------------------------------------------------   -------------------------
 26 player tokens (14 values) --MLP--+                             frame vectors (T x 128)
                                     +--> 2 self-attention --> pool --> 128 --(+)--> TCN, 8 blocks
 1 ball token (11 values) ----MLP----+    layers (64-d)       (321)       |    dilations 1..128
                                                                          |         |
 89 hand-made features + 40 missing flags --MLP----------------> 128 -----+    LayerNorm, Linear
                                                                                    |
                                                         logits: P(dead), P(home | alive) per frame
```

#### Inputs

Positions are scaled to [-1, 1] across the pitch (x / 52.5 - 1, y / 34 - 1). A missing value
(player not tracked, ball not seen) becomes 0, and a flag says whether the value exists.

**Player token**, 14 values, one per tracked player:

| Values | Content | Scaling |
|---|---|---|
| 2 | Position x, y | [-1, 1] |
| 2 | Velocity x, y: centred difference over 0.8 s (±2 grid steps) | clipped to ±12 m/s, / 6 |
| 1 | Team | +1 home, -1 away |
| 3 | Offset x, y to the ball and distance | / 20 m, distance capped at 60 m |
| 3 | Offset x, y and distance to where the ball was **last seen** | same |
| 1 | Is the ball seen in this frame | 0 / 1 |
| 1 | Has the ball been seen earlier in the half | 0 / 1 |
| 1 | Seconds since the ball was last seen | / 10 |

The "last seen" values let the network keep track of the ball through the 41% of graded frames
where it is missing. A frame keeps up to 26 players. The competition tracking has 48 player
slots over the match but never more than 25 tracked at once; the outside matches have 22. The
tracked players come first and the rest are padding, masked out everywhere.

**Ball token**, 11 values: position (2); velocity from a centred difference over 0.4 s,
clipped to ±40 m/s and divided by 15 (2); whether it is seen (1); seconds since and until it
is seen, / 10 (2); the last seen position (2) and the next seen position (2). The next seen
position and the time until the ball reappears look into the future. This is allowed because
the whole match is labelled offline.

**Hand-made features** (the hybrid input): the 89 features of section 2.2 for the frame,
standardised with a mean and standard deviation fitted on 12 random training halves, missing
values set to 0, clipped to ±6, plus a missing flag for each of the 40 features that can be
missing. That makes 129 values. The fitted statistics are saved with the networks
(`hybrid40_final_hybrid_stats.npz`), so prediction uses exactly the training scaling.

#### Set encoder (per frame), 118,850 parameters

1. **Embedding.** Each player token passes through a shared MLP (14 → 64 → 64, GELU), and the
   ball token through its own MLP (11 → 64 → 64).
2. **Self-attention.** The ball and the players (27 tokens) go through 2 Transformer encoder
   layers: width 64, 4 heads, feed-forward width 128, LayerNorm before each sub-layer
   (pre-norm), no dropout. Padding players are masked. Here each player sees every other
   player and the ball, so the network can learn relations such as "the nearest player to the
   ball, and how close the nearest opponent is". No player order is assumed, so the result
   does not depend on which tracking slot a player is in.
3. **Pooling** into one vector of 321 values: the ball token after attention (64); for each
   team, the mean (64) and the maximum (64) over its players; and the number of tracked
   players / 22 (1). A team with no tracked player gets mean 0 and maximum -10.
4. **Projection.** LayerNorm, then Linear 321 → 128 with GELU: the 128-value frame vector.

The hand-made features go through their own MLP (129 → 128 → 128, GELU, 33,152 parameters) and
are **added** to the frame vector. So the network starts from what the 89 features already know
and learns from the raw positions what they miss.

#### Temporal network (per half), 528,384 parameters

The frame vectors of the half go through 8 residual blocks of a non-causal dilated
convolutional network (TCN). Each block computes `x + Dropout(Conv1x1(GELU(Conv3(LayerNorm(x)))))`
with 128 channels, a kernel of 3 and dilations 1, 2, 4, …, 128 grid steps (dropout 0.1). Each
block widens the view by its dilation on each side, so every output frame sees 255 grid
steps, **about 51 s, in the past and in the future**. That is long enough to cover a whole
stoppage and the restart after it. At the edges of a half the convolutions pad with zeros.

#### Outputs and loss

A final LayerNorm and Linear 128 → 2 give two logits per frame: P(dead) and P(home | alive).
P(away | alive) is 1 - P(home | alive). The loss is the sum of two binary cross-entropies:

- **dead**: on every graded frame (not `X`) that has at least one tracked player;
- **home**: on the same frames when they are `H` or `A`.

Each term is averaged with per-piece weights (1 for optical sources; SkillCorner as in
section 2.1).

#### Training

- **Pieces.** Random 120 s pieces (600 grid steps) from random halves, drawn in proportion to
  half length; SkillCorner with probability 0.25. Each piece starts at a random offset of the
  native frames, so all frames of a 25 fps match are used. The inputs are computed with 60 s
  more on each side, so that velocities, "time since seen" and the hand-made features are
  correct at the piece edges; the network then sees the 120 s.
- **Augmentations**, drawn per piece in a fixed order:
  - 0.1 m of Gaussian noise on player positions;
  - home and away exchanged, with the pitch turned by 180 degrees and the labels `H` and `A`
    swapped (p = 0.5);
  - the pitch mirrored across its long axis (p = 0.5);
  - the ball hidden in periods of 2 to 90 s (mean 10 s) until 40% of the piece is hidden
    (p = 0.5);
  - player tracks removed: random tracks lose periods of mean 8 s until 2 to 15% of all player
    values are gone (p = 0.5). The competition tracking often misses players (about 9% of
    frames have fewer than 18), and the outside matches never do. We could not validate this
    step: no outside match has missing players (section 5).
- **Optimisation.** AdamW (learning rate 1e-3, weight decay 1e-2), one-cycle schedule (10% warm-up),
  batches of 8 pieces, 40 batches per epoch, 40 epochs (1,600 steps), gradient norm clipped
  at 1. No early stopping for the final networks; they train for the fixed 40 epochs on all 29
  matches. 3 seeds for the final networks, 2 for the fold networks of section 2.5. The seed
  sets the initial weights and every random draw above.

#### Prediction

For each half, each of the 5 grid offsets of the 25 fps match (so every native frame gets its
own prediction), each of 4 views (as tracked, teams exchanged, mirrored, both) and each of
the 3 seeds, the network runs over the **whole half** at once: 60 passes per half. P(home)
from a view with the teams exchanged is turned back (1 - p). The probabilities are averaged.
A frame with no tracked player gets P(dead) = 0.98 and P(home) = 0.5, since the network was
never trained on such frames.

The final networks are trained on an NVIDIA GPU (RTX 3060, PyTorch 2.15.0 with CUDA 13.2) with
PyTorch's deterministic mode on: deterministic kernels only, fixed cuDNN algorithms, full float32
(no TF32). All the randomness of training (pieces, augmentations, initial weights) comes from
the seed, so the same machine and package versions give bit-identical weights (section 6).
Prediction runs on the CPU (section 6).

### 2.4 Decoder

Per-frame answers flicker, and spellF1 counts possessions, so a Viterbi decoder
(`solution/decode.py`) finds the label sequence that maximises the sum of log probabilities
minus a fixed cost for every change of label. The three per-frame label costs are

- `H`: -log(1 - P(dead)) - log P(home | alive)
- `A`: -log(1 - P(dead)) - log(1 - P(home | alive))
- `D`: -log P(dead)

with probabilities kept within [1e-4, 1 - 1e-4]. The decoder runs on every native frame of each
half. The cost is 3 nats per change on the 5 fps grid
(scaled by fps / 5 when decoding every frame, so 15 nats at the 25 fps of this match). It was
the best value on the held-out outside matches; the score is flat from 1 to 4.

### 2.5 How the model and the settings were chosen

Outside matches are held out in two folds. Each fold holds out 3 SkillCorner matches and 3
optical matches (fold A: Metrica game 2, DFL J03WMX, DFL J03WN1; fold B: Metrica game 1, DFL
J03WOH, DFL J03WOY). Networks trained on 21 of the other matches score them. Two more, DFL J03WPY
and J03WQQ, are kept apart for early stopping: after each epoch the fold network is scored on
them (loss with the ball as tracked and hidden), and the best epoch is kept. Each held-out match is scored twice,
with the ball as tracked and with the ball hidden for long periods. The selection score is the
mean spellF1 of the 6 held-out optical matches in both conditions (12 cases). SkillCorner
held-out matches are reported but do not choose anything, because their tracking is less exact
than the competition's.

Network candidates, each at its best switch cost (`solution/weights/rawnet/fold_<tag>_<fold>.json`):

| Network (tag) | Epochs | Fold A | Fold B | Selection score |
|---|---|---|---|---|
| Raw positions only (`base`) | 24 | 0.619 | 0.635 | 0.627 |
| Raw positions only (`long40`) | 40 | 0.653 | 0.659 | 0.656 |
| Raw positions and the 89 features (`hybrid`) | 24 | 0.676 | 0.690 | 0.683 |
| **Raw positions and the 89 features (`hybrid40`, submitted)** | 40 | 0.685 | 0.702 | **0.693** |

The 89 features add about 0.03 to 0.06, and 40 epochs add about 0.01 to 0.03.

Other models on the same folds and selection score, built earlier; their code is not in this
repository:

| System | Fold A | Fold B | Selection score |
|---|---|---|---|
| CatBoost on the 89 features | 0.651 | 0.696 | 0.673 |
| LightGBM on the 89 features | 0.655 | 0.687 | 0.671 |
| XGBoost on the 89 features | 0.653 | 0.690 | 0.672 |
| TCN on the 89 features, no raw positions | 0.671 | 0.691 | 0.681 |
| Mean of the submitted network and CatBoost | 0.683 | 0.705 | 0.694 |

The mean with CatBoost ties with the network alone (within 0.001), so the simpler system is
submitted. No other mean of several systems did better.

The submitted network on each held-out optical match, at switch cost 3:

| Held-out match | spellF1, ball as tracked | spellF1, ball hidden |
|---|---|---|
| Metrica game 2 | 0.723 | 0.660 |
| DFL J03WMX | 0.721 | 0.677 |
| DFL J03WN1 | 0.690 | 0.637 |
| Metrica game 1 | 0.728 | 0.669 |
| DFL J03WOH | 0.750 | 0.696 |
| DFL J03WOY | 0.702 | 0.666 |

On the held-out SkillCorner matches it scores 0.46 to 0.59.

## 3. Assets used, and assets tried and dropped

**Used from the competition bundle:** the tactical-camera tracking only (player positions, team
of each slot, ball position).

One correction is applied to it (`solution/atd_fix.py`). The tracking has two goalkeeper slots,
and after a gap a slot often reappears at the other goal, following the other team's goalkeeper
with the wrong team attached. The rule is that a goalkeeper slot belongs to the team defending
the half it is in, and which team defends which goal is read from the outfield centroids. The
rule uses no labels.

**Used from outside:** the three public datasets in section 2.1, and PyTorch (open source).
CatBoost is still in `pyproject.toml` but nothing in the pipeline imports it.

**Tried and dropped.** None of these is in the submitted pipeline:

| Asset or idea | What we did with it | Why dropped |
|---|---|---|
| Smart Tagging | Set-piece and stoppage windows for dead-ball labels | Not available in the outside matches, so it could not be trained or validated |
| Broadcast recording and its tracking | Aligned to the tactical video with one offset per half; replays and close-ups as a stoppage signal, and a second ball track | Same |
| Tactical video: referee and assistants | Pose model (RTMPose via `rtmlib`) to detect a raised arm or flag | Same; also about 2.5 h of CPU and sensitive to detector differences between machines |
| Audio: referee whistle | A band-power detector and a pre-trained sound model (PANNs Cnn14) | The tactical microphone does not record the whistle above the crowd (AUC 0.61 against live play) |
| CatBoost, LightGBM, XGBoost and a TCN on the 89 features | Trained and validated like the network (section 2.5) | Lower selection score |
| `X` on unsure frames | Abstaining where the models were unsure, rule chosen on held-out outside matches | No reliable gain over the decoder |

## 4. Possession rules implemented

There are no hand-written possession rules at prediction time. The definition of possession
comes from the outside labels, one adjustment to them, and the decoder.

- **Control, not touch.** The outside labels change team at every touch (4.8 to 9.0 possessions
  per minute of live play). `CHALLENGE.md` says a touch without control is not a change of
  possession and offers "keeps the ball for a few seconds" as a sign of control. So in the
  training and validation labels, a spell shorter than 3 s that lies between two spells of the
  other team is given to the other team. The value 3 s is our reading of that text; it was not
  tuned. It changes 1.3% of the frames of a Metrica match and reduces its possessions from 328
  to 226.
- **Loose balls and challenges.** A loose or contested ball stays with the team that last
  controlled it until the other team shows sustained evidence of control. This follows from the
  3 s label adjustment, the 51 s of context that the network sees, and the switch cost. There is
  no fixed timeout.
- **Dead ball.** `D` follows the source definitions: ball out of play, or play stopped, until
  the restart is taken. The model sees this mainly through the ball leaving the pitch, the ball
  and the players slowing down, and the ball not being tracked.
- **Restarts.** Possession starts when the network says the ball is back in play; there is no
  fixed offset.
- **Unsure frames.** The file never abstains on a graded frame; the decoder keeps the current
  possession through uncertain frames.
- **Outside the halves** (before kick-off, half-time, after the final whistle): `X`.

## 5. Limitations

- **Stoppages are a weak point.** The tracking alone often cannot tell a short stoppage from
  slow play, and the ball is missing on 41% of graded frames. The video and Smart Tagging
  signals that find stoppages are not used, because we had no outside match to validate them
  on.
- **Missing players are not validated.** The competition tracking has fewer than 18 players on
  about 9% of frames; the outside matches always have 22. The network trains with player tracks
  removed, but no outside match can test whether that is enough.
- **Selection rests on six matches.** The held-out optical matches give twelve cases, so choices
  between close systems are weak. Differences under about 0.01 to 0.02 on that score are not
  meaningful. The network's gain over CatBoost (0.020) is at the edge of that.
- **Reproduction is verified on one machine only** (section 6), including from a fresh copy of
  the repository. Bit-identical retraining needs
  the same GPU model, driver and package versions; on other hardware retraining gives
  equivalent but not identical weights. The submitted file is reproduced from the saved
  weights, not by retraining.

With more time we would add outside matches that have video, so that stoppage signals from the
broadcast and the referee could be trained and validated the same way; add more optical matches
to the selection set; and build a held-out test with players removed.

## 6. Reproduction

Requirements: Python 3.11.18 and `uv`. All packages are pinned in `uv.lock`. On Windows and
Linux `uv` installs the CUDA 13.2 build of PyTorch 2.15.0 (`[tool.uv.sources]` in
`pyproject.toml`), which also runs without a GPU; on macOS it installs the PyPI build.

```bash
uv sync --locked
# put the competition data bundle in data/ or point MTAH_DATA_DIR at it (see DATA.md);
# it is not in the repository

# the submitted file, from the saved networks (a few minutes on a CPU):
uv run --locked python solution/run_v19.py --no-score
uv run --locked python evaluation/validate_submission.py evaluation/results/rawnet_v19.csv
```

This writes `evaluation/results/rawnet_v19.csv`. It reads only the tactical tracking and the
saved networks in `solution/weights/rawnet/`. Without `--no-score` it also prints the scores on
the published 5 minutes.

Prediction always runs on the CPU, even when a GPU is present, because GPU kernels (CUDA, Apple)
round differently from each other and from the CPU. The probabilities are rounded to 4 decimals
before the decoder, so that last-digit differences between processors cannot reach it. We
checked how sensitive the decoder is: random noise of up to 1e-5 added to the probabilities (far
more than the differences between processors) changed no frame of the file, with or without the
rounding.

Expected SHA-256:

| File | SHA-256 |
|---|---|
| `rawnet_v19.csv` (submitted) | `53234d8cdadc312cfb86607ca60520712116bf796b419867927babd5f9c8d754` |

The submitted file `john_football.csv` is `evaluation/results/rawnet_v19.csv` renamed to the team
name; the SHA-256 above applies to both.

We got this hash on Windows 10 (CPU prediction) from the saved networks, both in our working copy
and in a fresh copy of the repository: a clean `uv sync --locked`, no cached tracking (parsed
again from the raw bundle), and the three commands above, about 10 minutes in total. We have not
confirmed it on a second machine.

To rebuild the networks from the public data:

```bash
uv run --locked python solution/external.py --download      # 29 outside matches, 4.5 GB
# held-out folds (the selection scores of section 2.5):
uv run --locked python solution/raw_model.py fold --fold A --tag hybrid40 --set hybrid=true --epochs 40
uv run --locked python solution/raw_model.py fold --fold B --tag hybrid40 --set hybrid=true --epochs 40
uv run --locked python solution/raw_report.py --chosen hybrid40 --cost 3
# final networks on all 29 matches, about 27 min on an RTX 3060 (about 4 h on a CPU):
uv run --locked python solution/raw_model.py final --tag hybrid40 --seeds 0,1,2 --set hybrid=true --epochs 40
uv run --locked python solution/run_v19.py --no-score
```

`raw_model.py` picks the device itself (CUDA, then Apple GPU, then CPU); `RAWNET_DEVICE=cpu`
forces one. We trained the final networks twice from scratch on the same machine and got
bit-identical weights. On a different GPU model, driver or PyTorch version the rounding inside
the kernels differs, so retraining gives equivalent but not identical weights. That is why the
trained networks are in the repository, and why the submitted file is reproduced from them. The
fold results in `solution/weights/rawnet/` were produced on Apple silicon; rerunning the folds
elsewhere gives close but not identical scores.

## 7. File map

| File | Role |
|---|---|
| `solution/external.py` | Download and parse the outside matches |
| `solution/ext_features.py` | The 89 features |
| `solution/raw_model.py` | The network, outside labels and ball removal, training, held-out folds, final networks, decoder call |
| `solution/raw_report.py` | Collects the fold results into `weights/rawnet/report.json` |
| `solution/decode.py` | Viterbi decoder |
| `solution/load.py` | Reads the competition tracking |
| `solution/atd_fix.py` | Goalkeeper slot correction |
| `solution/run_v19.py` | Writes the submitted file (CPU prediction) |
| `solution/weights/rawnet/hybrid40_final_*` | The submitted networks: 3 seeds, feature standardisation, settings |
| `solution/weights/rawnet/hybrid_final_*` | Final networks of the 24-epoch `hybrid` candidate, not submitted |
| `solution/weights/rawnet/fold_*.json`, `report.json` | Fold results of the four candidates (section 2.5) |
| `solution/weights/rawnet/log_*.txt` | Training logs (`log_final_hybrid40.txt` holds every training run of the final networks, newest last) |
| `solution/README.md` | Short guide to the code |
| `games.py`, `starter/`, `evaluation/` (repository root) | Competition helpers used by the pipeline: data paths, the submission writer, the metric |
