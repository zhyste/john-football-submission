"""raw_model.py -- rawnet: an end-to-end network on the RAW tracking (self-contained rewrite).

    uv run python solution/raw_model.py fold  --fold A --tag base
    uv run python solution/raw_model.py final --tag hybrid40 --seeds 0,1,2 --set hybrid=true --epochs 40
    uv run python solution/raw_model.py speed

    Config            all hyper-parameters (one frozen dataclass; `--set key=json` overrides)
    labels / hiding   challenge_labels, lose_ball, hidden_mask, swap_labels, validation_hide
    Half              one oriented half on native frames (home attacks +x, 105 x 68 m)
    encode()          grid arrays -> Inputs(player tokens, ball token, optional v17 feature row)
    HybridStats       standardisation of the v17 feature row (fit / transform / save / load)
    RawNet            set encoder per frame -> dilated TCN -> (P(dead), P(home | alive))
    Augmenter         train-time augmentation (jitter, team swap, y mirror, ball hide, dropout)
    CropSampler       random 120 s training crops of the outside matches
    Trainer           trains one model
    Predictor         test-time prediction (all grid offsets x 4 views x seeds)
    decode / score    Viterbi decoding and spellF1 scoring
    run_fold / run_final   the two experiments; main() is a thin CLI over them

Outside dependencies that remain: external.py (the outside matches), ext_features.py (the v17
feature table and time helpers), decode.py (Viterbi), possession_metrics.py (scoring), and --
only for the competition match -- games.py and load.py.

Nothing of the competition match is read except its tracking (competition_match()); its
probabilities are saved, never scored.

Model, on the 5 fps grid of ext_features (105 x 68, home attacks +x):
  * one token per tracked player [x, y, vx, vy, team, ball dx/dy/dist, last-seen-ball dx/dy/dist,
    ball-seen flag] and one ball token [x, y, vx, vy, visible, secs since/until seen, last and next
    seen position];
  * a set encoder per frame (shared MLP -> self-attention layers with a padding mask -> pooling:
    ball token, mean and max per team) -> a frame embedding;
  * optional (hybrid): the v17 feature row of the frame (standardised, NaN -> 0 + missing flag)
    is projected and added to the frame embedding;
  * a non-causal dilated TCN over the half (receptive field +/- 51 s);
  * two heads: P(dead) on all graded frames and P(home | alive) on H/A frames.
Training: random 120 s crops; team swap (labels H<->A), y mirror, ball removal, player track
dropout (the competition match often misses players, the outside matches never do), jitter.
Prediction: all grid offsets, 4 views (as is / team swap / y mirror / both), mean over seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import external  # noqa: E402  the outside matches (labelled)
import ext_features as ef  # noqa: E402  v17 feature table + time helpers
import possession_metrics as pm  # noqa: E402  spellF1
from decode import costs as viterbi_costs, viterbi  # noqa: E402

# Deterministic GPU maths: the same machine, driver and library versions give the same weights
# run after run. cuBLAS reads this variable when it starts, so it is set before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(3)
torch.use_deterministic_algorithms(True)       # an op without a deterministic version raises
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False  # full float32 on the GPU, as on the CPU
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
H, A, D, X = external.H, external.A, external.D, external.X   # label codes (same as pm)
L, W = ef.LENGTH, ef.WIDTH                                    # 105 x 68 m
K_PLAYERS = 26                 # player tokens kept per frame (valid first); outside matches: 22
FP, FB = 14, 11                # player / ball token sizes
NO_PLAYERS_P_DEAD = 0.98       # P(dead) of a frame with no tracked player (SkillCorner replays)

PROBS_DIR = os.path.join(ROOT, "external_data", "probs", "rawnet")
WEIGHTS_DIR = os.path.join(HERE, "weights", "rawnet")

FOLDS = {
    "A": ["skillcorner_2016236", "skillcorner_2017461", "skillcorner_1874553",
          "metrica_game2", "dfl_J03WMX", "dfl_J03WN1"],
    "B": ["skillcorner_2016236", "skillcorner_2017461", "skillcorner_1874553",
          "metrica_game1", "dfl_J03WOH", "dfl_J03WOY"],
}
EARLY = ["dfl_J03WPY", "dfl_J03WQQ"]   # early stopping: train side of both folds, never held out
SELECT_ON = ("metrica", "dfl")         # the switch cost / config is chosen on optical matches only


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # optimisation
    epochs: int = 24
    crops_per_epoch: int = 320
    batch: int = 8
    lr: float = 1e-3
    wd: float = 1e-2
    restore_best: bool = True          # with early stopping: keep the best epoch
    # crops
    crop_s: int = 120
    margin_s: int = 60
    # model
    d_tok: int = 64
    d_frame: int = 128
    n_attn: int = 2
    tcn_dil: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.1
    hybrid: bool = False               # add the v17 feature row to the frame embedding
    # SkillCorner (TV tracking, less exact): sampled less, loss-weighted to match v17's weight
    sc_weight: float = 0.05
    sc_sample: float = 0.25
    # augmentation
    p_hide: float = 0.5                # remove the ball for long periods
    p_drop: float = 0.5                # drop player tracks (competition match misses players)
    jitter_m: float = 0.1
    # decoder
    switch_costs: tuple = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)

    @classmethod
    def from_dict(cls, data) -> "Config":
        """Build from a (possibly older / JSON) dict; unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in dict(data).items() if k in known}
        if "tcn_dil" in kw:
            kw["tcn_dil"] = tuple(int(i) for i in kw["tcn_dil"])
        if "switch_costs" in kw:
            kw["switch_costs"] = tuple(float(c) for c in kw["switch_costs"])
        return cls(**kw)

    @classmethod
    def coerce(cls, cfg) -> "Config":
        return cfg if isinstance(cfg, Config) else cls.from_dict(cfg)

    def with_overrides(self, **kw) -> "Config":
        unknown = set(kw) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError("unknown config keys: %s" % sorted(unknown))
        return Config.from_dict(dict(asdict(self), **kw))

    def to_dict(self) -> dict:
        return asdict(self)


def device() -> torch.device:
    if os.environ.get("RAWNET_DEVICE"):
        return torch.device(os.environ["RAWNET_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------
# Labels and ball hiding (formerly ext_model.py)
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelRules:
    # CHALLENGE.md: a touch without control is not a change of possession. An H/A spell shorter
    # than this, between two spells of the other team, goes to the other team.
    control_min_s: float = 3.0
    # sources whose ball track is perfect; the ball is removed as a real tracker loses it
    perfect_ball_sources: tuple = ("dfl",)
    dead_lost: float = 0.9             # chance that a dead-ball period loses the ball
    live_lost: float = 0.1             # share of live frames that lose the ball


@dataclass(frozen=True)
class HideRules:
    share: float = 0.4                 # share of a half whose ball is hidden
    mean_s: float = 10.0               # mean length of a hidden period
    range_s: tuple = (2.0, 90.0)


LABEL_RULES = LabelRules()
HIDE_RULES = HideRules()


def challenge_labels(label: np.ndarray, fps: float, min_s: Optional[float]) -> np.ndarray:
    """Remove short touches: an H or A spell shorter than `min_s` with the other team directly
    before and after it (no D between) goes to the other team. Left to right, so A-H-A-H-A with
    two short H spells becomes one A spell."""
    if not min_s:
        return label
    label = label.copy()
    limit = min_s * fps
    breaks = np.flatnonzero(label[1:] != label[:-1]) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, len(label)]
    for i in range(1, len(starts) - 1):
        first, stop = starts[i], stops[i]
        value = label[first]
        if value not in (H, A) or stop - first >= limit:
            continue
        other = A if value == H else H
        if label[first - 1] == other and label[stop] == other:
            label[first:stop] = other
    return label


def lose_ball(match: dict, rules: LabelRules = LABEL_RULES) -> dict:
    """Copy of a match with a perfect ball track, with the ball removed as a tracker loses it:
    dead-ball periods lose it (0-2 s after their start to 0-2 s before their end), live play in
    short periods (mean 2 s) on `live_lost` of its frames. Seeded by the match name."""
    rng = np.random.default_rng(sum(map(ord, match["name"])) + 17)
    label, fps = match["label"], match["fps"]
    lost = np.zeros(len(label), dtype=bool)
    edges = np.diff(np.concatenate(([0], (label == D).astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a = start + int(rng.uniform(0, 2) * fps)
        b = stop - int(rng.uniform(0, 2) * fps)
        if rng.random() < rules.dead_lost and b > a:
            lost[a:b] = True
    alive = np.flatnonzero((label == H) | (label == A))
    target = rules.live_lost * len(alive)
    live_lost = np.zeros(len(label), dtype=bool)
    while (live_lost & (label != D)).sum() < target:
        start = int(alive[rng.integers(len(alive))])
        live_lost[start:start + max(int(rng.exponential(2.0) * fps), fps // 5)] = True
    lost |= live_lost & (label != D)
    return dict(match, bx=np.where(lost, np.nan, match["bx"]).astype(np.float32),
                by=np.where(lost, np.nan, match["by"]).astype(np.float32))


_MATCH_CACHE: dict = {}


def load_match(name: str) -> dict:
    """An outside match in the common format of external.py, with challenge labels and, for
    perfect-ball sources, the ball removed as a tracker loses it. Cached."""
    if name not in _MATCH_CACHE:
        match = external.load(name)
        if name.split("_")[0] in LABEL_RULES.perfect_ball_sources:
            match = lose_ball(match)
        _MATCH_CACHE[name] = dict(match, label=challenge_labels(
            match["label"], match["fps"], LABEL_RULES.control_min_s))
    return _MATCH_CACHE[name]


def competition_match() -> dict:
    """The competition match in the same format, tracking only (no label)."""
    from games import GAME, GAMES, half_video_blocks
    from load import loadTracking
    tracking = loadTracking()
    team = tracking["team_of_slot"]
    return dict(name=GAME, fps=GAMES[GAME]["fps"], pitch=GAMES[GAME]["pitch"],
                px=tracking["players_x"], py=tracking["players_y"],
                is_home=team == 0, is_away=team == 1,
                bx=tracking["ball_x"], by=tracking["ball_y"],
                halves=[(int(h), int(a), int(b)) for h, a, b in half_video_blocks(GAME)])


def hidden_mask(n: int, rng: np.random.Generator, rules: HideRules = HIDE_RULES) -> np.ndarray:
    """Random periods of a half (n grid frames) where the ball is removed."""
    hidden = np.zeros(n, dtype=bool)
    low, high = (ef.seconds(s) for s in rules.range_s)
    while hidden.mean() < rules.share:
        length = int(np.clip(rng.exponential(rules.mean_s * ef.MODEL_FPS), low, high))
        start = int(rng.integers(0, n))
        hidden[start:start + length] = True
    return hidden


def validation_hide(match: dict) -> np.ndarray:
    """A fixed ball removal for one held-out match (seeded by its name), native frames."""
    rng = np.random.default_rng(sum(map(ord, match["name"])))
    hide = np.zeros(len(match["bx"]), dtype=bool)
    step = match["fps"] // ef.MODEL_FPS
    for _, first, last in match["halves"]:
        grid = hidden_mask((last - first + 1) // step + 1, rng)
        hide[first:last + 1] = np.repeat(grid, step)[:last - first + 1]
    return hide


def swap_labels(label: np.ndarray) -> np.ndarray:
    return np.where(label == H, A, np.where(label == A, H, label)).astype(np.int8)


# ---------------------------------------------------------------------------------------------
# One oriented half
# ---------------------------------------------------------------------------------------------
@dataclass
class Half:
    """Native frames first..last of one half, scaled to 105 x 68 m and turned so that home
    attacks +x. px, py: (frames, players); bx, by: (frames,); NaN = not tracked / not seen."""
    px: np.ndarray
    py: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    home: np.ndarray               # (players,) True for home players
    step: int                      # native frames per 5 fps grid step
    first: int
    last: int
    label: Optional[np.ndarray] = None
    name: str = ""

    def __len__(self) -> int:
        return len(self.px)

    @property
    def source(self) -> str:
        return self.name.split("_")[0]

    @classmethod
    def from_match(cls, match: dict, first: int, last: int, name: str = "") -> "Half":
        length, width = match["pitch"]
        home = np.asarray(match["is_home"], bool)
        away = np.asarray(match.get("is_away", ~home), bool)
        cols = home | away
        rows = slice(first, last + 1)
        px = match["px"][rows][:, cols].astype(np.float32) * (L / length)
        py = match["py"][rows][:, cols].astype(np.float32) * (W / width)
        bx = match["bx"][rows].astype(np.float32) * (L / length)
        by = match["by"][rows].astype(np.float32) * (W / width)
        home, away = home[cols], away[cols]
        step = match["fps"] // ef.MODEL_FPS
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            attacks_right = np.nanmean(px[::step][:, home]) < np.nanmean(px[::step][:, away])
        if not attacks_right:
            px, py, bx, by = L - px, W - py, L - bx, W - by
        label = match["label"][rows] if "label" in match else None
        return cls(px, py, bx, by, home, step, first, last, label, name or match.get("name", ""))

    @classmethod
    def all_of(cls, match: dict, name: str = "") -> list:
        return [cls.from_match(match, first, last, name) for _, first, last in match["halves"]]


# ---------------------------------------------------------------------------------------------
# Inputs: tokens + (optional) the v17 feature row
# ---------------------------------------------------------------------------------------------
class Inputs(NamedTuple):
    ptok: np.ndarray               # (T, K, FP) player tokens
    pmask: np.ndarray              # (T, K) True where the token is a tracked player
    ball: np.ndarray               # (T, FB) ball token
    has: np.ndarray                # (T,) frame has at least one tracked player
    hyb: Optional[np.ndarray]      # (T, n_hybrid) v17 feature row, or None


def _centred(v: np.ndarray, hw: int) -> np.ndarray:
    """(v[t+hw] - v[t-hw]) / (2 hw / 5 fps); NaN at the edges or where missing."""
    out = np.full(v.shape, np.nan, np.float32)
    if len(v) > 2 * hw:
        out[hw:-hw] = (v[2 * hw:] - v[:-2 * hw]) * ef.MODEL_FPS / (2 * hw)
    return out


def _tokens(px, py, bx, by, home):
    """Grid arrays (T, S) / (T,) -> player tokens, mask, ball token, has_players."""
    T = len(px)
    valid = ~np.isnan(px)
    vx, vy = _centred(px, 2), _centred(py, 2)
    team = np.where(home, 1.0, -1.0).astype(np.float32)[None, :].repeat(T, 0)
    seen = ~np.isnan(bx)
    secs_since = ef.seconds_since(seen).astype(np.float32)
    secs_until = ef.seconds_until(seen).astype(np.float32)
    idx = np.arange(T)
    last_i = np.maximum.accumulate(np.where(seen, idx, -1))
    next_i = np.minimum.accumulate(np.where(seen, idx, T)[::-1])[::-1]
    has_last, has_next = last_i >= 0, next_i < T
    lbx = np.where(has_last, bx[np.clip(last_i, 0, T - 1)], np.nan)
    lby = np.where(has_last, by[np.clip(last_i, 0, T - 1)], np.nan)
    nbx = np.where(has_next, bx[np.clip(next_i, 0, T - 1)], np.nan)
    nby = np.where(has_next, by[np.clip(next_i, 0, T - 1)], np.nan)
    bvx, bvy = _centred(bx, 1), _centred(by, 1)

    def z(a):
        return np.nan_to_num(a, nan=0.0).astype(np.float32)

    dx, dy = px - bx[:, None], py - by[:, None]
    ldx, ldy = px - lbx[:, None], py - lby[:, None]
    feats = np.stack([
        z((px - L / 2) / (L / 2)), z((py - W / 2) / (W / 2)),
        z(np.clip(vx, -12, 12) / 6), z(np.clip(vy, -12, 12) / 6), team,
        z(dx / 20), z(dy / 20), z(np.minimum(np.hypot(dx, dy), 60) / 20),
        z(ldx / 20), z(ldy / 20), z(np.minimum(np.hypot(ldx, ldy), 60) / 20),
        np.broadcast_to(seen[:, None], px.shape).astype(np.float32),
        np.broadcast_to(has_last[:, None], px.shape).astype(np.float32),
        np.broadcast_to((secs_since[:, None] / 10), px.shape).astype(np.float32),
    ], axis=-1)                                                        # (T, S, FP)
    order = np.argsort(~valid, axis=1, kind="stable")[:, :K_PLAYERS]   # valid players first
    S = px.shape[1]
    if S < K_PLAYERS:
        pad = K_PLAYERS - S
        feats = np.concatenate([feats, np.zeros((T, pad, FP), np.float32)], 1)
        valid = np.concatenate([valid, np.zeros((T, pad), bool)], 1)
        order = np.concatenate([order, np.arange(S, K_PLAYERS)[None].repeat(T, 0)], 1)
    ptok = np.take_along_axis(feats, order[:, :, None], 1)
    pmask = np.take_along_axis(valid, order, 1)
    ptok[~pmask] = 0.0
    ball = np.stack([
        z((bx - L / 2) / (L / 2)), z((by - W / 2) / (W / 2)),
        z(np.clip(bvx, -40, 40) / 15), z(np.clip(bvy, -40, 40) / 15), seen.astype(np.float32),
        secs_since / 10, secs_until / 10,
        z((lbx - L / 2) / (L / 2)), z((lby - W / 2) / (W / 2)),
        z((nbx - L / 2) / (L / 2)), z((nby - W / 2) / (W / 2)),
    ], axis=-1).astype(np.float32)
    return ptok.astype(np.float32), pmask, ball, valid.any(1)


class _GridHalf:
    """Duck-typed ef.Half built from oriented grid arrays (what ef.feature_table reads)."""

    def __init__(self, px, py, bx, by, home):
        self.px, self.py = px.astype(np.float64), py.astype(np.float64)
        self.bx, self.by = bx.astype(np.float64), by.astype(np.float64)
        self.home, self.away = home, ~home


def v17_table(px, py, bx, by, home) -> np.ndarray:
    """The version 17 feature table of one grid half, float32 (T, n_features), with NaNs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ef.feature_table(_GridHalf(px, py, bx, by, home)).values.astype(np.float32)


@dataclass
class HybridStats:
    """Standardisation of the v17 feature row: (x - mean) / std, NaN -> 0, plus a missing flag for
    the columns that had NaNs in the fitting sample. Fitted on training halves only."""
    mean: np.ndarray
    std: np.ndarray
    miss_cols: np.ndarray

    @property
    def width(self) -> int:
        return len(self.mean) + len(self.miss_cols)

    @classmethod
    def fit(cls, halves: Sequence[Half], rng: np.random.Generator, n_halves: int = 12):
        tables = []
        for _ in range(n_halves):
            h = halves[int(rng.integers(len(halves)))]
            rows = np.arange(0, len(h), h.step)
            tables.append(v17_table(h.px[rows], h.py[rows], h.bx[rows], h.by[rows], h.home))
        t = np.concatenate(tables)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = np.nanmean(t, 0), np.nanstd(t, 0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        return cls(np.nan_to_num(mean).astype(np.float32), std.astype(np.float32),
                   np.flatnonzero(np.isnan(t).any(0)))

    def transform(self, table: np.ndarray) -> np.ndarray:
        miss = np.isnan(table)
        std = np.clip(np.nan_to_num((table - self.mean) / self.std, nan=0.0), -6, 6)
        return np.concatenate([std, miss[:, self.miss_cols].astype(np.float32)], 1)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std, miss_cols=self.miss_cols)

    @classmethod
    def load(cls, path: str) -> "HybridStats":
        return cls.coerce(dict(np.load(path)))

    @classmethod
    def coerce(cls, stats) -> Optional["HybridStats"]:
        """Accept None, a HybridStats, or the dict that np.load of a saved file gives."""
        if stats is None or isinstance(stats, cls):
            return stats
        return cls(np.asarray(stats["mean"]), np.asarray(stats["std"]),
                   np.asarray(stats["miss_cols"]))


def encode(px, py, bx, by, home, stats: Optional[HybridStats] = None) -> Inputs:
    """Oriented grid arrays -> network inputs (the hybrid row only when `stats` is given)."""
    ptok, pmask, ball, has = _tokens(px, py, bx, by, home)
    hyb = None if stats is None else stats.transform(v17_table(px, py, bx, by, home))
    return Inputs(ptok, pmask, ball, has, hyb)


# ---------------------------------------------------------------------------------------------
# Network (parameter names are part of the saved-weight format: do not rename)
# ---------------------------------------------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, d, dil, p):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.mix = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):                       # (B, T, d)
        h = self.norm(x).transpose(1, 2)
        h = self.mix(F.gelu(self.conv(h))).transpose(1, 2)
        return x + self.drop(h)


class RawNet(nn.Module):
    def __init__(self, cfg, n_hybrid=0):
        super().__init__()
        cfg = Config.coerce(cfg)
        d, df = cfg.d_tok, cfg.d_frame
        self.pemb = nn.Sequential(nn.Linear(FP, d), nn.GELU(), nn.Linear(d, d))
        self.bemb = nn.Sequential(nn.Linear(FB, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.attn = nn.TransformerEncoder(layer, cfg.n_attn, enable_nested_tensor=False)
        self.frame = nn.Sequential(nn.LayerNorm(5 * d + 1), nn.Linear(5 * d + 1, df), nn.GELU())
        self.hyb = nn.Sequential(nn.Linear(n_hybrid, df), nn.GELU(), nn.Linear(df, df)) \
            if n_hybrid else None
        self.tcn = nn.Sequential(*[TCNBlock(df, k, cfg.dropout) for k in cfg.tcn_dil])
        self.out = nn.Sequential(nn.LayerNorm(df), nn.Linear(df, 2))

    def frames(self, ptok, pmask, ball):
        """(N, K, FP), (N, K), (N, FB) -> (N, d_frame)"""
        team = ptok[..., 4]
        x = torch.cat([self.bemb(ball)[:, None], self.pemb(ptok)], 1)
        pad = torch.cat([torch.zeros_like(pmask[:, :1]), ~pmask], 1)
        x = self.attn(x, src_key_padding_mask=pad)
        b, p = x[:, 0], x[:, 1:]
        pooled = [b]
        for sel in ((team > 0) & pmask, (team < 0) & pmask):
            m = sel[..., None].float()
            cnt = m.sum(1).clamp(min=1)
            pooled.append((p * m).sum(1) / cnt)
            pooled.append(torch.where(sel[..., None], p, torch.full_like(p, -1e4)).amax(1)
                          .clamp(min=-10))
        pooled.append(pmask.float().sum(1, keepdim=True) / 22.0)
        return self.frame(torch.cat(pooled, 1))

    def forward(self, ptok, pmask, ball, hyb=None, chunk=4096):
        """(B, T, ...) -> logits (B, T, 2): dead, home."""
        B, T = pmask.shape[:2]
        flat = [t.reshape(B * T, *t.shape[2:]) for t in (ptok, pmask, ball)]
        emb = torch.cat([self.frames(*[t[i:i + chunk] for t in flat])
                         for i in range(0, B * T, chunk)], 0).reshape(B, T, -1)
        if self.hyb is not None:
            emb = emb + self.hyb(hyb)
        return self.out(self.tcn(emb))


def loss_of(logits, label, has, w):
    """Mean BCE of P(dead) on graded frames + of P(home) on graded alive frames, weighted."""
    graded = (label != X) & has
    alive = graded & (label != D)
    wf = w[:, None].expand_as(label)
    ld = F.binary_cross_entropy_with_logits(logits[..., 0], (label == D).float(), reduction="none")
    lh = F.binary_cross_entropy_with_logits(logits[..., 1], (label == H).float(), reduction="none")
    gd, ga = graded.float() * wf, alive.float() * wf
    return (ld * gd).sum() / gd.sum().clamp(min=1e-3) + (lh * ga).sum() / ga.sum().clamp(min=1e-3)


def _to_device(arrays: Sequence[Optional[np.ndarray]], dev) -> list:
    """numpy arrays -> (1, ...) tensors on dev; None stays None."""
    return [None if a is None else torch.from_numpy(a)[None].to(dev) for a in arrays]


# ---------------------------------------------------------------------------------------------
# Views (flip the pitch / mirror it) -- used by augmentation and by test-time prediction
# ---------------------------------------------------------------------------------------------
def team_swap(px, py, bx, by, home):
    """Turn the pitch 180 degrees and exchange the teams."""
    return L - px, W - py, L - bx, W - by, ~home


def y_mirror(py, by):
    return W - py, W - by


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
class Augmenter:
    """Train-time augmentation of one crop. The order of the random draws is fixed."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, rng, px, py, bx, by, home, label):
        cfg = self.cfg
        if cfg.jitter_m:
            px = px + rng.normal(0, cfg.jitter_m, px.shape).astype(np.float32)
            py = py + rng.normal(0, cfg.jitter_m, py.shape).astype(np.float32)
        if rng.random() < 0.5:                                       # team swap
            px, py, bx, by, home = team_swap(px, py, bx, by, home)
            label = swap_labels(label)
        if rng.random() < 0.5:                                       # y mirror
            py, by = y_mirror(py, by)
        if rng.random() < cfg.p_hide:                                # ball removal
            hidden = hidden_mask(len(bx), rng)
            bx, by = np.where(hidden, np.nan, bx), np.where(hidden, np.nan, by)
        if rng.random() < cfg.p_drop:                                # player track dropout
            px, py = self._drop_tracks(rng, px, py)
        return px, py, bx, by, home, label

    @staticmethod
    def _drop_tracks(rng, px, py):
        share = rng.uniform(0.02, 0.15)
        lost = np.zeros(px.shape, bool)
        target = share * px.size
        while lost.sum() < target:
            j = int(rng.integers(px.shape[1]))
            s = int(rng.integers(len(px)))
            lost[s:s + max(1, int(rng.exponential(8.0) * ef.MODEL_FPS)), j] = True
        return np.where(lost, np.nan, px), np.where(lost, np.nan, py)


class Crop(NamedTuple):
    inputs: Inputs
    label: np.ndarray
    weight: float


class Batch(NamedTuple):
    ptok: torch.Tensor
    pmask: torch.Tensor
    ball: torch.Tensor
    hyb: Optional[torch.Tensor]
    label: torch.Tensor
    has: torch.Tensor
    w: torch.Tensor


def collate(crops: Sequence[Crop], dev) -> Batch:
    T = min(len(c.label) for c in crops)

    def stack(get):
        return torch.from_numpy(np.stack([get(c)[:T] for c in crops])).to(dev)

    return Batch(
        ptok=stack(lambda c: c.inputs.ptok), pmask=stack(lambda c: c.inputs.pmask),
        ball=stack(lambda c: c.inputs.ball),
        hyb=None if crops[0].inputs.hyb is None else stack(lambda c: c.inputs.hyb),
        label=stack(lambda c: c.label).long(), has=stack(lambda c: c.inputs.has),
        w=torch.tensor([c.weight for c in crops], device=dev))


class CropSampler:
    """Random crops of the oriented halves of the training matches.

    SkillCorner crops are drawn with probability sc_sample; their loss weight gives the same
    expected share of the gradient as v17's per-row weight sc_weight."""

    def __init__(self, names: Sequence[str], cfg: Config):
        self.cfg = cfg
        self.halves = [h for name in names for h in Half.all_of(load_match(name), name)]
        minutes = {"sc": 0.0, "opt": 0.0}
        for h in self.halves:
            minutes["sc" if h.source == "skillcorner" else "opt"] += (h.last - h.first) / h.step
        if minutes["sc"] and cfg.sc_weight > 0:
            self.p_sc = cfg.sc_sample
            self.w_sc = cfg.sc_weight * (minutes["sc"] / minutes["opt"]) * \
                ((1 - self.p_sc) / self.p_sc)
        else:
            self.p_sc, self.w_sc = 0.0, 0.0
        self.augment = Augmenter(cfg)

    def sample(self, rng, stats: Optional[HybridStats] = None) -> Crop:
        cfg = self.cfg
        use_sc = rng.random() < self.p_sc
        pool = [h for h in self.halves if (h.source == "skillcorner") == use_sc]
        sizes = np.array([h.last - h.first for h in pool], float)
        h = pool[rng.choice(len(pool), p=sizes / sizes.sum())]
        step = h.step
        off = int(rng.integers(step))
        n = (len(h) - off + step - 1) // step
        T, M = ef.seconds(cfg.crop_s), ef.seconds(cfg.margin_s)
        k0 = int(rng.integers(0, max(1, n - T)))
        a, b = max(0, k0 - M), min(n, k0 + T + M)               # the margin feeds the TCN context
        rows = off + step * np.arange(a, b)
        px, py, bx, by, home, label = self.augment(
            rng, h.px[rows].copy(), h.py[rows].copy(), h.bx[rows].copy(), h.by[rows].copy(),
            h.home.copy(), h.label[rows].copy())
        inputs = encode(px, py, bx, by, home, stats)
        c = slice(k0 - a, k0 - a + T)
        return Crop(Inputs(inputs.ptok[c], inputs.pmask[c], inputs.ball[c], inputs.has[c],
                           None if inputs.hyb is None else inputs.hyb[c]),
                    label[c], np.float32(self.w_sc if h.source == "skillcorner" else 1.0))


# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------
def tta_views(h: Half, offset: int, stats: Optional[HybridStats], hide=None):
    """The 4 views (as is / team swap / y mirror / both) of one grid offset of a half.
    hide: native-frame bool mask of the half where the ball is removed first.
    Returns (native rows of the grid, [(swapped, Inputs)])."""
    rows = np.arange(offset, len(h), h.step)
    px, py, bx, by = h.px[rows], h.py[rows], h.bx[rows], h.by[rows]
    if hide is not None:
        bx, by = np.where(hide[rows], np.nan, bx), np.where(hide[rows], np.nan, by)
    views = []
    for swap in (False, True):
        for mirror in (False, True):
            x, y, u, v, home = (team_swap(px, py, bx, by, h.home) if swap
                                else (px, py, bx, by, h.home))
            if mirror:
                y, v = y_mirror(y, v)
            views.append((swap, encode(x, y, u, v, home, stats)))
    return rows, views


class Predictor:
    """Mean of several models over all grid offsets and the 4 views."""

    def __init__(self, models, dev, stats=None):
        self.models = list(models)
        for m in self.models:
            m.eval()
        self.dev = dev
        self.stats = HybridStats.coerce(stats)

    @torch.no_grad()
    def half(self, h: Half, hide=None):
        n = len(h)
        p_dead, p_home = np.full(n, 0.5, np.float32), np.full(n, 0.5, np.float32)
        for off in range(h.step):
            rows, views = tta_views(h, off, self.stats, hide)
            dead, home = [], []
            for swapped, inp in views:
                args = _to_device([inp.ptok, inp.pmask, inp.ball, inp.hyb], self.dev)
                for model in self.models:
                    logit = model(*args)[0].float().cpu().numpy()
                    dead.append(1 / (1 + np.exp(-logit[:, 0])))
                    ph = 1 / (1 + np.exp(-logit[:, 1]))
                    home.append(1 - ph if swapped else ph)
            has = views[0][1].has
            p_dead[rows] = np.where(has, np.mean(dead, 0), NO_PLAYERS_P_DEAD)
            p_home[rows] = np.where(has, np.mean(home, 0), 0.5)
        return p_dead, p_home

    def match(self, match: dict, hide=None):
        """(p_dead, p_home), one value per native frame, 0.5 outside the halves."""
        n = len(match["bx"])
        p_dead, p_home = np.full(n, 0.5, np.float32), np.full(n, 0.5, np.float32)
        for _, first, last in match["halves"]:
            h = Half.from_match(match, first, last)
            d, p = self.half(h, None if hide is None else hide[first:last + 1])
            p_dead[first:last + 1], p_home[first:last + 1] = d, p
        return p_dead, p_home


def predict_match(models, match, dev, stats=None, hide=None):
    """Drop-in for the old raw_model.predict_match (run_v19.py uses it)."""
    return Predictor(models, dev, stats).match(match, hide)


# ---------------------------------------------------------------------------------------------
# Decoding and scoring (formerly ext_model.decode / score)
# ---------------------------------------------------------------------------------------------
def decode(p_dead, p_home, halves, fps, cost) -> np.ndarray:
    """H / A / D codes on the frames of the halves, X elsewhere. `cost` is per change on the 5 fps
    grid; a match with more frames a second has fps / 5 times the evidence, so it is scaled."""
    codes = np.full(len(p_dead), X, dtype=np.int8)
    for _, first, last in halves:
        rows = slice(first, last + 1)
        codes[rows] = viterbi(viterbi_costs(p_dead[rows], p_home[rows]),
                              cost * fps / ef.MODEL_FPS)
    return codes


def score(match: dict, codes: np.ndarray) -> float:
    """spellF1 of decoded codes against the labels of an outside match."""
    return pm.evaluate(match["label"].astype(np.int8), codes, bootstrap_n=0)["spell_f1_mean"]


def score_match(match, p_dead, p_home, costs) -> dict:
    return {str(c): score(match, decode(p_dead, p_home, match["halves"], match["fps"], c))
            for c in costs}


def selection(results: dict, costs) -> dict:
    """{cost: mean spellF1 over the held-out optical matches, ball seen and ball hidden}."""
    use = [n for n in results if n.split("_")[0] in SELECT_ON]
    return {str(c): float(np.mean([results[n][k][str(c)] for n in use for k in ("seen", "hidden")]))
            for c in costs}


def best_str(scores: dict) -> str:
    k = max(scores, key=scores.get)
    return "%.3f@%s" % (scores[k], k)


# ---------------------------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------------------------
class TrainResult(NamedTuple):
    model: RawNet
    stats: Optional[HybridStats]
    history: list


class Trainer:
    def __init__(self, cfg: Config, dev, log: Callable[[str], None] = print):
        self.cfg, self.dev, self.log = cfg, dev, log

    def _early_halves(self, names):
        out = []
        for name in names:
            match = load_match(name)
            hide = validation_hide(match)
            for h in Half.all_of(match, name):
                out.append((h, hide[h.first:h.last + 1]))
        return out

    @torch.no_grad()
    def _early_loss(self, model, early_halves, stats) -> float:
        """Mean loss on the early-stopping halves, offset 0, ball seen and ball hidden."""
        model.eval()
        total = []
        for h, hide in early_halves:
            for hd in (None, hide):
                rows, views = tta_views(h, 0, stats, hd)
                inp = views[0][1]
                logit = model(*_to_device([inp.ptok, inp.pmask, inp.ball, inp.hyb], self.dev))
                (label,) = _to_device([h.label[rows].astype(np.int64)], self.dev)
                (has,) = _to_device([inp.has], self.dev)
                total.append(loss_of(logit, label, has, torch.ones(1, device=self.dev)).item())
        model.train()
        return float(np.mean(total))

    def fit(self, names, seed, early=None, epochs=None) -> TrainResult:
        cfg, dev, log = self.cfg, self.dev, self.log
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        sampler = CropSampler(names, cfg)
        stats = HybridStats.fit(sampler.halves, np.random.default_rng(1000 + seed)) \
            if cfg.hybrid else None
        model = RawNet(cfg, 0 if stats is None else stats.width).to(dev)
        log("params %d | halves %d | sc weight per crop %.3f (p %.2f)" % (
            sum(p.numel() for p in model.parameters()), len(sampler.halves),
            sampler.w_sc, sampler.p_sc))
        epochs = epochs or cfg.epochs
        steps_per_epoch = cfg.crops_per_epoch // cfg.batch
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=epochs * steps_per_epoch,
                                                    pct_start=0.1)
        early_halves = self._early_halves(early) if early else []
        history, best, best_state = [], 1e9, None
        model.train()
        t0 = time.time()
        for epoch in range(epochs):
            run = []
            for _ in range(steps_per_epoch):
                b = collate([sampler.sample(rng, stats) for _ in range(cfg.batch)], dev)
                loss = loss_of(model(b.ptok, b.pmask, b.ball, b.hyb), b.label, b.has, b.w)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                run.append(loss.item())
            el = self._early_loss(model, early_halves, stats) if early_halves else float("nan")
            history.append(dict(epoch=epoch + 1, train=float(np.mean(run)), early=el))
            log("  seed %d epoch %2d train %.4f early %.4f (%.0f s)" % (
                seed, epoch + 1, np.mean(run), el, time.time() - t0))
            if early_halves and el < best:
                best, best_state = el, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if best_state is not None and cfg.restore_best:
            model.load_state_dict(best_state)
        model.eval()
        return TrainResult(model, stats, history)


# ---------------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------------
def run_fold(fold: str, tag: str, cfg: Config, seeds, dev, log) -> dict:
    """Train on everything but the fold's held-out matches (and the early-stopping ones), then
    score the held-out matches with the ball seen and hidden, at every switch cost."""
    held_out = FOLDS[fold]
    train = [n for n in external.match_names() if n not in held_out and n not in EARLY]
    log("fold %s: train %d, early %s, held-out %s" % (fold, len(train), EARLY, held_out))
    trainer = Trainer(cfg, dev, log)
    results = [trainer.fit(train, seed, early=EARLY) for seed in seeds]
    models, stats = [r.model for r in results], results[-1].stats
    predictor = Predictor(models, dev, stats)
    os.makedirs(PROBS_DIR, exist_ok=True)
    scores = {}
    for name in held_out:
        match = load_match(name)
        scores[name] = {}
        for kind, hide in (("seen", None), ("hidden", validation_hide(match))):
            p_dead, p_home = predictor.match(match, hide)
            np.savez(os.path.join(PROBS_DIR, "%s_fold%s_%s_%s.npz" % (tag, fold, name, kind)),
                     p_dead=p_dead.astype(np.float32), p_home=p_home.astype(np.float32))
            scores[name][kind] = score_match(match, p_dead, p_home, cfg.switch_costs)
        log("  %-22s seen %s | hidden %s" % (name, best_str(scores[name]["seen"]),
                                             best_str(scores[name]["hidden"])))
    return dict(fold=fold, tag=tag, cfg=cfg.to_dict(), seeds=list(seeds),
                history=[r.history for r in results], scores=scores,
                selection=selection(scores, cfg.switch_costs))


def run_final(tag: str, cfg: Config, seeds, dev, log) -> None:
    """Train on all outside matches (fixed epochs, no early stopping), save the weights and the
    probabilities of the competition match. Nothing of that match is scored here."""
    names = external.match_names()
    trainer = Trainer(cfg, dev, log)
    models, stats = [], None
    for seed in seeds:
        result = trainer.fit(names, seed)
        stats = result.stats
        torch.save(result.model.state_dict(),
                   os.path.join(WEIGHTS_DIR, "%s_final_seed%d.pt" % (tag, seed)))
        models.append(result.model)
    if stats is not None:
        stats.save(os.path.join(WEIGHTS_DIR, "%s_final_hybrid_stats.npz" % tag))
    with open(os.path.join(WEIGHTS_DIR, "%s_final_cfg.json" % tag), "w") as f:
        json.dump(dict(cfg=cfg.to_dict(), seeds=list(seeds), matches=names), f, indent=1)
    p_dead, p_home = Predictor(models, dev, stats).match(competition_match())
    os.makedirs(PROBS_DIR, exist_ok=True)
    np.savez(os.path.join(PROBS_DIR, "%s_competition.npz" % tag),
             p_dead=p_dead.astype(np.float32), p_home=p_home.astype(np.float32))
    log("wrote competition probabilities (not scored)")


def run_speed(cfg: Config, dev, log) -> None:
    """Time ten training steps on one fold's training matches."""
    names = [n for n in external.match_names() if n not in FOLDS["A"] and n not in EARLY]
    sampler = CropSampler(names, cfg)
    rng = np.random.default_rng(0)
    model = RawNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t = time.time()
    for i in range(10):
        b = collate([sampler.sample(rng) for _ in range(cfg.batch)], dev)
        t1 = time.time()
        loss = loss_of(model(b.ptok, b.pmask, b.ball), b.label, b.has, b.w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log("step %d loss %.3f data+step %.2f s (step %.2f)" % (
            i, loss.item(), time.time() - t, time.time() - t1))
        t = time.time()


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["fold", "final", "speed"])
    ap.add_argument("--fold", default="A", choices=sorted(FOLDS))
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a Config field, e.g. --set hybrid=true")
    return ap.parse_args(argv)


def build_config(args) -> Config:
    overrides = {}
    for kv in args.set:
        key, value = kv.split("=", 1)
        overrides[key] = json.loads(value)
    if args.epochs:
        overrides["epochs"] = args.epochs
    return Config().with_overrides(**overrides)


def make_logger(path: str) -> Callable[[str], None]:
    handle = open(path, "a")

    def log(message: str) -> None:
        print(message)
        handle.write(message + "\n")
        handle.flush()
        sys.stdout.flush()

    return log


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    dev = device()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    suffix = "_" + args.fold if args.mode == "fold" else ""
    log = make_logger(os.path.join(WEIGHTS_DIR, "log_%s_%s.txt" % (args.mode, args.tag + suffix)))
    log("device %s cfg %s" % (dev, json.dumps(cfg.to_dict())))
    if args.mode == "speed":
        run_speed(cfg, dev, log)
    elif args.mode == "fold":
        out = run_fold(args.fold, args.tag, cfg, seeds, dev, log)
        with open(os.path.join(WEIGHTS_DIR, "fold_%s_%s.json" % (args.tag, args.fold)), "w") as f:
            json.dump(out, f, indent=1)
        sel = out["selection"]
        best = max(sel, key=sel.get)
        log("SELECTION fold %s %s: best %.3f @ cost %s | %s" % (
            args.fold, args.tag, sel[best], best, " ".join("%s:%.3f" % kv for kv in sel.items())))
    else:
        run_final(args.tag, cfg, seeds, dev, log)


if __name__ == "__main__":
    main()