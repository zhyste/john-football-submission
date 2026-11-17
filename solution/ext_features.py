"""ext_features.py -- the same features for ANY match (version 17).

Version 17 trains on other matches (external.py) and only predicts on the competition match. So
every feature must exist in both: the features use the player positions and the ball position
only. No Smart Tagging, no broadcast, no referee (the outside matches do not have them).

The same code runs on the outside matches and on our match:

  * A match is given in the common format of external.py: px, py (frames, players) in metres,
    is_home (players,), bx, by (frames,), fps, pitch (length, width), halves.
  * All positions are scaled to a 105 m x 68 m pitch.
  * The pitch is turned so that the home team attacks towards +x in both halves. The direction
    comes from the data (the mean x of each team in the half).
  * Each half is read on a grid of MODEL_FPS = 5 frames a second (every 2nd frame of a 10 fps
    match, every 5th frame of a 25 fps match). The speeds and the time windows are calculated on
    that grid, in seconds, so a 10 fps match and a 25 fps match give the same kind of numbers.
    A match has fps / 5 such grids (the offsets); together they give every frame.
  * Speeds come from the positions (a difference over 0.8 s), not from a speed column.

player_features() does not use the ball. ball_features() gets the ball as an argument, so the
training can remove the ball for a period and calculate these features again (ext_model.py).

Left out on purpose, because they are not the same in all matches: the number of tracked players
(the SkillCorner files always have 22: the players that the camera does not show are
extrapolated), the minute of the half.
"""

import warnings

import numpy as np
import pandas as pd

MODEL_FPS = 5
LENGTH, WIDTH = 105.0, 68.0
MAX_SECONDS = 60.0
CONTACT_M = 1.5
CONTROL_MIN_STEPS = 2          # a control: one team in contact on 2 grid frames or more (0.4 s)


# player features that also get a mean over +/- 3 s and +/- 8 s
LONG_STEMS = ("spread_x_gap", "spread_y_gap", "advance_gap", "cx_gap", "speed_gap")
LONG_WINDOWS = (("3s", 3), ("8s", 8))


def seconds(value):
    """Seconds -> grid frames."""
    return int(round(value * MODEL_FPS))


def rolling(values, window, how="mean", side="centre"):
    """Rolling statistic over +/- window grid frames (centre), or the window frames before
    (past) or after (future), the frame itself included."""
    series = pd.Series(values)
    if side == "past":
        rolled = series.rolling(window, min_periods=1)
    elif side == "future":
        rolled = series[::-1].rolling(window, min_periods=1)
    else:
        rolled = series.rolling(2 * window + 1, center=True, min_periods=1)
    out = getattr(rolled, how)()
    return (out[::-1] if side == "future" else out).values


def seconds_since(flag):
    """Seconds since the last True, at most MAX_SECONDS. 0 on a True frame."""
    index = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, index, -1))
    out = np.where(last >= 0, (index - last) / MODEL_FPS, MAX_SECONDS)
    return np.minimum(out, MAX_SECONDS)


def seconds_until(flag):
    return seconds_since(flag[::-1])[::-1]


def centred_speed(x, y, half_width=2):
    """Speed in m/s from the positions: the distance between t - half_width and t + half_width.
    Works on (frames,) and (frames, players). NaN where one of the two positions is missing."""
    out = np.full(x.shape, np.nan)
    span = 2 * half_width
    if len(x) > span:
        out[half_width:-half_width] = np.hypot(x[span:] - x[:-span], y[span:] - y[:-span]) \
            * MODEL_FPS / span
    return out


def runs_of(values):
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(a), int(b), values[a]) for a, b in zip(bounds[:-1], bounds[1:])]


class Half:
    """One half of a match on one 5 fps grid, turned so that home attacks +x."""

    def __init__(self, match, first, last, offset):
        step = match["fps"] // MODEL_FPS
        self.frames = np.arange(first + offset, last + 1, step)
        length, width = match["pitch"]
        px = match["px"][self.frames].astype(np.float64) * (LENGTH / length)
        py = match["py"][self.frames].astype(np.float64) * (WIDTH / width)
        bx = match["bx"][self.frames].astype(np.float64) * (LENGTH / length)
        by = match["by"][self.frames].astype(np.float64) * (WIDTH / width)
        self.home = np.asarray(match["is_home"], dtype=bool)
        self.away = np.asarray(match.get("is_away", ~self.home), dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            right = np.nanmean(px[:, self.home]) < np.nanmean(px[:, self.away])
        if not right:                                  # turn the pitch 180 degrees
            px, py, bx, by = LENGTH - px, WIDTH - py, LENGTH - bx, WIDTH - by
        self.px, self.py, self.bx, self.by = px, py, bx, by


def halves_of(match):
    """[Half] for every half and every offset of the match."""
    step = match["fps"] // MODEL_FPS
    return [Half(match, first, last, offset)
            for _, first, last in match["halves"] for offset in range(step)]


def player_features(half):
    """Everything that does not need the ball: the shape, the position and the speed of the
    teams."""
    out = {}
    speed = centred_speed(half.px, half.py)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for side, cols, sign in (("h", half.home, 1.0), ("a", half.away, -1.0)):
            x, y = half.px[:, cols], half.py[:, cols]
            out["cx_" + side] = np.nanmean(x, axis=1)
            out["cy_" + side] = np.nanmean(y, axis=1)
            out["spread_x_" + side] = np.nanstd(x, axis=1)
            out["spread_y_" + side] = np.nanstd(y, axis=1)
            out["max_x_" + side] = np.nanmax(x, axis=1)
            out["min_x_" + side] = np.nanmin(x, axis=1)
            out["speed_" + side] = np.nanmean(speed[:, cols], axis=1)
            # how fast the team moves towards the goal that it attacks, over 1 s
            centroid = pd.Series(out["cx_" + side]).interpolate(limit_direction="both").values
            velocity = np.zeros(len(centroid))
            velocity[MODEL_FPS:] = sign * (centroid[MODEL_FPS:] - centroid[:-MODEL_FPS])
            out["advance_" + side] = rolling(velocity, seconds(1))
        both = half.home | half.away
        speed_all = np.nanmean(speed[:, both], axis=1)
        out["speed_all"] = speed_all
        out["slow_share"] = np.nanmean(np.where(np.isnan(speed[:, both]), np.nan,
                                                speed[:, both] < 1.0), axis=1)
    filled = pd.Series(speed_all).interpolate(limit_direction="both").values
    for name, window in (("1s", 1), ("2s", 2), ("5s", 5)):
        out["speed_all_" + name] = rolling(filled, seconds(window))
    out["speed_past_5s"] = rolling(filled, seconds(5), side="past")
    out["speed_future_5s"] = rolling(filled, seconds(5), side="future")
    out["slow_share_2s"] = rolling(pd.Series(out["slow_share"]).interpolate(
        limit_direction="both").values, seconds(2))
    out["cx_gap"] = out["cx_h"] - out["cx_a"]
    out["advance_gap"] = out["advance_h"] - out["advance_a"]
    # The team with the ball opens (wide and deep); the other team closes. The differences
    # between the teams, also as means over a longer time: possession lasts for seconds.
    out["spread_x_gap"] = out["spread_x_h"] - out["spread_x_a"]
    out["spread_y_gap"] = out["spread_y_h"] - out["spread_y_a"]
    out["speed_gap"] = out["speed_h"] - out["speed_a"]
    for stem in LONG_STEMS:
        filled = pd.Series(out[stem]).interpolate(limit_direction="both").values
        for name, window in LONG_WINDOWS:
            out["%s_%s" % (stem, name)] = rolling(filled, seconds(window))
    return pd.DataFrame(out)


def ball_features(half, bx=None, by=None):
    """Everything that needs the ball. bx, by: the ball on the grid of `half` (default: the
    ball of the half), so that the training can remove it for a period."""
    bx = half.bx if bx is None else bx
    by = half.by if by is None else by
    seen = ~np.isnan(bx)
    out = {"ball_seen": seen.astype(np.int8)}
    out["ball_x"] = bx
    out["ball_y_off_centre"] = np.abs(by - WIDTH / 2.0)
    with np.errstate(invalid="ignore"):
        to_line = np.minimum.reduce([bx, LENGTH - bx, by, WIDTH - by])
    out["ball_to_line"] = to_line
    with np.errstate(invalid="ignore"):
        out["ball_off_pitch"] = (seen & (to_line < -0.5)).astype(np.int8)

    speed = centred_speed(bx, by, 1)
    out["ball_speed"] = speed
    out["ball_speed_1s"] = rolling(speed, seconds(0.6), how="median")
    velocity = np.full(len(bx), np.nan)
    velocity[2:-2] = (bx[4:] - bx[:-4]) * MODEL_FPS / 4.0 if len(bx) > 4 else np.nan
    out["ball_vx"] = velocity                       # + = towards the goal that home attacks
    with np.errstate(invalid="ignore"):
        slow = seen & (out["ball_speed_1s"] < 1.0)
    index = np.arange(len(slow))
    run_start = np.maximum.accumulate(np.where(~slow, index + 1, 0))
    out["ball_slow_secs"] = np.minimum(np.where(slow, (index - run_start + 1) / MODEL_FPS, 0.0),
                                       MAX_SECONDS)

    out["secs_since_ball"] = seconds_since(seen)
    out["secs_until_ball"] = seconds_until(seen)
    out["last_ball_x"] = pd.Series(bx).ffill().values
    out["next_ball_x"] = pd.Series(bx).bfill().values
    out["last_ball_to_line"] = pd.Series(to_line).ffill().values
    out["next_ball_to_line"] = pd.Series(to_line).bfill().values

    distance = np.hypot(half.px - bx[:, None], half.py - by[:, None])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for side, cols in (("h", half.home), ("a", half.away)):
            d = distance[:, cols]
            out["near_" + side] = np.nanmin(d, axis=1)
            out["within2_" + side] = np.sum(d < 2.0, axis=1)
            out["within5_" + side] = np.sum(d < 5.0, axis=1)
    out["near_gap"] = out["near_a"] - out["near_h"]

    # +1 home in contact, -1 away, 0 nobody or no ball
    with np.errstate(invalid="ignore"):
        home_near = (out["near_h"] < CONTACT_M) & ~(out["near_a"] <= out["near_h"])
        away_near = (out["near_a"] < CONTACT_M) & ~(out["near_h"] <= out["near_a"])
        contact = np.where(home_near, 1.0, np.where(away_near, -1.0, 0.0))
    out["contact"] = contact
    for name, window in (("1s", 1), ("3s", 3), ("5s", 5)):
        out["vote_" + name] = rolling(contact, seconds(window))
    for name, window in (("3s", 3), ("8s", 8), ("15s", 15)):
        out["vote_past_" + name] = rolling(contact, seconds(window), side="past")
        out["vote_future_" + name] = rolling(contact, seconds(window), side="future")
    out["ball_vx_3s"] = rolling(velocity, seconds(3))
    touched = contact != 0
    marked = pd.Series(np.where(touched, contact, np.nan))
    out["last_contact"] = marked.ffill().fillna(0).values
    out["next_contact"] = marked.bfill().fillna(0).values
    out["secs_since_contact"] = seconds_since(touched)
    out["secs_until_contact"] = seconds_until(touched)

    # Controls: one team in contact for CONTROL_MIN_STEPS grid frames or more. The team of the
    # last and of the next control, the time to them, and the number of consecutive controls of
    # that team (a chain of passes).
    sign = np.full(len(bx), np.nan)
    chain = np.full(len(bx), np.nan)
    length = np.zeros(len(bx))
    count, last_team = 0, 0.0
    for a, b, value in runs_of(contact):
        if value == 0 or b - a < CONTROL_MIN_STEPS:
            continue
        count = count + 1 if value == last_team else 1
        last_team = value
        sign[a:b], chain[a:b], length[a:b] = value, count, (b - a) / MODEL_FPS
    controlled = ~np.isnan(sign)
    out["ctl_last"] = pd.Series(sign).ffill().fillna(0).values
    out["ctl_next"] = pd.Series(sign).bfill().fillna(0).values
    out["secs_since_ctl"] = seconds_since(controlled)
    out["secs_until_ctl"] = seconds_until(controlled)
    out["ctl_chain_last"] = pd.Series(chain).ffill().fillna(0).values * out["ctl_last"]
    out["ctl_chain_next"] = pd.Series(chain).bfill().fillna(0).values * out["ctl_next"]
    out["ctl_secs"] = length
    return pd.DataFrame(out)


def relation_features(players, ball):
    """The ball against the teams: is the last known ball in front of or behind each team's
    centre and its last line? + = nearer to the goal that home attacks."""
    out = {}
    for name in ("last_ball_x", "next_ball_x"):
        stem = name[:4]
        out[stem + "_ball_ahead_h"] = ball[name] - players["cx_h"]
        out[stem + "_ball_ahead_a"] = ball[name] - players["cx_a"]
    out["last_ball_past_line_h"] = ball["last_ball_x"] - players["min_x_h"]   # home's last line
    out["last_ball_past_line_a"] = ball["last_ball_x"] - players["max_x_a"]   # away's last line
    return pd.DataFrame(out)


def feature_table(half, players=None, bx=None, by=None):
    players = player_features(half) if players is None else players
    ball = ball_features(half, bx, by)
    return pd.concat([players, ball, relation_features(players, ball)], axis=1)


# Home <-> away swap (the pitch turned 180 degrees): used to train on both views of each frame
# and to predict with both views, so the model cannot prefer one team.
_PAIRS = ("cx", "cy", "spread_x", "spread_y", "speed", "advance", "near", "within2", "within5")
_NEGATE = ("advance_gap", "spread_x_gap", "spread_y_gap", "speed_gap", "near_gap", "contact",
           "vote_1s", "vote_3s", "vote_5s", "vote_past_3s", "vote_future_3s", "vote_past_8s",
           "vote_future_8s", "vote_past_15s", "vote_future_15s", "last_contact", "next_contact",
           "ctl_last", "ctl_next", "ctl_chain_last", "ctl_chain_next", "ball_vx", "ball_vx_3s") \
    + tuple("%s_%s" % (stem, name) for stem in LONG_STEMS if stem != "cx_gap"
            for name, _ in LONG_WINDOWS)
# these do not change: cx_gap = cx_h - cx_a becomes (L - cx_a) - (L - cx_h), the same value
_MIRROR_X = ("ball_x", "last_ball_x", "next_ball_x")


def swap_teams(table):
    """The feature table of the same frames with home and away exchanged."""
    out = table.copy()
    for stem in _PAIRS:
        out[stem + "_h"], out[stem + "_a"] = table[stem + "_a"], table[stem + "_h"]
    # positions: x -> LENGTH - x, y -> WIDTH - y, and the teams change names
    for side, other in (("h", "a"), ("a", "h")):
        out["cx_" + side] = LENGTH - table["cx_" + other]
        out["cy_" + side] = WIDTH - table["cy_" + other]
        out["max_x_" + side] = LENGTH - table["min_x_" + other]
        out["min_x_" + side] = LENGTH - table["max_x_" + other]
    for name in _NEGATE:
        out[name] = -table[name]
    for name in _MIRROR_X:
        out[name] = LENGTH - table[name]
    # ball - team centre: the sign changes and the teams change names
    for stem in ("last_ball_ahead", "next_ball_ahead"):
        out[stem + "_h"], out[stem + "_a"] = -table[stem + "_a"], -table[stem + "_h"]
    out["last_ball_past_line_h"] = -table["last_ball_past_line_a"]
    out["last_ball_past_line_a"] = -table["last_ball_past_line_h"]
    return out
