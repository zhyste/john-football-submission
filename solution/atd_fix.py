"""atd_fix.py -- correct the team of the two goalkeeper slots of the ATD tracking.

The ATD metadata has 46 slots ("Track 1".."Track 46"). Slot 23 ("Track 24", home) and slot 45
("Track 46", away) are the only slots with position_type "Goalkeeper". The tracker does not keep a
goalkeeper in his own slot: it puts "a goalkeeper" in one of the two goalkeeper slots. When a slot
comes back after a gap, it often comes back at the OTHER goal (a jump of 73 to 96 m; 12 times for
slot 23, 19 times for slot 45). Then the slot follows the other team's goalkeeper, and the ATD team
of the slot is wrong. Over the match:
  - slot 45 (away) is at the home goal for 1.3 + 0.6 + 1.3 + 1.3 + 0.3 min in the first half and
    the last 10 min of the second half (frames 184439 to the end) -- this is the "0% in the block
    from 48685" of the first check: 51132-51460, slot 45 at x = 4-11 m, the home goal;
  - slot 23 (home) is at the away goal for 8 short periods (3 to 19 s), e.g. 133552-133634 and
    134973-135441 -- the "0% in the block from 131185".
The rule: a goalkeeper slot belongs to the team that defends the half of the
pitch that it is in. Which team defends which goal comes from the data, not the metadata: in each
half of the match, the team whose outfield centroid has the smaller mean x defends the goal at
x = 0 (for BAR-ATH: home defends x = 0 in the first half, as the metadata says,
attack_direction_first_half = left_to_right).

The outfield slots: a test with the same idea (an outfield slot that is the deepest player at a
goal, behind the defending team's last outfield player, for a long time, would be an attacker or
a keeper in the wrong slot) is in check_outfield().

API
    corrected_team(tracking)      -> int8 [frame, slot] team (0 home, 1 away, -1 not tracked)
    corrected_tracking(tracking)  -> a copy of the tracking dict with 48 slots: each goalkeeper
                                     position that changes team moves to an extra slot of the
                                     correct team (46 = home, 47 = away), so every function that
                                     uses a 1-D team_of_slot works unchanged.
    overrides(tracking)           -> [(slot, start, stop, new_team)] the corrected runs.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from games import GAME, GAMES, half_video_blocks  # noqa: E402

LENGTH, WIDTH = GAMES[GAME]["pitch"]
HALFWAY = LENGTH / 2.0
# The goalkeeper slots of the metadata (position_type "Goalkeeper"); used if the xml is missing.
GOALKEEPER_SLOTS = (23, 45)
# load.loadTracking() applies corrected_tracking(). False gives the raw ATD slots.
ENABLED = True
# A goalkeeper slot run at the other team's goal shorter than this stays as it is: a keeper can
# come out over the halfway line for a moment (it never happens in BAR-ATH: 0 frames).
MIN_RUN_FRAMES = 1


def runs_of(values):
    """[(start, stop, value)] maximal constant runs, stop exclusive."""
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(a), int(b), values[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def goalkeeper_slots(meta_path=None):
    """The slots with position_type Goalkeeper in the ATD metadata."""
    if meta_path is None:
        meta_path = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
    if not os.path.exists(meta_path):
        return GOALKEEPER_SLOTS
    slots = []
    for index, player in enumerate(ET.parse(meta_path).getroot().iter("Player")):
        for parameter in player.iter("ProviderParameter"):
            if parameter.findtext("Name") == "position_type" \
                    and parameter.findtext("Value") == "Goalkeeper":
                slots.append(index)
    return tuple(slots) or GOALKEEPER_SLOTS


def half_of_frames(n_frames):
    """Half index (0, 1) of each frame; frames before the second half's first frame are half 0."""
    half = np.zeros(n_frames, dtype=np.int8)
    blocks = half_video_blocks(GAME)
    if len(blocks) > 1:
        half[blocks[1][1]:] = 1
    return half


def left_defender(tracking, keepers=None):
    """{half: team that defends the goal at x = 0}, from the mean x of each team's outfield
    centroid over the half."""
    keepers = goalkeeper_slots() if keepers is None else keepers
    px, team = tracking["players_x"], tracking["team_of_slot"]
    outfield = np.ones(len(team), dtype=bool)
    outfield[list(keepers)] = False
    half = half_of_frames(len(px))
    out = {}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for h in (0, 1):
            rows = half == h
            mean_x = [np.nanmean(px[rows][:, (team == t) & outfield]) for t in (0, 1)]
            out[h] = int(np.argmin(mean_x))
    return out


def corrected_team(tracking, keepers=None):
    """int8 [frame, slot]: the team of each tracked slot (-1 where the slot is not tracked)."""
    keepers = goalkeeper_slots() if keepers is None else keepers
    px, team = tracking["players_x"], tracking["team_of_slot"]
    n_frames = len(px)
    out = np.where(np.isnan(px), -1, team[None, :]).astype(np.int8)
    defends_left = np.array([left_defender(tracking, keepers)[h] for h in (0, 1)])
    left_team = defends_left[half_of_frames(n_frames)]
    for slot in keepers:
        x = px[:, slot]
        seen = ~np.isnan(x)
        with np.errstate(invalid="ignore"):
            owner = np.where(x < HALFWAY, left_team, 1 - left_team)
        new = np.where(seen, owner, -1).astype(np.int8)
        if MIN_RUN_FRAMES > 1:
            for a, b, v in runs_of(new != out[:, slot]):
                if v and b - a < MIN_RUN_FRAMES:
                    new[a:b] = out[a:b, slot]
        out[:, slot] = new
    return out


def overrides(tracking, keepers=None):
    """[(slot, start, stop, new_team)] runs where corrected_team differs from team_of_slot."""
    teams = corrected_team(tracking, keepers)
    base = tracking["team_of_slot"]
    out = []
    for slot in range(teams.shape[1]):
        changed = (teams[:, slot] >= 0) & (teams[:, slot] != base[slot])
        for a, b, v in runs_of(changed):
            if v:
                out.append((slot, a, b, int(teams[a, slot])))
    return out


def corrected_tracking(tracking, keepers=None):
    """A copy of `tracking` with two extra slots (46 home, 47 away). Each goalkeeper-slot frame
    whose corrected team differs from its ATD team moves to the extra slot of the corrected team.
    If both goalkeeper slots move to the same team on one frame (never in BAR-ATH), the second
    keeps its slot."""
    keepers = goalkeeper_slots() if keepers is None else keepers
    teams = corrected_team(tracking, keepers)
    base = tracking["team_of_slot"]
    n_frames, n_slots = tracking["players_x"].shape
    out = dict(tracking)
    for key in ("players_x", "players_y", "players_s"):
        out[key] = np.concatenate(
            [tracking[key], np.full((n_frames, 2), np.nan, tracking[key].dtype)], axis=1)
    out["team_of_slot"] = np.concatenate([base, np.array([0, 1], dtype=base.dtype)])
    for slot in keepers:
        moved = (teams[:, slot] >= 0) & (teams[:, slot] != base[slot])
        target = n_slots + teams[:, slot].clip(0, 1)
        for extra in (n_slots, n_slots + 1):
            rows = moved & (target == extra) & np.isnan(out["players_x"][:, extra])
            for key in ("players_x", "players_y", "players_s"):
                out[key][rows, extra] = out[key][rows, slot]
                out[key][rows, slot] = np.nan
    return out


def check_outfield(tracking, keepers=None, depth_m=3.0, goal_m=25.0, min_frames=10 * 25):
    """Outfield slot runs (>= min_frames) where the slot is deeper than every outfield player of
    the team that defends that goal by depth_m, within goal_m of the goal line, and is of the
    attacking team: candidates for a wrong team. Returns [(slot, start, stop)]."""
    import warnings
    keepers = goalkeeper_slots() if keepers is None else keepers
    px, team = tracking["players_x"], tracking["team_of_slot"]
    n_frames = len(px)
    outfield = np.ones(len(team), dtype=bool)
    outfield[list(keepers)] = False
    defends_left = np.array([left_defender(tracking, keepers)[h] for h in (0, 1)])
    left_team = defends_left[half_of_frames(n_frames)]
    flags = []
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        last = {}
        for t in (0, 1):
            cols = px[:, (team == t) & outfield]
            last[t] = (np.nanmin(cols, axis=1), np.nanmax(cols, axis=1))
        for slot in np.flatnonzero(outfield):
            x = px[:, slot]
            attacker_left = 1 - left_team  # attacking the goal at x = 0
            deep_left = (team[slot] == attacker_left) & (x < goal_m) \
                & (x < np.where(left_team == 0, last[0][0], last[1][0]) - depth_m)
            deep_right = (team[slot] == left_team) & (x > LENGTH - goal_m) \
                & (x > np.where(left_team == 0, last[1][1], last[0][1]) + depth_m)
            for a, b, v in runs_of(deep_left | deep_right):
                if v and b - a >= min_frames:
                    flags.append((int(slot), a, b))
    return flags


if __name__ == "__main__":
    from load import loadTracking
    tracking = loadTracking(fix_goalkeepers=False)
    print("goalkeeper slots:", goalkeeper_slots())
    print("defends x = 0 by half:", left_defender(tracking))
    rows = overrides(tracking)
    print("overrides (slot, start, stop, new team, seconds):")
    for slot, a, b, t in rows:
        if b - a >= 25:
            print("  %2d %6d %6d %s %6.1f s" % (slot, a, b, "HA"[t], (b - a) / 25.0))
    print("total frames changed: %d" % sum(b - a for _, a, b, _ in rows))
    print("outfield flags:", check_outfield(tracking))
