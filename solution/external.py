"""external.py -- labelled training data from OTHER matches (allowed by RULES.md).

    uv run python solution/external.py --download    # once: ~4.5 GB into external_data/ (git-ignored)
    uv run python solution/external.py               # parse + cache, print a summary per match

Sources (public, the organisers can access them -- list them in the write-up's training-data section):

  SkillCorner open data   https://github.com/SkillCorner/opendata   (MIT licence; credit SkillCorner)
      20 A-League 2024/25 matches, broadcast tracking at 10 fps.
      H/A  : the per-frame `possession.group` field (home team / away team), labelled by SkillCorner.
      D    : frames outside every "phase of play" -- SkillCorner: "phases of play are only defined
             when the ball is in play".

  Metrica Sports sample data   https://github.com/metrica-sports/sample-data   (public sample)
      Sample games 1 and 2, full tracking at 25 fps + human-coded event data.
      H/A  : the team of the latest on-ball event (pass, recovery, ball lost, shot, set piece).
      D    : ball out -> next set piece, foul -> next set piece, goal -> next kick-off,
             and the start of each half before the kick-off.

  DFL / Sportec open data   https://doi.org/10.1038/s41597-025-04505-y   (CC BY 4.0; credit
      Bassek, Rein, Weber and Memmert, "An integrated dataset of spatiotemporal and event data in
      elite soccer", Sci Data 12, 195 (2025); files on figshare, added for CatBoost version 17)
      7 matches of the Bundesliga and the 2. Bundesliga 2022/23, official optical tracking at
      25 fps. Each ball frame has the labels of the league's data collection:
      H/A  : `BallPossession` (1 = home team, 2 = guest team) where the ball is in play.
      D    : `BallStatus` = 0 (the ball is not in play).

Nothing here touches the competition match. Every match is converted to one common format:

    name, fps, pitch (length, width)
    px, py         (n_frames, n_players) metres, 0..length / 0..width, NaN = not tracked
    is_home        (n_players,) True for home players
    bx, by         (n_frames,) ball in metres, NaN = not seen
    label          (n_frames,) 0 = H, 1 = A, 2 = D, 3 = outside the halves
    halves         [(half, first_frame, last_frame)]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(ROOT, "external_data")
CACHE = os.path.join(EXT_DIR, "cache")
SC_DIR = os.path.join(EXT_DIR, "skillcorner")
METRICA_DIR = os.path.join(EXT_DIR, "metrica")

SC_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/"
SC_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/"
METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
DFL_DIR = os.path.join(EXT_DIR, "dfl")
# The figshare API (the web download link asks a browser check). File ids as in kloppy.
DFL_API = "https://api.figshare.com/v2/file/download/"
DFL_FILES = {   # match id -> (file id of the match information, file id of the positions)
    "J03WMX": (51643475, 51643514), "J03WN1": (51643472, 51643517),
    "J03WOH": (51643478, 51643520), "J03WOY": (51643481, 51643523),
    "J03WPY": (51643487, 51643526), "J03WQQ": (51643484, 51643529),
    "J03WR9": (51643490, 51643532),
}

H, A, D, X = 0, 1, 2, 3


# ---------------------------------------------------------------
# Download
# ---------------------------------------------------------------
def fetch(url, path):
    """Download url -> path (skip if already there). Uses curl, which Windows 10+ has built in."""
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("  downloading", os.path.relpath(path, ROOT))
    subprocess.run(["curl", "-sSfL", "-o", path + ".part", url], check=True)
    os.replace(path + ".part", path)


def download():
    fetch(SC_RAW + "matches.json", os.path.join(SC_DIR, "matches.json"))
    for match in json.load(open(os.path.join(SC_DIR, "matches.json"), encoding="utf-8")):
        mid = str(match["id"])
        folder = os.path.join(SC_DIR, mid)
        fetch(SC_RAW + "matches/{0}/{0}_match.json".format(mid), os.path.join(folder, mid + "_match.json"))
        fetch(SC_RAW + "matches/{0}/{0}_phases_of_play.csv".format(mid),
              os.path.join(folder, mid + "_phases_of_play.csv"))
        fetch(SC_LFS + "matches/{0}/{0}_tracking_extrapolated.jsonl".format(mid),
              os.path.join(folder, mid + "_tracking_extrapolated.jsonl"))
    for game in (1, 2):
        stem = "Sample_Game_{}/Sample_Game_{}_".format(game, game)
        for part in ("RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv"):
            fetch(METRICA_RAW + stem + part, os.path.join(METRICA_DIR, stem + part))
    for mid, (meta_id, tracking_id) in DFL_FILES.items():
        fetch(DFL_API + str(meta_id), os.path.join(DFL_DIR, mid + "_meta.xml"))
        fetch(DFL_API + str(tracking_id), os.path.join(DFL_DIR, mid + "_tracking.xml"))


# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------
def halves_of(period):
    """[(half, first, last)] from a per-frame period array (0 = no period)."""
    out = []
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if len(idx):
            out.append((half, int(idx[0]), int(idx[-1])))
    return out


def fill_team(team, alive):
    """Inside each alive stretch, carry the last known team forward (loose ball = last team keeps
    it), then backward for the first frames. Returns codes H/A, -1 where nothing is known."""
    out = team.copy()
    edges = np.diff(np.r_[0, alive.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        seg = pd.Series(out[a:b]).replace(-1, np.nan).ffill().bfill()
        out[a:b] = seg.fillna(-1).astype(int).to_numpy()
    return out


# ---------------------------------------------------------------
# SkillCorner
# ---------------------------------------------------------------
def parse_skillcorner(mid):
    folder = os.path.join(SC_DIR, mid)
    meta = json.load(open(os.path.join(folder, mid + "_match.json"), encoding="utf-8"))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    players = [p["id"] for p in meta["players"]]
    slot = {pid: i for i, pid in enumerate(players)}
    is_home = np.array([p["team_id"] == home_id for p in meta["players"]])

    rows = [json.loads(line) for line in open(os.path.join(folder, mid + "_tracking_extrapolated.jsonl"), encoding="utf-8")]
    n = max(r["frame"] for r in rows) + 1
    px = np.full((n, len(players)), np.nan, np.float32)
    py = np.full((n, len(players)), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    period = np.zeros(n, np.int8)
    group = np.full(n, -1, np.int8)
    for r in rows:
        f = r["frame"]
        period[f] = r["period"] or 0
        for p in r.get("player_data") or []:
            i = slot.get(p["player_id"])
            if i is not None and p["x"] is not None:
                px[f, i] = p["x"] + length / 2
                py[f, i] = p["y"] + width / 2
        b = r.get("ball_data") or {}
        if b.get("is_detected") and b.get("x") is not None:          # only a really seen ball
            bx[f] = b["x"] + length / 2
            by[f] = b["y"] + width / 2
        g = (r.get("possession") or {}).get("group")
        group[f] = H if g == "home team" else A if g == "away team" else -1

    # live = inside a phase of play; the phase's team is a fallback for frames with no group
    alive = np.zeros(n, bool)
    phase_team = np.full(n, -1, np.int8)
    for ph in pd.read_csv(os.path.join(folder, mid + "_phases_of_play.csv")).itertuples():
        alive[ph.frame_start:ph.frame_end + 1] = True
        phase_team[ph.frame_start:ph.frame_end + 1] = H if ph.team_in_possession_id == home_id else A
    alive &= period > 0
    team = np.where(group >= 0, group, -1).astype(np.int8)
    team[alive & (team < 0)] = phase_team[alive & (team < 0)]
    team = fill_team(team, alive)

    label = np.full(n, X, np.int8)
    label[period > 0] = D
    label[alive & (team >= 0)] = team[alive & (team >= 0)]
    return dict(name="skillcorner_" + mid, fps=10, pitch=(length, width), px=px, py=py,
                is_home=is_home, bx=bx, by=by, label=label, halves=halves_of(period))


# ---------------------------------------------------------------
# Metrica sample games
# ---------------------------------------------------------------
def read_metrica_team(path):
    table = pd.read_csv(path, skiprows=2)
    period = table["Period"].to_numpy()
    frame = table["Frame"].to_numpy() - 1                      # frames count from 1
    data = table.iloc[:, 3:].to_numpy(dtype=np.float32)       # x, y pairs, ball last
    xs, ys = data[:, 0:-2:2], data[:, 1:-2:2]
    return frame, period, xs, ys, data[:, -2], data[:, -1]


def parse_metrica(game, length=105.0, width=68.0):
    stem = os.path.join(METRICA_DIR, "Sample_Game_{0}".format(game), "Sample_Game_{0}_".format(game))
    frame, period, hx, hy, bx, by = read_metrica_team(stem + "RawTrackingData_Home_Team.csv")
    _, _, ax, ay, _, _ = read_metrica_team(stem + "RawTrackingData_Away_Team.csv")
    n = int(frame.max()) + 1
    px = np.full((n, hx.shape[1] + ax.shape[1]), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    px[frame] = np.hstack([hx, ax]) * length
    py[frame] = np.hstack([hy, ay]) * width
    ball_x = np.full(n, np.nan, np.float32)
    ball_y = np.full(n, np.nan, np.float32)
    ball_x[frame], ball_y[frame] = bx * length, by * width
    per = np.zeros(n, np.int8)
    per[frame] = period
    is_home = np.r_[np.ones(hx.shape[1], bool), np.zeros(ax.shape[1], bool)]

    events = pd.read_csv(stem + "RawEventsData.csv")
    events["Team"] = events["Team"].str.strip().str.lower()
    events["Type"] = events["Type"].str.strip().str.upper()
    events["Subtype"] = events["Subtype"].fillna("").str.upper()
    start = events["Start Frame"].to_numpy() - 1
    end = events["End Frame"].to_numpy() - 1

    # team: the team of the latest on-ball event
    team = np.full(n, -1, np.int8)
    on_ball = events["Type"].isin(["PASS", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE"]).to_numpy()
    for i in np.flatnonzero(on_ball):
        team[max(start[i], 0)] = H if events["Team"].iat[i] == "home" else A

    # dead: from the stoppage to the next set piece
    dead = np.zeros(n, bool)
    set_pieces = np.flatnonzero((events["Type"] == "SET PIECE").to_numpy())
    set_starts = start[set_pieces]
    for i in range(len(events)):
        kind, sub = events["Type"].iat[i], events["Subtype"].iat[i]
        if kind == "BALL OUT":
            begin = end[i]
        elif kind == "FAULT RECEIVED":
            begin = start[i]
        elif kind == "SHOT" and "GOAL" in sub:
            begin = end[i]
        else:
            continue
        later = set_starts[set_starts > begin]
        if len(later) and later[0] - begin < 120 * 25:          # restart within 2 minutes
            dead[max(begin, 0):later[0]] = True
    for half, first, _ in halves_of(per):                       # before each kick-off
        kickoffs = set_starts[(set_starts >= first)]
        if len(kickoffs):
            dead[first:kickoffs[0]] = True

    alive = (per > 0) & ~dead
    team = fill_team(np.where(alive, team, -1).astype(np.int8), alive)
    label = np.full(n, X, np.int8)
    label[per > 0] = D
    label[alive & (team >= 0)] = team[alive & (team >= 0)]
    return dict(name="metrica_game{}".format(game), fps=25, pitch=(length, width), px=px, py=py,
                is_home=is_home, bx=ball_x, by=ball_y, label=label, halves=halves_of(per))


# ---------------------------------------------------------------
# DFL / Sportec open data
# ---------------------------------------------------------------
def parse_dfl(mid):
    """One DFL match. The positions file has one <FrameSet> for each person (or the ball) and
    half, with one <Frame N=.. X=.. Y=..> line for each frame; the ball frames also have
    BallPossession and BallStatus."""
    import re
    meta = open(os.path.join(DFL_DIR, mid + "_meta.xml"), encoding="utf-8").read()
    length = float(re.search(r'PitchX="([0-9.]+)"', meta).group(1))
    width = float(re.search(r'PitchY="([0-9.]+)"', meta).group(1))
    role = {}
    for tag in re.findall(r"<Team [^>]*>", meta):
        role[re.search(r'TeamId="([^"]+)"', tag).group(1)] = re.search(r'Role="([^"]+)"', tag).group(1)

    attr = re.compile(r'(\w+)="([^"]*)"')
    sets = []                    # (section, team id, person id, N, X, Y, possession, status)
    current = None
    with open(os.path.join(DFL_DIR, mid + "_tracking.xml"), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                a = dict(attr.findall(line))
                current[3].append(int(a["N"]))
                current[4].append(float(a["X"]))
                current[5].append(float(a["Y"]))
                if "BallStatus" in a:
                    current[6].append(int(a["BallPossession"]))
                    current[7].append(int(a["BallStatus"]))
            elif line.startswith("<FrameSet "):
                a = dict(attr.findall(line))
                current = (a["GameSection"], a["TeamId"], a["PersonId"], [], [], [], [], [])
                sets.append(current)

    # frame index: the first half, then the second half directly after it
    sections = [name for name in ("firstHalf", "secondHalf") if any(s[0] == name for s in sets)]
    first_n, size, start = {}, {}, {}
    for name in sections:
        ball = next(s for s in sets if s[0] == name and s[1].upper() == "BALL")
        first_n[name], size[name] = min(ball[3]), max(ball[3]) - min(ball[3]) + 1
    total = 0
    for name in sections:
        start[name], total = total, total + size[name]

    people = sorted({(s[1], s[2]) for s in sets if s[1].upper() != "BALL" and s[1] in role
                     and role[s[1]] in ("home", "guest")})
    slot = {person: i for i, (_, person) in enumerate(people)}
    is_home = np.array([role[team] == "home" for team, _ in people])
    px = np.full((total, len(people)), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    bx = np.full(total, np.nan, np.float32)
    by = np.full(total, np.nan, np.float32)
    label = np.full(total, X, np.int8)
    period = np.zeros(total, np.int8)
    for section, team, person, frame, x, y, possession, status in sets:
        if section not in start:
            continue
        index = np.array(frame) - first_n[section] + start[section]
        ok = (index >= start[section]) & (index < start[section] + size[section])
        index = index[ok]
        x, y = np.array(x, np.float32)[ok] + length / 2, np.array(y, np.float32)[ok] + width / 2
        if team.upper() == "BALL":
            bx[index], by[index] = x, y
            possession, status = np.array(possession)[ok], np.array(status)[ok]
            label[index] = np.where(status == 1, np.where(possession == 1, H, A), D)
            period[index] = 1 if section == "firstHalf" else 2
        elif person in slot:
            px[index, slot[person]], py[index, slot[person]] = x, y
    return dict(name="dfl_" + mid, fps=25, pitch=(length, width), px=px, py=py,
                is_home=is_home, bx=bx, by=by, label=label, halves=halves_of(period))


# ---------------------------------------------------------------
# Load everything (cached)
# ---------------------------------------------------------------
def match_names():
    names = []
    listing = os.path.join(SC_DIR, "matches.json")
    if os.path.exists(listing):
        names += ["skillcorner_" + str(m["id"]) for m in json.load(open(listing, encoding="utf-8"))]
    names += ["metrica_game1", "metrica_game2"]
    names += ["dfl_" + mid for mid in sorted(DFL_FILES)
              if os.path.exists(os.path.join(DFL_DIR, mid + "_tracking.xml"))]
    return names


def load(name):
    """One outside match in the common format, parsed once and cached."""
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        out["name"], out["fps"] = str(out["name"]), int(out["fps"])
        out["pitch"] = tuple(float(v) for v in out["pitch"])
        out["halves"] = [tuple(int(v) for v in h) for h in out["halves"]]
        return out
    if name.startswith("skillcorner_"):
        match = parse_skillcorner(name.split("_", 1)[1])
    elif name.startswith("dfl_"):
        match = parse_dfl(name.split("_", 1)[1])
    else:
        match = parse_metrica(int(name[-1]))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **{k: (np.array(v, dtype=object) if k == "halves" else v)
                                 for k, v in match.items()})
    return match


def summary(match):
    """Label sanity: share of D, possessions, median length. Real labels, other matches."""
    label, fps = match["label"], match["fps"]
    graded = label != X
    lengths = []
    for _, first, last in match["halves"]:
        seg = label[first:last + 1]
        edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        lengths += [(b - a) / fps for a, b in zip(edges[:-1], edges[1:]) if seg[a] in (H, A)]
    minutes = graded.sum() / fps / 60
    return "{:<22} {:>5.0f} min  D {:>4.1f} %  possessions {:>4}  median {:>4.1f} s  ball seen {:>4.1f} %".format(
        match["name"], minutes, 100 * np.mean(label[graded] == D), len(lengths),
        np.median(lengths) if lengths else 0, 100 * np.mean(~np.isnan(match["bx"][graded])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    for name in match_names():
        print(summary(load(name)))
        sys.stdout.flush()
